
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <locale.h>
#include <map>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "custring.cuh"
#include "unicode/unicode_flags.h"
#include "unicode/charcases.h"
#include "Timing.h"


struct timing_record
{
    double compute_size_times;
    double operation_times;
    timing_record() : compute_size_times(0.0), operation_times(0.0) {}
    void add_time(double st, double ot) { compute_size_times += st; operation_times += ot; }
};

//
void printCudaError( cudaError_t err, const char* prefix )
{
    if( err != cudaSuccess )
        fprintf(stderr,"%s: %s(%d):%s\n",prefix,cudaGetErrorName(err),(int)err,cudaGetErrorString(err));
}

//
char32_t* to_char32( const char* ca )
{
    unsigned int size = (unsigned int)strlen(ca);
    unsigned int count = custring_view::chars_in_string(ca,size);
    char32_t* rtn = new char32_t[count+1];
    char32_t* optr = rtn;
    const char* iptr = ca;
    for( int i=0; i < size; ++i )
    {
        Char oc = 0;
        unsigned int cw = custring_view::char_to_Char(iptr,oc);
        iptr += cw;
        i += cw - 1;
        *optr++ = oc;
    }
    rtn[count] = 0;
    return rtn;
}

//
static unsigned char* d_unicode_flags = 0;
unsigned char* get_unicode_flags()
{
    if( !d_unicode_flags )
    {
        // leave this out of RMM since it is never freed
        cudaMalloc(&d_unicode_flags,65536);
        cudaMemcpy(d_unicode_flags,unicode_flags,65536,cudaMemcpyHostToDevice);
    }
    return d_unicode_flags;
}

static unsigned short* d_charcases = 0;
unsigned short* get_charcases()
{
    if( !d_charcases )
    {
        // leave this out of RMM since it is never freed
        cudaMalloc(&d_charcases,65536*sizeof(unsigned short));
        cudaMemcpy(d_charcases,charcases,65536*sizeof(unsigned short),cudaMemcpyHostToDevice);
    }
    return d_charcases;
}

//
NVStringsImpl::NVStringsImpl(unsigned int count) : bufferSize(0), memoryBuffer(0), stream_id(0)
{
    pList = new rmm::device_vector<custring_view*>(count,nullptr);
}

NVStringsImpl::~NVStringsImpl()
{
    if( memoryBuffer )
        RMM_FREE(memoryBuffer,0);
    memoryBuffer = 0;
    delete pList;
    pList = 0;
    bufferSize = 0;
}

char* NVStringsImpl::createMemoryFor( size_t* d_lengths )
{
    unsigned int count = (unsigned int)pList->size();
    auto execpol = rmm::exec_policy(stream_id);
    size_t outsize = thrust::reduce(execpol->on(stream_id), d_lengths, d_lengths+count);
    if( outsize==0 )
        return 0; // all sizes are zero
    RMM_ALLOC(&memoryBuffer,outsize,0);
    bufferSize = outsize;
    return memoryBuffer;
}


void NVStringsImpl::addOpTimes( const char* op, double sizeTime, double opTime )
{
    std::string name = op;
    if( mapTimes.find(name)==mapTimes.end() )
        mapTimes[name] = timing_record();
    mapTimes[name].add_time(sizeTime,opTime);
}

void NVStringsImpl::printTimingRecords()
{
    size_t count = pList->size();
    if( !count )
        return;
    for( auto itr = mapTimes.begin(); itr != mapTimes.end(); itr++ )
    {
        std::string opname = itr->first;
        timing_record tr = itr->second;
        double otavg = (tr.operation_times / (double)count) * 1000.0;
        printf("%s: ",opname.c_str());
        if( tr.compute_size_times )
        {
            double ctavg = (tr.compute_size_times / (double)count) * 1000.0;
            printf("avg compute size time = %g; ",ctavg);
        }
        printf("avg operation time = %g\n",otavg);
    }
}

//
int NVStrings_init_from_strings(NVStringsImpl* pImpl, const char** strs, unsigned int count )
{
    cudaError_t err = cudaSuccess;
    auto execpol = rmm::exec_policy(0);
    setlocale(LC_NUMERIC, "");
    // first compute the size of each string
    size_t nbytes = 0;
    thrust::host_vector<size_t> hoffsets(count+1,0);
    //hoffsets[0] = 0; --already set by this ----^
    thrust::host_vector<size_t> hlengths(count,0);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        const char* str = strs[idx];
        size_t len = ( str ? (strlen(str)+1) : 0 );
        size_t nsz = len; // include null-terminator
        if( len > 0 )     // len=0 is null, len=1 is empty string
        {
            hlengths[idx] = len; // just the string length
            int nchars = custring_view::chars_in_string(str,(int)len-1);
            nsz = custring_view::alloc_size((int)len-1,nchars);
        }
        nsz = ALIGN_SIZE(nsz);
        nbytes += nsz;
        hoffsets[idx+1] = nbytes;
    }
    // check if they are all null
    if( nbytes==0 )
        return 0;

    // Host serialization
    unsigned int cheat = 0;//sizeof(custring_view);
    char* h_flatstrs = (char*)malloc(nbytes);
    for( unsigned int idx = 0; idx < count; ++idx )
        memcpy(h_flatstrs + hoffsets[idx] + cheat, strs[idx], hlengths[idx]);

    // copy to device memory
    char* d_flatstrs = 0;
    rmmError_t rerr = RMM_ALLOC(&d_flatstrs,nbytes,0);
    if( rerr == RMM_SUCCESS )
        err = cudaMemcpy(d_flatstrs, h_flatstrs, nbytes, cudaMemcpyHostToDevice);
    free(h_flatstrs); // no longer needed
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-sts: alloc/copy %'lu bytes\n",nbytes);
        printCudaError(err);
        return (int)err;
    }

    // copy offsets and lengths to device memory
    rmm::device_vector<size_t> offsets(hoffsets);
    rmm::device_vector<size_t> lengths(hlengths);
    size_t* d_offsets = offsets.data().get();
    size_t* d_lengths = lengths.data().get();

    // initialize custring objects in device memory
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_flatstrs, d_offsets, d_lengths, cheat, d_strings] __device__(unsigned int idx){
            size_t len = d_lengths[idx];
            if( len < 1 )
                return; // null string
            size_t offset = d_offsets[idx];
            char* ptr = d_flatstrs + offset;
            char* str = ptr + cheat;
            d_strings[idx] = custring_view::create_from(ptr,str,(int)len-1);
        });
    //
    err = cudaDeviceSynchronize();
    if( err!=cudaSuccess )
    {
        fprintf(stderr,"nvs-sts: sync=%d copy %'u strings\n",(int)err,count);
        printCudaError(err);
    }

    pImpl->setMemoryBuffer(d_flatstrs,nbytes);

#if STR_STATS
    if( err==cudaSuccess )
    {
        size_t memSize = nbytes + (count * sizeof(custring_view*));
        // lengths are +1 the size of the string so readjust
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_lengths] __device__ (unsigned int idx) {
                size_t val = d_lengths[idx];
                val = (val ? val-1 : 0);
                d_lengths[idx] = val;
            });
        //size_t max = thrust::transform_reduce(execpol->on(0),d_dstLengths,d_dstLengths+count,thrust::identity<size_t>(),0,thrust::maximum<size_t>());
        size_t max = *thrust::max_element(execpol->on(0), lengths.begin(), lengths.end());
        size_t sum = thrust::reduce(execpol->on(0), lengths.begin(), lengths.end());
        size_t avg = 0;
        if( count > 0 )
            avg =sum / count;
        printf("nvs-sts: created %'u strings in device memory(%p) = %'lu bytes\n",count,d_flatstrs,memSize);
        printf("nvs-sts: largest string is %lu bytes, average string length is %lu bytes\n",max,avg);
    }
#endif

    return (int)err;
}

// build strings from array of device pointers and sizes
int NVStrings_init_from_indexes( NVStringsImpl* pImpl, std::pair<const char*,size_t>* indexes, unsigned int count, bool bdevmem, NVStrings::sorttype stype )
{
    setlocale(LC_NUMERIC, "");
    cudaError_t err = cudaSuccess;
    auto execpol = rmm::exec_policy(0);
    thrust::pair<const char*,size_t>* d_indexes = (thrust::pair<const char*,size_t>*)indexes;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_indexes,sizeof(std::pair<const char*,size_t>)*count,0);
        cudaMemcpy(d_indexes,indexes,sizeof(std::pair<const char*,size_t>)*count,cudaMemcpyHostToDevice);
    }

    // sort the list - helps reduce divergence
    if( stype )
    {
        thrust::sort(execpol->on(0), d_indexes, d_indexes + count,
            [stype] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
                if( lhs.first==0 || rhs.first==0 )
                    return rhs.first!=0; // null < non-null
                int diff = 0;
                if( stype & NVStrings::length )
                    diff = (unsigned int)(lhs.second - rhs.second);
                if( diff==0 && (stype & NVStrings::name) )
                    diff = custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second);
                return (diff < 0);
            });
        err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            printCudaError(err,"nvs-idx: sorting");
            if( !bdevmem )
                RMM_FREE(d_indexes,0);
            return (int)err;
        }
    }

    // first get the size we need to store these strings
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_indexes, d_sizes] __device__ (unsigned int idx) {
            const char* str = d_indexes[idx].first;
            size_t bytes = d_indexes[idx].second;
            if( str )
                d_sizes[idx] = ALIGN_SIZE(custring_view::alloc_size((char*)str,(int)bytes));
        });
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        printCudaError(err,"nvs-idx: computing sizes");
        if( !bdevmem )
            RMM_FREE(d_indexes,0);
        return (int)err;
    }

    // allocate device memory
    size_t nbytes = thrust::reduce(execpol->on(0),sizes.begin(),sizes.end());
    //printf("nvs-idx: %'lu bytes\n",nbytes);
    if( nbytes==0 )
        return 0;  // done, all the strings were null
    char* d_flatdstrs = 0;
    rmmError_t rerr = RMM_ALLOC(&d_flatdstrs,nbytes,0);
    if( rerr != RMM_SUCCESS )
    {
        fprintf(stderr,"nvs-idx: RMM_ALLOC(%p,%lu)=%d\n", d_flatdstrs,nbytes,(int)rerr);
        //printCudaError(err);
        if( !bdevmem )
            RMM_FREE(d_indexes,0);
        return (int)err;
    }

    // build offsets array
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());

    // now build the strings vector
    custring_view_array d_strings = pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_indexes, d_flatdstrs, d_offsets, d_sizes, d_strings] __device__(unsigned int idx){
            // add string to internal vector array
            const char* str = d_indexes[idx].first;
            size_t bytes = d_indexes[idx].second;
            size_t offset = d_offsets[idx];
            char* ptr = d_flatdstrs + offset;
            custring_view* dstr = 0;
            if( str )
                dstr = custring_view::create_from(ptr,(char*)str,(int)bytes);
            d_strings[idx] = dstr;
            d_sizes[idx] = bytes;
        });
    //
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-idx: sync=%d copying %'u strings\n",(int)err,count);
        printCudaError(err);
    }

    pImpl->setMemoryBuffer(d_flatdstrs,nbytes);

#ifdef STR_STATS
    if( err == cudaSuccess )
    {
        size_t memSize = nbytes + (count * sizeof(custring_view*)); // flat memory plus device_vector<custring_view*>
        //size_t max = thrust::transform_reduce(execpol->on(0),d_sizes,d_sizes+count,thrust::identity<size_t>(),0,thrust::maximum<size_t>());
        size_t max = *thrust::max_element(execpol->on(0), sizes.begin(), sizes.end());
        size_t sum = thrust::reduce(execpol->on(0), sizes.begin(), sizes.end());
        size_t avg = 0;
        if( count > 0 )
            avg =sum / count;
        //
        printf("nvs-idx: created %'u strings in device memory(%p) = %'lu bytes\n",count,d_flatdstrs,memSize);
        printf("nvs-idx: largest string is %lu bytes, average string length is %lu bytes\n",max,avg);
    }
#endif
    //printf("nvs-idx: processed %'u strings\n",count);

    if( !bdevmem )
        RMM_FREE(d_indexes,0);
    return (int)err;
}

// build strings from array of device pointers and sizes
int NVStrings_init_from_offsets( NVStringsImpl* pImpl, const char* strs, int count, const int* offsets, const unsigned char* bitmask, int nulls )
{
    if( count==nulls )
        return 0; // if all are nulls then we are done
    setlocale(LC_NUMERIC, "");
    cudaError_t err = cudaSuccess;
    auto execpol = rmm::exec_policy(0);

    // first compute the size of each string
    size_t nbytes = 0;
    thrust::host_vector<size_t> hoffsets(count+1,0);
    thrust::host_vector<size_t> hlengths(count,0);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int offset = offsets[idx];
        int len = offsets[idx+1] - offset;
        const char* str = strs + offset;
        int nchars = custring_view::chars_in_string(str,len);
        int bytes = custring_view::alloc_size(len,nchars);
        if( bitmask && ((bitmask[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            bytes = 0;
        hlengths[idx] = len;
        nbytes += ALIGN_SIZE(bytes);
        hoffsets[idx+1] = nbytes;
    }
    if( nbytes==0 )
        return 0; // should not happen

    // serialize host memory into a new buffer
    unsigned int cheat = 0;//sizeof(custring_view);
    char* h_flatstrs = (char*)malloc(nbytes);
    for( unsigned int idx = 0; idx < count; ++idx )
        memcpy(h_flatstrs + hoffsets[idx] + cheat, strs + offsets[idx], hlengths[idx]);

    // copy whole thing to device memory
    char* d_flatstrs = 0;
    rmmError_t rerr = RMM_ALLOC(&d_flatstrs,nbytes,0);
    if( rerr == RMM_SUCCESS )
        err = cudaMemcpy(d_flatstrs, h_flatstrs, nbytes, cudaMemcpyHostToDevice);
    free(h_flatstrs); // no longer needed
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-ofs: alloc/copy %'lu bytes\n",nbytes);
        printCudaError(err);
        return (int)err;
    }

    // copy offsets and lengths to device memory
    rmm::device_vector<size_t> doffsets(hoffsets);
    rmm::device_vector<size_t> dlengths(hlengths);
    size_t* d_offsets = doffsets.data().get();
    size_t* d_lengths = dlengths.data().get();

    // initialize custring objects in device memory
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_flatstrs, d_offsets, d_lengths, cheat, d_strings] __device__(unsigned int idx){
            size_t len = d_lengths[idx];
            size_t offset = d_offsets[idx];
            size_t size = d_offsets[idx+1] - offset;
            if( size < 1 )
                return; // null string
            char* ptr = d_flatstrs + offset;
            char* str = ptr + cheat;
            d_strings[idx] = custring_view::create_from(ptr,str,len);
        });
    //
    err = cudaDeviceSynchronize();
    if( err!=cudaSuccess )
    {
        fprintf(stderr,"nvs-ofs: sync=%d copy %'u strings\n",(int)err,count);
        printCudaError(err);
    }

    pImpl->setMemoryBuffer(d_flatstrs,nbytes);

#if STR_STATS
    if( err==cudaSuccess )
    {
        size_t memSize = nbytes + (count * sizeof(custring_view*));
        // lengths are +1 the size of the string so readjust
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_lengths] __device__ (unsigned int idx) {
                size_t val = d_lengths[idx];
                val = (val ? val-1 : 0);
                d_lengths[idx] = val;
            });
        //size_t max = thrust::transform_reduce(execpol->on(0),d_dstLengths,d_dstLengths+count,thrust::identity<size_t>(),0,thrust::maximum<size_t>());
        size_t max = *thrust::max_element(execpol->on(0), lengths.begin(), lengths.end());
        size_t sum = thrust::reduce(execpol->on(0), lengths.begin(), lengths.end());
        size_t avg = 0;
        if( count > 0 )
            avg =sum / count;
        printf("nvs-ofs: created %'u strings in device memory(%p) = %'lu bytes\n",count,d_flatstrs,memSize);
        printf("nvs-ofs: largest string is %lu bytes, average string length is %lu bytes\n",max,avg);
    }
#endif

    return (int)err;;
}

int NVStrings_copy_strings( NVStringsImpl* pImpl, std::vector<NVStrings*>& strslist )
{
    auto execpol = rmm::exec_policy(0);
    auto pList = pImpl->pList;
    unsigned int count = (unsigned int)pList->size();
    size_t nbytes = 0;
    for( auto itr=strslist.begin(); itr!=strslist.end(); itr++ )
        nbytes += (*itr)->memsize();

    custring_view_array d_results = pList->data().get();
    char* d_buffer = 0;
    RMM_ALLOC(&d_buffer,nbytes,0);
    size_t offset = 0;
    size_t memoffset = 0;

    for( auto itr=strslist.begin(); itr!=strslist.end(); itr++ )
    {
        NVStrings* strs = *itr;
        unsigned int size = strs->size();
        size_t memsize = strs->memsize();
        if( size==0 )
            continue;
        rmm::device_vector<custring_view*> strings(size,nullptr);
        custring_view** d_strings = strings.data().get();
        strs->create_custring_index(d_strings);
        if( memsize )
        {
            // checking pointer values to find the first non-null one
            custring_view** first = thrust::min_element(execpol->on(0),d_strings,d_strings+size,
                [] __device__ (custring_view* lhs, custring_view* rhs) {
                    return (lhs && rhs) ? (lhs < rhs) : rhs==0;
                });
            char* baseaddr = 0;
            cudaError_t err = cudaMemcpy(&baseaddr,first,sizeof(custring_view*),cudaMemcpyDeviceToHost);
            if( err!=cudaSuccess )
                fprintf(stderr, "copy-strings: cudaMemcpy(%p,%p,%d)=%d\n",&baseaddr,first,(int)sizeof(custring_view*),(int)err);
            // copy string memory
            char* buffer = d_buffer + memoffset;
            err = cudaMemcpy((void*)buffer,(void*)baseaddr,memsize,cudaMemcpyDeviceToDevice);
            if( err!=cudaSuccess )
                fprintf(stderr, "copy-strings: cudaMemcpy(%p,%p,%ld)=%d\n",buffer,baseaddr,memsize,(int)err);
            // adjust pointers
            custring_view_array results = d_results + offset;
            thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), size,
                [buffer, baseaddr, d_strings, results] __device__(unsigned int idx){
                    char* dstr = (char*)d_strings[idx];
                    if( !dstr )
                        return;
                    size_t diff = dstr - baseaddr;
                    char* newaddr = buffer + diff;
                    results[idx] = (custring_view*)newaddr;
            });
        }
        offset += size;
        memoffset += memsize;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if( err!=cudaSuccess )
        printCudaError(err,"nvs-cs");
    pImpl->setMemoryBuffer(d_buffer,nbytes);
    return count;
}
