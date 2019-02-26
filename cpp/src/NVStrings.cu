
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <locale.h>
#include <map>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "custring_view.cuh"
#include "custring.cuh"
#include "util.h"
#include "regex/regex.cuh"
#include "unicode/is_flags.h"
#include "unicode/charcases.h"
#include "Timing.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif
//
typedef custring_view** custring_view_array;

#define ALIGN_SIZE(v)  (((v+7)/8)*8)


struct timing_record
{
    double compute_size_times;
    double operation_times;
    timing_record() : compute_size_times(0.0), operation_times(0.0) {}
    void add_time(double st, double ot) { compute_size_times += st; operation_times += ot; }
};

//
static void printCudaError( cudaError_t err, const char* prefix="\t" )
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

// defined in util.cu
__host__ __device__ unsigned int u2u8( unsigned int unchr );
__host__ __device__ unsigned int u82u( unsigned int utf8 );

unsigned char* d_unicode_flags = 0;
unsigned char* get_unicode_flags()
{
    if( !d_unicode_flags )
    {
        cudaMalloc(&d_unicode_flags,65536); // leave this out of RMM since it is never freed
        cudaMemcpy(d_unicode_flags,unicode_flags,65536,cudaMemcpyHostToDevice);
    }
    return d_unicode_flags;
}

unsigned short* d_charcases = 0;
unsigned short* get_charcases()
{
    if( !d_charcases )
    {
        cudaMalloc(&d_charcases,65536*sizeof(unsigned short)); // leave this out of RMM since it is never freed
        cudaMemcpy(d_charcases,charcases,65536*sizeof(unsigned short),cudaMemcpyHostToDevice);
    }
    return d_charcases;
}

//
class NVStringsImpl
{
public:
    // this holds the strings in device memory
    // so operations can be performed on them through python calls
    rmm::device_vector<custring_view*>* pList;
    char* memoryBuffer;
    size_t bufferSize; // size of memoryBuffer only
    std::map<std::string,timing_record> mapTimes;
    cudaStream_t stream_id;

    //
    NVStringsImpl(unsigned int count) : bufferSize(0), memoryBuffer(0), stream_id(0)
    {
        pList = new rmm::device_vector<custring_view*>(count,nullptr);
    }

    ~NVStringsImpl()
    {
        if( memoryBuffer )
            RMM_FREE(memoryBuffer,0);
        memoryBuffer = 0;
        delete pList;
        pList = 0;
        bufferSize = 0;
    }

    char* createMemoryFor( size_t* d_lengths )
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

    inline custring_view_array getStringsPtr()
    {
        return pList->data().get();
    }

    inline char* getMemoryPtr()
    {
        return memoryBuffer;
    }

    inline size_t getMemorySize()
    {
        return bufferSize;
    }

    inline cudaStream_t getStream()
    {
        return stream_id;
    }

    inline void setMemoryBuffer( void* ptr, size_t memSize )
    {
        memoryBuffer = (char*)ptr;
        bufferSize = memSize;
    }

    void addOpTimes( const char* op, double sizeTime, double opTime )
    {
        std::string name = op;
        if( mapTimes.find(name)==mapTimes.end() )
            mapTimes[name] = timing_record();
        mapTimes[name].add_time(sizeTime,opTime);
    }

    void printTimingRecords()
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
};

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
    //printf("nvs: indexes=%p, count=%'lu\n",d_indexes,count);

    // sort the list - helps reduce divergence
    if( stype )
    {
        thrust::sort(execpol->on(0), d_indexes, d_indexes + count,
            [stype] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
                //if( lhs.first==0 || rhs.first==0 )
                //    return lhs.first==0; // non-null > null
                //return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
                //return lhs.second > rhs.second;
                bool cmp = false;
                if( lhs.first==0 || rhs.first==0 )
                    cmp = lhs.first==0; // null < non-null
                else
                {   // allow sorting by name and length
                    int diff = 0;
                    if( stype & NVStrings::length )
                        diff = (unsigned int)(rhs.second - lhs.second);
                    if( diff==0 && (stype & NVStrings::name) )
                        diff = custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) > 0;
                    cmp = (diff > 0);
                }
                return cmp;
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
        if( size==0 || memsize==0 )
            continue;
        rmm::device_vector<custring_view*> strings(size,nullptr);
        custring_view** d_strings = strings.data().get();
        strs->create_custring_index(d_strings);
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
        offset += size;
        memoffset += memsize;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if( err!=cudaSuccess )
        printCudaError(err,"nvs-cs");
    pImpl->setMemoryBuffer(d_buffer,nbytes);
    return count;
}

// ctor and dtor are private to control the memory allocation in a single shared-object module
NVStrings::NVStrings(unsigned int count)
{
    pImpl = new NVStringsImpl(count);
}

NVStrings::NVStrings()
{
    pImpl = new NVStringsImpl(0);
}

NVStrings::NVStrings(const NVStrings& strsIn)
{
    NVStrings& strs = (NVStrings&)strsIn;
    unsigned int count = strs.size();
    pImpl = new NVStringsImpl(count);
    if( count )
    {
        std::vector<NVStrings*> strslist;
        strslist.push_back(&strs);
        NVStrings_copy_strings(pImpl,strslist);
    }
}
    
NVStrings& NVStrings::operator=(const NVStrings& strsIn)
{
    delete pImpl;
    NVStrings& strs = (NVStrings&)strsIn;
    unsigned int count = strs.size();
    pImpl = new NVStringsImpl(count);
    if( count )
    {
        std::vector<NVStrings*> strslist;
        strslist.push_back(&strs);
        NVStrings_copy_strings(pImpl,strslist);
    }
    return *this;
}

NVStrings::~NVStrings()
{
    delete pImpl;
}

NVStrings* NVStrings::create_from_array( const char** strs, unsigned int count)
{
    NVStrings* rtn = new NVStrings(count);
    if( count )
        NVStrings_init_from_strings(rtn->pImpl,strs,count);
    return rtn;
}

NVStrings* NVStrings::create_from_index(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem, sorttype stype)
{
    NVStrings* rtn = new NVStrings(count);
    if( count )
        NVStrings_init_from_indexes(rtn->pImpl,strs,count,devmem,stype);
    return rtn;
}

NVStrings* NVStrings::create_from_offsets(const char* strs, int count, const int* offsets, const unsigned char* nullbitmask, int nulls)
{
    NVStrings* rtn = new NVStrings(count);
    if( count )
        NVStrings_init_from_offsets(rtn->pImpl,strs,count,offsets,nullbitmask,nulls);
    return rtn;
}

NVStrings* NVStrings::create_from_strings( std::vector<NVStrings*> strs )
{
    unsigned int count = 0;
    for( auto itr=strs.begin(); itr!=strs.end(); itr++ )
        count += (*itr)->size();
    NVStrings* rtn = new NVStrings(count);
    if( count )
        NVStrings_copy_strings(rtn->pImpl,strs);
    return rtn;
}

void NVStrings::destroy(NVStrings* inst)
{
    delete inst;
}

size_t NVStrings::memsize() const
{
    return pImpl->bufferSize;
}

void NVStrings::printTimingRecords()
{
    pImpl->printTimingRecords();
}

NVStrings* NVStrings::copy()
{
    unsigned int count = size();
    NVStrings* rtn = new NVStrings(count);
    if( count )
    {
        std::vector<NVStrings*> strslist;
        strslist.push_back(this);
        NVStrings_copy_strings(rtn->pImpl,strslist);
    }
    return rtn;
}

//
void NVStrings::print( int start, int end, int maxwidth, const char* delimiter )
{
    unsigned int count = size();
    if( end < 0 || end > count )
        end = count;
    if( start < 0 )
        start = 0;
    if( start >= end )
        return;
    count = end - start;
    //
    auto execpol = rmm::exec_policy(0);
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> lens(count,0);
    size_t* d_lens = lens.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, maxwidth, d_lens] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = dstr->size();
            if( maxwidth > 0 )
                len = dstr->byte_offset_for(maxwidth);
            d_lens[idx-start] = len +1; // include null-terminator;
        });

    // allocate large device buffer to hold all the strings
    size_t msize = thrust::reduce(execpol->on(0),lens.begin(),lens.end());
    if( msize==0 )
    {
        printf("all %d strings are null\n",count);
        return;
    }
    char* d_buffer = 0;
    RMM_ALLOC(&d_buffer,msize,0);
    // convert lengths to offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // copy strings into single buffer
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, maxwidth, d_offsets, d_lens, d_buffer] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            size_t offset = d_offsets[idx-start];
            char* optr = d_buffer + offset;
            if( dstr )
            {
                dstr->copy(optr,maxwidth);
                size_t len = d_lens[idx-start];
                //memcpy(optr,dstr->data(),len-1);
                *(optr+len-1) = 0;
            }
        });
    //
    cudaDeviceSynchronize();
    // copy strings to host
    char* h_buffer = new char[msize];
    cudaMemcpy(h_buffer, d_buffer, msize, cudaMemcpyDeviceToHost);
    RMM_FREE(d_buffer,0);
    // print strings to stdout
    thrust::host_vector<custring_view*> h_strings(*(pImpl->pList)); // just for checking nulls
    thrust::host_vector<size_t> h_lens(lens);
    char* hstr = h_buffer;
    for( int idx=0; idx < count; ++idx )
    {
        printf("%d:",idx);
        if( !h_strings[idx] )
            printf("<null>");
        else
            printf("[%s]",hstr);
        printf("%s",delimiter);
        hstr += h_lens[idx];
    }
    delete h_buffer;
}

//
int NVStrings::to_host(char** list, int start, int end)
{
    unsigned int count = size();
    if( end < 0 || end > count )
        end = count;
    if( start >= end )
        return 0;
    count = end - start;

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<size_t> lens(count,0);
    size_t* d_lens = lens.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, d_lens] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_lens[idx-start] = dstr->size()+1; // include space for null terminator
        });

    cudaError_t err = cudaSuccess;
    size_t msize = thrust::reduce(execpol->on(0),lens.begin(),lens.end());
    if( msize==0 )
    {   // every string is null
        memset(list,0,count*sizeof(char*));
        return 0;
    }

    char* d_buffer = 0;
    rmmError_t rerr = RMM_ALLOC(&d_buffer,msize,0);
    if( rerr != RMM_SUCCESS )
    {
        fprintf(stderr,"nvs-to_host: RM_ALLOC(%p,%lu)=%d\n", d_buffer,msize,(int)rerr);
        //printCudaError(err);
        return (int)err;
    }
    // allocate large device buffer to hold all the strings
    // convert lengths to offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // copy strings into single buffer
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, d_offsets, d_buffer] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            size_t offset = d_offsets[idx-start];
            char* optr = d_buffer + offset;
            if( dstr )
            {
                int len = dstr->size();
                memcpy(optr,dstr->data(),len);
                *(optr + len) = 0;
            }
        });
    //
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        printCudaError(err,"nvs-to_host: copying strings device to device");
        RMM_FREE(d_buffer,0);
        return (int)err;
    }

    // copy strings to host
    char* h_buffer = new char[msize];
    err = cudaMemcpy(h_buffer, d_buffer, msize, cudaMemcpyDeviceToHost);
    RMM_FREE(d_buffer,0);
    if( err != cudaSuccess )
    {
        printCudaError(err, "nvs-to_host: copying strings device to host");
        delete h_buffer;
        return (int)err;
    }

    // Host deserialization
    thrust::host_vector<custring_view*> h_strings(*(pImpl->pList)); // just for checking nulls
    thrust::host_vector<size_t> h_offsets(offsets);
    h_offsets.push_back(msize); // include size as last offset
    for( unsigned int idx=0; idx < count; ++idx )
    {
        if( h_strings[idx]==0 )
        {
            list[idx] = 0;
            continue;
        }
        size_t offset = h_offsets[idx];
        size_t len = h_offsets[idx+1] - offset;
        const char* p_data = h_buffer + offset;
        char* h_data = new char[len]; // make memory on the host
        h_data[len-1] = 0; // null terminate for the caller
        memcpy(h_data, p_data, len-1);
        list[idx] = h_data;
    }
    delete h_buffer;
    return 0;
}

// build a string-index from this instances strings
int NVStrings::create_index(std::pair<const char*,size_t>* strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_indexes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                d_indexes[idx].first = (const char*)dstr->data();
                d_indexes[idx].second = (size_t)dstr->size();
            }
            else
            {
                d_indexes[idx].first = 0;
                d_indexes[idx].second = 0;
            }
        });

    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        printCudaError(err,"nvs-create_index");
        return (int)err;
    }
    if( bdevmem )
        cudaMemcpy( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToDevice );
    else
        cudaMemcpy( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToHost );
    return 0;
}

//
int NVStrings::create_custring_index( custring_view** strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;
    custring_view_array d_strings = pImpl->getStringsPtr();
    if( bdevmem )
        cudaMemcpy( strs, d_strings, count * sizeof(custring_view*), cudaMemcpyDeviceToDevice );
    else
        cudaMemcpy( strs, d_strings, count * sizeof(custring_view*), cudaMemcpyDeviceToHost );
    return 0;
}

// copy strings into memory provided
int NVStrings::create_offsets( char* strs, int* offsets, unsigned char* nullbitmask, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;
    if( strs==0 || offsets==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // first compute offsets/nullbitmask
    int* d_offsets = offsets;
    unsigned char* d_nulls = nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_offsets,(count+1)*sizeof(int),0);
        if( nullbitmask )
        {
            RMM_ALLOC(&d_nulls,((count+7)/8)*sizeof(unsigned char),0);
            cudaMemset(d_nulls,0,((count+7)/8));
        }
    }
    //
    rmm::device_vector<int> sizes(count+1,0);
    int* d_sizes = sizes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_sizes[idx] = (int)dstr->size();
        });
    // ^^^-- these two for-each-n's can likely be combined --vvv
    if( d_nulls )
    {
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), (count+7)/8,
            [d_strings, count, d_nulls] __device__(unsigned int idx){
                unsigned int ndx = idx * 8;
                unsigned char nb = 0;
                for( int i=0; i<8; ++i )
                {
                    nb = nb >> 1;
                    if( ndx+i < count )
                    {
                        custring_view* dstr = d_strings[ndx+i];
                        if( dstr )
                            nb |= 128;
                    }
                }
                d_nulls[idx] = nb;
            });
    }
    //
    thrust::exclusive_scan(execpol->on(0),d_sizes, d_sizes+(count+1), d_offsets);
    // compute/allocate of memory
    size_t totalbytes = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count);
    char* d_strs = strs;
    if( !bdevmem )
        RMM_ALLOC(&d_strs,totalbytes,0);
    // shuffle strings into memory
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strs, d_offsets] __device__(unsigned int idx){
            char* buffer = d_strs + d_offsets[idx];
            custring_view* dstr = d_strings[idx];
            if( dstr )
                memcpy(buffer,dstr->data(),dstr->size());
        });
    // copy memory to parameters (if necessary)
    if( !bdevmem )
    {
        cudaMemcpy(offsets,d_offsets,(count+1)*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(strs,d_strs,totalbytes,cudaMemcpyDeviceToHost);
        if( nullbitmask )
            cudaMemcpy(nullbitmask,d_nulls,((count+7)/8)*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    }
    return 0;
}

// create a new instance containing only the strings at the specified positions
// position values can be in any order and can even be repeated
NVStrings* NVStrings::gather( int* pos, unsigned int elems, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || elems==0 || pos==0 )
        return new NVStrings(0);

    auto execpol = rmm::exec_policy(0);
    int* d_pos = pos;
    if( !bdevmem )
    {   // copy indexes to device memory
        RMM_ALLOC(&d_pos,elems*sizeof(int),0);
        cudaMemcpy(d_pos,pos,elems*sizeof(int),cudaMemcpyHostToDevice);
    }
    // get individual sizes
    rmm::device_vector<size_t> sizes(elems,0);
    size_t* d_sizes = sizes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elems,
        [d_strings, d_pos, count, d_sizes] __device__(unsigned int idx){
            int pos = d_pos[idx];
            if( (pos < 0) || (pos >= count) )
                return;
            custring_view* dstr = d_strings[pos];
            if( dstr )
                d_sizes[idx] = ALIGN_SIZE(dstr->alloc_size());
        });
    // create output object
    NVStrings* rtn = new NVStrings(elems);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer ) // if all values are not null
    {
        // create offsets
        rmm::device_vector<size_t> offsets(elems,0);
        thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
        // copy strings
        custring_view_array d_results = rtn->pImpl->getStringsPtr();
        size_t* d_offsets = offsets.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elems,
            [d_strings, d_buffer, d_offsets, d_pos, count, d_results] __device__(unsigned int idx){
                int pos = d_pos[idx];
                if( (pos < 0) || (pos >= count) )
                    return;
                custring_view* dstr = d_strings[pos];
                if( !dstr )
                    return;
                char* buffer = d_buffer + d_offsets[idx];
                d_results[idx] = custring_view::create_from(buffer,*dstr);
            });
        //
        printCudaError(cudaDeviceSynchronize(),"nvs-gather");
    }
    if( !bdevmem )
        RMM_FREE(d_pos,0);
    return rtn;
}

NVStrings* NVStrings::sublist( unsigned int start, unsigned int end, unsigned int step )
{
    unsigned int count = size();
    if( end > count )
        end = count;
    if( start >= end )
        return new NVStrings(0);
    if( step==0 )
        step = 1;
    unsigned int elems = (end - start + step -1)/step;
    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<unsigned int> indexes(elems);
    thrust::sequence(execpol->on(0),indexes.begin(),indexes.end(),start,step);
    return gather((int*)indexes.data().get(),elems,true);
}

// remove the specified strings and return a new instance
NVStrings* NVStrings::remove_strings( unsigned int* pos, unsigned int elems, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || elems==0 || pos==0 )
        return 0; // return copy of ourselves?

    auto execpol = rmm::exec_policy(0);
    unsigned int* dpos = pos;
    if( !bdevmem )
    {
        RMM_ALLOC(&dpos,elems*sizeof(unsigned int),0);
        cudaMemcpy(dpos,pos,elems*sizeof(unsigned int),cudaMemcpyHostToDevice);
    }
    // sort the position values
    thrust::sort(execpol->on(0),dpos,dpos+elems,thrust::greater<unsigned int>());
    // also should remove duplicates
    unsigned int* nend = thrust::unique(execpol->on(0),dpos,dpos+elems,thrust::equal_to<unsigned int>());
    elems = (unsigned int)(nend - dpos);
    if( count < elems )
    {
        if( !bdevmem )
            RMM_FREE(dpos,0);
        fprintf(stderr,"nvs.remove_strings: more positions (%u) specified than the number of strings (%u)\n",elems,count);
        return 0;
    }

    // build array to hold positions which are not to be removed by marking deleted positions with -1
    rmm::device_vector<int> dnpos(count);
    thrust::sequence(execpol->on(0),dnpos.begin(),dnpos.end());
    int* d_npos = dnpos.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elems,
        [dpos, d_npos, count] __device__ (unsigned int idx) {
            unsigned int pos = dpos[idx];
            if( pos < count )
                d_npos[pos] = -1;
        });

    // now remove the positions marked with -1
    int* dend = thrust::remove_if(execpol->on(0),d_npos,d_npos+count,[] __device__ (int val) { return val < 0; });
    unsigned int newCount = (unsigned int)(dend-d_npos);
    // gather string pointers based on indexes in dnpos (new-positions)
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<custring_view*> newList(newCount,nullptr);           // newList will hold
    custring_view_array d_newList = newList.data().get();                      // all the remaining
    thrust::gather(execpol->on(0),d_npos,d_npos+newCount,d_strings,d_newList); // strings ptrs

    // get individual sizes for the new strings list
    rmm::device_vector<size_t> sizes(newCount,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), newCount,
        [d_newList, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_newList[idx];
            if( dstr )
                d_sizes[idx] = ALIGN_SIZE(dstr->alloc_size());
        });
    // create output object
    NVStrings* rtn = new NVStrings(newCount);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
    {
        if( !bdevmem )
            RMM_FREE(dpos,0);
        return rtn;
    }
    // create offsets
    rmm::device_vector<size_t> offsets(newCount,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // finally, copy the strings
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), newCount,
        [d_newList, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_newList[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            d_results[idx] = custring_view::create_from(buffer,*dstr);
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-remove_strings");
    if( !bdevmem )
        RMM_FREE(dpos,0);
    return rtn;
}

// fills in a bitarray with 0 for null values and 1 for non-null values
// if emptyIsNull=true, empty strings will have bit values of 0 as well
unsigned int NVStrings::set_null_bitarray( unsigned char* bitarray, bool emptyIsNull, bool devmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    unsigned int size = (count + 7)/8; // round up to byte align
    unsigned char* d_bitarray = bitarray;
    if( !devmem )
        RMM_ALLOC(&d_bitarray,size,0);

    // count nulls in range for return value
    custring_view** d_strings = pImpl->getStringsPtr();
    unsigned int ncount = thrust::count_if(execpol->on(0), d_strings, d_strings + count,
       [emptyIsNull] __device__ (custring_view*& dstr) { return (dstr==0) || (emptyIsNull && !dstr->size()); });

    // fill in the bitarray
    // the bitmask is in arrow format which means for each byte
    // the null indicator is in bit position right-to-left: 76543210
    // logic sets the high-bit and shifts to the right
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), size,
        [d_strings, count, emptyIsNull, d_bitarray] __device__(unsigned int byteIdx){
            unsigned char byte = 0; // set one byte per thread -- init to all nulls
            for( unsigned int i=0; i < 8; ++i )
            {
                unsigned int idx = i + (byteIdx*8);  // compute d_strings index
                byte = byte >> 1;                    // shift until we are done
                if( idx < count )                    // check boundary
                {
                    custring_view* dstr = d_strings[idx];
                    if( dstr && (!emptyIsNull || dstr->size()) )
                        byte |= 128;                 // string is not null, set high bit
                }
            }
            d_bitarray[byteIdx] = byte;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-set_null_bitarray(%p,%d,%d) size=%u\n",bitarray,(int)emptyIsNull,(int)devmem,count);
        printCudaError(err);
    }
    //
    if( !devmem )
    {
        cudaMemcpy(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost);
        RMM_FREE(d_bitarray,0);
    }
    return ncount;
}

// set int array with position of null strings
unsigned int NVStrings::get_nulls( unsigned int* array, bool emptyIsNull, bool devmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<int> darray(count,-1);
    int* d_array = darray.data().get();

    custring_view** d_strings = pImpl->getStringsPtr();
    //unsigned int ncount = thrust::count_if(execpol->on(0), d_strings, d_strings + count,
    //   [emptyIsNull] __device__ (custring_view*& dstr) { return (dstr==0) || (emptyIsNull && !dstr->size()); });

    // fill in the array
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, emptyIsNull, d_array] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr && (!emptyIsNull || dstr->size()) )
                d_array[idx] = -1; // not null
            else
                d_array[idx] = idx;  // null
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-get_nulls(%p,%d,%d) size=%u\n",array,(int)emptyIsNull,(int)devmem,count);
        printCudaError(err);
    }
    // compact out the negative values
    int* newend = thrust::remove_if(execpol->on(0), d_array, d_array + count, [] __device__ (int val) {return val<0;});
    unsigned int ncount = (unsigned int)(newend - d_array);

    //
    if( array )
    {
        if( devmem )
            cudaMemcpy(array,d_array,sizeof(int)*ncount,cudaMemcpyDeviceToDevice);
        else
            cudaMemcpy(array,d_array,sizeof(int)*ncount,cudaMemcpyDeviceToHost);
    }
    return ncount;
}


// number of strings in this instance
unsigned int NVStrings::size() const
{
    return (unsigned int)pImpl->pList->size();
}

// this will return the number of characters for each string
unsigned int NVStrings::len(int* lengths, bool todevice)
{
    unsigned int count = size();
    if( lengths==0 || count==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = lengths;
    if( !todevice )
        RMM_ALLOC(&d_rtn,sizeof(int)*count,0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->chars_count();
            else
                d_rtn[idx] = -1;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-len");
    double et = GetTime();
    pImpl->addOpTimes("len",0.0,(et-st));
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(lengths,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

// this will return the number of bytes for each string
size_t NVStrings::byte_count(int* lengths, bool todevice)
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = lengths;
    if( !lengths )
        todevice = false; // makes sure we free correctly
    if( !todevice )
        RMM_ALLOC(&d_rtn,sizeof(int)*count,0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->size();
            else
                d_rtn[idx] = -1;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-bytes");
    double et = GetTime();
    pImpl->addOpTimes("byte_count",0.0,(et-st));
    size_t size = thrust::reduce(execpol->on(0), d_rtn, d_rtn+count, (size_t)0,
         []__device__(int lhs, int rhs) {
            if( lhs < 0 )
                lhs = 0;
            if( rhs < 0 )
                rhs = 0;
            return lhs + rhs;
         });
    if( !todevice )
    {   // copy result back to host
        if( lengths )
            cudaMemcpy(lengths,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return size;
}

//
unsigned int NVStrings::compare( const char* str, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || results==0 || count==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str);
    if( bytes==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,sizeof(int)*count,0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->compare(d_str,bytes);
            else
                d_rtn[idx] = (d_str ? -1: 0);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-compare(%s,%p,%d)\n",str,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("compare",0.0,(et-st));

    RMM_FREE(d_str,0);
    //
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::hash(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->hash();
            else
                d_rtn[idx] = 0;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-hash");
    double et = GetTime();
    pImpl->addOpTimes("hash",0.0,(et-st));

    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::stoi(int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stoi();
            else
                d_rtn[idx] = 0;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-stoi");
    double et = GetTime();
    pImpl->addOpTimes("stoi",0.0,(et-st));

    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::stof(float* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    float* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(float),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stof();
            else
                d_rtn[idx] = (float)0;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-stof");
    double et = GetTime();
    pImpl->addOpTimes("stof",0.0,(et-st));
    //
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::htoi(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr || dstr->empty() )
            {
                d_rtn[idx] = 0;
                return;
            }
            long result = 0, base = 1;
            const char* str = dstr->data();
            int len = dstr->size()-1;
            for( int i=len; i >= 0; --i )
            {
                char ch = str[i];
                if( ch >= '0' && ch <= '9' )
                {
                    result += (long)(ch-48) * base;
                    base *= 16;
                }
                else if( ch >= 'A' && ch <= 'Z' )
                {
                    result += (long)(ch-55) * base;
                    base *= 16;
                }
                else if( ch >= 'a' && ch <= 'z' )
                {
                    result += (long)(ch-87) * base;
                    base *= 16;
                }
            }
            d_rtn[idx] = (unsigned int)result;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-htoi");
    double et = GetTime();
    pImpl->addOpTimes("htoi",0.0,(et-st));

    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
NVStrings* NVStrings::cat( NVStrings* others, const char* separator, const char* narep )
{
    if( others==0 )
        return 0; // return a copy of ourselves?
    unsigned int count = size();
    if( others->size() != count )
        return 0; // this is not allowed: use assert?

    auto execpol = rmm::exec_policy(0);
    unsigned int seplen = 0;
    if( separator )
        seplen = (unsigned int)strlen(separator);
    char* d_sep = 0;
    if( seplen )
    {
        RMM_ALLOC(&d_sep,seplen,0);
        cudaMemcpy(d_sep,separator,seplen,cudaMemcpyHostToDevice);
    }
    unsigned int narlen = 0;
    if( narep )
        narlen = (unsigned int)strlen(narep);
    char* d_narep = 0;
    if( narlen )
    {
        RMM_ALLOC(&d_narep,narlen,0);
        cudaMemcpy(d_narep,narep,narlen,cudaMemcpyHostToDevice);
    }

    custring_view_array d_strings = pImpl->getStringsPtr();
    custring_view_array d_others = others->pImpl->getStringsPtr();

    // first compute the size of the output
    double st1 = GetTime();
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_others, d_sep, seplen, d_narep, narlen, d_sizes] __device__(unsigned int idx){
            custring_view* dstr1 = d_strings[idx];
            custring_view* dstr2 = d_others[idx];
            if( (!dstr1 || !dstr2) && !d_narep )
                return; // null case
            int nchars = 0;
            int bytes = 0;
            // left side
            if( dstr1 )
            {
                nchars = dstr1->chars_count();
                bytes = dstr1->size();
            }
            else if( d_narep )
            {
                nchars = custring_view::chars_in_string(d_narep,narlen);
                bytes = narlen;
            }
            // separator
            if( d_sep )
            {
                nchars += custring_view::chars_in_string(d_sep,seplen);
                bytes += seplen;
            }
            // right side
            if( dstr2 )
            {
                nchars += dstr2->chars_count();
                bytes += dstr2->size();
            }
            else if( d_narep )
            {
                nchars += custring_view::chars_in_string(d_narep,narlen);
                bytes += narlen;
            }
            int size = custring_view::alloc_size(bytes,nchars);
            //printf("cat:%lu:size=%d\n",idx,size);
            size = ALIGN_SIZE(size);
            d_sizes[idx] = size;
        });

    // allocate the memory for the output
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
    {
        if( d_sep )
            RMM_FREE(d_sep,0);
        if( d_narep )
            RMM_FREE(d_narep,0);
        return rtn;
    }
    double et1 = GetTime();
    cudaMemset(d_buffer,0,rtn->pImpl->getMemorySize());
    // compute the offset
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_others, d_sep, seplen, d_narep, narlen, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dstr1 = d_strings[idx];
            custring_view* dstr2 = d_others[idx];
            if( (!dstr1 || !dstr2) && !d_narep )
                return; // if null, an no null rep, done
            custring_view* dout = custring_view::create_from(buffer,0,0); // init empty string
            if( dstr1 )
                dout->append(*dstr1);        // add left side
            else if( d_narep )               // (or null rep)
                dout->append(d_narep,narlen);
            if( d_sep )
                dout->append(d_sep,seplen);  // add separator
            if( dstr2 )
                dout->append(*dstr2);        // add right side
            else if( d_narep )               // (or null rep)
                dout->append(d_narep,narlen);
            //printf("cat:%lu:[]=%d\n",idx,dout->size());
            d_results[idx] = dout;
    });
    printCudaError(cudaDeviceSynchronize(),"nvs-cat: combining strings");
    double et2 = GetTime();
    pImpl->addOpTimes("cat",(et1-st1),(et2-st2));

    if( d_sep )
        RMM_FREE(d_sep,0);
    if( d_narep )
        RMM_FREE(d_narep,0);
    return rtn;
}

//
int NVStrings::split( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->split_size(d_delimiter,bytes,0,maxsplit);
        });
    //cudaDeviceSynchronize();

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = d_sizes + d_offsets[idx];
            int dcount = d_counts[idx];
            d_totals[idx] = dstr->split_size(d_delimiter,bytes,dsizes,dcount);
            //printf("[%s]=%d split bytes\n",dstr->data(),d_totals[idx]);
        });
    //
    cudaDeviceSynchronize();

    // now build an array of custring_views* arrays for each value
    int totalNewStrings = 0;
    thrust::host_vector<int> h_counts(counts);
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int splitCount = h_counts[idx];
        if( splitCount==0 )
        {
            results.push_back(0);
            continue;
        }

        NVStrings* splitResult = new NVStrings(splitCount);
        results.push_back(splitResult);
        h_splits[idx] = splitResult->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,totalSize,0);
        splitResult->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;

        totalNewStrings += splitCount;
    }

    //
    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the splits and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int d_count = d_counts[idx];
            if( d_count < 1 )
                return;
            char* buffer = (char*)d_buffers[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            custring_view_array d_strs = d_splits[idx];
            for( int i=0; i < d_count; ++i )
            {
                int size = ALIGN_SIZE(dsizes[i]);
                d_strs[i] = (custring_view*)buffer;
                buffer += size;
            }
            dstr->split(d_delimiter,bytes,d_count,d_strs);
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-split");
    RMM_FREE(d_delimiter,0);
    //
    return totalNewStrings;
}

//
int NVStrings::rsplit( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->rsplit_size(d_delimiter,bytes,0,maxsplit);
        });

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int dcount = d_counts[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            d_totals[idx] = dstr->rsplit_size(d_delimiter,bytes,dsizes,dcount);
        });

    cudaDeviceSynchronize();

    // now build an array of custring_views* arrays for each value
    int totalNewStrings = 0;
    thrust::host_vector<int> h_counts(counts);
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    for( int idx=0; idx < count; ++idx )
    {
        int splitCount = h_counts[idx];
        if( splitCount==0 )
        {
            results.push_back(0);
            continue;
        }
        NVStrings* splitResult = new NVStrings(splitCount);
        results.push_back(splitResult);
        h_splits[idx] = splitResult->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,totalSize,0);
        splitResult->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;

        totalNewStrings += splitCount;
    }

    //
    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the splits and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int d_count = d_counts[idx];
            if( d_count < 1 )
                return;
            char* buffer = (char*)d_buffers[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            custring_view_array d_strs = d_splits[idx];
            for( int i=0; i < d_count; ++i )
            {
                d_strs[i] = (custring_view*)buffer;
                int size = ALIGN_SIZE(dsizes[i]);
                buffer += size;
                //printf("%d:%d=%d\n",(int)idx,i,size);
            }
            dstr->rsplit(d_delimiter,bytes,d_count,d_strs);
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-rsplit");
    RMM_FREE(d_delimiter,0);

    return totalNewStrings;
}

// This will create new columns by splitting the array of strings vertically.
// All the first tokens go in the first column, all the second tokens go in the second column, etc.
unsigned int NVStrings::split_column( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    int bytes = (int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);

    // need to count how many output strings per string
    //double st = GetTime();
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->split_size(d_delimiter,bytes,0,maxsplit);
        });
    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    //double et = GetTime();
    //printf("%d columns: %gs\n",columnsCount,(et-st));

    // create each column
    for( int col=0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        //st = GetTime();
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, d_delimiter, bytes, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = 0;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr )
                    return;
                // dcount already accounts for the maxsplit value
                int dcount = d_counts[idx];
                if( col >= dcount )
                    return; // passed the end for this string
                // skip delimiters until we reach this column
                int spos = 0, nchars = dstr->chars_count();
                int epos = nchars;
                for( int c=0; c < (dcount-1); ++c )
                {
                    epos = dstr->find(d_delimiter,bytes,spos);
                    if( epos < 0 )
                    {
                        epos = nchars;
                        break;
                    }
                    if( c==col )  // found our column
                        break;
                    spos = epos + bytes;
                    epos = nchars;
                }
                // this will be the string for this column
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
            });
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-split_column(%s,%d), col=%d\n",delimiter,maxsplit,col);
            printCudaError(err);
        }
        //et = GetTime();
        //printf("%3d split-index = %gs, ",col,(et-st));
        //
        //st = GetTime();
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
        //et = GetTime();
        //printf("create = %gs\n",(et-st));
    }
    //
    RMM_FREE(d_delimiter,0);
    return (unsigned int)columnsCount;
}

// split-from-the-right version of split_column
unsigned int NVStrings::rsplit_column( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    int bytes = (int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->rsplit_size(d_delimiter,bytes,0,maxsplit);
        });

    //int columnsCount = thrust::transform_reduce(execpol->on(0),d_counts,d_counts+count,thrust::identity<int>(),0,thrust::maximum<int>());
    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    //printf("columns=%d\n",columnsCount);

    // create each column
    for( int col = 0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, columnsCount, d_delimiter, bytes, d_counts, d_indexes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            d_indexes[idx].first = 0;   // initialize to
            d_indexes[idx].second = 0;  // null string
            if( !dstr )
                return;
            // dcount already accounts for the maxsplit value
            int dcount = d_counts[idx];
            if( col >= dcount )
                return; // passed the end for this string
            // skip delimiters until we reach this column
            int spos = 0, nchars = dstr->chars_count();
            int epos = nchars;
            for( int c=(dcount-1); c > 0; --c )
            {
                spos = dstr->rfind(d_delimiter,bytes,0,epos);
                if( spos < 0 )
                {
                    spos = 0;
                    break;
                }
                if( c==col ) // found our column
                {
                    spos += bytes;  // do not include delimiter
                    break;
                }
                epos = spos;
                spos = 0;
            }
            // this will be the string for this column
            if( spos < epos )
            {
                spos = dstr->byte_offset_for(spos); // convert char pos
                epos = dstr->byte_offset_for(epos); // to byte offset
                d_indexes[idx].first = dstr->data() + spos;
                d_indexes[idx].second = (epos-spos);
            }
        });
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-rsplit_column(%s,%d)\n",delimiter,maxsplit);
            printCudaError(err);
        }
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    RMM_FREE(d_delimiter,0);
    return (unsigned int)columnsCount;
}

//
// Split the string at the first occurrence of delimiter, and return 3 elements containing
// the part before the delimiter, the delimiter itself, and the part after the delimiter.
// If the delimiter is not found, return 3 elements containing the string itself, followed by two empty strings.
//
// >>> import pandas as pd
// >>> strs = pd.Series(['héllo', None, 'a_bc_déf', 'a__bc', '_ab_cd', 'ab_cd_'])
// >>> strs.str.partition('_')
//        0     1       2
// 0  héllo
// 1   None  None    None
// 2      a     _  bc_déf
// 3      a     _     _bc
// 4            _   ab_cd
// 5     ab     _     cd_
//
int NVStrings::partition( const char* delimiter, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    // copy delimiter to device
    char* d_delimiter = 0;
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);
    int d_asize = custring_view::alloc_size((char*)delimiter,bytes);
    d_asize = ALIGN_SIZE(d_asize);

    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    // build int arrays to hold each string's partition sizes
    int totalSizes = 2 * count;
    rmm::device_vector<int> sizes(totalSizes,0), totals(count,0);
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_delimiter, bytes, d_asize, d_sizes, d_totals] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = &(d_sizes[idx*2]);
            d_totals[idx] = dstr->split_size(d_delimiter,bytes,dsizes,2) + d_asize;
        });

    cudaDeviceSynchronize();

    // build an output array of custring_views* arrays for each value
    // there will always be 3 per string
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    for( int idx=0; idx < count; ++idx )
    {
        NVStrings* result = new NVStrings(3);
        results.push_back(result);
        h_splits[idx] = result->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,totalSize,0);
        result->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;
    }

    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the partition and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    [d_strings, d_delimiter, bytes, d_buffers, d_sizes, d_splits] __device__(unsigned int idx){
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        char* buffer = (char*)d_buffers[idx];
        int* dsizes = &(d_sizes[idx*2]);
        custring_view_array d_strs = d_splits[idx];

        d_strs[0] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[0]);
        d_strs[1] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[1]);
        d_strs[2] = custring_view::create_from(buffer,0,0);

        //
        int dcount = dstr->rsplit_size(d_delimiter,bytes,0,2);
        dstr->split(d_delimiter,bytes,2,d_strs);
        if( dcount==2 )
        {   // insert delimiter element in the middle
            custring_view* tmp  = d_strs[1];
            d_strs[1] = custring_view::create_from(buffer,d_delimiter,bytes);
            d_strs[2] = tmp;
        }
    });

    printCudaError(cudaDeviceSynchronize(),"nvs-partition");
    RMM_FREE(d_delimiter,0);
    return count;
}

//
// This follows most of the same logic as partition above except that the delimiter
// search starts from the end of the string. Also, if no delimiter is found the
// resulting array includes two empty strings followed by the original string.
//
// >>> import pandas as pd
// >>> strs = pd.Series(['héllo', None, 'a_bc_déf', 'a__bc', '_ab_cd', 'ab_cd_'])
// >>> strs.str.rpartition('_')
//        0     1      2
// 0               héllo
// 1   None  None   None
// 2   a_bc     _    déf
// 3     a_     _     bc
// 4    _ab     _     cd
// 5  ab_cd     _
//
int NVStrings::rpartition( const char* delimiter, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    // copy delimiter to device
    char* d_delimiter = 0;
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);
    int d_asize = custring_view::alloc_size((char*)delimiter,bytes);
    d_asize = ALIGN_SIZE(d_asize);

    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    // build int arrays to hold each string's partition sizes
    int totalSizes = 2 * count;
    rmm::device_vector<int> sizes(totalSizes,0), totals(count,0);
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_asize, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = &(d_sizes[idx*2]);
            //d_totals[idx] = dstr->rpartition_size(d_delimiter,bytes,dsizes);
            d_totals[idx] = dstr->rsplit_size(d_delimiter,bytes,dsizes,2) + d_asize;
        });

    cudaDeviceSynchronize();

    // now build an output array of custring_views* arrays for each value
    // there will always be 3 per string
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    for( int idx=0; idx < count; ++idx )
    {
        NVStrings* result = new NVStrings(3);
        results.push_back(result);
        h_splits[idx] = result->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,totalSize,0);
        result->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;
    }

    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the partition and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    [d_strings, d_delimiter, bytes, d_buffers, d_sizes, d_splits] __device__(unsigned int idx){
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        char* buffer = (char*)d_buffers[idx];
        int* dsizes = &(d_sizes[idx*2]);
        custring_view_array d_strs = d_splits[idx];

        d_strs[0] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[0]);
        d_strs[1] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[1]);
        d_strs[2] = custring_view::create_from(buffer,0,0);

        //
        int dcount = dstr->rsplit_size(d_delimiter,bytes,0,2);
        dstr->rsplit(d_delimiter,bytes,2,d_strs);
        // reorder elements
        if( dcount==1 )
        {   // if only one element, it goes on the end
            custring_view* tmp  = d_strs[2];
            d_strs[2] = d_strs[0];
            d_strs[0] = tmp;
        }
        if( dcount==2 )
        {   // insert delimiter element in the middle
            custring_view* tmp  = d_strs[1];
            d_strs[1] = custring_view::create_from(buffer,d_delimiter,bytes);
            d_strs[2] = tmp;
        }
    });

    printCudaError(cudaDeviceSynchronize(),"nvs-rpartition");
    RMM_FREE(d_delimiter,0);
    return count;
}

// Extract character from each component at specified position
NVStrings* NVStrings::get(unsigned int pos)
{
    return slice(pos,pos+1,1);
}

// duplicate and concatenate the string the number of times specified
NVStrings* NVStrings::repeat(unsigned int reps)
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();

    auto execpol = rmm::exec_policy(0);
    // compute size of output buffer
    double st1 = GetTime();
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, reps, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int bytes = dstr->size();
            unsigned int nchars = dstr->chars_count();
            if( reps > 1 )
            {
                bytes += (bytes * (reps-1));
                nchars += (nchars * (reps-1));
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the repeat
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, reps, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dout = custring_view::create_from(buffer,*dstr);
            int count = (reps > 1 ? reps : 1);
            while( --count > 0 )
                dout->append(*dstr); // *dout += *dstr; works too
            d_results[idx] = dout;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-repeat");
    double et2 = GetTime();
    pImpl->addOpTimes("repeat",(et1-st1),(et2-st2));

    return rtn;
}

// Add specified padding to each string.
// Side:{'left','right','both'}, default is 'left'.
NVStrings* NVStrings::pad(unsigned int width, padside side, const char* fillchar )
{
    if( side==right ) // pad to the right
        return ljust(width,fillchar);
    if( side==both )  // pad both ends
        return center(width,fillchar);
    // default is pad to the left
    return rjust(width,fillchar);
}

// Pad the end of each string to the minimum width.
NVStrings* NVStrings::ljust( unsigned int width, const char* fillchar )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();

    auto execpol = rmm::exec_policy(0);
    if( !fillchar )
        fillchar = " ";
    Char d_fillchar = 0;
    unsigned int fcbytes = custring_view::char_to_Char(fillchar,d_fillchar);

    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, fcbytes, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int bytes = dstr->size();
            unsigned int nchars = dstr->chars_count();
            if( width > nchars )
            {
                unsigned int pad = width - nchars;
                bytes += fcbytes * pad;
                nchars += pad;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn; // all strings are null
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the padding
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, d_fillchar, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            // create init string with size enough for inserts
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dout = custring_view::create_from(buffer,*dstr);
            unsigned int nchars = dstr->chars_count();
            if( width > nchars ) // add pad character to the end
                dout->insert(nchars,width-nchars,d_fillchar);
            d_results[idx] = dout;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-ljust");
    double et2 = GetTime();
    pImpl->addOpTimes("ljust",(et1-st1),(et2-st2));

    return rtn;
}

// Pad the beginning and end of each string to the minimum width.
NVStrings* NVStrings::center( unsigned int width, const char* fillchar )
{
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();

    auto execpol = rmm::exec_policy(0);
    if( !fillchar )
        fillchar = " ";
    Char d_fillchar = 0;
    int fcbytes = custring_view::char_to_Char(fillchar,d_fillchar);

    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, fcbytes, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int bytes = dstr->size();
            unsigned int nchars = dstr->chars_count();
            if( width > nchars )
            {
                unsigned int pad = width - nchars;
                bytes += fcbytes * pad;
                nchars += pad;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    // create offsets
    double et1 = GetTime();
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the padding
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, d_fillchar, fcbytes, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            // create init string with buffer sized enough the inserts
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dout = custring_view::create_from(buffer,*dstr);
            unsigned int nchars = dstr->chars_count();
            if( width > nchars )
            {
                unsigned int pad = width - nchars;
                unsigned int left = pad/2;
                unsigned int right = pad - left;
                dout->insert(nchars,right,d_fillchar);
                dout->insert(0,left,d_fillchar);
            }
            d_results[idx] = dout;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-center");
    double et2 = GetTime();
    pImpl->addOpTimes("center",(et1-st1),(et2-st2));

    return rtn;
}

// Pad the beginning of each string to the minimum width.
NVStrings* NVStrings::rjust( unsigned int width, const char* fillchar )
{
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();

    auto execpol = rmm::exec_policy(0);
    if( !fillchar )
        fillchar = " ";
    Char d_fillchar = 0;
    int fcbytes = custring_view::char_to_Char(fillchar,d_fillchar);

    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, fcbytes, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int bytes = dstr->size();
            unsigned int nchars = dstr->chars_count();
            if( width > nchars )
            {
                unsigned int pad = width - nchars;
                bytes += fcbytes * pad;
                nchars += pad;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the padding
    custring_view** d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, width, d_fillchar, d_buffer, d_offsets, d_results] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            // create init string with size enough for inserts
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dout = custring_view::create_from(buffer,*dstr);
            unsigned int nchars = dstr->chars_count();
            if( width > nchars ) // add pad character to the beginning
                dout->insert(0,width-nchars,d_fillchar);
            d_results[idx] = dout;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-rjust");
    double et2 = GetTime();
    pImpl->addOpTimes("rjust",(et1-st1),(et2-st2));

    return rtn;
}

// Pad the beginning of each string with 0s honoring any sign prefix.
NVStrings* NVStrings::zfill( unsigned int width )
{
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();

    auto execpol = rmm::exec_policy(0);
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int bytes = dstr->size();
            unsigned int nchars = dstr->chars_count();
            if( width > nchars )
            {
                unsigned int pad = width - nchars;
                bytes += pad;
                nchars += pad;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the fill
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, width, d_buffer, d_offsets, d_results] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            // create init string with buffer sized enough for the inserts
            char* buffer = d_buffer + d_offsets[idx];
            char* sptr = dstr->data();
            unsigned int sz = dstr->size();
            custring_view* dout = custring_view::create_from(buffer,sptr,sz);
            unsigned int nchars = dstr->chars_count();
            if( width > nchars )
            {
                char fchr = ((sz <= 0) ? 0 : *sptr );                        // check for sign and shift
                unsigned int pos = (((fchr == '-') || (fchr == '+')) ? 1:0); // insert pos if necessary
                dout->insert(pos,width-nchars,'0');  // insert characters
            }
            d_results[idx] = dout;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-zfill");
    double et2 = GetTime();
    pImpl->addOpTimes("zfill",(et1-st1),(et2-st2));

    return rtn;
}

// All strings are substr'd with the same (start,stop) position values.
NVStrings* NVStrings::slice( int start, int stop, int step )
{
    if( (stop > 0) && (start > stop) )
        return 0;

    auto execpol = rmm::exec_policy(0);
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, start, stop, step, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = ( stop <= 0 ? dstr->chars_count() : stop ) - start;
            unsigned int size = dstr->substr_size((unsigned)start,(unsigned)len,(unsigned)step);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // slice it and dice it
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, start, stop, step, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            int len = ( stop <= 0 ? dstr->chars_count() : stop ) - start;
            d_results[idx] = dstr->substr((unsigned)start,(unsigned)len,(unsigned)step,buffer);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-slice(%d,%d,%d)\n",start,stop,step);
        printCudaError(err);
    }
    pImpl->addOpTimes("slice",(et1-st1),(et2-st2));
    return rtn;
}

// Each string is substr'd according to the individual (start,stop) position values
NVStrings* NVStrings::slice_from( int* starts, int* stops )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, starts, stops, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int start = (starts ? starts[idx]:0);
            int stop = (stops ? stops[idx]: -1);
            int len = ( stop <= 0 ? dstr->chars_count() : stop ) - start;
            unsigned int size = dstr->substr_size((unsigned)start,(unsigned)len);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // slice, slice, baby
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, starts, stops, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int start = (starts ? starts[idx]:0);
            int stop = (stops ? stops[idx]: -1);
            char* buffer = d_buffer + d_offsets[idx];
            int len = ( stop <= 0 ? dstr->chars_count() : stop ) - start;
            d_results[idx] = dstr->substr((unsigned)start,(unsigned)len,1,buffer);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-slice_from(%p,%p)\n",starts,stops);
        printCudaError(err);
    }
    pImpl->addOpTimes("slice",(et1-st1),(et2-st2));
    return rtn;
}

//
NVStrings* NVStrings::slice_replace( const char* repl, int start, int stop )
{
    if( !repl )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned int replen = (unsigned int)strlen(repl);
    char* d_repl = 0;
    RMM_ALLOC(&d_repl,replen,0);
    cudaMemcpy(d_repl,repl,replen,cudaMemcpyHostToDevice);
    // compute size of output buffer
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_repl, replen, start, stop, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int len = dstr->replace_size((unsigned)start,(unsigned)(stop-start),d_repl,replen);
            len = ALIGN_SIZE(len);
            d_lengths[idx] = (size_t)len;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
    {
        if( d_repl )
            RMM_FREE(d_repl,0);
        return rtn;
    }
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the slice and replace
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_repl, replen, start, stop, d_buffer, d_offsets, d_results] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dout = dstr->replace((unsigned)start,(unsigned)(stop-start),d_repl,replen,buffer);
            d_results[idx] = dout;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-slice_replace(%s,%d,%d)\n",repl,start,stop);
        printCudaError(err);
    }
    if( d_repl )
        RMM_FREE(d_repl,0);
    pImpl->addOpTimes("slice_replace",(et1-st1),(et2-st2));
    return rtn;
}

// this should replace multiple occurrences up to maxrepl
NVStrings* NVStrings::replace( const char* str, const char* repl, int maxrepl )
{
    if( !str || !*str )
        return 0; // null and empty string not allowed
    auto execpol = rmm::exec_policy(0);
    unsigned int ssz = (unsigned int)strlen(str);
    char* d_str = 0;
    RMM_ALLOC(&d_str,ssz,0);
    cudaMemcpy(d_str,str,ssz,cudaMemcpyHostToDevice);
    unsigned int sszch = custring_view::chars_in_string(str,ssz);

    if( !repl )
        repl = "";
    unsigned int rsz = (unsigned int)strlen(repl);
    char* d_repl = 0;
    RMM_ALLOC(&d_repl,rsz,0);
    cudaMemcpy(d_repl,repl,rsz,cudaMemcpyHostToDevice);
    unsigned int rszch = custring_view::chars_in_string(repl,rsz);

    // compute size of the output
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, ssz, sszch, d_repl, rsz, rszch, maxrepl, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int mxn = maxrepl;
            if( mxn < 0 )
                mxn = dstr->chars_count(); //max possible replaces for this string
            unsigned int bytes = dstr->size(), nchars = dstr->chars_count();
            int pos = dstr->find(d_str,ssz);
            // counting bytes and chars
            while((pos >= 0) && (mxn > 0))
            {
                bytes += rsz - ssz;
                nchars += rszch - sszch;
                pos = dstr->find(d_str,ssz,(unsigned)pos+sszch); // next one
                --mxn;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            d_sizes[idx] = ALIGN_SIZE(size);
        });
    //
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
    {
        RMM_FREE(d_str,0);
        RMM_FREE(d_repl,0);
        return rtn; // all strings are null
    }
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, ssz, sszch, d_repl, rsz, d_buffer, d_offsets, maxrepl, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int mxn = maxrepl;
            if( mxn < 0 )
                mxn = dstr->chars_count(); //max possible replaces for this string
            //
            char* buffer = d_buffer + d_offsets[idx];
            char* sptr = dstr->data();
            char* optr = buffer;
            unsigned int size = dstr->size();
            int pos = dstr->find(d_str,ssz), lpos=0;
            while((pos >= 0) && (mxn > 0))
            {                                                 // i:bbbbsssseeee
                int spos = dstr->byte_offset_for(pos);        //       ^
                memcpy(optr,sptr+lpos,spos-lpos);             // o:bbbb
                optr += spos - lpos;                          //       ^
                memcpy(optr,d_repl,rsz);                      // o:bbbbrrrr
                optr += rsz;                                  //           ^
                lpos = spos + ssz;                            // i:bbbbsssseeee
                pos = dstr->find(d_str,ssz,pos+sszch);        //           ^
                --mxn;
            }
            memcpy(optr,sptr+lpos,size-lpos);                 // o:bbbbrrrreeee
            unsigned int nsz = (unsigned int)(optr - buffer) + size - lpos;
            d_results[idx] = custring_view::create_from(buffer,buffer,nsz);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-replace(%s,%s,%d)\n",str,repl,maxrepl);
        printCudaError(err);
    }
    pImpl->addOpTimes("replace",(et1-st1),(et2-st2));

    RMM_FREE(d_str,0);
    RMM_FREE(d_repl,0);
    return rtn;
}

// same as above except parameter is regex
NVStrings* NVStrings::replace_re( const char* pattern, const char* repl, int maxrepl )
{
    if( !pattern || !*pattern )
        return 0; // null and empty string not allowed

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-replace: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return 0;
    }
    //
    // copy replace string to device memory
    if( !repl )
        repl = "";
    unsigned int rsz = (unsigned int)strlen(repl);
    char* d_repl = 0;
    RMM_ALLOC(&d_repl,rsz,0);
    cudaMemcpy(d_repl,repl,rsz,cudaMemcpyHostToDevice);
    unsigned int rszch = custring_view::chars_in_string(repl,rsz);

    // compute size of the output
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, d_repl, rsz, rszch, maxrepl, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int mxn = maxrepl;
            if( mxn < 0 )
                mxn = dstr->chars_count(); //max possible replaces for this string
            unsigned int bytes = dstr->size(), nchars = dstr->chars_count();
            int begin = 0, end = (int)nchars;
            int result = prog->find(dstr,begin,end);
            while((result > 0) && (mxn > 0))
            {
                bytes += rsz - (dstr->byte_offset_for(end)-dstr->byte_offset_for(begin));
                nchars += rszch - (end-begin);
                begin = end;
                end = (int)nchars;
                result = prog->find(dstr,begin,end); // next one
                --mxn;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            d_sizes[idx] = ALIGN_SIZE(size);
        });
    //
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
    {
        dreprog::destroy(prog);
        RMM_FREE(d_repl,0);
        return rtn; // all strings are null
    }
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the replace
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, d_repl, rsz, d_buffer, d_offsets, maxrepl, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int mxn = maxrepl;
            int nchars = (int)dstr->chars_count();
            if( mxn < 0 )
                mxn = nchars; //max possible replaces for this string
            char* buffer = d_buffer + d_offsets[idx];  // output buffer
            char* sptr = dstr->data();                 // input buffer
            char* optr = buffer;                       // running output pointer
            unsigned int size = dstr->size();          // number of byte in input string
            int lpos = 0, begin = 0, end = nchars;     // working vars
            // copy input to output replacing strings as we go
            int result = prog->find(dstr,begin,end);
            while((result > 0) && (mxn > 0))
            {                                                 // i:bbbbsssseeee
                int spos = dstr->byte_offset_for(begin);      //       ^
                memcpy(optr,sptr+lpos,spos-lpos);             // o:bbbb
                optr += spos - lpos;                          //       ^
                memcpy(optr,d_repl,rsz);                      // o:bbbbrrrr
                optr += rsz;                                  //           ^
                lpos = dstr->byte_offset_for(end);            // i:bbbbsssseeee
                begin = end;                                  //           ^
                end = nchars;
                result = prog->find(dstr,begin,end);
                --mxn;
            }                                                 // copy the rest:
            memcpy(optr,sptr+lpos,size-lpos);                 // o:bbbbrrrreeee
            unsigned int nsz = (unsigned int)(optr - buffer) + size - lpos;
            d_results[idx] = custring_view::create_from(buffer,buffer,nsz);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-replace_re(%s,%s,%d)\n",pattern,repl,maxrepl);
        printCudaError(err);
    }
    pImpl->addOpTimes("replace_re",(et1-st1),(et2-st2));
    //
    dreprog::destroy(prog);
    RMM_FREE(d_repl,0);
    return rtn;
}


// remove the target characters from the beginning of each string
NVStrings* NVStrings::lstrip( const char* to_strip )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);

    char* d_strip = 0;
    if( to_strip )
    {
        int len = (int)strlen(to_strip) + 1; // include null
        RMM_ALLOC(&d_strip,len,0);
        cudaMemcpy(d_strip,to_strip,len,cudaMemcpyHostToDevice);
    }

    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strip, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int len = dstr->lstrip_size(d_strip);
            len = ALIGN_SIZE(len);
            d_lengths[idx] = (size_t)len;
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
    {
        if( d_strip )
            RMM_FREE(d_strip,0);
        return rtn; // all strings are null
    }

    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the strip
    custring_view** d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strip, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            d_results[idx] = dstr->lstrip(d_strip,buffer);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-lstrip(%s)\n",to_strip);
        printCudaError(err);
    }
    pImpl->addOpTimes("lstrip",(et1-st1),(et2-st2));
    if( d_strip )
        RMM_FREE(d_strip,0);
    return rtn;
}

// remove the target character from the beginning and the end of each string
NVStrings* NVStrings::strip( const char* to_strip )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);

    char* d_strip = 0;
    if( to_strip )
    {
        int len = (int)strlen(to_strip) + 1; // include null
        RMM_ALLOC(&d_strip,len,0);
        cudaMemcpy(d_strip,to_strip,len,cudaMemcpyHostToDevice);
    }

    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strip, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int len = dstr->strip_size(d_strip);
            len = ALIGN_SIZE(len);
            d_lengths[idx] = (size_t)len;
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
    {
        if( d_strip )
            RMM_FREE(d_strip,0);
        return rtn;
    }

    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the strip
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strip, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            d_results[idx] = dstr->strip(d_strip,buffer);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-strip(%s)\n",to_strip);
        printCudaError(err);
    }
    pImpl->addOpTimes("strip",(et1-st1),(et2-st2));
    if( d_strip )
        RMM_FREE(d_strip,0);
    return rtn;
}

// remove the target character from the end of each string
NVStrings* NVStrings::rstrip( const char* to_strip )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);

    char* d_strip = 0;
    if( to_strip )
    {
        int len = (int)strlen(to_strip) + 1; // include null
        RMM_ALLOC(&d_strip,len,0);
        cudaMemcpy(d_strip,to_strip,len,cudaMemcpyHostToDevice);
    }

    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strip, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int len = dstr->rstrip_size(d_strip);
            len = ALIGN_SIZE(len);
            d_lengths[idx] = (size_t)len;
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
    {
        if( d_strip )
            RMM_FREE(d_strip,0);
        return rtn; // all strings are null
    }

    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the strip
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strip, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            d_results[idx] = dstr->rstrip(d_strip,buffer);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-rstrip(%s)\n",to_strip);
        printCudaError(err);
    }
    pImpl->addOpTimes("rstrip",(et1-st1),(et2-st2));
    if( d_strip )
        RMM_FREE(d_strip,0);
    return rtn;
}

//
NVStrings* NVStrings::lower()
{
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(0);
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    unsigned char* d_flags = get_unicode_flags();
    unsigned short* d_cases = get_charcases();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    //thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    //    [d_strings, d_lengths] __device__(unsigned int idx){
    //        custring_view* dstr = d_strings[idx];
    //        if( dstr )
    //            d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
    //    });
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int chw = custring_view::bytes_in_char(chr);
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( IS_UPPER(flg) )
                        chw = custring_view::bytes_in_char(u2u8(d_cases[uni]));
                    bytes += chw;
                }
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes,dstr->chars_count()));
            }
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                char* buffer = d_buffer + d_offsets[idx];
                //custring_view* dout = custring_view::create_from(buffer,*dstr);
                //dout->lower(); // inplace function
                //d_results[idx] = dout;
                char* ptr = buffer;
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( IS_UPPER(flg) )
                        chr = u2u8(d_cases[uni]);
                    unsigned int chw = custring_view::Char_to_char(chr,ptr);
                    ptr += chw;
                    bytes += chw;
                }
                d_results[idx] = custring_view::create_from(buffer,buffer,bytes);
            }
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-lower()");
    pImpl->addOpTimes("lower",(et1-st1),(et2-st2));
    return rtn;
}

//
NVStrings* NVStrings::upper()
{
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(0);
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    unsigned char* d_flags = get_unicode_flags();
    unsigned short* d_cases = get_charcases();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    //thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    //    [d_strings, d_lengths] __device__(unsigned int idx){
    //        custring_view* dstr = d_strings[idx];
    //        if( dstr )
    //            d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
    //    });
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int chw = custring_view::bytes_in_char(chr);
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( IS_LOWER(flg) )
                        chw = custring_view::bytes_in_char(u2u8(d_cases[uni]));
                    bytes += chw;
                }
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes,dstr->chars_count()));
            }
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            //char* buffer = d_output + (size_t)((char*)dstr - d_input);
            //custring_view* dout = custring_view::create_from(buffer,*dstr);
            //dout->upper(); // inplace
            //d_results[idx] = dout;
            char* ptr = buffer;
            unsigned int bytes = 0;
            for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
            {
                Char chr = *itr;
                unsigned int uni = u82u(*itr);
                unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                if( IS_LOWER(flg) )
                    chr = u2u8(d_cases[uni]);
                int chw = custring_view::Char_to_char(chr,ptr);
                ptr += chw;
                bytes += chw;
            }
            d_results[idx] = custring_view::create_from(buffer,buffer,bytes);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-upper()");
    pImpl->addOpTimes("upper",(et1-st1),(et2-st2));
    return rtn;
}

//
NVStrings* NVStrings::swapcase()
{
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(0);
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    unsigned char* d_flags = get_unicode_flags();
    unsigned short* d_cases = get_charcases();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    //thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    //    [d_strings, d_lengths] __device__(unsigned int idx){
    //        custring_view* dstr = d_strings[idx];
    //        if( dstr )
    //            d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
    //    });
    //
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int chw = custring_view::bytes_in_char(chr);
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( IS_LOWER(flg) || IS_UPPER(flg) )
                        chw = custring_view::bytes_in_char(u2u8(d_cases[uni]));
                    bytes += chw;
                }
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes,dstr->chars_count()));
            }
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                char* buffer = d_buffer + d_offsets[idx];
                //custring_view* dout = custring_view::create_from(buffer,*dstr);
                //dout->swapcase();
                //d_results[idx] = dout;
                char* ptr = buffer;
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( IS_LOWER(flg) || IS_UPPER(flg) )
                        chr = u2u8(d_cases[uni]);
                    int chw = custring_view::Char_to_char(chr,ptr);
                    ptr += chw;
                    bytes += chw;
                }
                d_results[idx] = custring_view::create_from(buffer,buffer,bytes);
            }
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-swapcase()");
    pImpl->addOpTimes("swapcase",(et1-st1),(et2-st2));
    return rtn;
}

//
NVStrings* NVStrings::capitalize()
{
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(0);
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    unsigned char* d_flags = get_unicode_flags();
    unsigned short* d_cases = get_charcases();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int chw = custring_view::bytes_in_char(chr);
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( (bytes && IS_UPPER(flg)) || (!bytes && IS_LOWER(flg)) )
                    {
                        uni = (uni <= 0x00FFF ? d_cases[uni] : uni);
                        chr = u2u8(uni);
                        chw = custring_view::bytes_in_char(chr);
                    }
                    bytes += chw;
                }
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes,dstr->chars_count()));
            }
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                char* buffer = d_buffer + d_offsets[idx];
                //custring_view* dout = custring_view::create_from(buffer,*dstr);
                //dout->capitalize();
                //d_results[idx] = dout;
                char* ptr = buffer;
                unsigned int bytes = 0;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( (bytes && IS_UPPER(flg)) || (!bytes && IS_LOWER(flg)) )
                    {
                        uni = (uni <= 0x00FFF ? d_cases[uni] : uni);
                        chr = u2u8(uni);
                    }
                    unsigned int chw = custring_view::Char_to_char(chr,ptr);
                    ptr += chw;
                    bytes += chw;
                }
                d_results[idx] = custring_view::create_from(buffer,buffer,bytes);
            }
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
        printCudaError(err,"mvs-capitalize()");
    pImpl->addOpTimes("capitalize",(et1-st1),(et2-st2));
    return rtn;
}

// returns titlecase for each string
NVStrings* NVStrings::title()
{
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(0);
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    unsigned char* d_flags = get_unicode_flags();
    unsigned short* d_cases = get_charcases();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                int bytes = 0;
                bool bcapnext = true;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( !IS_ALPHA(flg) )
                    {
                        bcapnext = true;
                        bytes += custring_view::bytes_in_char(chr);
                        continue;
                    }
                    if( (bcapnext && IS_LOWER(flg)) || (!bcapnext && IS_UPPER(flg)) )
                        uni = (unsigned int)(uni <= 0x00FFFF ? d_cases[uni] : uni);
                    bcapnext = false;
                    bytes += custring_view::bytes_in_char(u2u8(uni));
                }
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes,dstr->chars_count()));
            }
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // do the title thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                char* buffer = d_buffer + d_offsets[idx];
                //custring_view* dout = custring_view::create_from(buffer,*dstr);
                //dout->titlecase();
                //d_results[idx] = dout;
                char* ptr = buffer;
                int bytes = 0;
                bool bcapnext = true;
                for( auto itr = dstr->begin(); (itr != dstr->end()); itr++ )
                {
                    Char chr = *itr;
                    unsigned int uni = u82u(chr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    if( !IS_ALPHA(flg) )
                        bcapnext = true;
                    else
                    {
                        if( (bcapnext && IS_LOWER(flg)) || (!bcapnext && IS_UPPER(flg)) )
                        {
                            uni = (unsigned int)(uni <= 0x00FFFF ? d_cases[uni] : uni);
                            chr = u2u8(uni);
                        }
                        bcapnext = false;
                    }
                    int chw = custring_view::Char_to_char(chr,ptr);
                    bytes += chw;
                    ptr += chw;
                }
                d_results[idx] = custring_view::create_from(buffer,buffer,bytes);
            }
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-title()");
    pImpl->addOpTimes("",(et1-st1),(et2-st2));
    return rtn;
}

// searches from the beginning of each string
unsigned int NVStrings::find( const char* str, int start, int end, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string
    if( start < 0 )
        start = 0;

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

    custring_view** d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, start, end, d_rtn] __device__(unsigned int idx){
            //__shared__ char tgt[24];
            char* dtgt = d_str;
            //if( bytes<24  )
            //{
            //    dtgt = tgt;
            //    if( threadIdx.x==0 )
            //        memcpy(dtgt,d_str,bytes);
            //}
            //__syncthreads();
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->find(dtgt,bytes-1,start,end-start);
            else
                d_rtn[idx] = -2; // indicate null to caller
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-find(%s,%d,%d,%p,%d)\n",str,start,end,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("find",0.0,(et-st));

    RMM_FREE(d_str,0);
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

// searches from the beginning of each string and specified individual starting positions
unsigned int NVStrings::find_from( const char* str, int* starts, int* ends, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, starts, ends, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                int pos = (starts ? starts[idx] : 0);
                int len = (ends ? (ends[idx]-pos) : -1);
                d_rtn[idx] = dstr->find(d_str,bytes-1,pos,len);
            }
            else
                d_rtn[idx] = -2; // indicate null to caller
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-find_from(%s,%p,%p,%p,%d)\n",str,starts,ends,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("find_from",0.0,(et-st));

    RMM_FREE(d_str,0);
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

// searches from the end of each string
unsigned int NVStrings::rfind( const char* str, int start, int end, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1;
    if( start < 0 )
        start = 0;

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

    custring_view** d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, start, end, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->rfind(d_str,bytes-1,start,end-start);
            else
                d_rtn[idx] = -2; // indicate null to caller
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-rfind(%s,%d,%d,%p,%d)\n",str,start,end,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("rfind",0.0,(et-st));
    RMM_FREE(d_str,0);
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

//
unsigned int NVStrings::find_multiple( NVStrings& strs, int* results, bool todevice )
{
    unsigned int count = size();
    unsigned int tcount = strs.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,tcount*count*sizeof(int),0);
    //
    custring_view_array d_strings = pImpl->getStringsPtr();
    custring_view_array d_targets = strs.pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_targets, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_targets[jdx];
                d_rtn[(idx*tcount)+jdx] = ( (dstr && dtgt) ? dstr->find(*dtgt) : -2 );
            }
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-find_multiple(%u,%p,%d)\n",tcount,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("find_multiple",0.0,(et-st));
    //
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count*tcount,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

// for each string, return substring(s) which match specified pattern
int NVStrings::findall( const char* pattern, std::vector<NVStrings*>& results )
{
    if( pattern==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-findall: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }

    // compute counts of each match and size of the buffers
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> sizes(count,0);
    int* d_sizes = sizes.data().get();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, d_counts, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int tsize = 0;;
            int fnd = 0, end = (int)dstr->chars_count();
            int spos = 0, epos = end;
            int result = prog->find(dstr,spos,epos);
            while(result > 0)
            {
                unsigned int bytes = (dstr->byte_offset_for(epos)-dstr->byte_offset_for(spos));
                unsigned int nchars = (epos-spos);
                unsigned int size = custring_view::alloc_size(bytes,nchars);
                tsize += ALIGN_SIZE(size);
                spos = epos;
                epos = end;
                ++fnd;
                result = prog->find(dstr,spos,epos); // next one
            }
            d_sizes[idx] = tsize;
            d_counts[idx] = fnd;
        });
    cudaDeviceSynchronize();
    //
    // create rows of buffers
    thrust::host_vector<int> hcounts(counts); // copies counts from device
    thrust::host_vector<custring_view_array> hrows(count,nullptr);
    thrust::host_vector<char*> hbuffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int rcount = hcounts[idx];
        NVStrings* row = new NVStrings(rcount);
        results.push_back(row);
        if( rcount==0 )
            continue;
        hrows[idx] = row->pImpl->getStringsPtr();
        int size = sizes[idx];
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,size,0);
        row->pImpl->setMemoryBuffer(d_buffer,size);
        hbuffers[idx] = d_buffer;
    }
    // copy substrings into buffers
    rmm::device_vector<custring_view_array> rows(hrows); // copies hrows to device
    custring_view_array* d_rows = rows.data().get();
    rmm::device_vector<char*> buffers(hbuffers); // copies hbuffers to device
    char** d_buffers = buffers.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, d_counts, d_buffers, d_sizes, d_rows] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int dcount = d_counts[idx];
            if( dcount < 1 )
                return;
            char* buffer = (char*)d_buffers[idx];
            custring_view_array drow = d_rows[idx];
            int spos = 0, nchars = (int)dstr->chars_count();
            for( int i=0; i < dcount; ++i )
            {
                int epos = nchars;
                prog->find(dstr,spos,epos);
                custring_view* str = dstr->substr((unsigned)spos,(unsigned)(epos-spos),1,buffer);
                drow[i] = str;
                buffer += ALIGN_SIZE(str->alloc_size());
                spos = epos;
            }
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-findall");
    return count;
}

// same as findall but strings are returned organized in column-major
int NVStrings::findall_column( const char* pattern, std::vector<NVStrings*>& results )
{
    if( pattern==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-findall_column: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }

    // compute counts of each match and size of the buffers
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int fnd = 0, nchars = (int)dstr->chars_count();
            int begin = 0, end = nchars;
            int result = prog->find(dstr,begin,end);
            while(result > 0)
            {
                ++fnd;
                begin = end;
                end = nchars;
                result = prog->find(dstr,begin,end); // next one
            }
            d_counts[idx] = fnd;
        });
    int columns = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    //printf("%d columns\n",columns);
    //
    // create columns of nvstrings
    for( int col=0; col < columns; ++col )
    {
        // build index for each string -- collect pointers and lengths
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [prog, d_strings, d_counts, col, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = 0;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr || (col >= d_counts[idx]) )
                    return;
                int spos = 0, nchars = (int)dstr->chars_count();
                int epos = nchars;
                prog->find(dstr,spos,epos);
                for( int c=0; c < col; ++c )
                {
                    spos = epos;    // update
                    epos = nchars;  // parameters
                    prog->find(dstr,spos,epos);
                }
                // this will be the string for this column
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
            });
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-findall_column(%s): col=%d\n",pattern,col);
            printCudaError(err);
        }
        // build new instance from the index
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }

    return columns;
}

// does specified string occur in each string
int NVStrings::contains( const char* str, bool* results, bool todevice )
{
    if( str==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->find(d_str,bytes-1)>=0;
            else
                d_rtn[idx] = false;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-contains(%s,%p,%d)\n",str,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("contains",0.0,(et-st));

    RMM_FREE(d_str,0);
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)rtn;
}

// regex version of contain() above
int NVStrings::contains_re( const char* pattern, bool* results, bool todevice )
{
    if( pattern==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-contains: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, prog, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = prog->contains(dstr)==1;
            else
                d_rtn[idx] = false;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-contains_re(%s,%p,%d)\n",pattern,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("contains_re",0.0,(et-st));

    dreprog::destroy(prog);

    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val){ return val; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)rtn;
}

// match is like contains() except the pattern must match the beginning of the string only
int NVStrings::match( const char* pattern, bool* results, bool todevice )
{
    if( pattern==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-match: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, prog, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = prog->match(dstr)==1;
            else
                d_rtn[idx] = false;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-match(%s,%p,%d)\n",pattern,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("match",0.0,(et-st));

    dreprog::destroy(prog);

    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val){ return val; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)rtn;
}


// counts number of times the regex pattern matches a string within each string
int NVStrings::count_re( const char* pattern, int* results, bool todevice )
{
    if( pattern==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-count: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }

    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, prog, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            int fnd = -1;
            if( dstr )
            {
                fnd = 0;
                int nchars = (int)dstr->chars_count();
                int begin = 0, end = nchars;
                int result = prog->find(dstr,begin,end);
                while(result > 0)
                {
                    ++fnd; // count how many we find
                    begin = end;
                    end = nchars;
                    result = prog->find(dstr,begin,end);
                }
            }
            d_rtn[idx] = fnd;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-count_re(%s,%p,%d)\n",pattern,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("count_re",0.0,(et-st));

    dreprog::destroy(prog);

    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val>0; });
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)rtn;
}

//
unsigned int NVStrings::startswith( const char* str, bool* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->starts_with(d_str,bytes-1);
            else
                d_rtn[idx] = false;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-startswith(%s,%p,%d)\n",str,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("startswith",0.0,(et-st));

    RMM_FREE(d_str,0);
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

//
unsigned int NVStrings::endswith( const char* str, bool* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = 0;
    RMM_ALLOC(&d_str,bytes,0);
    cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->ends_with(d_str,bytes-1);
            else
                d_rtn[idx] = false;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-endswith(%s,%p,%d)\n",str,results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("endswith",0.0,(et-st));

    RMM_FREE(d_str,0);
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

//
int NVStrings::extract( const char* pattern, std::vector<NVStrings*>& results)
{
    if( pattern==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-extract: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }
    //
    int groups = prog->group_counts();
    if( groups==0 )
    {
        dreprog::destroy(prog);
        return 0;
    }
    // compute lengths of each group for each string
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> lengths(count*groups,0);
    int* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, groups, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int begin = 0, end = dstr->chars_count();
            if( prog->find(dstr,begin,end) <=0 )
                return;
            int* sizes = d_lengths + (idx*groups);
            for( int col=0; col < groups; ++col )
            {
                int spos=begin, epos=end;
                if( prog->extract(dstr,spos,epos,col) <=0 )
                    continue;
                unsigned int size = dstr->substr_size(spos,epos);
                sizes[col] = (size_t)ALIGN_SIZE(size);
            }
        });
    //
    cudaDeviceSynchronize();
    // this part will be slow for large number of strings
    rmm::device_vector<custring_view_array> strings(count,nullptr);
    rmm::device_vector<char*> buffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        NVStrings* row = new NVStrings(groups);
        results.push_back(row);
        int* sizes = d_lengths + (idx*groups);
        int size = thrust::reduce(execpol->on(0), sizes, sizes+groups);
        if( size==0 )
            continue;
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,size,0);
        row->pImpl->setMemoryBuffer(d_buffer,size);
        strings[idx] = row->pImpl->getStringsPtr();
        buffers[idx] = d_buffer;
    }
    // copy each subgroup into each rows memory
    custring_view_array* d_rows = strings.data().get();
    char** d_buffers = buffers.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, d_buffers, d_lengths, groups, d_rows] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int begin = 0, end = dstr->chars_count(); // these could have been saved above
            if( prog->find(dstr,begin,end) <=0 )      // to avoid this call again here
                return;
            int* sizes = d_lengths + (idx*groups);
            char* buffer = (char*)d_buffers[idx];
            custring_view_array d_row = d_rows[idx];
            for( int col=0; col < groups; ++col )
            {
                int spos=begin, epos=end;
                if( prog->extract(dstr,spos,epos,col) <=0 )
                    continue;
                d_row[col] = dstr->substr((unsigned)spos,(unsigned)(epos-spos),1,buffer);
                buffer += sizes[col];
            }
        });
        //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-extract(%s): groups=%d\n",pattern,groups);
        printCudaError(err);
    }
    return groups;
}

// column-major version of extract() method above
int NVStrings::extract_column( const char* pattern, std::vector<NVStrings*>& results)
{
    if( pattern==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    int numInsts = prog->inst_counts();
    if( numInsts > 64 ) // 64 = LISTBYTES<<3 (from regexec.cu)
    {   // pre-check prevents crashing
        fprintf(stderr,"nvs-extract_column: pattern %s exceeds instances limit (64) for regex execution.\n",pattern);
        dreprog::destroy(prog);
        return -2;
    }
    //
    int groups = prog->group_counts();
    if( groups==0 )
    {
        dreprog::destroy(prog);
        return 0;
    }
    //
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> begins(count,0);
    int* d_begins = begins.data().get();
    rmm::device_vector<int> ends(count,0);
    int* d_ends = ends.data().get();
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    // build strings vector for each group (column)
    for( int col=0; col < groups; ++col )
    {
        // first, build two vectors of (begin,end) position values;
        // also get the lengths of the substrings
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [prog, d_strings, col, d_begins, d_ends, d_lengths] __device__(unsigned int idx) {
                custring_view* dstr = d_strings[idx];
                d_begins[idx] = -1;
                d_ends[idx] = -1;
                if( !dstr )
                    return;
                int begin=0, end=dstr->chars_count();
                int result = prog->find(dstr,begin,end);
                if( result > 0 )
                    result = prog->extract(dstr,begin,end,col);
                if( result > 0 )
                {
                    d_begins[idx] = begin;
                    d_ends[idx] = end;
                    unsigned int size = dstr->substr_size(begin,end-begin);
                    d_lengths[idx] = (size_t)ALIGN_SIZE(size);
                }
            });
        cudaDeviceSynchronize();
        // create list of strings for this group
        NVStrings* column = new NVStrings(count);
        results.push_back(column); // append here so continue statement will work
        char* d_buffer = column->pImpl->createMemoryFor(d_lengths);
        if( d_buffer==0 )
            continue;
        rmm::device_vector<size_t> offsets(count,0);
        thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
        // copy the substrings into the new object
        custring_view_array d_results = column->pImpl->getStringsPtr();
        size_t* d_offsets = offsets.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, d_begins, d_ends, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                if( !dstr )
                    return;
                int start = d_begins[idx];
                int stop = d_ends[idx];
                if( stop > start )
                    d_results[idx] = dstr->substr((unsigned)start,(unsigned)(stop-start),1,d_buffer+d_offsets[idx]);
            });
        //
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-extract_column(%s): col=%d\n",pattern,col);
            printCudaError(err);
        }
        // column already added to results above
    }

    return groups;
}

//
unsigned int NVStrings::isalnum( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // alnum requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_ALPHANUM(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isalnum(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isalnum",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isalpha( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // alpha requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_ALPHA(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isalpha(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isalpha",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

//
unsigned int NVStrings::isdigit( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // digit requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_DIGIT(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isdigit(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isdigit",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isspace( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // space requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_SPACE(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isspace(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isspace",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isdecimal( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // decimal requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_DECIMAL(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isdecimal(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isdecimal",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isnumeric( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // numeric requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_NUMERIC(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isnumeric(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isnumeric",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::islower( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    brc = !IS_ALPHA(flg) || IS_LOWER(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-islower(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("islower",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isupper( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    brc = !IS_ALPHA(flg) || IS_UPPER(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-islower(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("islower",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

//
NVStrings* NVStrings::translate( std::pair<unsigned,unsigned>* utable, unsigned int tableSize )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);

    // convert unicode table into utf8 table
    thrust::host_vector< thrust::pair<Char,Char> > htable(tableSize);
    for( unsigned int idx=0; idx < tableSize; ++idx )
    {
        htable[idx].first = u2u8(utable[idx].first);
        htable[idx].second = u2u8(utable[idx].second);
    }
    // could sort on the device; this table should not be very big
    thrust::sort(thrust::host, htable.begin(), htable.end(),
        [] __host__ (thrust::pair<Char,Char> p1, thrust::pair<Char,Char> p2) { return p1.first > p2.first; });

    // copy translate table to device memory
    rmm::device_vector< thrust::pair<Char,Char> > table(htable);
    thrust::pair<Char,Char>* d_table = table.data().get();

    // compute size of each new string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    int tsize = tableSize;
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_table, tsize, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            const char* sptr = dstr->data();
            unsigned int bytes = dstr->size();
            unsigned int nchars = dstr->chars_count();
            for( unsigned int i=0; i < nchars; ++i )
            {
                Char ch = dstr->at(i);
                Char nch = ch;
                for( int t=0; t < tsize; ++t ) // replace with faster lookup
                    nch = ( ch==d_table[t].first ? d_table[t].second : nch );
                int bic = custring_view::bytes_in_char(ch);
                int nbic = (nch ? custring_view::bytes_in_char(nch) : 0);
                bytes += nbic - bic;
                if( nch==0 )
                    --nchars;
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the translate
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_buffer, d_offsets, d_table, tsize, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            const char* sptr = dstr->data();
            unsigned int nchars = dstr->chars_count();
            char* optr = buffer;
            int nsz = 0;
            for( unsigned int i=0; i < nchars; ++i )
            {
                Char ch = 0;
                unsigned int cw = custring_view::char_to_Char(sptr,ch);
                Char nch = ch;
                for( int t=0; t < tsize; ++t ) // replace with faster lookup
                    nch = ( ch==d_table[t].first ? d_table[t].second : nch );
                sptr += cw;
                if( nch==0 )
                    continue;
                unsigned int nbic = custring_view::Char_to_char(nch,optr);
                optr += nbic;
                nsz += nbic;
            }
            d_results[idx] = custring_view::create_from(buffer,buffer,nsz);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-translate(...,%d)\n",(int)tableSize);
        printCudaError(err);
    }
    pImpl->addOpTimes("translate",(et1-st1),(et2-st2));
    return rtn;
}

// this returns one giant string joining all the strings
// in the list with the delimiter string between each one
NVStrings* NVStrings::join( const char* delimiter, const char* narep )
{
    if( delimiter==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned int dellen = (unsigned int)strlen(delimiter);
    char* d_delim = 0;
    if( dellen > 0 )
    {
        RMM_ALLOC(&d_delim,dellen,0);
        cudaMemcpy(d_delim,delimiter,dellen,cudaMemcpyHostToDevice);
    }
    unsigned int narlen = 0;
    if( narep )
        narlen = (unsigned int)strlen(narep);
    char* d_narep = 0;
    if( narlen > 0 )
    {
        RMM_ALLOC(&d_narep,narlen,0);
        cudaMemcpy(d_narep,narep,narlen,cudaMemcpyHostToDevice);
    }

    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();

    // need to compute the giant buffer size
    rmm::device_vector<size_t> lens(count,0);
    size_t* d_lens = lens.data().get();
    rmm::device_vector<size_t> chars(count,0);
    size_t* d_chars = chars.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delim, dellen, d_narep, narlen, count, d_lens, d_chars] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            int dlen = dellen;
            int nchars = 0;
            int bytes = 0;
            if( idx+1 >= count )
                dlen = 0; // no trailing delimiter
            if( dstr )
            {
                nchars = dstr->chars_count();
                bytes = dstr->size();
            }
            else if( d_narep )
            {
                nchars = custring_view::chars_in_string(d_narep,narlen);
                bytes = narlen;
            }
            else
                dlen = 0; // for null, no delimiter
            if( dlen )
            {
                nchars += custring_view::chars_in_string(d_delim,dellen);
                bytes += dellen;
            }
            d_lens[idx] = bytes;
            d_chars[idx] = nchars;
        });

    cudaDeviceSynchronize();
    // compute how much space is required for the giant string
    size_t totalBytes = thrust::reduce(execpol->on(0), lens.begin(), lens.end());
    size_t totalChars = thrust::reduce(execpol->on(0), chars.begin(), chars.end());
    //printf("totalBytes=%ld, totalChars=%ld\n",totalBytes,totalChars);
    size_t allocSize = custring_view::alloc_size((unsigned int)totalBytes,(unsigned int)totalChars);
    //printf("allocSize=%ld\n",allocSize);

    // convert the lens values into offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // create one big buffer to hold the strings
    NVStrings* rtn = new NVStrings(1);
    char* d_buffer = 0;
    RMM_ALLOC(&d_buffer,allocSize,0);
    custring_view_array d_result = rtn->pImpl->getStringsPtr();
    rtn->pImpl->setMemoryBuffer(d_buffer,allocSize);
    // copy the strings into it
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_buffer, d_offsets, count, d_delim, dellen, d_narep, narlen] __device__(unsigned int idx){
            char* sptr = d_buffer + 8 + d_offsets[idx];
            char* dlim = d_delim;
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {   // copy string to output
                int ssz = dstr->size();
                memcpy(sptr,dstr->data(),ssz);
                sptr += ssz;
            }
            else if( d_narep )
            {   // or copy null-replacement to output
                memcpy(sptr,d_narep,narlen);
                sptr += narlen;
            }
            else // or copy nothing to output
                dlim = 0; // prevent delimiter copy below
            // copy delimiter to output
            if( (idx+1 < count) && dlim )
                memcpy(sptr,dlim,dellen);
        });

    // assign to resulting custring_view
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), 1,
        [d_buffer, totalBytes, d_result] __device__ (unsigned int idx){
            char* sptr = d_buffer + 8;
            d_result[0] = custring_view::create_from(d_buffer,sptr,totalBytes);
        });
    printCudaError(cudaDeviceSynchronize(),"nvs-join");

    if( d_delim )
        RMM_FREE(d_delim,0);
    if( d_narep )
        RMM_FREE(d_narep,0);
    return rtn;
}

// Essentially inserting new-line chars into appropriate places in the string to ensure that each 'line' is
// no longer than width characters. Along the way, tabs may be expanded (8 spaces) or replaced.
// and long words may be broken up or reside on their own line.
//    expand_tabs = false         (tab = 8 spaces)
//    replace_whitespace = true   (replace with space)
//    drop_whitespace = false     (no spaces after new-line)
//    break_long_words = false
//    break_on_hyphens = false
NVStrings* NVStrings::wrap( unsigned int width )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);

    // need to compute the size of each new string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            // replacing space with new-line does not change the size
            // -- this is oversimplification since 'expand' and 'drop' options would change the size of the string
            d_sizes[idx] = ALIGN_SIZE(dstr->alloc_size());
        });

    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
        return rtn;
    double et1 = GetTime();
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the wrap logic
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    double st2 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, width, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            char* sptr = dstr->data();
            unsigned int sz = dstr->size();
            // start by copying whole string into buffer
            char* optr = buffer;
            memcpy( optr, sptr, sz );
            // replace appropriate spaces with new-line
            // - this should be way more complicated with all the permutations of flags
            unsigned int nchars = dstr->chars_count();
            int charOffsetToLastSpace = -1, byteOffsetToLastSpace = -1, spos=0, bidx=0;
            for( unsigned int pos=0; pos < nchars; ++pos )
            {
                Char chr = dstr->at(pos);
                if( chr <= ' ' )
                {   // convert all whitespace to space
                    optr[bidx] = ' ';
                    byteOffsetToLastSpace = bidx;
                    charOffsetToLastSpace = pos;
                }
                if( (pos - spos) >= width )
                {
                    if( byteOffsetToLastSpace >=0 )
                    {
                        optr[byteOffsetToLastSpace] = '\n';
                        spos = charOffsetToLastSpace;
                        byteOffsetToLastSpace = charOffsetToLastSpace = -1;
                    }
                }
                bidx += (int)custring_view::bytes_in_char(chr);
            }
            d_results[idx] = custring_view::create_from(buffer,buffer,sz);
    });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-wrap(%d)\n",width);
        printCudaError(err);
    }
    pImpl->addOpTimes("wrap",(et1-st1),(et2-st2));
    return rtn;
}

// this now sorts the strings into a new instance;
// a sorted strings list can improve performance by reducing divergence
NVStrings* NVStrings::sort( sorttype stype, bool ascending )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);
    // get the lengths so they can be sorted too
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
        });
    //
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn; // all are null so we are done
    // copy the pointers to temporary vector and sort them along with the alloc-lengths
    rmm::device_vector<custring_view*> sortvector(count,nullptr);
    custring_view_array d_sortvector = sortvector.data().get();
    cudaMemcpy(d_sortvector,d_strings,sizeof(custring_view*)*count,cudaMemcpyDeviceToDevice);
    thrust::sort_by_key(execpol->on(0), d_sortvector, d_sortvector+count, d_lengths,
        [ stype, ascending ] __device__( custring_view*& lhs, custring_view*& rhs ) {
            bool cmp = false;
            if( lhs==0 || rhs==0 )
                cmp = lhs==0; // non-null > null
            else
            {   // allow sorting by name and length
                int diff = 0;
                if( stype & NVStrings::length )
                    diff = rhs->size() - lhs->size();
                if( diff==0 && (stype & NVStrings::name) )
                    diff = rhs->compare(*lhs) > 0;
                cmp = (diff > 0);
            }
            return ( ascending ? cmp : !cmp );
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-sort(0x%0x,%d):by-key\n",(int)stype,(int)ascending);
        printCudaError(err);
    }
    // create offsets from the sorted lengths
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // gather the sorted results into the new memory
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_sortvector, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_sortvector[idx];
            if( dstr )
            {
                char* buffer = d_buffer + d_offsets[idx];
                d_results[idx] = custring_view::create_from(buffer,*dstr);
            }
        });
    //
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-sort(0x%0x,%d)\n",(unsigned)stype,(int)ascending);
        printCudaError(err);
    }
    return rtn;
}

// just provide the index order and leave the strings intact
int NVStrings::order( sorttype stype, bool ascending, unsigned int* indexes, bool todevice )
{
    unsigned int count = size();
    unsigned int* d_indexes = indexes;
    auto execpol = rmm::exec_policy(0);
    if( !todevice )
        RMM_ALLOC(&d_indexes,count*sizeof(unsigned int),0);
    thrust::sequence(execpol->on(0), d_indexes, d_indexes+count);
    //
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::sort(execpol->on(0), d_indexes, d_indexes+count,
        [ d_strings, stype, ascending ] __device__( unsigned int& lidx, unsigned int& ridx ) {
            custring_view* lhs = d_strings[lidx];
            custring_view* rhs = d_strings[ridx];
            bool cmp = false;
            if( lhs==0 || rhs==0 )
                cmp = lhs==0; // non-null > null
            else
            {   // allow sorting by name and length
                int diff = 0;
                if( stype & NVStrings::length )
                    diff = rhs->size() - lhs->size();
                if( diff==0 && (stype & NVStrings::name) )
                    diff = rhs->compare(*lhs) > 0;
                cmp = (diff > 0);
            }
            return ( ascending ? cmp : !cmp );
        });
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-order(0x%0x,%d,%p,%d)\n",(int)stype,(int)ascending,indexes,(int)todevice);
        printCudaError(err);
    }
    //
    if( !todevice )
    {
        cudaMemcpy(indexes,d_indexes,count*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        RMM_FREE(d_indexes,0);
    }
    return 0;
}
