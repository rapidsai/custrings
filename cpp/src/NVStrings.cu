
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"


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

    // compute size of specified strings
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
    {
        memset(list,0,count*sizeof(char*));
        return 0; // every string is null so we are done
    }

    // allocate device memory to copy strings temporarily
    char* d_buffer = 0;
    rmmError_t rerr = RMM_ALLOC(&d_buffer,msize,0);
    if( rerr != RMM_SUCCESS )
    {
        fprintf(stderr,"nvs-to_host: RM_ALLOC(%p,%lu)=%d\n", d_buffer,msize,(int)rerr);
        //printCudaError(err);
        return (int)err;
    }
    // convert lengths to offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // copy strings into temporary buffer
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
    RMM_FREE(d_buffer,0); // done with device buffer
    if( err != cudaSuccess )
    {
        printCudaError(err, "nvs-to_host: copying strings device to host");
        delete h_buffer;
        return (int)err;
    }

    // Deserialization host memory to memory provided by the caller
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
        //char* h_data = new char[len]; // make memory on the host
        //h_data[len-1] = 0; // null terminate for the caller
        //memcpy(h_data, p_data, len-1);
        //list[idx] = h_data;
        if( list[idx] )
            memcpy(list[idx], p_data, len-1);
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

