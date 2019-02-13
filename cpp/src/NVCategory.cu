
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <locale.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVCategory.h"
#include "NVStrings.h"
#include "custring_view.cuh"
#include "custring.cuh"

//
typedef custring_view** custring_view_array;

#define ALIGN_SIZE(v)  (((v+7)/8)*8)

//static void printDeviceInts( const char* title, int* d_ints, int count )
//{
//    thrust::host_vector<int> ints(count);
//    int* h_ints = ints.data();
//    cudaMemcpy( h_ints, d_ints, count * sizeof(int), cudaMemcpyDeviceToHost);
//    if( title )
//        printf("%s:\n",title);
//    for( int i=0; i < count; ++i )
//        printf(" %d",h_ints[i]);
//    printf("\n");
//}

//
class NVCategoryImpl
{
public:
    //
    rmm::device_vector<custring_view*>* pList;
    rmm::device_vector<int>* pMap;
    void* memoryBuffer;
    size_t bufferSize; // total memory size
    cudaStream_t stream_id;

    //
    NVCategoryImpl() : bufferSize(0), memoryBuffer(0), pList(0), pMap(0), stream_id(0)
    {}

    ~NVCategoryImpl()
    {
        if( memoryBuffer )
            RMM_FREE(memoryBuffer,0);
        delete pList;
        delete pMap;
        memoryBuffer = 0;
        bufferSize = 0;
    }

    inline custring_view_array getStringsPtr()
    {
        return pList->data().get();
    }

    inline int* getMapPtr()
    {
        return pMap->data().get();
    }

    inline void addMemoryBuffer( void* ptr, size_t memSize )
    {
        bufferSize += memSize;
        memoryBuffer = ptr;
    }
};

//
NVCategory::NVCategory()
{
    pImpl = new NVCategoryImpl;
}

NVCategory::~NVCategory()
{
    delete pImpl;
}

// Utility to create category instance data from array of string pointers (in device memory).
// It does all operations using the given pointers (or copies) to build the map.
// This method can be given the index values from the NVStrings::create_index.
// So however an NVStrings can be created can also create an NVCategory.
void NVCategoryImpl_init(NVCategoryImpl* pImpl, std::pair<const char*,size_t>* pairs, size_t count, bool bdevmem, bool bindexescopied=false )
{
    cudaError_t err = cudaSuccess;
    auto execpol = rmm::exec_policy(0);

    // make a copy of the indexes so we can sort them, etc
    thrust::pair<const char*,size_t>* d_pairs = 0;
    if( bdevmem )
    {
        if( bindexescopied )                                    // means caller already made a temp copy
            d_pairs = (thrust::pair<const char*,size_t>*)pairs; // and we can just use it here
        else
        {
            RMM_ALLOC(&d_pairs,sizeof(thrust::pair<const char*,size_t>)*count,0);
            cudaMemcpy(d_pairs,pairs,sizeof(thrust::pair<const char*,size_t>)*count,cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        RMM_ALLOC(&d_pairs,sizeof(thrust::pair<const char*,size_t>)*count,0);
        cudaMemcpy(d_pairs,pairs,sizeof(thrust::pair<const char*,size_t>)*count,cudaMemcpyHostToDevice);
    }

    //
    // example strings used in comments                                e,a,d,b,c,c,c,e,a
    //
    rmm::device_vector<int> indexes(count);
    thrust::sequence(execpol->on(0),indexes.begin(),indexes.end()); // 0,1,2,3,4,5,6,7,8
    int* d_indexes = indexes.data().get();
    // sort by key (string)                                            a,a,b,c,c,c,d,e,e
    // and indexes go along for the ride                               1,8,3,4,5,6,2,0,7
    thrust::sort_by_key(execpol->on(0), d_pairs, d_pairs+count, d_indexes,
        [] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
            if( lhs.first==0 || rhs.first==0 )
                return lhs.first==0; // non-null > null
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
        });

    // build the map; this will let us lookup strings by index
    rmm::device_vector<int>* pMap = new rmm::device_vector<int>(count,0);
    int* d_map = pMap->data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count,
        [d_pairs, d_map] __device__ (int idx) {
            if( idx==0 )
                return;
            const char* ptr1 = d_pairs[idx-1].first;
            const char* ptr2 = d_pairs[idx].first;
            unsigned int len1 = (unsigned int)d_pairs[idx-1].second, len2 = (unsigned int)d_pairs[idx].second;
            //d_map[idx] = (int)(custr::compare(ptr1,len1,ptr2,len2)!=0);
            int cmp = 0; // vvvvv - probably faster than - ^^^^^
            if( !ptr1 || !ptr2 )
                cmp = (int)(ptr1!=ptr2);
            else if( len1 != len2 )
                cmp = 1;
            else
                for( int i=0; !cmp && (i < len1); ++i)
                    cmp = (int)(*ptr1++ != *ptr2++);
            d_map[idx] = cmp;
        });
    //
    // d_map now identifies just string changes                        0,0,1,1,0,0,1,1,0
    int ucount = thrust::reduce(execpol->on(0), pMap->begin(), pMap->end()) + 1;
    // scan converts to index values                                   0,0,1,2,2,2,3,4,4
    thrust::inclusive_scan(execpol->on(0), pMap->begin(), pMap->end(), pMap->begin());
    // re-sort will complete the map                                   4,0,3,1,2,2,2,4,0
    thrust::sort_by_key(execpol->on(0), indexes.begin(), indexes.end(), pMap->begin());
    pImpl->pMap = pMap;  // index -> str is now just a lookup in the map

    // now remove duplicates from string list                          a,b,c,d,e
    thrust::unique(execpol->on(0), d_pairs, d_pairs+count,
        [] __device__ ( thrust::pair<const char*,size_t> lhs, thrust::pair<const char*,size_t> rhs ) {
            if( lhs.first==0 || rhs.first==0 )
                return lhs.first==rhs.first;
            if( lhs.second != rhs.second )
                return false;
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0;
        });

    // finally, create new string vector of just the keys
    {
        // add up the lengths
        rmm::device_vector<size_t> lengths(ucount,0);
        size_t* d_lengths = lengths.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
            [d_pairs, d_lengths] __device__(size_t idx){
                const char* str = d_pairs[idx].first;
                int bytes = (int)d_pairs[idx].second;
                if( str )
                    d_lengths[idx] = ALIGN_SIZE(custring_view::alloc_size((char*)str,bytes));
            });
        // create output buffer to hold the string keys
        size_t outsize = thrust::reduce(execpol->on(0), lengths.begin(), lengths.end());
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,outsize,0);
        rmm::device_vector<size_t> offsets(ucount,0);
        thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
        size_t* d_offsets = offsets.data().get();
        // create the vector to hold the pointers
        rmm::device_vector<custring_view*>* pList = new rmm::device_vector<custring_view*>(ucount,nullptr);
        custring_view_array d_results = pList->data().get();
        pImpl->addMemoryBuffer(d_buffer,outsize);
        // copy keys strings to new memory buffer
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
            [d_pairs, d_buffer, d_offsets, d_results] __device__ (size_t idx) {
                const char* str = d_pairs[idx].first;
                int bytes = (int)d_pairs[idx].second;
                if( str )
                    d_results[idx] = custring_view::create_from(d_buffer+d_offsets[idx],(char*)str,bytes);
            });
        pImpl->pList = pList;
    }

    err = cudaDeviceSynchronize();
    if( err!=cudaSuccess )
        printf("category: error(%d) creating %'d strings\n",(int)err,ucount);
    if( !bindexescopied )
        RMM_FREE(d_pairs,0);
}

NVCategory* NVCategory::create_from_index(std::pair<const char*,size_t>* strs, size_t count, bool devmem )
{
    NVCategory* rtn = new NVCategory;
    NVCategoryImpl_init(rtn->pImpl,strs,count,devmem);
    return rtn;
}

NVCategory* NVCategory::create_from_array(const char** strs, int count)
{
    NVCategory* rtn = new NVCategory;
    //NVCategoryImpl_init(rtn->pImpl,strs,count);
    NVStrings* dstrs = NVStrings::create_from_array(strs,count);
    std::pair<const char*,size_t>* indexes = 0;
    RMM_ALLOC(&indexes, count * sizeof(std::pair<const char*,size_t>),0);
    dstrs->create_index(indexes);
    NVCategoryImpl_init(rtn->pImpl,indexes,count,true,true);
    RMM_FREE(indexes,0);
    NVStrings::destroy(dstrs);
    return rtn;
}

NVCategory* NVCategory::create_from_strings(NVStrings& strs)
{
    NVCategory* rtn = new NVCategory;
    int count = strs.size();
    std::pair<const char*,size_t>* indexes = 0;
    RMM_ALLOC(&indexes, count * sizeof(std::pair<const char*,size_t>),0);
    strs.create_index(indexes);
    NVCategoryImpl_init(rtn->pImpl,indexes,count,true,true);
    RMM_FREE(indexes,0);
    return rtn;
}

NVCategory* NVCategory::create_from_strings(std::vector<NVStrings*>& strs)
{
    NVCategory* rtn = new NVCategory;
    unsigned int count = 0;
    for( unsigned int idx=0; idx < (unsigned int)strs.size(); idx++ )
        count += strs[idx]->size();
    std::pair<const char*,size_t>* indexes = 0;
    RMM_ALLOC(&indexes, count * sizeof(std::pair<const char*,size_t>),0);
    std::pair<const char*,size_t>* ptr = indexes;
    for( unsigned int idx=0; idx < (unsigned int)strs.size(); idx++ )
    {
        strs[idx]->create_index(ptr);
        ptr += strs[idx]->size();
    }
    NVCategoryImpl_init(rtn->pImpl,indexes,count,true,true);
    RMM_FREE(indexes,0);
    return rtn;
}

NVCategory* NVCategory::create_from_offsets(const char* strs, int count, const int* offsets, const unsigned char* nullbitmask, int nulls)
{
    NVCategory* rtn = new NVCategory;
    NVStrings* dstrs = NVStrings::create_from_offsets(strs,count,offsets,nullbitmask,nulls);
    std::pair<const char*,size_t>* indexes = 0;
    RMM_ALLOC(&indexes, count * sizeof(std::pair<const char*,size_t>),0);
    dstrs->create_index(indexes);
    NVCategoryImpl_init(rtn->pImpl,indexes,count,true,true);
    RMM_FREE(indexes,0);
    NVStrings::destroy(dstrs);
    return rtn;
}

void NVCategory::destroy(NVCategory* inst)
{
    delete inst;
}

// return number of items
unsigned int NVCategory::size()
{
    return pImpl->pMap->size();
}

// return number of keys
unsigned int NVCategory::keys_size()
{
    return pImpl->pList->size();
}

// true if any null values exist
bool NVCategory::has_nulls()
{
    unsigned int count = keys_size();
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    int n = thrust::count_if(execpol->on(0), d_strings, d_strings+count,
            []__device__(custring_view* dstr) { return dstr==0; } );
    return n > 0;
}

// bitarray is for the values
// return the number of null values found
int NVCategory::set_null_bitarray( unsigned char* bitarray, bool devmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned int size = (count + 7)/8;
    unsigned char* d_bitarray = bitarray;
    if( !devmem )
        RMM_ALLOC(&d_bitarray,size,0);

    int nidx = -1;
    {
        custring_view_array d_strings = pImpl->getStringsPtr();
        rmm::device_vector<int> nulls(1,-1);
        thrust::copy_if( execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(keys_size()), nulls.begin(),
                [d_strings] __device__ (unsigned int idx) { return d_strings[idx]==0; } );
        nidx = nulls[0]; // should be the index of the null entry (or -1)
    }

    if( nidx < 0 )
    {   // no nulls, set everything to 1s
        cudaMemset(d_bitarray,255,size); // actually sets more bits than we need to
        if( !devmem )
        {
            cudaMemcpy(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost);
            RMM_FREE(d_bitarray,0);
        }
        return 0; // no nulls;
    }

    // count nulls in range for return value
    int* d_map = pImpl->getMapPtr();
    unsigned int ncount = thrust::count_if(execpol->on(0), d_map, d_map + count,
        [nidx] __device__ (int index) { return (index==nidx); });
    // fill in the bitarray
    // the bitmask is in arrow format which means for each byte
    // the null indicator is in bit position right-to-left: 76543210
    // logic sets the high-bit and shifts to the right
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), size,
        [d_map, nidx, count, d_bitarray] __device__(unsigned int byteIdx){
            unsigned char byte = 0; // init all bits to zero
            for( unsigned int i=0; i < 8; ++i )
            {
                unsigned int idx = i + (byteIdx*8);
                byte = byte >> 1;
                if( idx < count )
                {
                    int index = d_map[idx];
                    byte |= (unsigned char)((index!=nidx) << 7);
                }
            }
            d_bitarray[byteIdx] = byte;
        });
    cudaDeviceSynchronize();
    if( !devmem )
    {
        cudaMemcpy(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost);
        RMM_FREE(d_bitarray,0);
    }
    return ncount; // number of nulls
}

// build a string-index from this instances strings
int NVCategory::create_index(std::pair<const char*,size_t>* strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    int* d_map = pImpl->getMapPtr();
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_map, d_indexes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[d_map[idx]];
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

    cudaDeviceSynchronize();
    //
    if( bdevmem )
        cudaMemcpy( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToDevice );
    else
        cudaMemcpy( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToHost );
    return 0;
}

// return strings keys for this instance
NVStrings* NVCategory::get_keys()
{
    int count = keys_size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_indexes] __device__(size_t idx){
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

    cudaDeviceSynchronize();

    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

//
int NVCategory::get_value(unsigned int index)
{
    if( index >= size() )
        return -1;
    int* d_map = pImpl->getMapPtr();
    int rtn = -1;
    cudaMemcpy(&rtn,d_map+index,sizeof(int),cudaMemcpyDeviceToHost);
    return rtn;
}

//
int NVCategory::get_value(const char* str)
{
    char* d_str = 0;
    unsigned int bytes = 0;
    auto execpol = rmm::exec_policy(0);
    if( str )
    {
        bytes = (unsigned int)strlen(str);
        RMM_ALLOC(&d_str,bytes+1,0);
        cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);
    }
    int count = keys_size();
    custring_view_array d_strings = pImpl->getStringsPtr();

    // find string in this instance
    //rmm::device_vector<size_t> indexes(count,0);
    //size_t* d_indexes = indexes.data().get();
    //thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
    //        [d_strings, d_str, bytes, d_indexes] __device__(size_t idx){
    //            custring_view* dstr = d_strings[idx];
    //            if( (char*)dstr==d_str ) // only true if both are null
    //                d_indexes[idx] = idx+1;
    //            else if( dstr && dstr->compare(d_str,bytes)==0 )
    //                d_indexes[idx] = idx+1;
    //        });
    //// should only be one non-zero value in the result
    //size_t cidx = thrust::reduce(execpol->on(0), indexes.begin(), indexes.end());
    //// cidx==0 means string was not found
    //return cidx-1; // -1 for not found, otherwise the key-index value

    rmm::device_vector<int> keys(1,-1);
    thrust::copy_if( execpol->on(0), thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), keys.begin(),
         [d_strings, d_str, bytes] __device__ (int idx) {
             custring_view* dstr = d_strings[idx];
             if( (char*)dstr==d_str ) // only true if both are null
                 return true;
             return ( dstr && dstr->compare(d_str,bytes)==0 );
         } );
    cudaDeviceSynchronize();
    if( d_str )
        RMM_FREE(d_str,0);
    return keys[0];
}

// return category values for all indexes
int NVCategory::get_values( int* results, bool bdevmem )
{
    int count = (int)size();
    int* d_map = pImpl->getMapPtr();
    if( bdevmem )
        cudaMemcpy(results,d_map,count*sizeof(int),cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(results,d_map,count*sizeof(int),cudaMemcpyDeviceToHost);
    return count;
}

const int* NVCategory::values_cptr()
{
    return pImpl->getMapPtr();
}

int NVCategory::get_indexes_for( unsigned int index, unsigned int* results, bool bdevmem )
{
    unsigned int count = size();
    if( index >= count )
        return -1;

    auto execpol = rmm::exec_policy(0);
    int* d_map = pImpl->getMapPtr();
    int matches = thrust::count_if( execpol->on(0), d_map, d_map+count, [index] __device__(int idx) { return idx==(int)index; });
    if( matches <= 0 )
        return 0; // done, found nothing, not likely
    if( results==0 )
        return matches; // caller just wants the count

    unsigned int* d_results = results;
    if( !bdevmem )
        RMM_ALLOC(&d_results,matches*sizeof(unsigned int),0);

    thrust::counting_iterator<unsigned int> itr(0);
    thrust::copy_if( execpol->on(0), itr, itr+count, d_results,
                     [index, d_map] __device__(unsigned int idx) { return d_map[idx]==(int)index; });
    cudaDeviceSynchronize();
    if( !bdevmem )
    {
        cudaMemcpy(results,d_results,matches*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        RMM_FREE(d_results,0);
    }
    return matches;
}

int NVCategory::get_indexes_for( const char* str, unsigned int* results, bool bdevmem )
{
    int id = get_value(str);
    if( id < 0 )
        return id;
    return get_indexes_for((unsigned int)id, results, bdevmem);
}

// creates a new instance incorporating the new strings
NVCategory* NVCategory::add_strings(NVStrings& strs)
{
    // create one large index of both datasets
    unsigned int count1 = size();
    unsigned int count2 = strs.size();
    unsigned int count = count1 + count2;
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    create_index((std::pair<const char*,size_t>*)d_indexes,count1);
    strs.create_index((std::pair<const char*,size_t>*)d_indexes+count1,count2);
    // build the category from this new set
    return create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

//
NVCategory* NVCategory::merge_category(NVCategory& cat2)
{
    auto execpol = rmm::exec_policy(0);
    unsigned int count1 = keys_size();
    unsigned int mcount1 = size();
    unsigned int count2 = cat2.keys_size();
    unsigned int mcount2 = cat2.size();
    NVCategory* rtn = new NVCategory();
    if( (count1==0) && (count2==0) )
        return rtn;
    unsigned int count12 = count1 + count2;
    unsigned int mcount = mcount1 + mcount2;
    // if either category is empty, just copy the non-empty one
    // copying category probably should be a utility
    if( (count1==0) || (count2==0) )
    {
        unsigned int ucount = count12;
        rmm::device_vector<custring_view*>* pNewList = new rmm::device_vector<custring_view*>(ucount,nullptr);
        rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
        NVCategory* dcat = ((count1==0) ? &cat2 : this);
        // copy map values from non-empty category instance
        cudaMemcpy( pNewMap->data().get(), dcat->pImpl->pMap->data().get(), mcount*sizeof(int), cudaMemcpyDeviceToDevice );
        rtn->pImpl->pMap = pNewMap;
        // copy key strings buffer
        char* d_buffer = (char*)dcat->pImpl->memoryBuffer;
        size_t bufsize = dcat->pImpl->bufferSize;
        char* d_newbuffer = 0;
        RMM_ALLOC(&d_newbuffer,bufsize,0);
        cudaMemcpy(d_newbuffer,d_buffer,bufsize,cudaMemcpyDeviceToDevice);
        // need to set custring_view ptrs
        custring_view_array d_strings = dcat->pImpl->getStringsPtr();
        custring_view_array d_results = pNewList->data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
            [d_strings, d_buffer, d_newbuffer, d_results] __device__ (size_t idx) {
                custring_view* dstr = d_strings[idx];
                char* buffer = d_newbuffer + (size_t)dstr - (size_t)d_buffer;
                if( dstr )
                    d_results[idx] = (custring_view*)buffer;
            });
        rtn->pImpl->pList = pNewList;
        rtn->pImpl->addMemoryBuffer( d_newbuffer, bufsize );
        return rtn;
    }
    // both this cat and cat2 are non-empty
    // init working vars
    custring_view_array d_keys1 = pImpl->getStringsPtr();
    int* d_map1 = pImpl->pMap->data().get();
    custring_view_array d_keys2 = cat2.pImpl->getStringsPtr();
    int* d_map2 = cat2.pImpl->pMap->data().get();
    // create some vectors we can sort
    rmm::device_vector<custring_view*> wstrs(count12); // w = key2 + keys1
    custring_view_array d_w = wstrs.data().get();
    cudaMemcpy(d_w, d_keys2, count2*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_w+count2, d_keys1, count1*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    rmm::device_vector<int> x(count12);  // 0,1,....count2,-1,...,-count1
    int* d_x = x.data().get();
    // sequence and for-each-n could be combined into for-each-n logic
    thrust::sequence( execpol->on(0), d_x, d_x+count2 );   // first half is 0...count2
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count1,
        [d_x, count2] __device__ (int idx) { d_x[idx+count2]= -idx-1; }); // 2nd half is -1...-count1
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w + count12, d_x,  // preserves order for
        [] __device__ (custring_view*& lhs, custring_view*& rhs) {        // strings that match
            return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (lhs==0));
        });
    rmm::device_vector<int> y(count12,0); // y-vector will identify overlapped keys
    int* d_y = y.data().get();
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), (count12-1),
        [d_y, d_w] __device__ (int idx) {
            custring_view* lhs = d_w[idx];
            custring_view* rhs = d_w[idx+1];
            if( lhs && rhs )
                d_y[idx] = (int)(lhs->compare(*rhs)==0);
            else
                d_y[idx] = (int)(lhs==rhs);
        });
    int matched = thrust::reduce( execpol->on(0), d_y, d_y + count12 ); // how many keys matched
    unsigned int ncount = count2 - (unsigned int)matched; // new keys count
    unsigned int ucount = count1 + ncount; // total unique keys count
    rmm::device_vector<custring_view*>* pNewList = new rmm::device_vector<custring_view*>(ucount,nullptr);
    custring_view_array d_keys = pNewList->data().get(); // this will hold the merged keyset
    rmm::device_vector<int> nidxs(ucount); // needed for various gather methods below
    int* d_nidxs = nidxs.data().get(); // indexes of 'new' keys from key2 not in key1
    {
        thrust::counting_iterator<int> citr(0);
        thrust::copy_if( execpol->on(0), citr, citr + (count12), d_nidxs,
            [d_x, d_y] __device__ (const int& idx) { return (d_x[idx]>=0) && (d_y[idx]==0); });
    }
    // first half of merged keyset is direct copy of key1
    cudaMemcpy( d_keys, d_keys1, count1*sizeof(custring_view*), cudaMemcpyDeviceToDevice);
    // append the 'new' keys from key2: extract them from w as identified by nidxs
    thrust::gather( execpol->on(0), d_nidxs, d_nidxs + ncount, d_w, d_keys + count1 );
    int* d_ubl = d_x; // reuse d_x for unique-bias-left values
    thrust::unique_by_key( execpol->on(0), d_w, d_w + count12, d_ubl,
         [] __device__ (custring_view* lhs, custring_view* rhs) {
            return ((lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs));
         });  // ubl now contains new index values for key2
    int* d_sws = d_y; // reuse d_y for sort-with-seq values
    thrust::sequence( execpol->on(0), d_sws, d_sws + ucount); // need to assign new index values
    rmm::device_vector<custring_view*> keySort(ucount);    // for all the original key2 values
    cudaMemcpy( keySort.data().get(), d_keys, ucount * sizeof(custring_view*), cudaMemcpyDeviceToDevice);
    thrust::sort_by_key( execpol->on(0), keySort.begin(), keySort.end(), d_sws,
        [] __device__ (custring_view*& lhs, custring_view*& rhs ) {
            return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (lhs==0));
        }); // sws is now key index values for the new keyset
    //printDeviceInts("d_sws",d_sws,ucount);
    {
        thrust::counting_iterator<int> citr(0); // generate subset of just the key2 values
        thrust::copy_if( execpol->on(0), citr, citr + ucount, d_nidxs, [d_ubl] __device__ (const int& idx) { return d_ubl[idx]>=0; });
    }
    // nidxs has the indexes to the key2 values in the new keyset but they are sorted when key2 may not have been
    rmm::device_vector<int> remap2(count2); // need to remap the indexes to the original positions
    int* d_remap2 = remap2.data().get();       // do this by de-sorting the key2 values from the full keyset
    thrust::gather( execpol->on(0), d_nidxs, d_nidxs + count2, d_sws, d_remap2 ); // here grab new positions for key2
    // first, remove the key1 indexes from the sorted sequence values; ubl will then have only key2 orig. pos values
    thrust::remove_if( execpol->on(0), d_ubl, d_ubl + ucount, [] __device__ (int v) { return v<0; });
    thrust::sort_by_key( execpol->on(0), d_ubl, d_ubl+count2, d_remap2 ); // does a de-sort of key2 only
    // build new map
    rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
    int* d_map = pNewMap->data().get(); // first half is identical to map1
    cudaMemcpy( d_map, d_map1, mcount1 * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy( d_map+mcount1, d_map2, mcount2 * sizeof(int), cudaMemcpyDeviceToDevice);
    // remap map2 values to their new positions in the full keyset
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(mcount1), mcount2,
        [d_map, d_remap2] __device__ (int idx) { d_map[idx] = d_remap2[d_map[idx]]; });
    // finally, need to copy the pNewList keys strings to new memory
    {   // copying strings should likely be a utility
        // add up the lengths
        rmm::device_vector<size_t> lengths(ucount,0);
        size_t* d_lengths = lengths.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
            [d_keys, d_lengths] __device__(size_t idx){
                custring_view* dstr = d_keys[idx];
                if( dstr )
                    d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
            });
        // create output buffer to hold the string keys
        size_t outsize = thrust::reduce(execpol->on(0), lengths.begin(), lengths.end());
        char* d_buffer = 0;
        RMM_ALLOC(&d_buffer,outsize,0);
        rmm::device_vector<size_t> offsets(ucount,0);
        thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
        size_t* d_offsets = offsets.data().get();
        // old ptrs are replaced with new ones using d_buffer and d_offsets
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
            [d_keys, d_buffer, d_offsets] __device__ (size_t idx) {
                custring_view* dstr = d_keys[idx];
                if( dstr )
                    d_keys[idx] = custring_view::create_from(d_buffer+d_offsets[idx],*dstr);
            });
        rtn->pImpl->addMemoryBuffer(d_buffer,outsize);
    }
    rtn->pImpl->pList = pNewList;
    rtn->pImpl->pMap = pNewMap;
    return rtn;
}

// creates a new instance without the specified strings
NVCategory* NVCategory::remove_strings(NVStrings& strs)
{
    auto execpol = rmm::exec_policy(0);
    unsigned int count = size();
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    create_index((std::pair<const char*,size_t>*)d_indexes,count);

    unsigned int delete_count = strs.size();
    rmm::device_vector< thrust::pair<const char*,size_t> > deletes(delete_count);
    thrust::pair<const char*,size_t>* d_deletes = deletes.data().get();
    strs.create_index((std::pair<const char*,size_t>*)d_deletes,delete_count);

    // this would be inefficient if strs is very large
    thrust::pair<const char*,size_t>* newend = thrust::remove_if(execpol->on(0), d_indexes, d_indexes + count,
        [d_deletes,delete_count] __device__ (thrust::pair<const char*,size_t> lhs) {
            for( unsigned int idx=0; idx < delete_count; ++idx )
            {
                thrust::pair<const char*,size_t> rhs = d_deletes[idx];
                if( lhs.first == rhs.first )
                    return true;
                if( lhs.second != rhs.second )
                    continue;
                if( custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0 )
                    return true;
            }
            return false;
        });
    // return value ensures a dev-sync has already been performed by thrust
    count = (unsigned int)(newend - d_indexes); // new count of strings
    // build the category from this new set
    return create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// basically recreates the original string list
NVStrings* NVCategory::to_strings()
{
    int count = (int)size();
    int* d_map = pImpl->getMapPtr();
    custring_view** d_strings = pImpl->getStringsPtr();
    // use the map to build the indexes array
    auto execpol = rmm::exec_policy(0);
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_map, d_indexes] __device__(size_t idx){
            int stridx = d_map[idx];
            custring_view* dstr = d_strings[stridx];
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
    //
    cudaDeviceSynchronize();
    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// creates a new NVStrings instance using the specified index values
NVStrings* NVCategory::gather_strings( unsigned int* pos, unsigned int count, bool bdevmem )
{
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_pos = pos;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_pos,count*sizeof(unsigned int),0);
        cudaMemcpy(d_pos,pos,count*sizeof(unsigned int),cudaMemcpyHostToDevice);
    }

    custring_view** d_strings = pImpl->getStringsPtr();
    // use the map to build the indexes array
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_pos, d_indexes] __device__(size_t idx){
            int stridx = d_pos[idx];
            custring_view* dstr = d_strings[stridx];
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
    //
    cudaDeviceSynchronize();
    if( !bdevmem )
        RMM_FREE(d_pos,0);
    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}
