
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
#include <locale.h>
#include "NVCategory.h"
#include "NVStrings.h"
#include "custring_view.cuh"
#include "custring.cuh"

//
typedef custring_view** custring_view_array;

#define ALIGN_SIZE(v)  (((v+7)/8)*8)

//
class NVCategoryImpl
{
public:
    //
    thrust::device_vector<custring_view*>* pList;
    thrust::device_vector<int>* pMap;
    void* memoryBuffer;
    size_t bufferSize; // total memory size

    //
    NVCategoryImpl() : bufferSize(0), memoryBuffer(0), pList(0), pMap(0)
    {}

    ~NVCategoryImpl()
    {
        if( memoryBuffer )
            cudaFree(memoryBuffer);
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
//
void NVCategoryImpl_init(NVCategoryImpl* pImpl, const char** strs, int count )
{
    thrust::device_vector<custring_view*> allstrings(count,nullptr);
    custring_view_array d_strings = allstrings.data().get();
    char* d_flatstrs = 0;

    // scoping here so intermediate data is freed when not needed
    // probably should be a utility function
    {
        // first, calculate total size of all the strings
        size_t nbytes = 0;
        thrust::host_vector<size_t> hoffsets(count+1,0);
        hoffsets[0] = 0;
        thrust::host_vector<size_t> hlengths(count,0);
        for( int idx=0; idx < count; ++idx )
        {
            const char* str = strs[idx];
            size_t len = ( str ? (strlen(str)+1) : 0 );
            size_t nsz = len; // include null-terminator
            if( len > 0 ) // len=0 is null, len=1 is empty string
            {
                int nchars = custring_view::chars_in_string(str,(int)len-1);
                nsz = custring_view::alloc_size((int)len-1,nchars);
                hlengths[idx] = len;
            }
            nsz = ALIGN_SIZE(nsz);
            nbytes += nsz;
            hoffsets[idx+1] = nbytes;
        }

        // host serialization -- copy all strings to one contiguous host buffer
        char* h_flatstrs = (char*)malloc(nbytes);
        for( int idx = 0; idx< count; ++idx )
            memcpy(h_flatstrs + hoffsets[idx], strs[idx], hlengths[idx]);
        // copy entire host buffer to device memory
        cudaMalloc(&d_flatstrs,nbytes);
        cudaMemcpy(d_flatstrs, h_flatstrs, nbytes, cudaMemcpyHostToDevice);
        free(h_flatstrs); // host memory no longer needed

        // copy offsets and lengths to device memory
        thrust::device_vector<size_t> offsets(hoffsets);
        thrust::device_vector<size_t> lengths(hlengths);
        size_t* d_offsets = offsets.data().get();
        size_t* d_lengths = lengths.data().get();
        // create d_strings from device memory
        thrust::for_each_n(thrust::device,
            thrust::make_counting_iterator<size_t>(0), count,
                [d_flatstrs, d_offsets, d_lengths, d_strings] __device__(size_t idx){
                  size_t len = d_lengths[idx];
                  if( len < 1 )
                      return; // null string; len==1 is empty string
                  size_t offset = d_offsets[idx];
                  char* ptr = d_flatstrs + offset;
                  d_strings[idx] = custring_view::create_from(ptr,ptr,(int)len-1);
            });
    }

    // sort the strings so we can remove duplicates easily
    // indexes are also moved so we can map them back to the original values
    thrust::device_vector<int> indexes(count);
    thrust::sequence(indexes.begin(),indexes.end());
    int* d_indexes = indexes.data().get();
    thrust::sort_by_key(thrust::device, d_strings, d_strings+count, d_indexes,
        [] __device__( custring_view*& lhs, custring_view*& rhs ) { return ( (!lhs || !rhs) ? (lhs==0) : (rhs->compare(*lhs)>0) ); });
    // build keys map
    thrust::device_vector<int>* pMap = new thrust::device_vector<int>(count,0);
    int* d_map = pMap->data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<int>(0), count,
        [d_strings, d_map] __device__ (int idx) {
            if( idx==0 )
                return;
            custring_view* dstr1 = d_strings[idx-1];
            custring_view* dstr2 = d_strings[idx];
            if( !dstr1 || !dstr2 )
                d_map[idx] = (int)(dstr1!=dstr2);
            else
                d_map[idx] = (int)dstr1->compare(*dstr2)!=0;
        });
    // each value in the map is index to the keys (string values)
    int ucount = thrust::reduce(thrust::device, pMap->begin(), pMap->end()) + 1;
    thrust::inclusive_scan(thrust::device, pMap->begin(), pMap->end(), pMap->begin());
    thrust::sort_by_key(thrust::device, indexes.begin(), indexes.end(), pMap->begin());
    pImpl->pMap = pMap;

    // now remove duplicates from string list
    thrust::unique(thrust::device, d_strings, d_strings+count,
        [] __device__ ( custring_view* lhs, custring_view* rhs ) { return ( (!lhs || !rhs) ? (lhs==rhs) : (rhs->compare(*lhs)==0) ); });

    // create new string vector of just the keys
    {
        // add up the lengths
        thrust::device_vector<size_t> lengths(ucount,0);
        size_t* d_lengths = lengths.data().get();
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), ucount,
            [d_strings, d_lengths] __device__(size_t idx){
                custring_view* dstr = d_strings[idx];
                if( dstr )
                    d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
            });
        // create output buffer to hold the string keys
        size_t outsize = thrust::reduce(thrust::device, lengths.begin(), lengths.end());
        char* d_buffer = 0;
        cudaMalloc(&d_buffer,outsize);
        thrust::device_vector<size_t> offsets(ucount,0);
        thrust::exclusive_scan(thrust::device,lengths.begin(),lengths.end(),offsets.begin());
        size_t* d_offsets = offsets.data().get();
        // create the vector to hold the pointers
        thrust::device_vector<custring_view*>* pList = new thrust::device_vector<custring_view*>(ucount,nullptr);
        custring_view_array d_results = pList->data().get();
        pImpl->addMemoryBuffer(d_buffer,outsize);
        // copy keys strings to new memory buffer
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), ucount,
            [d_strings, d_buffer, d_offsets, d_results] __device__ (size_t idx) {
                custring_view* dstr = d_strings[idx];
                if( dstr )
                    d_results[idx] = custring_view::create_from(d_buffer+d_offsets[idx],*dstr);
            });
        pImpl->pList = pList;
    }
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err!=cudaSuccess )
        printf("category1: error(%d) creating %'d unique strings\n",(int)err,ucount);
    cudaFree(d_flatstrs);
}

NVCategory* NVCategory::create_from_array(const char** strs, int count)
{
    NVCategory* rtn = new NVCategory;
    NVCategoryImpl_init(rtn->pImpl,strs,count);
    return rtn;
}

void NVCategoryImpl_init(NVCategoryImpl* pImpl, std::pair<const char*,size_t>* pairs, size_t count, bool bdevmem, bool bindexescopied=false )
{
    cudaError_t err = cudaSuccess;

    // make a copy of the indexes so we can sort them, etc
    thrust::pair<const char*,size_t>* d_pairs = 0;
    if( bdevmem )
    {
        if( bindexescopied )                                    // means caller already made a temp copy
            d_pairs = (thrust::pair<const char*,size_t>*)pairs; // and we can just use it here
        else
        {
            cudaMalloc(&d_pairs,sizeof(thrust::pair<const char*,size_t>)*count);
            cudaMemcpy(d_pairs,pairs,sizeof(thrust::pair<const char*,size_t>)*count,cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        cudaMalloc(&d_pairs,sizeof(thrust::pair<const char*,size_t>)*count);
        cudaMemcpy(d_pairs,pairs,sizeof(thrust::pair<const char*,size_t>)*count,cudaMemcpyHostToDevice);
    }
        
    //
    // example strings used in comments                                e,a,d,b,c,c,c,e,a
    //
    thrust::device_vector<int> indexes(count);
    thrust::sequence(thrust::device,indexes.begin(),indexes.end()); // 0,1,2,3,4,5,6,7,8
    int* d_indexes = indexes.data().get();
    // sort by key (string)                                            a,a,b,c,c,c,d,e,e
    // and indexes go along for the ride                               1,8,3,4,5,6,2,0,7
    thrust::sort_by_key(thrust::device, d_pairs, d_pairs+count, d_indexes,
        [] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
            if( lhs.first==0 || rhs.first==0 )
                return lhs.first==0; // non-null > null
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
        });

    // build the map; this will let us lookup strings by index
    thrust::device_vector<int>* pMap = new thrust::device_vector<int>(count,0);
    int* d_map = pMap->data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<int>(0), count,
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
    int ucount = thrust::reduce(thrust::device, pMap->begin(), pMap->end()) + 1;
    // scan converts to index values                                   0,0,1,2,2,2,3,4,4
    thrust::inclusive_scan(thrust::device, pMap->begin(), pMap->end(), pMap->begin());
    // re-sort will complete the map                                   4,0,3,1,2,2,2,4,0
    thrust::sort_by_key(thrust::device, indexes.begin(), indexes.end(), pMap->begin());
    pImpl->pMap = pMap;  // index -> str is now just a lookup in the map

    // now remove duplicates from string list                          a,b,c,d,e
    thrust::unique(thrust::device, d_pairs, d_pairs+count,
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
        thrust::device_vector<size_t> lengths(ucount,0);
        size_t* d_lengths = lengths.data().get();
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), ucount,
            [d_pairs, d_lengths] __device__(size_t idx){
                const char* str = d_pairs[idx].first;
                int bytes = (int)d_pairs[idx].second;
                if( str )
                    d_lengths[idx] = ALIGN_SIZE(custring_view::alloc_size((char*)str,bytes));
            });
        // create output buffer to hold the string keys
        size_t outsize = thrust::reduce(thrust::device, lengths.begin(), lengths.end());
        char* d_buffer = 0;
        cudaMalloc(&d_buffer,outsize);
        thrust::device_vector<size_t> offsets(ucount,0);
        thrust::exclusive_scan(thrust::device,lengths.begin(),lengths.end(),offsets.begin());
        size_t* d_offsets = offsets.data().get();
        // create the vector to hold the pointers
        thrust::device_vector<custring_view*>* pList = new thrust::device_vector<custring_view*>(ucount,nullptr);
        custring_view_array d_results = pList->data().get();
        pImpl->addMemoryBuffer(d_buffer,outsize);
        // copy keys strings to new memory buffer
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), ucount,
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
        printf("category2: error(%d) creating %'d strings\n",(int)err,ucount);
    if( !bindexescopied )
        cudaFree(d_pairs);
}

NVCategory* NVCategory::create_from_index(std::pair<const char*,size_t>* strs, size_t count, bool devmem )
{
    NVCategory* rtn = new NVCategory;
    NVCategoryImpl_init(rtn->pImpl,strs,count,devmem);
    return rtn;
}

void NVCategoryImpl_init(NVCategoryImpl* pImpl, NVStrings& strs )
{
    int count = strs.size();
    std::pair<const char*,size_t>* indexes = 0;
    cudaMalloc( &indexes, count * sizeof(std::pair<const char*,size_t>) );
    strs.create_index(indexes);
    NVCategoryImpl_init(pImpl,indexes,count,true,true);
    cudaFree(indexes);
}

void NVCategoryImpl_init(NVCategoryImpl* pImpl, std::vector<NVStrings*>& strs )
{
    unsigned int count = 0;
    for( unsigned int idx=0; idx < (unsigned int)strs.size(); idx++ )
        count += strs[idx]->size();
    std::pair<const char*,size_t>* indexes = 0;
    cudaMalloc( &indexes, count * sizeof(std::pair<const char*,size_t>) );
    std::pair<const char*,size_t>* ptr = indexes;
    for( unsigned int idx=0; idx < (unsigned int)strs.size(); idx++ )
    {
        strs[idx]->create_index(ptr);
        ptr += strs[idx]->size();
    }
    NVCategoryImpl_init(pImpl,indexes,count,true,true);
    cudaFree(indexes);
}

NVCategory* NVCategory::create_from_strings(NVStrings& strs)
{
    NVCategory* rtn = new NVCategory;
    NVCategoryImpl_init(rtn->pImpl,strs);
    return rtn;
}

NVCategory* NVCategory::create_from_strings(std::vector<NVStrings*>& strs)
{
    NVCategory* rtn = new NVCategory;
    NVCategoryImpl_init(rtn->pImpl,strs);
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

//
int NVCategory::create_null_bitarray( unsigned char* bitarray, bool emptyIsNull, bool devmem )
{
    int count = (int)size();
    if( count==0 )
        return 0;
    int size = (count + 7)/8;
    unsigned char* d_bitarray = bitarray;
    if( devmem )
        cudaMalloc(&d_bitarray,size);

    // get the null value index
    int nidx = get_value((const char*)0); // null index
    int eidx = get_value("");             // empty string index
    if( (nidx < 0) && (!emptyIsNull || (eidx < 0)) )
    {
        // no nulls, set everything to 1s
        cudaMemset(d_bitarray,255,size); // actually sets more bits than we need to
        if( !devmem )
        {
            cudaMemcpy(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost);
            cudaFree(d_bitarray);
        }
        return 0; // no nulls;
    }

    // count nulls in range for return value
    int* d_map = pImpl->getMapPtr();
    int ncount = thrust::count_if(thrust::device, d_map, d_map + count,
       [emptyIsNull,nidx,eidx] __device__ (int index) {
            return (index==nidx) || (emptyIsNull && (index==eidx));
         });
    // fill in the bitarray
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), size,
        [d_map, eidx, nidx, count, emptyIsNull, d_bitarray] __device__(size_t byteIdx){
            unsigned char byte = 0; // init all bits to zero
            for( int i=0; i < 8; ++i )
            {
                int idx = i + (byteIdx*8);
                byte = byte << 1;
                if( idx < count )
                {
                    int index = d_map[idx];
                    if( (index!=nidx) && (!emptyIsNull || (index!=eidx)) )
                        byte |= 1;
                }
            }
            d_bitarray[byteIdx] = byte;
        });
    cudaDeviceSynchronize();
    if( !devmem )
    {
        cudaMemcpy(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost);
        cudaFree(d_bitarray);
    }
    return ncount; // number of nulls
}

// build a string-index from this instances strings
int NVCategory::create_index(std::pair<const char*,size_t>* strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    custring_view_array d_strings = pImpl->getStringsPtr();
    int* d_map = pImpl->getMapPtr();
    thrust::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<unsigned int>(0), count,
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

    thrust::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), count,
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
    if( str )
    {
        bytes = (unsigned int)strlen(str);
        cudaMalloc(&d_str,bytes+1);
        cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice);
    }
    int count = keys_size();
    custring_view_array d_strings = pImpl->getStringsPtr();

    // find string in this instance
    thrust::device_vector<size_t> indexes(count,0);
    size_t* d_indexes = indexes.data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), count,
            [d_strings, d_str, bytes, d_indexes] __device__(size_t idx){
                custring_view* dstr = d_strings[idx];
                if( (char*)dstr==d_str ) // only true if both are null
                    d_indexes[idx] = idx+1;
                else if( dstr && dstr->compare(d_str,bytes)==0 )
                    d_indexes[idx] = idx+1;
            });
    // should only be one non-zero value in the result
    size_t cidx = thrust::reduce(thrust::device, indexes.begin(), indexes.end());
    // cidx==0 means string was not found
    if( d_str )
        cudaFree(d_str);
    return cidx-1; // -1 for not found, otherwise the key-index value
}

// return category values for all indexes
int NVCategory::get_values( unsigned int* results, bool bdevmem )
{
    int count = (int)size();
    int* d_map = pImpl->getMapPtr();
    if( bdevmem )
        cudaMemcpy(results,d_map,count*sizeof(unsigned int),cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(results,d_map,count*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    return count;
}

int NVCategory::get_indexes_for( unsigned int index, unsigned int* results, bool bdevmem )
{
    unsigned int count = size();
    if( index >= count )
        return -1;

    int* d_map = pImpl->getMapPtr();
    int matches = thrust::count_if( thrust::device, d_map, d_map+count, [index] __device__(int idx) { return idx==(int)index; });
    if( matches <= 0 )
        return 0; // done, found nothing, not likely
    if( results==0 )
        return matches; // caller just wants the count

    unsigned int* d_results = results;
    if( !bdevmem )
        cudaMalloc(&d_results,matches*sizeof(unsigned int));

    thrust::counting_iterator<unsigned int> itr(0);
    thrust::copy_if( thrust::device, itr, itr+count, d_results,
                     [index, d_map] __device__(unsigned int idx) { return d_map[idx]==(int)index; });
    cudaDeviceSynchronize();
    if( !bdevmem )
    {
        cudaMemcpy(results,d_results,matches*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        cudaFree(d_results);
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
    thrust::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    create_index((std::pair<const char*,size_t>*)d_indexes,count1);
    strs.create_index((std::pair<const char*,size_t>*)d_indexes+count1,count2);
    // build the category from this new set
    return create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// creates a new instance without the specified strings
NVCategory* NVCategory::remove_strings(NVStrings& strs)
{
    unsigned int count = size();
    thrust::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    create_index((std::pair<const char*,size_t>*)d_indexes,count);

    unsigned int delete_count = strs.size();
    thrust::device_vector< thrust::pair<const char*,size_t> > deletes(delete_count);
    thrust::pair<const char*,size_t>* d_deletes = deletes.data().get();
    strs.create_index((std::pair<const char*,size_t>*)d_deletes,delete_count);

    // this would be inefficient if strs is very large
    thrust::pair<const char*,size_t>* newend = thrust::remove_if(thrust::device, d_indexes, d_indexes + count,
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
    thrust::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), count,
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
    unsigned int* d_pos = pos;
    if( !bdevmem )
    {
        cudaMalloc(&d_pos,count*sizeof(unsigned int));
        cudaMemcpy(d_pos,pos,count*sizeof(unsigned int),cudaMemcpyHostToDevice);
    }

    custring_view** d_strings = pImpl->getStringsPtr();
    // use the map to build the indexes array
    thrust::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), count,
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
        cudaFree(d_pos);
    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}
