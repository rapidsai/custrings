/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <stdlib.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"


// create a new instance containing only the strings at the specified positions
// position values can be in any order and can even be repeated
NVStrings* NVStrings::gather( const int* pos, unsigned int elems, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || elems==0 || pos==0 )
        return new NVStrings(0);

    auto execpol = rmm::exec_policy(0);
    const int* d_pos = pos;
    if( !bdevmem )
    {   // copy indexes to device memory
        RMM_ALLOC((void**)&d_pos,elems*sizeof(int),0);
        cudaMemcpy((void*)d_pos,pos,elems*sizeof(int),cudaMemcpyHostToDevice);
    }
    // get individual sizes
    rmm::device_vector<long> sizes(elems,0);
    long* d_sizes = sizes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elems,
        [d_strings, d_pos, count, d_sizes] __device__(unsigned int idx){
            int pos = d_pos[idx];
            if( (pos < 0) || (pos >= count) )
            {
                d_sizes[idx] = -1;
                return;
            }
            custring_view* dstr = d_strings[pos];
            if( dstr )
                d_sizes[idx] = ALIGN_SIZE(dstr->alloc_size());
        });
    // check for any out-of-range values
    long* first = thrust::min_element(execpol->on(0),d_sizes,d_sizes+elems);
    long hfirst = 0;
    cudaMemcpy(&hfirst,first,sizeof(long),cudaMemcpyDeviceToHost);
    if( hfirst < 0 )
    {
        if( !bdevmem )
            RMM_FREE((void*)d_pos,0);
        throw std::out_of_range("");
    }

    // create output object
    NVStrings* rtn = new NVStrings(elems);
    char* d_buffer = rtn->pImpl->createMemoryFor((size_t*)d_sizes);
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
                //if( (pos < 0) || (pos >= count) )
                //    return;  -- should no longer happen
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
        RMM_FREE((void*)d_pos,0);
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
NVStrings* NVStrings::remove_strings( const int* pos, unsigned int elems, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || elems==0 || pos==0 )
        return 0; // return copy of ourselves?

    auto execpol = rmm::exec_policy(0);
    int* dpos = 0;
    RMM_ALLOC(&dpos,elems*sizeof(unsigned int),0);
    if( bdevmem )
       cudaMemcpy((void*)dpos,pos,elems*sizeof(unsigned int),cudaMemcpyDeviceToDevice);
    else
       cudaMemcpy((void*)dpos,pos,elems*sizeof(unsigned int),cudaMemcpyHostToDevice);
    // sort the position values
    thrust::sort(execpol->on(0),dpos,dpos+elems,thrust::greater<int>());
    // also should remove duplicates
    int* nend = thrust::unique(execpol->on(0),dpos,dpos+elems,thrust::equal_to<int>());
    elems = (unsigned int)(nend - dpos);
    if( count < elems )
    {
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
    rmm::device_vector<custring_view*> newList(newCount,nullptr);              // newList will hold
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
    RMM_FREE(dpos,0);
    return rtn;
}


// this now sorts the strings into a new instance;
// a sorted strings list can improve performance by reducing divergence
NVStrings* NVStrings::sort( sorttype stype, bool ascending, bool nullfirst )
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
        [stype, ascending, nullfirst] __device__( custring_view*& lhs, custring_view*& rhs ) {
            if( lhs==0 || rhs==0 )
                return (nullfirst ? rhs!=0 : lhs!=0); // null < non-null
            // allow sorting by name and length
            int diff = 0;
            if( stype & NVStrings::length )
                diff = lhs->size() - rhs->size();
            if( diff==0 && (stype & NVStrings::name) )
                diff = lhs->compare(*rhs);
            return (ascending ? (diff < 0) : (diff > 0));
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
int NVStrings::order( sorttype stype, bool ascending, unsigned int* indexes, bool nullfirst, bool todevice )
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
        [d_strings, stype, ascending, nullfirst] __device__( unsigned int& lidx, unsigned int& ridx ) {
            custring_view* lhs = d_strings[lidx];
            custring_view* rhs = d_strings[ridx];
            if( lhs==0 || rhs==0 )
                return (nullfirst ? rhs!=0 : lhs!=0);
            // allow sorting by name and length
            int diff = 0;
            if( stype & NVStrings::length )
                diff = lhs->size() - rhs->size();
            if( diff==0 && (stype & NVStrings::name) )
                diff = lhs->compare(*rhs);
            return (ascending ? (diff < 0) : (diff > 0));
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
