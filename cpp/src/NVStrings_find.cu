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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "regex/regex.cuh"
#include "Timing.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

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
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, 0);

    RMM_FREE(d_str,0);
    //
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
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
int NVStrings::findall_record( const char* pattern, std::vector<NVStrings*>& results )
{
    if( pattern==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;

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
            int result = prog->find(idx,dstr,spos,epos);
            while(result > 0)
            {
                unsigned int bytes = (dstr->byte_offset_for(epos)-dstr->byte_offset_for(spos));
                unsigned int nchars = (epos-spos);
                unsigned int size = custring_view::alloc_size(bytes,nchars);
                tsize += ALIGN_SIZE(size);
                spos = epos;
                epos = end;
                ++fnd;
                result = prog->find(idx,dstr,spos,epos); // next one
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
                prog->find(idx,dstr,spos,epos);
                custring_view* str = dstr->substr((unsigned)spos,(unsigned)(epos-spos),1,buffer);
                drow[i] = str;
                buffer += ALIGN_SIZE(str->alloc_size());
                spos = epos;
            }
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-findall_record");
    dreprog::destroy(prog);
    return (int)results.size();
}

// same as findall but strings are returned organized in column-major
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
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;

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
            int result = prog->find(idx,dstr,begin,end);
            while(result > 0)
            {
                ++fnd;
                begin = end;
                end = nchars;
                result = prog->find(idx,dstr,begin,end); // next one
            }
            d_counts[idx] = fnd;
        });
    int columns = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columns==0 )
        results.push_back(new NVStrings(count));

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
                prog->find(idx,dstr,spos,epos);
                for( int c=0; c < col; ++c )
                {
                    spos = epos;    // update
                    epos = nchars;  // parameters
                    prog->find(idx,dstr,spos,epos);
                }
                // this will be the string for this column
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
                else
                {   // create empty string instead of a null one
                    d_indexes[idx].first = dstr->data();
                }

            });
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-findall(%s): col=%d\n",pattern,col);
            printCudaError(err);
        }
        // build new instance from the index
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    dreprog::destroy(prog);
    return (unsigned int)results.size();
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
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return matches;
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
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, prog, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = prog->contains(idx,dstr)==1;
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
    // destroy compiled regex
    dreprog::destroy(prog);

    // count the number of successful finds
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return matches;
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
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;

    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, prog, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = prog->match(idx,dstr)==1;
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
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return matches;
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
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;

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
                int result = prog->find(idx,dstr,begin,end);
                while(result > 0)
                {
                    ++fnd; // count how many we find
                    begin = end;
                    end = nchars;
                    result = prog->find(idx,dstr,begin,end);
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
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return matches;
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
    unsigned int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return matches;
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
    unsigned int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return matches;
}
