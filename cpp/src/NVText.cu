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

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "custring_view.cuh"
#include "custring.cuh"
#include "NVText.h"

//static void printCudaError( cudaError_t err, const char* prefix="\t" )
//{
//    if( err != cudaSuccess )
//        fprintf(stderr,"%s: %s(%d):%s\n",prefix,cudaGetErrorName(err),(int)err,cudaGetErrorString(err));
//}

// return unique set of tokens within all the strings using the specified delimiter
NVStrings* NVText::unique_tokens(NVStrings& strs, const char* delimiter )
{
    int bytes = (int)strlen(delimiter);
    char* d_delimiter = nullptr;
    auto execpol = rmm::exec_policy(0);
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);

    // need to count how many output strings per string
    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->split_size(d_delimiter,bytes,0,-1);
        });

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );

    // build an index for each column and then sort/unique it
    rmm::device_vector< thrust::pair<const char*,size_t> > vocab;
    for( int col=0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, d_delimiter, bytes, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = nullptr;   // initialize to
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
        //cudaError_t err = cudaDeviceSynchronize();
        //if( err != cudaSuccess )
        //{
        //    fprintf(stderr,"unique_tokens:col=%d\n",col);
        //    printCudaError(err);
        //}
        // add column values to vocab list
        vocab.insert(vocab.end(),indexes.begin(),indexes.end());
        //printf("vocab size = %lu\n",vocab.size());
        thrust::pair<const char*,size_t>* d_vocab = vocab.data().get();
        // sort the list
        thrust::sort(execpol->on(0), d_vocab, d_vocab + vocab.size(),
            [] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
                if( lhs.first==0 || rhs.first==0 )
                    return lhs.first==0; // non-null > null
                return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
            });
        // unique the list
        thrust::pair<const char*,size_t>* newend = thrust::unique(execpol->on(0), d_vocab, d_vocab + vocab.size(),
            [] __device__ ( thrust::pair<const char*,size_t> lhs, thrust::pair<const char*,size_t> rhs ) {
                if( lhs.first==rhs.first )
                    return true;
                if( lhs.second != rhs.second )
                    return false;
                return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0;
            });
        // truncate list to the unique set
        // the above unique() call does an implicit dev-sync
        vocab.resize((size_t)(newend - d_vocab));
    }
    // remove the inevitable 'null' token
    thrust::pair<const char*,size_t>* d_vocab = vocab.data().get();
    auto end = thrust::remove_if(execpol->on(0), d_vocab, d_vocab + vocab.size(), [] __device__ ( thrust::pair<const char*,size_t> w ) { return w.first==0; } );
    unsigned int vsize = (unsigned int)(end - d_vocab); // may need new size
    // done
    RMM_FREE(d_delimiter,0);
    // build strings object from vocab elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_vocab,vsize);
}

// return a count of the number of tokens for each string when applying the specified delimiter
unsigned int NVText::token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool bdevmem )
{
    int bytes = (int)strlen(delimiter);
    char* d_delimiter = nullptr;
    auto execpol = rmm::exec_policy(0);
    RMM_ALLOC(&d_delimiter,bytes,0);
    cudaMemcpy(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice);

    unsigned int count = strs.size();
    unsigned int* d_counts = results;
    if( !bdevmem )
        RMM_ALLOC(&d_counts,count*sizeof(unsigned int),0);

    // count how many strings per string
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            int tc = 0;
            if( dstr )
                tc = dstr->split_size(d_delimiter,bytes,0,-1);
            d_counts[idx] = tc;
        });
    //
    if( !bdevmem )
    {
        cudaMemcpy(results,d_counts,count*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        RMM_FREE(d_counts,0);
    }
    RMM_FREE(d_delimiter,0);
    return 0;
}

// return boolean value for each token if found in the provided strings
unsigned int NVText::contains_strings( NVStrings& strs, NVStrings& tkns, bool* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,tcount*count*sizeof(bool),0);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                d_rtn[(idx*tcount)+jdx] = ((dstr && dtgt) ? dstr->find(*dtgt) : -2) >=0 ;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count*tcount,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

// return the number of occurrences of each string within a set of strings
// this will fill in the provided memory as a matrix:
//           'aa'  'bbb'  'c' ...
// "aaaabc"    2     0     1
// "aabbcc"    1     0     2
// "abbbbc"    0     1     1
// ...
unsigned int NVText::strings_counts( NVStrings& strs, NVStrings& tkns, unsigned int* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,tcount*count*sizeof(unsigned int),0);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                int fnd = 0;
                if( dstr && dtgt )
                {
                    int pos = dstr->find(*dtgt);
                    while( pos >= 0 )
                    {
                        pos = dstr->find(*dtgt,pos+dtgt->chars_count());
                        ++fnd;
                    }
                }
                d_rtn[(idx*tcount)+jdx] = fnd;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

// return the number of occurrences of each string within a set of strings
// this will fill in the provided memory as a matrix:
//              'aa'  'bbb'  'c' ...
// "aa aa b c"    2     0     1
// "aa bb c c"    1     0     2
// "a bbb ccc"    0     1     0
// ...
unsigned int NVText::tokens_counts( NVStrings& strs, NVStrings& tkns, const char* delimiter, unsigned int* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,tcount*count*sizeof(unsigned int),0);
    int dellen = (int)strlen(delimiter);
    char* d_delimiter = nullptr;
    RMM_ALLOC(&d_delimiter,dellen,0);
    cudaMemcpy(d_delimiter,delimiter,dellen,cudaMemcpyHostToDevice);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_delimiter, dellen, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                int fnd = 0;
                if( dstr && dtgt )
                {
                    int pos = dstr->find(*dtgt);
                    while( pos >= 0 )
                    {
                        int epos = pos + dtgt->chars_count();
                        if( ((pos==0) || (dstr->find(d_delimiter,dellen,pos-1)==(pos-1))) &&
                            ((epos>=dstr->chars_count()) || (dstr->find(d_delimiter,dellen,epos)==epos)) )
                            ++fnd;
                        pos = dstr->find(*dtgt,pos+dtgt->chars_count());
                    }
                }
                d_rtn[(idx*tcount)+jdx] = fnd;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

// Documentation here: https://www.cuelogic.com/blog/the-levenshtein-algorithm
// And here: https://en.wikipedia.org/wiki/Levenshtein_distances
struct editdistance_levenshtein_algorithm
{
    custring_view** d_strings; // trying match
    custring_view* d_tgt;      // match with this
    custring_view** d_tgts;    // or these
    short* d_buffer;           // compute buffer
    size_t* d_offsets;         // locate sub-buffer
    unsigned int* d_results;   // edit-distances

    // single string
    editdistance_levenshtein_algorithm( custring_view** strings, custring_view* tgt, short* buffer, size_t* offsets, unsigned int* results )
    : d_strings(strings), d_tgt(tgt), d_tgts(0), d_buffer(buffer), d_offsets(offsets), d_results(results) {}

    // multiple strings
    editdistance_levenshtein_algorithm( custring_view** strings, custring_view** tgts, short* buffer, size_t* offsets, unsigned int* results )
    : d_strings(strings), d_tgt(0), d_tgts(tgts), d_buffer(buffer), d_offsets(offsets), d_results(results) {}

    __device__ void operator() (unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        short* buf = (short*)d_buffer + d_offsets[idx];
        custring_view* dtgt = d_tgt;
        if( !d_tgt )
            dtgt = d_tgts[idx];
        d_results[idx] = compute_distance(dstr,dtgt,buf);
    }

    __device__ unsigned int compute_distance( custring_view* dstr, custring_view* dtgt, short* buf )
    {
        if( !dstr || dstr->empty() )
            return dtgt ? dtgt->chars_count() : 0;
        if( !dtgt || dtgt->empty() )
            return dstr->chars_count();
        //
        custring_view* strA = dstr;
        custring_view* strB = dtgt;
        int lenA = (int)dstr->chars_count();
        int lenB = (int)dtgt->chars_count();
        if( lenA > lenB )
        {
            lenB = lenA;
            lenA = dtgt->chars_count();
            strA = dtgt;
            strB = dstr;
        }
        //
        short* line2 = buf;
        short* line1 = line2 + lenA;
        short* line0 = line1 + lenA;
        int range = lenA + lenB - 1;
        for (int i = 0; i < range; i++)
        {
            short* tmp = line2;
            line2 = line1;
            line1 = line0;
            line0 = tmp;

            for(int x = (i < lenB ? 0 : i - lenB + 1); (x < lenA) && (x < i+1); x++)
            {
                int y = i - x;
                short u = y > 0 ? line1[x] : x + 1;
                short v = x > 0 ? line1[x - 1] : y + 1;
                short w;
                if((x > 0) && (y > 0))
                    w = line2[x - 1];
                else if(x > y)
                    w = x;
                else
                    w = y;
                u++; v++;
                Char c1 = strA->at(x);
                Char c2 = strB->at(y);
                if(c1 != c2)
                    w++;
                short value = u;
                if(v < value)
                    value = v;
                if(w < value)
                    value = w;
                line0[x] = value;
            }
        }
        return (unsigned int)line0[lenA-1];
    }
};

unsigned int NVText::edit_distance( distance_type algo, NVStrings& strs, const char* str, unsigned int* results, bool bdevmem )
{
    if( algo != levenshtein || str==0 || results==0 )
        throw std::invalid_argument("invalid algorithm");
    unsigned int count = strs.size();
    if( count==0 )
        return 0; // nothing to do
    auto execpol = rmm::exec_policy(0);
    unsigned int len = strlen(str);
    unsigned int alcsz = custring_view::alloc_size(str,len);
    custring_view* d_tgt = nullptr;
    RMM_ALLOC(&d_tgt,alcsz,0);
    custring_view::create_from_host(d_tgt,str,len);

    // setup results vector
    unsigned int* d_rtn = results;
    if( !bdevmem )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // calculate the size of the compute-buffer: 6 * length of string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tgt, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = dstr->chars_count();
            if( d_tgt->chars_count() < len )
                len = d_tgt->chars_count();
            d_sizes[idx] = len * 3;
        });
    //
    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(), sizes.end(), offsets.begin() );
    // compute edit distance
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        editdistance_levenshtein_algorithm(d_strings, d_tgt, d_buffer, d_offsets, d_rtn));
    //
    if( !bdevmem )
    {
        cudaMemcpy(results,d_rtn,count*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_tgt,0);
    return 0;
}

unsigned int NVText::edit_distance( distance_type algo, NVStrings& strs1, NVStrings& strs2, unsigned int* results, bool bdevmem )
{
    if( algo != levenshtein )
        throw std::invalid_argument("invalid algorithm");
    unsigned int count = strs1.size();
    if( count != strs2.size() )
        throw std::invalid_argument("sizes must match");
    if( count==0 )
        return 0; // nothing to do

    // setup results vector
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !bdevmem )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings1(count,nullptr);
    custring_view** d_strings1 = strings1.data().get();
    strs1.create_custring_index(d_strings1);
    rmm::device_vector<custring_view*> strings2(count,nullptr);
    custring_view** d_strings2 = strings2.data().get();
    strs2.create_custring_index(d_strings2);

    // calculate the size of the compute-buffer: 6 * length of string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings1, d_strings2, d_sizes] __device__(unsigned int idx){
            custring_view* dstr1 = d_strings1[idx];
            custring_view* dstr2 = d_strings2[idx];
            if( !dstr1 || !dstr2 )
                return;
            int len1 = dstr1->chars_count();
            int len = dstr2->chars_count();
            if( len1 < len )
                len = len1;
            d_sizes[idx] = len * 3;
        });
    //
    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(), sizes.end(), offsets.begin() );
    // compute edit distance
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        editdistance_levenshtein_algorithm(d_strings1, d_strings2, d_buffer, d_offsets, d_rtn));
    //
    if( !bdevmem )
    {
        cudaMemcpy(results,d_rtn,count*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return 0;
}
