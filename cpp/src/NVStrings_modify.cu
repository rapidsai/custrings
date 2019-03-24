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
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "regex/regex.cuh"
#include "regex/backref.h"
#include "unicode/is_flags.h"
#include "Timing.h"

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
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(count);
    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;
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
            int result = prog->find(idx,dstr,begin,end);
            while((result > 0) && (mxn > 0))
            {
                bytes += rsz - (dstr->byte_offset_for(end)-dstr->byte_offset_for(begin));
                nchars += rszch - (end-begin);
                begin = end;
                end = (int)nchars;
                result = prog->find(idx,dstr,begin,end); // next one
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
            int result = prog->find(idx,dstr,begin,end);
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
                result = prog->find(idx,dstr,begin,end);
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

// not even close to the others
NVStrings* NVStrings::replace_with_backrefs( const char* pattern, const char* repl )
{
    if( !pattern || !*pattern )
        return 0; // null and empty string not allowed
    unsigned int count = size();
    if( count==0 || repl==0 )
        return new NVStrings(count); // returns all nulls
    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags(),count);
    delete ptn32;
    //
    // parse the repl string for backref indicators
    std::vector<thrust::pair<int,int> > brefs;
    std::string srepl = parse_backrefs(repl,brefs);
    unsigned int rsz = (unsigned int)srepl.size();
    char* d_repl = 0;
    RMM_ALLOC(&d_repl,rsz,0);
    cudaMemcpy(d_repl,srepl.c_str(),rsz,cudaMemcpyHostToDevice);
    unsigned int rszch = custring_view::chars_in_string(srepl.c_str(),rsz);
    rmm::device_vector<thrust::pair<int,int> > dbrefs(brefs);
    auto d_brefs = dbrefs.data().get();
    unsigned int refcount = (unsigned int)dbrefs.size();

    // compute size of the output
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    double st1 = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [prog, d_strings, rsz, rszch, d_brefs, refcount, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            unsigned int bytes = rsz, nchars = rszch; // start with template
            int begin = 0, end = (int)dstr->chars_count(); // constants
            if( prog->find(idx,dstr,begin,end) > 0 )
            {
                for( unsigned int j=0; j < refcount; ++j ) // eval each ref
                {
                    int refidx = d_brefs[j].first; // backref indicator
                    int spos=begin, epos=end;      // modified by extract
                    if( (prog->extract(idx,dstr,spos,epos,refidx-1)<=0) || (epos <= spos) )
                        continue; // no value for this ref
                    nchars += epos - spos;  // add up chars
                    spos = dstr->byte_offset_for(spos); // convert to bytes
                    bytes += dstr->byte_offset_for(epos) - spos; // add up bytes
                }
            }
            unsigned int size = custring_view::alloc_size(bytes,nchars);
            d_sizes[idx] = ALIGN_SIZE(size); // new size for this string
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
        [prog, d_strings, d_repl, rsz, d_offsets, d_brefs, refcount, d_buffer, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return; // nulls create nulls                                     a\1bc\2d
            char* buffer = d_buffer + d_offsets[idx]; // output buffer            _________
            char* optr = buffer; // running output pointer                        ^
            char* sptr = d_repl; // input buffer                                  abcd
            int lpos = 0, begin = 0, end = (int)dstr->chars_count();
            // insert extracted strings left-to-right
            if( prog->find(idx,dstr,begin,end) > 0 )
            {
                for( unsigned int j=0; j < refcount; ++j ) // eval each ref
                {
                    int refidx = d_brefs[j].first; // backref indicator               abcd        abcd
                    int ipos = d_brefs[j].second;  // input position                   ^             ^
                    int len = ipos - lpos; // bytes to copy from input
                    memcpy(optr,sptr,len); // copy left half                          a________   axxbc____
                    optr += len;  // move output ptr                                   ^               ^
                    sptr += len;  // move input ptr                                   abcd        abcd
                    lpos += len;  // update last-position                              ^             ^
                    int spos=begin, epos=end;  // these are modified by extract
                    if( (prog->extract(idx,dstr,spos,epos,refidx-1)<=0) ||
                        (epos <= spos) )                                  //          xx          yyy
                        continue; // no value for this ref
                    spos = dstr->byte_offset_for(spos); // convert to bytes
                    int bytes = dstr->byte_offset_for(epos) - spos;
                    memcpy(optr,dstr->data()+spos,bytes); //                          axx______   axxbcyyy_
                    optr += bytes; // move output ptr                                    ^                ^
                }
            }
            if( lpos < rsz )
            {   // copy any remaining characters from input string
                memcpy(optr,sptr,rsz-lpos);                                 //    axxbcyyyd
                optr += rsz-lpos;                                           //             ^
            }
            unsigned int nsz = (unsigned int)(optr - buffer); // compute output size
            d_results[idx] = custring_view::create_from(buffer,buffer,nsz); // new string
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et2 = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-replace_with_backref(%s,%s)\n",pattern,repl);
        printCudaError(err);
    }
    pImpl->addOpTimes("replace_with_backref",(et1-st1),(et2-st2));
    //
    dreprog::destroy(prog);
    RMM_FREE(d_repl,0);
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

//
// This will create a new instance replacing any nulls with the provided string.
// The parameter can be an empty string or any other string but not null.
NVStrings* NVStrings::fillna( const char* str )
{
    if( str==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned int ssz = (unsigned int)strlen(str);
    unsigned int asz = custring_view::alloc_size(str,ssz);
    char* d_str = 0;
    RMM_ALLOC(&d_str,ssz+1,0);
    cudaMemcpy(d_str,str,ssz+1,cudaMemcpyHostToDevice);

    // compute size of the output
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, asz, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            unsigned int size = asz;
            if( dstr )
                size = dstr->alloc_size();
            d_sizes[idx] = ALIGN_SIZE(size);
        });
    //
    NVStrings* rtn = new NVStrings(count); // create output object
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    rmm::device_vector<size_t> offsets(count,0); // create offsets
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, ssz, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            char* buffer = d_buffer + d_offsets[idx];
            if( dstr )
                dstr = custring_view::create_from(buffer,*dstr);
            else
                dstr = custring_view::create_from(buffer,d_str,ssz);
            d_results[idx] = dstr;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-fillna(%s)\n",str);
        printCudaError(err);
    }
    RMM_FREE(d_str,0);
    return rtn;
}
