
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <locale.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "custring_view.cuh"
#include "custring.cuh"
#include "Rave.h"

static void printCudaError( cudaError_t err, const char* prefix="\t" )
{
    if( err != cudaSuccess )
        fprintf(stderr,"%s: %s(%d):%s\n",prefix,cudaGetErrorName(err),(int)err,cudaGetErrorString(err));
}

// return unique set of tokens within all the strings using the specified delimiter
NVStrings* Rave::unique_tokens(NVStrings& strs, const char* delimiter )
{
    int bytes = (int)strlen(delimiter);
    char* d_delimiter = 0;
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
            fprintf(stderr,"unique_tokens:col=%d\n",col);
            printCudaError(err);
        }
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
unsigned int Rave::token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool bdevmem )
{
    int bytes = (int)strlen(delimiter);
    char* d_delimiter = 0;
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
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
        printCudaError(err,"token_count:");
    //
    RMM_FREE(d_delimiter,0);
    if( !bdevmem )
    {
        cudaMemcpy(results,d_counts,count*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        RMM_FREE(d_counts,0);
    }
    return 0;
}

// return boolean value for each token if found in the provided strings
unsigned int Rave::contains_strings( NVStrings& strs, NVStrings& tkns, bool* results, bool todevice )
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
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"contains-strings(%u,%p,%d)\n",tcount,results,(int)todevice);
        printCudaError(err);
    }
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
unsigned int Rave::strings_counts( NVStrings& strs, NVStrings& tkns, unsigned int* results, bool todevice )
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
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"strings-count(%u,%p,%d)\n",tcount,results,(int)todevice);
        printCudaError(err);
    }
    //
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return 0;
}
