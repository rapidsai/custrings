
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "Timing.h"

//
int NVStrings::split_record( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    unsigned int dellen = 0;
    if( delimiter )
    {
        dellen = (unsigned int)strlen(delimiter);
        RMM_ALLOC(&d_delimiter,dellen+1,0);
        cudaMemcpy(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice);
    }

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->split_size(d_delimiter,dellen,0,maxsplit);
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
        [d_strings, d_delimiter, dellen, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = d_sizes + d_offsets[idx];
            int dcount = d_counts[idx];
            d_totals[idx] = dstr->split_size(d_delimiter,dellen,dsizes,dcount);
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
        [d_strings, d_delimiter, dellen, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
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
            dstr->split(d_delimiter,dellen,d_count,d_strs);
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-split_record");
    RMM_FREE(d_delimiter,0);
    //
    return totalNewStrings;
}

//
int NVStrings::rsplit_record( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    unsigned int dellen = 0;
    if( delimiter )
    {
        dellen = (unsigned int)strlen(delimiter);
        RMM_ALLOC(&d_delimiter,dellen+1,0);
        cudaMemcpy(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice);
    }

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->rsplit_size(d_delimiter,dellen,0,maxsplit);
        });

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int dcount = d_counts[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            d_totals[idx] = dstr->rsplit_size(d_delimiter,dellen,dsizes,dcount);
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
        [d_strings, d_delimiter, dellen, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
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
            dstr->rsplit(d_delimiter,dellen,d_count,d_strs);
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-rsplit_record");
    RMM_FREE(d_delimiter,0);

    return totalNewStrings;
}

// This will create new columns by splitting the array of strings vertically.
// All the first tokens go in the first column, all the second tokens go in the second column, etc.
unsigned int NVStrings::split( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    unsigned int dellen = 0;
    if( delimiter )
    {
        dellen = (unsigned int)strlen(delimiter);
        RMM_ALLOC(&d_delimiter,dellen+1,0);
        cudaMemcpy(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice);
    }

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->split_size(d_delimiter,dellen,0,maxsplit);
        });
    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columnsCount==0 )
        results.push_back(new NVStrings(count));

    // create each column
    for( int col=0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        //st = GetTime();
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, d_delimiter, dellen, d_counts, d_indexes] __device__(unsigned int idx){
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
                int dchars = 1;
                if( d_delimiter && dellen )
                    dchars = custring_view::chars_in_string(d_delimiter,dellen);
                int spos = 0, nchars = dstr->chars_count();
                int epos = nchars, pos = 0;
                for( int c=0; c < (dcount-1); ++c )
                {
                    if( d_delimiter && dellen )
                        epos = dstr->find(d_delimiter,dellen,spos);
                    else
                    {
                        epos = -1;
                        char* sptr = dstr->data();
                        while(pos < dstr->size())
                        {
                            unsigned char ch = (unsigned char)sptr[pos++];
                            if( ch <= ' ')
                            {
                                epos = custring_view::chars_in_string(sptr,pos-1);
                                break;
                            }
                        }
                    }
                    if( epos < 0 )
                    {
                        epos = nchars;
                        break;
                    }
                    if( c==col )  // found our column
                        break;
                    spos = epos + dchars;
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
                else
                {   // this will create empty string instead of null one
                    d_indexes[idx].first = dstr->data();
                }
            });
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-split(%s,%d), col=%d\n",delimiter,maxsplit,col);
            printCudaError(err);
        }
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    if( d_delimiter )
        RMM_FREE(d_delimiter,0);
    return (unsigned int)results.size();
}

// split-from-the-right version of split
unsigned int NVStrings::rsplit( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = 0;
    unsigned int dellen = 0;
    if( delimiter )
    {
        dellen = (unsigned int)strlen(delimiter);
        RMM_ALLOC(&d_delimiter,dellen+1,0);
        cudaMemcpy(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice);
    }

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, maxsplit, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->rsplit_size(d_delimiter,dellen,0,maxsplit);
        });

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columnsCount==0 )
        results.push_back(new NVStrings(count));

    // create each column
    for( int col = 0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, columnsCount, d_delimiter, dellen, d_counts, d_indexes] __device__(unsigned int idx){
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
                int dchars = 1;
                if( d_delimiter && dellen )
                    dchars = custring_view::chars_in_string(d_delimiter,dellen);
                int spos = 0, nchars = dstr->chars_count();
                int epos = nchars, pos = dstr->size()-1;
                for( int c=(dcount-1); c > 0; --c )
                {
                    if( d_delimiter && dellen )
                        spos = dstr->rfind(d_delimiter,dellen,0,epos);
                    else
                    {
                        spos = -1;
                        char* sptr = dstr->data();
                        while( pos >=0 )
                        {
                            unsigned char ch = (unsigned char)sptr[pos--];
                            if( ch <= ' ')
                            {
                                spos = custring_view::chars_in_string(sptr,pos+1);
                                break;
                            }
                        }
                    }
                    if( spos < 0 )
                    {
                        spos = 0;
                        break;
                    }
                    if( c==col ) // found our column
                    {
                        spos += dchars;  // do not include delimiter
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
                else
                {   // this will create empty string instead of null one
                    d_indexes[idx].first = dstr->data();
                }
            });
        cudaError_t err = cudaDeviceSynchronize();
        if( err != cudaSuccess )
        {
            fprintf(stderr,"nvs-rsplit(%s,%d)\n",delimiter,maxsplit);
            printCudaError(err);
        }
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    if( d_delimiter )
        RMM_FREE(d_delimiter,0);
    return (unsigned int)results.size();
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
