
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "Timing.h"


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
