
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "unicode/is_flags.h"
#include "Timing.h"


// this will return the number of characters for each string
unsigned int NVStrings::len(int* lengths, bool todevice)
{
    unsigned int count = size();
    if( lengths==0 || count==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = lengths;
    if( !todevice )
        RMM_ALLOC(&d_rtn,sizeof(int)*count,0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->chars_count();
            else
                d_rtn[idx] = -1;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-len");
    double et = GetTime();
    pImpl->addOpTimes("len",0.0,(et-st));
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(lengths,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

// this will return the number of bytes for each string
size_t NVStrings::byte_count(int* lengths, bool todevice)
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = lengths;
    if( !lengths )
        todevice = false; // makes sure we free correctly
    if( !todevice )
        RMM_ALLOC(&d_rtn,sizeof(int)*count,0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->size();
            else
                d_rtn[idx] = -1;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-bytes");
    double et = GetTime();
    pImpl->addOpTimes("byte_count",0.0,(et-st));
    size_t size = thrust::reduce(execpol->on(0), d_rtn, d_rtn+count, (size_t)0,
         []__device__(int lhs, int rhs) {
            if( lhs < 0 )
                lhs = 0;
            if( rhs < 0 )
                rhs = 0;
            return lhs + rhs;
         });
    if( !todevice )
    {   // copy result back to host
        if( lengths )
            cudaMemcpy(lengths,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return size;
}


//
unsigned int NVStrings::isalnum( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // alnum requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_ALPHANUM(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isalnum(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isalnum",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isalpha( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // alpha requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_ALPHA(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isalpha(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isalpha",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

//
unsigned int NVStrings::isdigit( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // digit requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_DIGIT(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isdigit(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isdigit",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isspace( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // space requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_SPACE(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isspace(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isspace",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isdecimal( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // decimal requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_DECIMAL(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isdecimal(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isdecimal",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isnumeric( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // numeric requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_NUMERIC(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-isnumeric(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("isnumeric",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::islower( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    brc = !IS_ALPHA(flg) || IS_LOWER(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-islower(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("islower",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}

unsigned int NVStrings::isupper( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    double st = GetTime();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    brc = !IS_ALPHA(flg) || IS_UPPER(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    double et = GetTime();
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-islower(%p,%d)\n",results,(int)todevice);
        printCudaError(err);
    }
    pImpl->addOpTimes("islower",0.0,(et-st));
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(bool val) {return val;} );
    if( !todevice )
    {   // copy result back to host
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}
