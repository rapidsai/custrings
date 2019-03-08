
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"
#include "Timing.h"

#ifdef __INTELLISENSE__
double log10(double);
#endif

//
unsigned int NVStrings::hash(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->hash();
            else
                d_rtn[idx] = 0;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-hash");
    double et = GetTime();
    pImpl->addOpTimes("hash",0.0,(et-st));

    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::stoi(int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stoi();
            else
                d_rtn[idx] = 0;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-stoi");
    double et = GetTime();
    pImpl->addOpTimes("stoi",0.0,(et-st));

    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::stof(float* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    float* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(float),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stof();
            else
                d_rtn[idx] = (float)0;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-stof");
    double et = GetTime();
    pImpl->addOpTimes("stof",0.0,(et-st));
    //
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

//
unsigned int NVStrings::htoi(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    double st = GetTime();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr || dstr->empty() )
            {
                d_rtn[idx] = 0;
                return;
            }
            long result = 0, base = 1;
            const char* str = dstr->data();
            int len = dstr->size()-1;
            for( int i=len; i >= 0; --i )
            {
                char ch = str[i];
                if( ch >= '0' && ch <= '9' )
                {
                    result += (long)(ch-48) * base;
                    base *= 16;
                }
                else if( ch >= 'A' && ch <= 'Z' )
                {
                    result += (long)(ch-55) * base;
                    base *= 16;
                }
                else if( ch >= 'a' && ch <= 'z' )
                {
                    result += (long)(ch-87) * base;
                    base *= 16;
                }
            }
            d_rtn[idx] = (unsigned int)result;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-htoi");
    double et = GetTime();
    pImpl->addOpTimes("htoi",0.0,(et-st));

    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

// build strings from given integers
NVStrings* NVStrings::itos(const int* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    int* d_values = (int*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(int),0);
        cudaMemcpy(d_values,values,count*sizeof(int),cudaMemcpyHostToDevice);
        if( nullbitmask )
        {
            RMM_ALLOC(&d_nulls,((count+7)/8)*sizeof(unsigned char),0);
            cudaMemcpy(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice);
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            int value = d_values[idx];
            if( value==0 )
            {   // yes, zero is a digit man
                d_sizes[idx] = ALIGN_SIZE(custring_view::alloc_size(1,1));
                return;
            }
            bool sign = value < 0;
            if( sign )
                value = -value;
            int digits = 0; // count the digits
            while( value > 0 )
            {
                ++digits;
                value = value/10;
            }
            int bytes = digits + (int)sign;
            int size = custring_view::alloc_size(bytes,bytes);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = 0;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            int value = d_values[idx];
            if( value==0 )
            {
                d_strings[idx] = custring_view::create_from(str,"0",1);
                return;
            }
            char* ptr = str;
            bool sign = value < 0;
            if( sign )
                value = -value;
            while( value > 0 )
            {
                char ch = '0' + (value % 10);
                *ptr++ = ch;
                value = value/10;
            }
            if( sign )
                *ptr++ = '-';
            // number is backwards, so let's reverse it
            // should make this a custring method
            int len = (int)(ptr-str);
            for( int j=0; j<(len/2); ++j )
            {
                char ch1 = str[j];
                char ch2 = str[len-j-1];
                str[j] = ch2;
                str[len-j-1] = ch1;
            }
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-itos");
    // done
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

unsigned int NVStrings::ip2int( unsigned int* results, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return count;
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !bdevmem )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr || dstr->empty() )
            {
                d_rtn[idx] = 0;
                return; // empty or null string
            }
            int tokens = dstr->split_size(".",1,0,-1);
            if( tokens != 4 )
            {
                d_rtn[idx] = 0;
                return; // invalid format
            }
            unsigned int vals[4];
            unsigned int* pval = vals;
            const char* str = dstr->data();
            int len = dstr->size();
            for( int i=0; i < len; ++i )
            {
                char ch = str[i];
                if( ch >= '0' && ch <= '9' )
                {
                    *pval *= 10;
                    *pval += (unsigned int)(ch-'0');
                }
                else if( ch=='.' )
                {
                    ++pval;
                    *pval = 0;
                }
            }
            unsigned int result = (vals[0] * 16777216) + (vals[1] * 65536) + (vals[2] * 256) + vals[3];
            d_rtn[idx] = result;
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-ip2int");
    if( !bdevmem )
    {
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return count;
}

NVStrings* NVStrings::int2ip( const unsigned int* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem )
{
    if( values==0 || count==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    unsigned int* d_values = (unsigned int*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(unsigned int),0);
        cudaMemcpy(d_values,values,count*sizeof(unsigned int),cudaMemcpyHostToDevice);
        if( nullbitmask )
        {
            RMM_ALLOC(&d_nulls,((count+7)/8)*sizeof(unsigned char),0);
            cudaMemcpy(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice);
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            unsigned int ipnum = d_values[idx];
            int bytes = 3; // 3 dots: xxx.xxx.xxx.xxx
            for( int j=0; j < 4; ++j )
            {
                unsigned int value = (ipnum & 255)+1; // don't want log(0)
                bytes += (int)log10((double)value)+1; // number of base10 digits
                ipnum = ipnum >> 8;
            }
            int size = custring_view::alloc_size(bytes,bytes);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_strings[idx] = 0;
                return;
            }
            unsigned int ipnum = d_values[idx];
            char* str = d_buffer + d_offsets[idx];
            char* ptr = str;
            for( int j=0; j < 4; ++j )
            {
                int value = ipnum & 255;
                do {
                    char ch = '0' + (value % 10);
                    *ptr++ = ch;
                    value = value/10;
                } while( value > 0 );
                if( j < 3 )
                    *ptr++ = '.';
                ipnum = ipnum >> 8;
            }
            int len = (int)(ptr-str);
            for( int j=0; j<(len/2); ++j )
            {
                char ch1 = str[j];
                char ch2 = str[len-j-1];
                str[j] = ch2;
                str[len-j-1] = ch1;
            }
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-itos");
    // done
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}
