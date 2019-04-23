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

#include <exception>
#include <math.h>  // for isnan, isinf; cmath does not work here
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include "NVStrings.h"
#include "NVStringsImpl.h"
#include "custring_view.cuh"

//
int NVStrings::hash(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stoi(int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(int),0);

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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stol(long* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    long* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(long),0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stol();
            else
                d_rtn[idx] = 0L;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(long)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stof(float* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    float* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(float),0);

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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stod(double* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    double* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(double),0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stod();
            else
                d_rtn[idx] = 0.0;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(double)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::htoi(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned int),0);

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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

// build strings from given integers
NVStrings* NVStrings::itos(const int* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::itos values or count invalid");
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
            int size = custring_view::ltos_size(value);
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
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            int value = d_values[idx];
            d_strings[idx] = custring_view::ltos(value,str);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

NVStrings* NVStrings::ltos(const long* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::ltos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    long* d_values = (long*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(long),0);
        cudaMemcpy(d_values,values,count*sizeof(long),cudaMemcpyHostToDevice);
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
            long value = d_values[idx];
            int size = custring_view::ltos_size(value);
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
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            long value = d_values[idx];
            d_strings[idx] = custring_view::ltos(value,str);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

struct ftos_converter
{
    // significant digits is independent of scientific notation range
    // digits more than this may require using long values instead of ints
    const unsigned int significant_digits = 10;
    // maximum power-of-10 that will fit in 32-bits
    const unsigned int nine_digits = 1000000000; // 1x10^9
    // Range of numbers here is for normalizing the value.
    // If the value is above or below the following limits, the output is converted to
    // scientific notation in order to show (at most) the number of significant digits.
    const double upper_limit = 1000000000; // max is 1x10^9
    const double lower_limit = 0.0001; // printf uses scientific notation below this
    // Tables for doing normalization: converting to exponent form
    // IEEE double float has maximum exponent of 305 so these should cover everthing
    const double upper10[9]  = { 10, 100, 10000, 1e8,  1e16,  1e32,  1e64,  1e128,  1e256 };
    const double lower10[9]  = { .1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256 };
    const double blower10[9] = { 1.0, .1, .001,  1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255 };

    // utility for quickly converting known integer range to character array
    __device__ char* int2str( int value, char* output )
    {
        if( value==0 )
        {
            *output++ = '0';
            return output;
        }
        char buffer[10]; // should be big-enough for 10 significant digits
        char* ptr = buffer;
        while( value > 0 )
        {
            *ptr++ = (char)('0' + (value % 10));
            value /= 10;
        }
        while( ptr != buffer )
            *output++ = *--ptr;  // 54321 -> 12345
        return output;
    }

    //
    // dissect value into parts
    // return decimal_places
    __device__ int dissect_value( double value, unsigned int& integer, unsigned int& decimal, int& exp10 )
    {
        // dissect float into parts
        int decimal_places = significant_digits-1;
        // normalize step puts value between lower-limit and upper-limit
        // by adjusting the exponent up or down
        exp10 = 0;
        if( value > upper_limit )
        {
            int fx = 256;
            for( int idx=8; idx >= 0; --idx )
            {
                if( value >= upper10[idx] )
                {
                    value *= lower10[idx];
                    exp10 += fx;
                }
                fx = fx >> 1;
            }
        }
        else if( (value > 0.0) && (value < lower_limit) )
        {
            int fx = 256;
            for( int idx=8; idx >= 0; --idx )
            {
                if( value < blower10[idx] )
                {
                    value *= upper10[idx];
                    exp10 -= fx;
                }
                fx = fx >> 1;
            }
        }
        //
        unsigned int max_digits = nine_digits;
        integer = (unsigned int)value;
        for( unsigned int i=integer; i >= 10; i/=10 )
        {
            --decimal_places;
            max_digits /= 10;
        }
        double remainder = (value - (double)integer) * (double)max_digits;
        //printf("remainder=%g,value=%g,integer=%u,sd=%u\n",remainder,value,integer,max_digits);
        decimal = (unsigned int)remainder;
        remainder -= (double)decimal;
        //printf("remainder=%g,decimal=%u\n",remainder,decimal);
        decimal += (unsigned int)(2.0*remainder);
        if( decimal >= max_digits )
        {
            decimal = 0;
            ++integer;
            if( exp10 && (integer >= 10) )
            {
                ++exp10;
                integer = 1;
            }
        }
        //
        while( (decimal % 10)==0 && (decimal_places > 0) )
        {
            decimal /= 10;
            --decimal_places;
        }
        return decimal_places;
    }

    //
    // Converts value to string into output
    // Output need not be more than significant_digits+7
    // 7 = 1 sign, 1 decimal point, 1 exponent ('e'), 1 exponent-sign, 3 digits for exponent
    //
    __device__ int float_to_string( double value, char* output )
    {
        // check for valid value
        if( isnan(value) )
        {
            memcpy(output,"NaN",3);
            return 3;
        }
        bool bneg = false;
        if( value < 0.0 )
        {
            value = -value;
            bneg = true;
        }
        if( isinf(value) )
        {
            if( bneg )
                memcpy(output,"-Inf",4);
            else
                memcpy(output,"Inf",3);
            return bneg ? 4 : 3;
        }

        // dissect value into components
        unsigned int integer = 0, decimal = 0;
        int exp10 = 0;
        int decimal_places = dissect_value(value,integer,decimal,exp10);
        //
        // now build the string from the
        // components: sign, integer, decimal, exp10, decimal_places
        //
        // sign
        char* ptr = output;
        if( bneg )
            *ptr++ = '-';
        // integer
        ptr = int2str(integer,ptr);
        // decimal
        if( decimal_places )
        {
            *ptr++ = '.';
            char buffer[10];
            char* pb = buffer;
            while( decimal_places-- )
            {
                *pb++ = (char)('0' + (decimal % 10));
                decimal /= 10;
            }
            while( pb != buffer )  // reverses the digits
                *ptr++ = *--pb;    // e.g. 54321 -> 12345
        }
        // exponent
        if( exp10 )
        {
            *ptr++ = 'e';
            if( exp10 < 0 )
            {
                *ptr++ ='-';
                exp10 = -exp10;
            }
            else
                *ptr++ ='+';
            if( exp10 < 10 )
                *ptr++ = '0'; // extra zero-pad
            ptr = int2str(exp10,ptr);
        }
        // done
        //*ptr = 0; // null-terminator

        return (int)(ptr-output);
    }

    // need to compute how much memory is needed to
    // hold the output string (not including null)
    __device__ int compute_ftos_size( double value )
    {
        if( isnan(value) )
            return 3; // NaN
        bool bneg = false;
        if( value < 0.0 )
        {
            value = -value;
            bneg = true;
        }
        if( isinf(value) )
            return 3 + (int)bneg; // Inf

        // dissect float into parts
        unsigned int integer = 0, decimal = 0;
        int exp10 = 0;
        int decimal_places = dissect_value(value,integer,decimal,exp10);
        // now count up the components
        // sign
        int count = (int)bneg;
        // integer
        count += (int)(integer==0);
        while( integer > 0 )
        {
            integer /= 10;
            ++count;
        } // log10(integer)
        // decimal
        if( decimal_places )
        {
            ++count; // decimal point
            count += decimal_places;
        }
        // exponent
        if( exp10 )
        {
            count += 2; // 'e±'
            if( exp10 < 0 )
                exp10 = -exp10;
            count += (int)(exp10<10); // padding
            while( exp10 > 0 )
            {
                exp10 /= 10;
                ++count;
            } // log10(exp10)
        }

        return count;
    }
};

// build strings from given floats
NVStrings* NVStrings::ftos(const float* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::ftos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    float* d_values = (float*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(float),0);
        cudaMemcpy(d_values,values,count*sizeof(float),cudaMemcpyHostToDevice);
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
            float value = d_values[idx];
            ftos_converter fts;
            int bytes = fts.compute_ftos_size((double)value);
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
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            float value = d_values[idx];
            ftos_converter fts;
            int len = fts.float_to_string((double)value,str);
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

// build strings from given doubles
NVStrings* NVStrings::dtos(const double* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::dtos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    double* d_values = (double*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(double),0);
        cudaMemcpy(d_values,values,count*sizeof(double),cudaMemcpyHostToDevice);
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
            double value = d_values[idx];
            ftos_converter fts;
            int bytes = fts.compute_ftos_size(value);
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
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            double value = d_values[idx];
            ftos_converter fts;
            int len = fts.float_to_string(value,str);
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

// convert IPv4 to integer
int NVStrings::ip2int( unsigned int* results, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;
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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !bdevmem )
    {
        cudaMemcpy(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

NVStrings* NVStrings::int2ip( const unsigned int* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem )
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::int2ip values or count invalid");
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
                d_strings[idx] = nullptr;
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
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

// this parses ISO8601 date/time characters into long timestamp
struct parse_iso8601
{
    custring_view_array d_strings;
    unsigned long* d_timestamps;
    NVStrings::timestamp_units units;

    parse_iso8601( custring_view_array dstrs, NVStrings::timestamp_units unts, unsigned long* results )
    : d_strings(dstrs), d_timestamps(results), units(unts) {}

    // could use the custring::stoi but this should be faster since we need know the data limits
    __device__ int str2int( const char* str, unsigned int bytes )
    {
        const char* ptr = str;
        int value = 0;
        for( unsigned int idx=0; idx < bytes; ++idx )
        {
            char chr = *ptr++;
            if( chr < '0' || chr > '9' )
                break;
            value = (value * 10) + (int)(chr - '0');
        }
        return value;
    }

    // parses the string without any format checking
     __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( (dstr==0) || (dstr->size()<4) )
        {   // we must at least have the year (4 bytes)
            d_timestamps[idx] = 0;
            return;
        }
        //
        int bytes = (int)dstr->size();
        int timeparts[9] = {0}; // year,month,day,hour,minute,second,milli,tzh,tzm
        int partsizes[9] = {4,2,2,2,2,2,3,2,2}; // char length of each part
        const char* sptr = dstr->data();
        int tzsign = 1;
        int part = 0; // starts with year, dissect each part
        while( (part < 9) && (bytes > 0) )
        {
            int len = partsizes[part];
            if( len > bytes )
                break;
            timeparts[part++] = str2int(sptr,len);
            sptr += len;
            bytes -= len;
            if( bytes==0 )
                break;
            char ch = *sptr;
            if( ch>='0' && ch<='9')
                continue;
            ++sptr;  // skip over any
            --bytes; // separators
            if( part == 7 ) // tz section
            {
                if( ch=='Z' )
                    break;
                if( ch=='-' )
                    tzsign = -1;
            }
        }
        //
        long timestamp = 0;
        int year = timeparts[0];
        int month = timeparts[1];
        int day = timeparts[2];
        // The months are shifted so that March is the starting month and February
        // (possible leap day in it) is the last month for the linear calculation
        year -= (month <= 2) ? 1 : 0;
        // date cycle repeats every 400 years (era)
        const int erasInDays = 146097;
        const int erasInYears = (erasInDays / 365);
        const int era = (year >= 0 ? year : year - 399) / erasInYears;
        const int yoe = year - era * erasInYears;
        const int doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
        const int doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy;
        int days = (era * erasInDays) + doe - 719468; // 719468 = days from 0000-00-00 to 1970-03-01
        int hour = timeparts[3];
        int minute = timeparts[4];
        int second = timeparts[5];
        int millisecond = timeparts[6];
        int tzadjust = (timeparts[7] * 3600) + (timeparts[8] * 60) * tzsign;
        //
        timestamp = (days * 24L * 3600L) + (hour * 3600L) + (minute * 60L) + second + tzadjust;
        if( units==NVStrings::milliseconds )
        {
            timestamp *= 1000L;
            timestamp += millisecond;
        }
        d_timestamps[idx] = timestamp;
    }
};

// convert ISO8601 date format into timestamp long integer
int NVStrings::timestamp2long( unsigned long* results, timestamp_units units, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;
    auto execpol = rmm::exec_policy(0);
    unsigned long* d_rtn = results;
    if( !bdevmem )
        RMM_ALLOC(&d_rtn,count*sizeof(unsigned long),0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        parse_iso8601(d_strings,units,d_rtn));
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !bdevmem )
    {
        cudaMemcpy(results,d_rtn,sizeof(unsigned long)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

// converts long timestamp into ISO8601 string (YYYY-MM-DDThh:mm:ss.sssZ)
struct iso8601_formatter
{
    unsigned long* d_timestamps;
    custring_view_array d_strings;
    unsigned char* d_nulls;
    char* d_buffer;
    size_t* d_offsets;
    NVStrings::timestamp_units units; // seconds, milliseconds

    iso8601_formatter( NVStrings::timestamp_units unts, char* buffer, size_t* offsets, unsigned char* nulls, unsigned long* timestamps, custring_view_array strings)
    : d_timestamps(timestamps), d_buffer(buffer), d_offsets(offsets), d_nulls(nulls), d_strings(strings), units(unts) {}

    __device__ void dissect_timestamp( long timestamp, int& year, int& month, int& day,
                                       int& hour, int& minute, int& second, int& millisecond )
    {
        const int daysInEra = 146097; // (400*365)+97
        const int daysInCentury = 36524; // (100*365) + 24;
        const int daysIn4Years = 1461; // (4*365) + 1;
        const int daysInYear = 365;
        // day offsets for each month:   Mar Apr May June July  Aug  Sep  Oct  Nov  Dec  Jan  Feb
        const int monthDayOffset[] = { 0, 31, 61, 92, 122, 153, 184, 214, 245, 275, 306, 337, 366 };
        // start with date
        // code logic handles leap years in chunks: 400y,100y,4y,1y
        long seconds = timestamp;
        if( units==NVStrings::milliseconds )
            seconds = seconds / 1000L;
        int days = (int)(seconds / 86400L) + 719468; // 86400 = 24 * 3600, 719468 is days between 0000-00-00 and 1970-01-01
        year = 400 * (days / daysInEra);
        days = days % daysInEra;
        int leapy = days / daysInCentury;
        days = days % daysInCentury;
        if( leapy==4 )
        {   // landed exactly on a leap century
            days += daysInCentury;
            --leapy;
        }
        year += 100 * leapy;
        year += 4 * (days / daysIn4Years);
        days = days % daysIn4Years;
        leapy = days / daysInYear;
        days = days % daysInYear;
        if( leapy==4 )
        {   // landed exactly on a leap year
            days += daysInYear;
            --leapy;
        }
        year += leapy;
        month = 12;
        for( int idx=0; idx < month; ++idx )
        {   // find the month
            if( days < monthDayOffset[idx+1] )
            {
                month = idx;
                break;
            }
        }
        day = days - monthDayOffset[month] +1; // compute day of month
        if( month >= 10 )
            ++year;
        month = ((month + 2) % 12) +1; // adjust Jan-Mar offset
        // now compute time
        int tms = (int)(timestamp % 86400);
        if( units==NVStrings::milliseconds )
        {
            tms = (int)(timestamp % 86400000);
            millisecond = tms % 1000;
            tms = tms / 1000;
        }
        hour = tms / 3600;
        tms -= hour * 3600;
        minute = tms / 60;
        second = tms - (minute * 60);
    }

    // utility to create 0-padded integers (up to 4 bytes)
    __device__ char* int2str( char* str, int len, int val )
    {
        char tmpl[4] = {'0','0','0','0'};
        char* ptr = tmpl;
        while( val > 0 )
        {
            int digit = val % 10;
            *ptr++ = '0' + digit;
            val = val / 10;
        }
        ptr = tmpl + len-1;
        while( len > 0 )
        {
            *str++ = *ptr--;
            --len;
        }
        return str;
    }

    __device__ void operator()( unsigned int idx )
    {
        if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            d_strings[idx] = nullptr;
            return;
        }
        long timestamp = d_timestamps[idx];
        int year, month, day, hour, minute, second, millisecond;
        dissect_timestamp(timestamp,year,month,day,hour,minute,second,millisecond);
        // convert to characters
        char* str = d_buffer + d_offsets[idx];
        char dtmpl[24];
        char* ptr = dtmpl;
        ptr = int2str(ptr,4,year);
        *ptr++ = '-';
        ptr = int2str(ptr,2,month);
        *ptr++ = '-';
        ptr = int2str(ptr,2,day);
        *ptr++ = 'T';
        ptr = int2str(ptr,2,hour);
        *ptr++ = ':';
        ptr = int2str(ptr,2,minute);
        *ptr++ = ':';
        ptr = int2str(ptr,2,second);
        if( units==NVStrings::milliseconds )
        {
            *ptr++ = '.';
            ptr = int2str(ptr,3,millisecond);
        }
        *ptr++ = 'Z';
        int len = (int)(ptr - dtmpl);
        d_strings[idx] = custring_view::create_from(str,dtmpl,len);
    }
};


NVStrings* NVStrings::long2timestamp( const unsigned long* values, unsigned int count, timestamp_units units, const unsigned char* nullbitmask, bool bdevmem )
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::long2timestamp values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    unsigned long* d_values = (unsigned long*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(unsigned long),0);
        cudaMemcpy(d_values,values,count*sizeof(unsigned long),cudaMemcpyHostToDevice);
        if( nullbitmask )
        {
            RMM_ALLOC(&d_nulls,((count+7)/8)*sizeof(unsigned char),0);
            cudaMemcpy(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice);
        }
    }

    // compute size of memory we'll need
    // each string will be the same size with the following form
    const char* fixedform = "YYYY-MM-DDThh:mm:ssZ";
    if( units==milliseconds )
        fixedform = "YYYY-MM-DDThh:mm:ss.sssZ";
    int d_size = custring_view::alloc_size(fixedform,strlen(fixedform));
    // we only need to account for any null strings
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_size, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
                d_sizes[idx] = 0;
            else
                d_sizes[idx] = ALIGN_SIZE(d_size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build iso8601 strings from timestamps
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        iso8601_formatter(units,d_buffer, d_offsets, d_nulls, d_values, d_strings));
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

int NVStrings::to_bools( bool* results, const char* true_string, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    // copy parameter to device memory
    char* d_true = nullptr;
    int d_len = 0;
    if( true_string )
    {
        d_len = (int)strlen(true_string);
        RMM_ALLOC(&d_true,d_len+1,0);
        cudaMemcpy(d_true,true_string,d_len+1,cudaMemcpyHostToDevice);
    }
    //
    bool* d_rtn = results;
    if( !bdevmem )
        RMM_ALLOC(&d_rtn,count*sizeof(bool),0);

    // set the values
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_true, d_len, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->compare(d_true,d_len)==0;
            else
                d_rtn[idx] = (d_true==0); // let null be a thing
        });
    //
    // calculate the number of falses (to include nulls too)
    int falses = thrust::count(execpol->on(0),d_rtn,d_rtn+count,false);
    if( !bdevmem )
    {
        cudaMemcpy(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost);
        RMM_FREE(d_rtn,0);
    }
    if( d_true )
        RMM_FREE(d_true,0);
    return (int)count-falses;
}

NVStrings* NVStrings::create_from_bools(const bool* values, unsigned int count, const char* true_string, const char* false_string, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::create_from_bools values or count invalid");
    if( true_string==0 || false_string==0 )
        throw std::invalid_argument("nvstrings::create_from_bools false and true strings must not be null");

    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    int d_len_true = strlen(true_string);
    char* d_true = nullptr;
    RMM_ALLOC(&d_true,d_len_true+1,0);
    cudaMemcpy(d_true,true_string,d_len_true+1,cudaMemcpyHostToDevice);
    int d_as_true = custring_view::alloc_size(true_string,d_len_true);
    int d_len_false = strlen(false_string);
    char* d_false = nullptr;
    RMM_ALLOC(&d_false,d_len_false+1,0);
    cudaMemcpy(d_false,false_string,d_len_false+1,cudaMemcpyHostToDevice);
    int d_as_false = custring_view::alloc_size(false_string,d_len_false);

    bool* d_values = (bool*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(bool),0);
        cudaMemcpy(d_values,values,count*sizeof(bool),cudaMemcpyHostToDevice);
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
        [d_values, d_nulls, d_as_true, d_as_false, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            bool value = d_values[idx];
            int size = value ? d_as_true : d_as_false;
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings of booleans
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_true, d_len_true, d_false, d_len_false, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = nullptr; // null string
                return;
            }
            char* buf = d_buffer + d_offsets[idx];
            bool value = d_values[idx];
            if( value )
                d_strings[idx] = custring_view::create_from(buf,d_true,d_len_true);
            else
                d_strings[idx] = custring_view::create_from(buf,d_false,d_len_false);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    RMM_FREE(d_true,0);
    RMM_FREE(d_false,0);
    return rtn;
}
