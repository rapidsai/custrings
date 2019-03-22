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
#include <thrust/count.h>
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
int NVStrings::hash(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost);
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
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        cudaMemcpy(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost);
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
    printCudaError(cudaDeviceSynchronize(),"nvs-ip2int");
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
    printCudaError(cudaDeviceSynchronize(),"nvs-timestamp2long");
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
            d_strings[idx] = 0;
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
        return 0;
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
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-long2timestamp");
    // done
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
    char* d_true = 0;
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
    double st = GetTime();
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
    printCudaError(cudaDeviceSynchronize(),"nvs-str2bool");
    double et = GetTime();
    pImpl->addOpTimes("to_bools",0.0,(et-st));
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
        return 0;
    if( true_string==0 || false_string==0 )
        return 0; // throw exception here

    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    int d_len_true = strlen(true_string);
    char* d_true = 0;
    RMM_ALLOC(&d_true,d_len_true+1,0);
    cudaMemcpy(d_true,true_string,d_len_true+1,cudaMemcpyHostToDevice);
    int d_as_true = custring_view::alloc_size(true_string,d_len_true);
    int d_len_false = strlen(false_string);
    char* d_false = 0;
    RMM_ALLOC(&d_false,d_len_false+1,0);
    cudaMemcpy(d_false,false_string,d_len_false+1,cudaMemcpyHostToDevice);
    int d_as_false = custring_view::alloc_size(false_string,d_len_false);
    
    bool* d_values = (bool*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        RMM_ALLOC(&d_values,count*sizeof(bool),0);
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
                d_strings[idx] = 0; // null string
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
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess )
        printCudaError(err,"nvs-bool2str");
    // done
    if( !bdevmem )
        RMM_FREE(d_values,0);
    RMM_FREE(d_true,0);
    RMM_FREE(d_false,0);
    return rtn;
}
