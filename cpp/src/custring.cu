
#include <cuda_runtime.h>
#include <memory.h>
#include "custring.cuh"


namespace custr
{
    // convert string with numerical characters to number
    __device__ int stoi( const char* str, size_t bytes )
    {
        const char* ptr = str;
        if( !ptr || !bytes )
            return 0; // probably should be an assert
        int value = 0, sign = 1, size = (int)bytes;
        if( *ptr == '-' || *ptr == '+' )
        {
            sign = (*ptr=='-' ? -1:1);
            ++ptr;
            --size;
        }
        for( int idx=0; idx < size; ++idx )
        {
            char chr = *ptr++;
            if( chr < '0' || chr > '9' )
                break;
            value = (value * 10) + (int)(chr - '0');
        }
        return value * sign;
    }

    __device__ long stol( const char* str, size_t bytes )
    {
        const char* ptr = str;
        if( !ptr || !bytes )
            return 0; // probably should be an assert
        long value = 0;
        int sign = 1, size = (int)bytes;
        if( *ptr == '-' || *ptr == '+' )
        {
            sign = (*ptr=='-' ? -1:1);
            ++ptr;
            --size;
        }
        for( int idx=0; idx < size; ++idx )
        {
            char chr = *ptr++;
            if( chr < '0' || chr > '9' )
                break;
            value = (value * 10) + (long)(chr - '0');
        }
        return value * sign;
    }

    __device__ unsigned long stoul( const char* str, size_t bytes )
    {
        const char* ptr = str;
        if( !ptr || !bytes )
            return 0; // probably should be an assert
        
        unsigned long value = 0;
        int size = (int)bytes;
        for( int idx=0; idx < size; ++idx )
        {
            char chr = *ptr++;
            if( chr < '0' || chr > '9' )
                break;
            value = (value * 10) + (unsigned long)(chr - '0');
        }
        return value;
    }
    
    __device__ float stof( const char* str, size_t bytes )
    {
        const char* ptr = str;
        if( !ptr || !bytes )
            return 0.0f; // probably should be an assert
    
        float value = 0, factor = 1;
        int size = (int)bytes;
        if(*ptr == '-' || *ptr == '+')
        {
            factor = (*ptr=='-' ? -1:1);
            ++ptr;
            --size;
        }
        bool decimal = false;
        for(int idx = 0; idx < size; ++idx )
        {
            char chr = *ptr++;
            if( chr == '.' )
            {
                decimal = true;
                continue;
            }
            if( chr < '0' || chr > '9' )
                break;
            if( decimal )
                factor /= 10.0f;
            value = value * 10.0f + (float)(chr - '0'); // this seems like we could run out of space in value
        }
        return value * factor;
    }
    
    __device__ double stod( const char* str, size_t bytes )
    {
        const char* ptr = str;
        if( !ptr || !bytes )
            return 0.0; // probably should be an assert
    
        double value = 0, factor = 1;
        int size = (int)bytes;
        if(*ptr == '-' || *ptr == '+')
        {
            factor = (*ptr=='-' ? -1:1);
            ++ptr;
            --size;
        }
        bool decimal = false;
        for(int idx = 0; idx < size; ++idx )
        {
            char chr = *ptr++;
            if( chr == '.' )
            {
                decimal = true;
                continue;
            }
            if( chr < '0' || chr > '9' )
                break;
            if( decimal )
                factor /= 10.0;
            value = value * 10.0 + (double)(chr - '0'); // see float above
        }
        return value * factor;
    }

    __device__ unsigned int hash( const char* str, unsigned int bytes )
    {
        unsigned int seed = 31; // prime number
        unsigned int hash = 0;
        for( unsigned int i = 0; i < bytes; i++ )
            hash = hash * seed + str[i];
        return hash;
    }
    
    __device__ int compare(const char* src, unsigned int sbytes, const char* tgt, unsigned int tbytes )
    {
        const char* ptr1 = src;
        if( !ptr1 )
            return -1;
        const char* ptr2 = tgt;
        if( !ptr2 )
            return 1;
        unsigned int len1 = sbytes;
        unsigned int len2 = tbytes;
        unsigned int idx;
        for(idx = 0; (idx < len1) && (idx < len2); ++idx)
        {
            if (*ptr1 != *ptr2)
                return (unsigned int)*ptr1 - (unsigned int)*ptr2;
            ptr1++;
            ptr2++;
        }
        if( idx < len1 )
            return 1;
        if( idx < len2 )
            return -1;
        return 0;
    }

    //
    __device__ int find( const char* sptr, unsigned int sz, const char* str, unsigned int bytes )
    {
        if(!sptr || !str || (sz < bytes))
            return -1;
        unsigned int end = sz - bytes;
        char* ptr1 = (char*)sptr;
        char* ptr2 = (char*)str;
        for(int idx=0; idx < end; ++idx)
        {
            bool match = true;
            for( int jdx=0; jdx < bytes; ++jdx )
            {
                if(ptr1[jdx] == ptr2[jdx] )
                    continue;
                match = false;
                break;
            }
            if( match )
                return idx; // chars_in_string(sptr,idx);
            ptr1++;
        }
        return -1;
    }

    __device__ int rfind( const char* sptr, unsigned int sz, const char* str, unsigned int bytes )
    {
        if(!sptr || !str || (sz < bytes) )
            return -1;
        unsigned end = sz - bytes;
        char* ptr1 = (char*)sptr + end;
        char* ptr2 = (char*)str;
        for(int idx=0; idx < end; ++idx)
        {
            bool match = true;
            for( int jdx=0; jdx < bytes; ++jdx )
            {
                if(ptr1[jdx] == ptr2[jdx] )
                    continue;
                match = false;
                break;
            }
            if( match )
                return sz - bytes - idx; //chars_in_string(sptr,end - idx);
            ptr1--; // go backwards
        }
        return -1;
    }

    //__device__ int find_first_of( const char* src, unsigned int bytes1, const char* chars, unsigned int bytes2 )
    //{
    //    return -1;
    //}
    //
    //__device__ int find_first_not_of( const char* src, unsigned int bytes1, const char* chars, unsigned int bytes2 )
    //{
    //    return -1;
    //}
    //
    //__device__ int find_last_of( const char* src, unsigned int bytes1, const char* chars, unsigned int bytes2 )
    //{
    //    return -1;
    //}
    //__device__ int find_last_not_of( const char* src, unsigned int bytes1, const char* chars, unsigned int bytes2 )
    //{
    //    return -1;
    //}

    //
    __device__ void copy( char* dst, unsigned int bytes, const char* src )
    {
        memcpy(dst,src,bytes);
    }

    //
    __device__ void lower( char* str, unsigned int bytes )
    {}
    __device__ void upper( char* str, unsigned int bytes )
    {}
    __device__ void swapcase( char* str, unsigned int bytes )
    {}
    
    // some utilities for handling individual UTF-8 characters
    #if 0
    __host__ __device__ int bytes_in_char( Char chr )
    {
        int count = 1;
        // no if-statements means no divergence
        count += (int)((chr & (unsigned)0x0000FF00 ) > 0);
        count += (int)((chr & (unsigned)0x00FF0000 ) > 0);
        count += (int)((chr & (unsigned)0xFF000000 ) > 0);
        return count;
    }

    __host__ __device__ int Char char_to_Char( const char* str )
    {
        int chwidth = _bytes_in_char((BYTE)*pSrc);
        Char ret = (Char)(*pSrc++) & 0xFF;
        if (chwidth > 1)
        {
            ret |= ((Char)(*pSrc++) & 0xFF) << 8;
            if (chwidth > 2)
            {
                ret |= ((Char)(*pSrc++) & 0xFF) << 16;
                if (chwidth > 3)
                    ret |= ((Char)(*pSrc++) & 0xFF) << 24;
            }
        }
        return ret;
    }

    __host__ __device__ int Char_to_char( Char chr, char* str )
    {
        int chwidth = bytes_in_char(chr);
        (*pDst++) = (char)chr & 0xFF;
        if(chwidth > 1)
        {
            (*pDst++) = (char)((chr >> 8) & 0xFF);
            if(chwidth > 2)
            {
                (*pDst++) = (char)((chr >> 16) & 0xFF);
                if(chwidth > 3)
                    (*pDst++) = (char)((chr >> 24) & 0xFF);
            }
        }
        return chwidth;
    }

    __host__ __device__ int chars_in_string( const char* str, unsigned int bytes )
    {
        if( str==0 || bytes==0 )
            return 0;
        // cannot get this to compile -- dynamic parallelism this is
        //auto citr = thrust::make_counting_iterator<int>(0);
        //int nchars = thrust::transform_reduce(thrust::device,
        //    citr, citr + bytes,
        //    [str] __device__( int idx ){
        //        BYTE chr = (BYTE)str[idx];
        //        return (int)((chr & 0xC0) != 0x80); // ignore 'extra' bytes
        //    },0,thrust::plus<size_t>());
        //cudaDeviceSynchronize(); -- this too
        // going manual; performance is not bad, especially for small strings
        int nchars = 0;
        for( int idx=0; idx < bytes; ++idx )
            nchars += (int)(((BYTE)str[idx] & 0xC0) != 0x80);
        return nchars;
    }
    #endif
}