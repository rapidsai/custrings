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
#pragma once

#include <cuda_runtime.h>

//
// This is used by the create_from_ipc and create_ipc_transfer methods.
//
struct nvstrings_ipc_transfer
{
    char* base_address;
    cudaIpcMemHandle_t hstrs;
    unsigned int count;
    void* strs;

    cudaIpcMemHandle_t hmem;
    size_t size;
    void* mem;

    nvstrings_ipc_transfer()
    : base_address(0), count(0), strs(0), size(0), mem(0) {}

    ~nvstrings_ipc_transfer()
    {
    }

    void setStrsHandle(void* in, char* base, unsigned int c)
    {
        count = c;
        base_address = base;
        cudaIpcGetMemHandle(&hstrs,in);
    }

    void setMemHandle(void* in, size_t sz)
    {
        size = sz;
        cudaIpcGetMemHandle(&hmem,in);
    }

    void* getStringsPtr()
    {
        if( !strs && count )
        {
            cudaError_t err = cudaIpcOpenMemHandle((void**)&strs,hstrs,cudaIpcMemLazyEnablePeerAccess);
            if( err!=cudaSuccess )
                printf("%d nvs-getStringsPtr", err);
        }
        return strs;
    }

    void* getMemoryPtr()
    {
        if( !mem && size )
        {
            cudaError_t err = cudaIpcOpenMemHandle((void**)&mem,hmem,cudaIpcMemLazyEnablePeerAccess);
            if( err!=cudaSuccess )
                printf("%d nvs-getMemoryPtr", err);
        }
        return mem;
    }
};

struct nvcategory_ipc_transfer
{
    char* base_address;
    cudaIpcMemHandle_t hstrs;
    unsigned int keys;
    void* strs;

    cudaIpcMemHandle_t hmem;
    size_t size;
    void* mem;

    cudaIpcMemHandle_t hmap;
    unsigned int count;
    void* vals;

    nvcategory_ipc_transfer()
    : base_address(0), keys(0), strs(0), size(0), mem(0), count(0), vals(0) {}

    ~nvcategory_ipc_transfer()
    {
        if( strs )
            cudaIpcCloseMemHandle(strs);
        if( mem )
            cudaIpcCloseMemHandle(mem);
        if( vals )
            cudaIpcCloseMemHandle(vals);
    }

    void setStrsHandle(void* in, char* base, unsigned int n)
    {
        keys = n;
        base_address = base;
        cudaIpcGetMemHandle(&hstrs,in);
    }

    void setMemHandle(void* in, size_t sz)
    {
        size = sz;
        cudaIpcGetMemHandle(&hmem,in);
    }

    void setMapHandle(void* in, unsigned int n)
    {
        count = n;
        cudaIpcGetMemHandle(&hmap,in);
    }

    void* getStringsPtr()
    {
        if( !strs && keys )
            cudaIpcOpenMemHandle((void**)&strs,hstrs,cudaIpcMemLazyEnablePeerAccess);
        return strs;
    }

    void* getMemoryPtr()
    {
        if( !mem && size )
            cudaIpcOpenMemHandle((void**)&mem,hmem,cudaIpcMemLazyEnablePeerAccess);
        return mem;
    }

    void* getMapPtr()
    {
        if( !vals && count )
            cudaIpcOpenMemHandle((void**)&vals,hmap,cudaIpcMemLazyEnablePeerAccess);
        return vals;
    }
};
