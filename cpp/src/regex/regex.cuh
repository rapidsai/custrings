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
struct Reinst;
struct Reljunk;
struct Reljunk_Sub;
class custring_view;

//
class dreclass
{
public:
    int builtins;
    int count;
    char32_t* chrs;
    unsigned char* uflags;

    __device__ inline dreclass(unsigned char* uflags);
    __device__ inline bool is_match(char32_t ch);
};

//
class dreprog
{
    int startinst_id, num_capturing_groups;
    int insts_count, classes_count;
    unsigned char* unicode_flags;
    void* relists_mem;

    dreprog() {}
    ~dreprog() {}

    void free_relists();

    //
    __device__ inline int regexec( custring_view* dstr, Reljunk& jnk, int& begin, int& end, int groupid=0 );

public:
    //
    static dreprog* create_from(const char32_t* pattern, unsigned char* uflags, unsigned int strscount=0);
    static void destroy(dreprog* ptr);

    int inst_counts();
    int group_counts();

    __host__ __device__ inline Reinst* get_inst(int idx);
    //__device__ char32_t* get_class(int idx, int& len);
    __device__ inline int get_class(int idx, dreclass& cls);

    //
    __device__ inline int contains( custring_view* dstr );
    __device__ inline int match( custring_view* dstr );
    __device__ inline int find( custring_view* dstr, int& begin, int& end );
    __device__ inline int extract( custring_view* str, int& begin, int& end, int col );

    __device__ inline int contains( unsigned int idx, custring_view* dstr );
    __device__ inline int match( unsigned int idx, custring_view* dstr );
    __device__ inline int find( unsigned int idx, custring_view* dstr, int& begin, int& end );
    __device__ inline int extract( unsigned int idx, custring_view* str, int& begin, int& end, int col );

};

#include "regexec.inl"
