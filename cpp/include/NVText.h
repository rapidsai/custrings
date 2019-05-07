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

class NVStrings;
//
//
class NVText
{

public:

    //
    static NVStrings* tokenize(NVStrings& strs, const char* delimiter );

    //
    static NVStrings* tokenize(NVStrings& strs, NVStrings& delimiters );

    //
    static NVStrings* unique_tokens(NVStrings& strs, const char* delimiter = " ");

    //
    static unsigned int token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool devmem=true );

    //
    static unsigned int contains_strings( NVStrings& strs, NVStrings& tokens, bool* results, bool devmem=true );

    //
    static unsigned int strings_counts( NVStrings& strs, NVStrings& tokens, unsigned int* results, bool devmem=true );

    //
    static unsigned int tokens_counts( NVStrings& strs, NVStrings& tokens, const char* delimiter, unsigned int* results, bool devmem=true );

    // edit distance algorithm types
    enum distance_type {
        levenshtein
    };
    static unsigned int edit_distance( distance_type algo, NVStrings& strs, const char* str, unsigned int* results, bool devmem=true );
    static unsigned int edit_distance( distance_type algo, NVStrings& strs1, NVStrings& strs2, unsigned int* results, bool devmem=true );
};
