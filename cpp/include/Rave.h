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
#include <cstddef>

class NVStrings;
//
//
class Rave
{

public:

    // previously - create_vocab
    static NVStrings* unique_tokens(NVStrings& strs, const char* delimiter = " ");

    // previously - word_count
    static unsigned int token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool devmem=true );

    //
    static unsigned int contains_strings( NVStrings& strs, NVStrings& tokens, bool* results, bool devmem=true );

    //
    static unsigned int strings_counts( NVStrings& strs, NVStrings& tokens, unsigned int* results, bool devmem=true );
};
