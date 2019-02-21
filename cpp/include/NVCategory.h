/*
* Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/
#include <cstddef>
#include <vector>

class NVStrings;
class NVCategoryImpl;
//
// Manages a list of strings for a category and their associated indexes.
// Unique strings are assigned unique integer values within this instance.
//
class NVCategory
{
    NVCategoryImpl* pImpl;
    NVCategory();
    NVCategory(const NVCategory&);
    ~NVCategory();
    NVCategory& operator=(const NVCategory&);

public:

    // create instance from array of null-terminated host strings
    static NVCategory* create_from_array(const char** strs, int count);
    // create instance from array of strings/length pairs
    static NVCategory* create_from_index(std::pair<const char*,size_t>* strs, size_t count, bool devmem=true);
    // create instance from host buffer with offsets; null-bitmask is arrow-ordered
    static NVCategory* create_from_offsets(const char* strs, int count, const int* offsets, const unsigned char* nullbitmask=0, int nulls=0);
    // create instance from NVStrings instance
    static NVCategory* create_from_strings(NVStrings& strs);
    static NVCategory* create_from_strings(std::vector<NVStrings*>& strs);
    // create instance from list of categories; values are remapped
    static NVCategory* create_from_categories(std::vector<NVCategory*>& cats);
    // use this method to free any instance create by methods in this class
    static void destroy(NVCategory* inst);

    // return number of items
    unsigned int size();
    // return number of keys
    unsigned int keys_size();
    // true if any null values exist
    bool has_nulls();

    // set bit-array identifying the null values
    int set_null_bitarray( unsigned char* bitarray, bool devmem=true );
    // build a string-index from this instances strings
    int create_index(std::pair<const char*,size_t>* strs, bool bdevmem=true );
    //
    NVCategory* copy();

    // return key strings for this instance
    NVStrings* get_keys();

    // return single category value given string or index
    int get_value(unsigned int index);
    int get_value(const char* str);

    // return category values for all indexes
    int get_values( int* results, bool devmem=true );
    // returns pointer to internal values array in device memory
    const int* values_cptr();

    // return values positions for given key
    int get_indexes_for( unsigned int index, int* results, bool devmem=true );
    int get_indexes_for( const char* str, int* results, bool devmem=true );

    // creates a new instance incorporating the new strings
    NVCategory* add_strings(NVStrings& strs);
    // creates a new instance without the specified strings
    NVCategory* remove_strings(NVStrings& strs);

    // creates a new instance adding the specified strings as keys and remapping the values
    NVCategory* add_keys_and_remap(NVStrings& strs);
    // creates a new instance removing the keys matching the specified strings and remapping the values
    NVCategory* remove_keys_and_remap(NVStrings& strs);
    // creates a new instance using the specified strings as keys causing add/remove as appropriate
    // values are also remapped
    NVCategory* set_keys_and_remap(NVStrings& strs);
    //
    NVCategory* merge_category(NVCategory& cat);
    NVCategory* merge_and_remap(NVCategory& cat);

    // convert to original strings list
    NVStrings* to_strings();
    // create a new strings instance identified by the specified index values
    NVStrings* gather_strings( int* pos, unsigned int elems, bool devmem=true );
};
