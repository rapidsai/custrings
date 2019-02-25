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

class custring_view;
class NVStringsImpl;
//
// This maps indirectly to the methods in the nvstrings class in the nvstrings.py source file.
// It is a host object that manages vectors of strings stored in device memory.
// Each operation performs (in parallel) against all strings in this instance.
//
class NVStrings
{
    NVStringsImpl* pImpl;

    // ctors/dtor are made private to control memory allocation
    NVStrings();
    NVStrings(unsigned int count);
    NVStrings(const NVStrings&);
    NVStrings& operator=(const NVStrings&);
    ~NVStrings();

public:
    // sort by length and name sorts by length first
    enum sorttype { none=0, length=1, name=2 };

    // create instance from array of null-terminated host strings
    static NVStrings* create_from_array(const char** strs, unsigned int count);
    // create instance from array of string/length pairs
    static NVStrings* create_from_index(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem=true, sorttype st=none );
    // create instance from host buffer with offsets; null-bitmask is arrow-ordered
    static NVStrings* create_from_offsets(const char* strs, int count, const int* offsets, const unsigned char* nullbitmask=0, int nulls=0);
    // create instance from NVStrings instances
    static NVStrings* create_from_strings( std::vector<NVStrings*> strs );
    // use this method to free any instance created by methods in this class
    static void destroy(NVStrings* inst);

    // return the number of device bytes used by this instance
    size_t memsize() const;
    // number of strings managed by this instance
    unsigned int size() const;

    // copy the list of strings back into the provided host memory
    int to_host(char** list, int start, int end);
    // create index for device strings contained in this instance; array must hold at least size() elements
    int create_index(std::pair<const char*,size_t>* strs, bool devmem=true );
    int create_custring_index( custring_view** strs, bool devmem=true );
    // copy strings into memory provided
    int create_offsets( char* strs, int* offsets, unsigned char* nullbitmask=0, bool devmem=true );
    // set bit-array identifying the null strings; returns the number of nulls found
    unsigned int set_null_bitarray( unsigned char* bitarray, bool emptyIsNull=false, bool todevice=true );
    // set int array with position of null strings
    unsigned int get_nulls( unsigned int* pos, bool emptyIsNull=false, bool todevice=true );

    // create a new instance from this instance
    NVStrings* copy();
    // create a new instance containing only the strings in the specified range
    NVStrings* sublist( unsigned int start, unsigned int end, unsigned int step=0 );
    // returns strings in the order of the specified position values
    NVStrings* gather( int* pos, unsigned int count, bool devmem=true );
    // return a new instance without the specified strings
    NVStrings* remove_strings( unsigned int* pos, unsigned int count, bool devmem=true );

    // return the number of characters in each string
    unsigned int len(int* lengths, bool todevice=true);
    // return the number of bytes for each string
    size_t byte_count(int* lengths, bool todevice=true);

    // adds the given string(s) to this list of strings and returns as new strings
    NVStrings* cat( NVStrings* others, const char* separator, const char* narep=0);
    // concatenates all strings into one new string
    NVStrings* join( const char* delimiter, const char* narep=0 );

    // each string is split into a list of new strings
    int split( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    int rsplit( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    // split each string into a new column -- number of columns = string with the most delimiters
    unsigned int split_column( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    unsigned int rsplit_column( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    // each string is split into two strings on the first delimiter found
    // three strings are returned for each string: left-half, delimiter itself, right-half
    int partition( const char* delimiter, std::vector<NVStrings*>& results);
    int rpartition( const char* delimiter, std::vector<NVStrings*>& results);

    // return a specific character (as a string) by position for each string
    NVStrings* get(unsigned int pos);
    // concatenate each string with itself the number of times specified
    NVStrings* repeat(unsigned int count);
    // add padding to each string as specified by the parameters
    enum padside { left, right, both };
    NVStrings* pad(unsigned int width, padside side, const char* fillchar=0);
    NVStrings* ljust( unsigned int width, const char* fillchar=0 );
    NVStrings* center( unsigned int width, const char* fillchar=0 );
    NVStrings* rjust( unsigned int width, const char* fillchar=0 );
    // pads string with number with leading zeros
    NVStrings* zfill( unsigned int width );
    // this inserts new-line characters into each string
    NVStrings* wrap( unsigned int width );

    // returns a substring of each string
    NVStrings* slice( int start=0, int stop=-1, int step=1 );
    NVStrings* slice_from( int* starts=0, int* ends=0 );
    // inserts the specified string (repl) into each string
    NVStrings* slice_replace( const char* repl, int start=0, int stop=-1 );
    // replaces occurrences of str with repl
    NVStrings* replace( const char* str, const char* repl, int maxrepl=-1 );
    NVStrings* replace_re( const char* pat, const char* repl, int maxrepl=-1 );
    // translate characters in each string using the character-mapping table provided
    NVStrings* translate( std::pair<unsigned,unsigned>* table, unsigned int count );

    // remove specified character if found at the beginning of each string
    NVStrings* lstrip( const char* to_strip );
    // remove specified character if found at the beginning or end of each string
    NVStrings* strip( const char* to_strip );
    // remove specified character if found at the end each string
    NVStrings* rstrip( const char* to_strip );

    // return new strings with modified character case
    NVStrings* lower();
    NVStrings* upper();
    NVStrings* capitalize();
    NVStrings* swapcase();
    NVStrings* title();

    // compare single arg string to all the strings
    unsigned int compare( const char* str, int* results, bool todevice=true );
    // search for a string within each string
    // the index/rindex methods just use these too
    // return value is the number of positive (>=0) results
    unsigned int find( const char* str, int start, int end, int* results, bool todevice=true );
    unsigned int rfind( const char* str, int start, int end, int* results, bool todevice=true );
    unsigned int find_from( const char* str, int* starts, int* ends, int* results, bool todevice=true );
    unsigned int find_multiple( NVStrings& strs, int* results, bool todevice=true );
    // return all occurrences of the specified regex pattern in each string
    int findall( const char* ptn, std::vector<NVStrings*>& results );
    int findall_column( const char* ptn, std::vector<NVStrings*>& results );
    // search for string or regex pattern within each string
    int contains( const char* str, bool* results, bool todevice=true );
    int contains_re( const char* ptn, bool* results, bool todevice=true );
    // match alrogithm is unique in that only the beginning of each string is checked
    int match( const char* ptn, bool* results, bool todevice=true );
    // return count of the regex pattern occurrences in each string
    int count_re( const char* ptn, int* results, bool todevice=true );
    // compares the beginning of each string with the specified string
    unsigned int startswith( const char* str, bool* results, bool todevice=true );
    // compares the end of each string with the specified string
    unsigned int endswith( const char* str, bool* results, bool todevice=true );

    // returns a list of strings for each group specified in the specified regex pattern
    int extract( const char* ptn, std::vector<NVStrings*>& results );
    // same as extract() but group results are returned in column-major
    int extract_column( const char* ptn, std::vector<NVStrings*>& results );
    //
    unsigned int isalnum( bool* results, bool todevice=true );
    unsigned int isalpha( bool* results, bool todevice=true );
    unsigned int isdigit( bool* results, bool todevice=true );
    unsigned int isspace( bool* results, bool todevice=true );
    unsigned int isdecimal( bool* results, bool todevice=true );
    unsigned int isnumeric( bool* results, bool todevice=true );
    unsigned int islower( bool* results, bool todevice=true );
    unsigned int isupper( bool* results, bool todevice=true );

    // returns integer values represented by each string
    unsigned int stoi(int* results, bool todevice=true);
    unsigned int htoi(unsigned int* results, bool todevice=true);
    // returns float values represented by each string
    unsigned int stof(float* results, bool todevice=true);
    // return unsigned 32-bit hash value for each string
    unsigned int hash( unsigned int* results, bool todevice=true );

    // sorts the strings managed by this instance
    NVStrings* sort( sorttype st, bool ascending=true );
    // returns new row index positions only; strings order is not modified
    int order( sorttype st, bool ascending, unsigned int* indexes, bool todevice=true );

    // output strings to stdout
    void print( int pos=0, int end=-1, int maxwidth=-1, const char* delimiter = "\n" );
    // for performance analysis
    void printTimingRecords();
};
