#pragma once

// utf8 characters are 1-4 bytes
typedef unsigned int Char;

//
// This class represents and manages a single character array in device memory.
// The character array is expected as a UTF-8 encoded string.
// All index values and lengths are in characters and not bytes.
// Some methods change the string in-place while others will return new strings.
// Any method where it is possible to shorten the string will return a new string.
//
// All memory must be device memory provided by the caller and the class may not
// be created on the stack or new'd. Use the alloc_size() methods to determine how
// much memory is required for an instance to manage a string. Use the create_from()
// methods to create an instance over the provided memory segment. Caller must
// ensure the memory is large enough as per the alloc_size() value.
//
// Methods like replace() and strip() require new memory buffer to hold the
// resulting string. The insert() and append() methods require the string's
// original memory to be large enough hold the additional characters.
//
// The 'unsigned int' sizes here allow for a string to be 4 billion bytes long.
// This seems impractical given that the purpose of this class is to run parallel
// operations across many strings. You could only parallelize 8 strings of this
// size on a single 32GB GPU device. Using `unsigned short` would allow for 65KB
// strings which should be sufficient for handling tweets (144 characters) or
// the average Wikipedia article which is 5KB:
//    https://en.wikipedia.org/wiki/Wikipedia:Size_comparisons
// Wikipedia recommends breaking up any article over 50KB.
// A different string class that handles parallelizing over characters seems
// more appropriate for processing single large documents like tokenizing CSV files or
// Wikipedia articles.
//
class custring_view
{
    custring_view(); // prevent creating instance directly
                     // use the create_from methods to provide memory for the object to reside
protected:
    unsigned int m_bytes;  // combining these two did not save space
    unsigned int m_chars;  // number of characters
    //char* m_data;        // all variable length data including char array;
                           // pointer is now calculated on demand

    __device__ static custring_view* create_from(void* buffer);
    __device__ void init_fields(unsigned int bytes);
    __device__ unsigned int offset_for_char_pos(unsigned int chpos) const;
    __device__ void offsets_for_char_pos(unsigned int& spos, unsigned int& epos) const;
    __device__ unsigned int char_offset(unsigned int bytepos) const;

public:

    // returns the amount of memory required to manage the given character array
    __host__ __device__ static unsigned int alloc_size(const char* data, unsigned int size);
    // returns the amount of memory needed to manage character array of this size
    __host__ __device__ static unsigned int alloc_size(unsigned int bytes, unsigned int chars);
    // these can be used to create instances in already allocated memory
    __device__ static custring_view* create_from(void* buffer, const char* data, unsigned int size);
    __device__ static custring_view* create_from(void* buffer, custring_view& str);

    // return how much memory is used by this instance
    __device__ unsigned int alloc_size() const;
    //
    __device__ unsigned int size() const;        // same as length()
    __device__ unsigned int length() const;      // number of bytes
    __device__ unsigned int chars_count() const; // number of characters
    __device__ char* data();            // raw pointer, use at your own risk
    __device__ const char* data() const;
    // returns true if string has no characters
    __device__ bool empty() const;
    // computes a hash based on the characters in the array
    __device__ unsigned int hash() const;

    // iterator is read-only
    class iterator
    {
        const char* p;
        unsigned int cpos, offset;
    public:
        __device__ iterator(custring_view& str,unsigned int initPos);
        __device__ iterator(const iterator& mit);
        __device__ iterator& operator++();
        __device__ iterator operator++(int);
        __device__ bool operator==(const iterator& rhs) const;
        __device__ bool operator!=(const iterator& rhs) const;
        __device__ Char operator*() const;
    };
    // iterator methods
    __device__ iterator begin();
    __device__ iterator end();

    // return character (UTF-8) at given position
    __device__ Char at(unsigned int pos) const;
    // this is read-only right now since modifying an individual character may change the memory requirements
    __device__ Char operator[](unsigned int pos) const;
    // return the byte offset for a character position
    __device__ unsigned int byte_offset_for(unsigned int pos) const;

    // return 0 if arg string matches
    // return <0 or >0 depending first different character
    __device__ int compare(const custring_view& str) const;
    __device__ int compare(const char* data, unsigned int bytes) const;

    __device__ bool operator==(const custring_view& rhs);
    __device__ bool operator!=(const custring_view& rhs);
    __device__ bool operator<(const custring_view& rhs);
    __device__ bool operator>(const custring_view& rhs);
    __device__ bool operator<=(const custring_view& rhs);
    __device__ bool operator>=(const custring_view& rhs);

    // return character position if arg string is contained in this string
    // return -1 if string is not found
    // (pos,pos+count) is the range of this string that is scanned
    __device__ int find( const custring_view& str, unsigned int pos=0, int count=-1 ) const;
    __device__ int find( const char* str, unsigned int bytes, unsigned int pos=0, int count=-1 ) const;
    __device__ int find( Char chr, unsigned int pos=0, int count=-1 ) const;
    // same as find() but searches from the end of this string
    __device__ int rfind( const custring_view& str, unsigned int pos=0, int count=-1 ) const;
    __device__ int rfind( const char* str, unsigned int bytes, unsigned int pos=0, int count=-1 ) const;
    __device__ int rfind( Char chr, unsigned int pos=0, int count=-1 ) const;
    // these are for parity with std::string
    __device__ int find_first_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ int find_first_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ int find_first_of( Char ch, unsigned int pos=0 ) const;
    __device__ int find_first_not_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ int find_first_not_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ int find_first_not_of( Char ch, unsigned int pos=0 ) const;
    __device__ int find_last_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ int find_last_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ int find_last_of( Char ch, unsigned int pos=0 ) const;
    __device__ int find_last_not_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ int find_last_not_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ int find_last_not_of( Char ch, unsigned int pos=0 ) const;

    // return substring based on character position and length
    // caller must provide memory for the resulting object
    __device__ custring_view* substr( unsigned int pos, unsigned int length, unsigned int step, void* mem );
    __device__ unsigned int substr_size( unsigned int pos, unsigned int length, unsigned int step=1 ) const;
    // copy the character array to the given device memory pointer
    __device__ unsigned int copy( char* str, int count, unsigned int pos=0 );

    // append string or character to this string
    // orginal string must have been created with enough memory for this operation
    __device__ custring_view& operator+=( const custring_view& str );
    __device__ custring_view& operator+=( Char chr );
    __device__ custring_view& operator+=( const char* str );
    // append argument string to this one
    __device__ custring_view& append( const char* str, unsigned int bytes );
    __device__ custring_view& append( const custring_view& str );
    __device__ custring_view& append( Char chr, unsigned int count=1 );
    __device__ unsigned int append_size( const char* str, unsigned int bytes ) const;
    __device__ unsigned int append_size( const custring_view& str ) const;
    __device__ unsigned int append_size( Char chr, unsigned int count=1 ) const;

    // insert the given string into the character position specified
    // orginal string must have been created with enough memory for this operation
    __device__ custring_view& insert( unsigned int pos, const char* data, unsigned int bytes );
    __device__ custring_view& insert( unsigned int pos, custring_view& str );
    __device__ custring_view& insert( unsigned int pos, unsigned int count, Char chr );
    __device__ unsigned int insert_size( const char* str, unsigned int bytes ) const;
    __device__ unsigned int insert_size( const custring_view& str ) const;
    __device__ unsigned int insert_size( Char chr, unsigned int count=1 ) const;

    // replace the given range of characters with the arg string
    // caller must provide memory for the resulting object
    __device__ custring_view* replace( unsigned int pos, unsigned int length, const char* data, unsigned int bytes, void* mem );
    __device__ custring_view* replace( unsigned int pos, unsigned int length, const custring_view& str, void* mem );
    __device__ custring_view* replace( unsigned int pos, unsigned int length, unsigned int count, Char chr, void* mem );
    __device__ unsigned int replace_size( unsigned int pos, unsigned int length, const char* data, unsigned int bytes ) const;
    __device__ unsigned int replace_size( unsigned int pos, unsigned int length, const custring_view& str ) const;
    __device__ unsigned int replace_size( unsigned int pos, unsigned int length, unsigned int count, Char chr ) const;

    // tokenizes string around the given delimiter string upto count
    // call with strs=0, will return the number of string tokens
    __device__ unsigned int split( const char* delim, unsigned int bytes, int count, custring_view** strs );
    __device__ unsigned int split_size( const char* delim, unsigned int bytes, int* sizes, int count ) const;
    __device__ unsigned int rsplit( const char* delim, unsigned int bytes, int count, custring_view** strs );
    __device__ unsigned int rsplit_size( const char* delim, unsigned int bytes, int* sizes, int count ) const;

    // return new string with given character from the beginning/end removed from this string
    // caller must provide memory for the resulting object
    __device__ custring_view* strip( Char chr, void* mem );
    __device__ unsigned int strip_size( Char chr=' ') const;
    __device__ custring_view* lstrip( Char chr, void* mem );
    __device__ unsigned int lstrip_size( Char chr=' ' ) const;
    __device__ custring_view* rstrip( Char chr, void* mem );
    __device__ unsigned int rstrip_size( Char chr=' ' ) const;

    // these will change the characters in this string
    __device__ custring_view& lower();
    __device__ custring_view& upper();
    __device__ custring_view& capitalize();
    __device__ custring_view& swapcase();
    __device__ custring_view& titlecase();

    // return numeric value represented by the characters in this string
    __device__ int stoi() const;
    __device__ long stol() const;
    __device__ unsigned long stoul() const;
    __device__ float stof() const;
    __device__ double stod() const;

    //
    __device__ bool starts_with( const char* str, unsigned int bytes ) const;
    __device__ bool starts_with( custring_view& str ) const;
    __device__ bool ends_with( const char* str, unsigned int bytes ) const;
    __device__ bool ends_with( custring_view& str ) const;

    //
    __device__ bool islower() const;
    __device__ bool isupper() const;
    __device__ bool isspace() const;
    __device__ bool isdecimal() const;
    __device__ bool isnumeric() const;
    __device__ bool isdigit() const;

    // some utilities for handling individual UTF-8 characters
    __host__ __device__ static unsigned int bytes_in_char( Char chr );
    __host__ __device__ static unsigned int char_to_Char( const char* str, Char& chr );
    __host__ __device__ static unsigned int Char_to_char( Char chr, char* str );
    __host__ __device__ static unsigned int chars_in_string( const char* str, unsigned int bytes );
};
