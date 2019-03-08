//
// 8-bit flag for each unicode code-point. 
// Bit values assigned as follows (76543210):
//    7 - reserved 
//    5 - islower 
//    6 - isupper 
//    4 - isspace 
//    3 - isalpha 
//    2 - isdigit 
//    1 - isnumeric 
//    0 - isdecimal 
//
// Note that isalnum can be identified by (https://docs.python.org/3/library/stdtypes.html#str.isalnum): 
//   isalpha, isdecimal, isdigit, isnumeric 
//
#define IS_SPACE(x) ((x & 16)>0) 
#define IS_ALPHA(x) ((x & 8)>0) 
#define IS_DIGIT(x) ((x & 4)>0) 
#define IS_NUMERIC(x) ((x & 2)>0) 
#define IS_DECIMAL(x) ((x & 1)>0) 
#define IS_ALPHANUM(x) ((x & 15)>0) 
#define IS_UPPER(x) ((x & 32)>0) 
#define IS_LOWER(x) ((x & 64)>0) 

