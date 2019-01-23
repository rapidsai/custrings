
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
