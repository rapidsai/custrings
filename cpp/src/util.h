

// csv parser flags
#define CSV_SORT_LENGTH    1
#define CSV_SORT_NAME      2
#define CSV_NULL_IS_EMPTY  8

// this has become a thing
NVStrings* createFromCSV(std::string csvfile, unsigned int column, unsigned int lines=0, unsigned int flags=0);
