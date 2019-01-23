#include <memory>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "../NVStrings.h"

//
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 csv.cu -L.. -lNVStrings -o CSVTest --linker-options -rpath,.:
//

// csv file contents in device memory
void* d_fileContents = 0;

// return a vector of DString's we wish to process
std::pair<const char*,size_t>* setupTest(int& linesCount, int column)
{
    //FILE* fp = fopen("../../data/1420-rows.csv", "rb");
    //FILE* fp = fopen("../../data/7584-rows.csv", "rb");
    FILE* fp = fopen("../../data/36634-rows.csv", "rb");
    if( !fp )
    {
        printf("missing csv file\n");
        return 0;
    }
	fseek(fp, 0, SEEK_END);
	int fileSize = (int)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    printf("File size = %d bytes\n", fileSize);
    if( fileSize < 2 )
    {
        fclose(fp);
        return 0;
    }
    // load file into memory
    int contentsSize = fileSize+2;
    char* contents = new char[contentsSize+2];
    fread(contents, 1, fileSize, fp);
    contents[fileSize] = '\r'; // line terminate
	contents[fileSize+1] = 0;  // and null-terminate
	fclose(fp);

    // find lines -- compute offsets vector values
    thrust::host_vector<int> lineOffsets;
    char* ptr = contents;
    while( *ptr )
    {
        char ch = *ptr;
        if( ch=='\r' )
        {
            *ptr = 0;
            while(ch && (ch < ' ')) ch = *(++ptr);
            lineOffsets.push_back((int)(ptr - contents));
            continue;
        }    
        ++ptr;
    }
    linesCount = (int)lineOffsets.size();
    printf("Found %d lines\n",linesCount);
    // copy file contents into device memory
    char* d_contents = 0;
    cudaMalloc(&d_contents,contentsSize);
    cudaMemcpy(d_contents,contents,contentsSize,cudaMemcpyHostToDevice);
    delete contents; // done with the host data

    // copy offsets vector into device memory
    thrust::device_vector<int> offsets(lineOffsets);
    int* d_offsets = offsets.data().get();
    // build empty output vector of DString*'s
    --linesCount; // removed header line
    std::pair<const char*,size_t>* d_column1 = 0;
    cudaMalloc(&d_column1, linesCount * sizeof(std::pair<const char*,size_t>));

    // create a vector of DStrings using the first column of each line
    thrust::for_each_n(thrust::device,
      thrust::make_counting_iterator<size_t>(0), linesCount, 
      [d_contents, d_offsets, column, d_column1] __device__(size_t idx){
        // probably some more elegant way to do this
        int lineOffset = d_offsets[idx];
        int lineLength = d_offsets[idx+1] - lineOffset;
        d_column1[idx].first = (const char*)0;
        if( lineLength < 1 )
            return;
        char* line = &(d_contents[lineOffset]);
        char* stringStart = line;
        int columnLength = 0, col = 0;
        for( int i=0; (i < lineLength); ++i )
        {
            if( line[i] && line[i] != ',' )
            {
                ++columnLength;
                continue;
            }
            if( col++ >= column )
                break;
            stringStart = line + i + 1;
            columnLength = 0;
        }
        if( columnLength==0 )
            return;
        // add string to vector array
        d_column1[idx].first = (const char*)stringStart;
        d_column1[idx].second = (size_t)columnLength;
      });
    //
    cudaThreadSynchronize();
    d_fileContents = d_contents;
    return d_column1;
}

int main( int argc, char** argv )
{
    //NVStrings::initLibrary();

    int count = 0;
    std::pair<const char*,size_t>* column1 = setupTest(count,2);
    if( column1==0 )
        return -1;

    NVStrings dstrs( column1, count );

    cudaFree(d_fileContents); // csv data not needed once dstrs is created
    cudaFree(column1);        // string index data has done its job as well

    // simple strings op
    int* rtn = new int[count];
    dstrs.len(rtn,false);
    for( int idx=0; idx < count; ++idx )
        printf("%d,",rtn[idx]);
    printf("\n");
    delete rtn;

    // show column values
    char** list = new char*[count];
    dstrs.to_host(list,0,count);
    for( int idx=0; idx < count; ++idx )
        printf("%s,",list[idx]);
    printf("\n");
    delete list;

    return 0;
}