
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../NVStrings.h"
//
// This can be compile using:
//   nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 indexes.cu -L.. -lNVStrings -o indexes --linker-options -rpath,.:
// or
//   g++ -w -std=c++11 -I/usr/local/cuda/include indexes.cpp -L.. -lNVStrings -L/usr/local/cuda/lib64 -lcudart -lrt -o indexes -Wl,-rpath,.:
// since it does not have any kernels or device code in it.
//
int main( int argc, char** argv )
{
    size_t count = 10000000; // 10 million
    char* buffer = new char[count*16];
    memset(buffer, 'a', count*16 );

    char* d_buffer = 0;
    cudaMalloc(&d_buffer,count*16);
    cudaMemcpy(d_buffer,buffer,count*16,cudaMemcpyHostToDevice);
    std::pair<const char*,size_t>* column = new std::pair<const char*,size_t>[count];
    char* d_ptr = d_buffer;
    for( int idx=0; idx < count; ++idx )
    {
        column[idx].first = d_ptr;
        column[idx].second = 16;
        d_ptr += 16;
    }

    std::pair<const char*,size_t>* d_column = 0;
    cudaMalloc(&d_column,count*sizeof(std::pair<const char*,size_t>));
    cudaMemcpy(d_column,column,count*sizeof(std::pair<const char*,size_t>),cudaMemcpyHostToDevice);
    
    NVStrings* strs = new NVStrings(d_column,count);
    
    cudaFree(d_buffer);
    cudaFree(d_column);

    printf("number of strings = %'lu\n", strs->size());
    delete strs;

    // simple strings op
    //int* rtn = new int[count];
    //strs->len(rtn,false);
    //for( int idx=0; idx < count; ++idx )
    //    printf("%d,",rtn[idx]);
    //printf("\n");
    //delete rtn;

    // show column values
    //char** list = new char*[count];
    //strs->to_host(list,0,count);
    //for( int idx=0; idx < count; ++idx )
    //    printf("%s,",list[idx]);
    //printf("\n");
    //delete list;

    return 0;
}