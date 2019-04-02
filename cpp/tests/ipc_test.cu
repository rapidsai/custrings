
#include <stdio.h>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include "../include/NVStrings.h"
#include "../include/ipc_transfer.h"


//
// cd ../cpp/build
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 ../tests/ipc_test.cu -L. -lNVStrings -lNVCategory -o ipc_test --linker-options -rpath,.:
//

int main( int argc, const char** argv )
{
    if( argc < 2 )
    {
        printf("require parameter: 'server' or values for pointers\n");
        return 0;
    }

    std::string mode = argv[1];
    NVStrings* strs = 0;
    if( mode.compare("client")==0 )
    {
        nvstrings_ipc_transfer ipc;
        FILE* fh = fopen("ipctx.bin","rb");
        fread(&ipc,1,sizeof(ipc),fh);
        fclose(fh);
        printf("%p %ld %ld\n", ipc.base_address, ipc.count, ipc.size);
        strs = NVStrings::create_from_ipc(ipc);
        strs->print();
        printf("%u strings in %ld bytes\n", strs->size(), strs->memsize() );
    }
    else
    {
        const char* hstrs[] = { "John Smith", "Joe Blow", "Jane Smith" };
        strs = NVStrings::create_from_array(hstrs,3);
        nvstrings_ipc_transfer ipc;
        strs->create_ipc_transfer(ipc);
        printf("%p %ld %ld\n", ipc.base_address, ipc.count, ipc.size);
        strs->print();
        printf("%u strings in %ld bytes\n", strs->size(), strs->memsize() );
        FILE* fh = fopen("ipctx.bin","wb");
        fwrite((void*)&ipc,1,sizeof(ipc),fh);
        fclose(fh);
        printf("Server ready. Press enter to terminate.\n");
        std::cin.ignore();
        // just checking
        strs->print();
    }

    NVStrings::destroy(strs);
    return 0;
}
