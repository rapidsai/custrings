//
#include <memory.h>
#include <cuda_runtime.h>
#include "regex.cuh"
#include "regcomp.h"

dreprog* dreprog::create_from(const char32_t* pattern, unsigned char* uflags, unsigned int strscount )
{
    // compile pattern
    Reprog* prog = Reprog::create_from(pattern);
    // compute size to hold prog
    int insts_count = (int)prog->inst_count();
    int classes_count = (int)prog->classes_count();
    int insts_size = insts_count * sizeof(Reinst);
    int classes_size = classes_count * sizeof(int); // offsets
    for( int idx=0; idx < classes_count; ++idx )
        classes_size += (int)((prog->class_at(idx).chrs.size())*sizeof(char32_t)) + (int)sizeof(int);
    // allocate memory to store prog
    size_t memsize = sizeof(dreprog) + insts_size + classes_size;
    u_char* buffer = (u_char*)malloc(memsize);
    dreprog* rtn = (dreprog*)buffer;
    buffer += sizeof(dreprog);
    Reinst* insts = (Reinst*)buffer;
    memcpy( insts, prog->insts_data(), insts_size);
    buffer += insts_size;
    // classes are variable size so create offsets array
    int* offsets = (int*)buffer;
    buffer += classes_count * sizeof(int);
    char32_t* classes = (char32_t*)buffer;
    int offset = 0;
    for( int idx=0; idx < classes_count; ++idx )
    {
        Reclass& cls = prog->class_at(idx);
        memcpy( classes++, &(cls.builtins), sizeof(int) );
        int len = (int)cls.chrs.size();
        memcpy( classes, cls.chrs.c_str(), len*sizeof(char32_t) );
        offset += 1 + len;
        offsets[idx] = offset;
        classes += len;
    }
    // initialize the rest of the elements
    rtn->startinst_id = prog->get_start_inst();
    rtn->num_capturing_groups = prog->groups_count();
    rtn->insts_count = insts_count;
    rtn->classes_count = classes_count;
    rtn->unicode_flags = uflags;
    rtn->relists_mem = 0;
    // allocate memory for relist if necessary
    if( (insts_count > LISTSIZE) && strscount )
    {
        int rsz = Relist::size_for(insts_count);
        size_t rlmsz = rsz*2*strscount; // Reljunk has 2 Relist ptrs
        void* rmem = 0;
        RMM_ALLOC(&rmem,rlmsz,0);//cudaMalloc(&rmem,rlmsz);
        rtn->relists_mem = rmem;
    }

    // compiled prog copied into flat memory
    delete prog;

    // copy flat prog to device memory
    dreprog* d_rtn = 0;
    RMM_ALLOC(&d_rtn,memsize,0);//cudaMalloc(&d_rtn,memsize);
    cudaMemcpy(d_rtn,rtn,memsize,cudaMemcpyHostToDevice);
    free(rtn);
    return d_rtn;
}

void dreprog::destroy(dreprog* prog)
{
    prog->free_relists();
    RMM_FREE(prog,0);//cudaFree(prog);
}

void dreprog::free_relists()
{
    void* cptr = 0; // this magic works but only as member function
    cudaMemcpy(&cptr,&relists_mem,sizeof(void*),cudaMemcpyDeviceToHost);
    if( cptr )
        RMM_FREE(cptr,0);//cudaFree(cptr);
}

int dreprog::inst_counts()
{
    int count = 0;
    cudaMemcpy(&count,&insts_count,sizeof(int),cudaMemcpyDeviceToHost);
    return count;
}

int dreprog::group_counts()
{
    int count = 0;
    cudaMemcpy(&count,&num_capturing_groups,sizeof(int),cudaMemcpyDeviceToHost);
    return count;
}

