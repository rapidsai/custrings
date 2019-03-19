/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <memory.h>
#include <cuda_runtime.h>
#include <rmm/rmm.h>
#include "regex.cuh"
#include "regcomp.h"
#include "../custring_view.cuh"

// from is_flags.h -- need to put these somewhere else
#define IS_SPACE(x) ((x & 16)>0)
#define IS_ALPHA(x) ((x & 8)>0)
#define IS_DIGIT(x) ((x & 4)>0)
#define IS_NUMERIC(x) ((x & 2)>0)
#define IS_DECIMAL(x) ((x & 1)>0)
#define IS_ALPHANUM(x) ((x & 15)>0)
#define IS_UPPER(x) ((x & 32)>0)
#define IS_LOWER(x) ((x & 64)>0)
// defined in util.cu
__host__ __device__ unsigned int u82u( unsigned int utf8 );

//
#define LISTBYTES 12
#define LISTSIZE (LISTBYTES<<3)
//
struct Relist
{
    short size, listsize;
    int pad; // keep data on 8-byte bounday
    int2* ranges;//[LISTSIZE];
    u_char* inst_ids;//[LISTSIZE];
    u_char* mask;//[LISTBYTES];
    u_char data[(9*LISTSIZE)+LISTBYTES]; // always last

    __host__ __device__ static int size_for(int insts)
    {
        int size = 0;
        size += sizeof(short);                  // size
        size += sizeof(short);                  // listsize
        size += sizeof(int);                    // pad
        size += sizeof(u_char*)*3;              // 3 pointers
        size += sizeof(int2)*insts;             // ranges bytes
        size += sizeof(u_char)*insts;           // inst_ids bytes
        size += sizeof(u_char)*((insts+7)/8);   // mask bytes
        size = ((size+7)/8)*8;   // align it too
        return size;
    }

    __host__ __device__ Relist()
    {
        //listsize = LISTSIZE;
        //reset();
        set_listsize(LISTSIZE);
    }

    __host__ __device__ inline void set_listsize(short ls)
    {
        listsize = ls;
        u_char* ptr = (u_char*)data;
        ranges = (int2*)ptr;
        ptr += listsize * sizeof(int2);
        inst_ids = ptr;
        ptr += listsize;
        mask = ptr;
        reset();
    }

    __host__ __device__ inline void reset()
    {
        //memset(mask, 0, LISTBYTES);
        memset(mask, 0, (listsize+7)/8);
        size = 0;
    }

    __device__ inline bool activate(int i, int begin, int end)
    {
        //if ( i >= listsize )
        //    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        {
            if (!readMask(i))
            {
                writeMask(true, i);
                inst_ids[size] = (u_char)i;

                int2 range;
                range.x = begin;
                range.y = end;
                ranges[size] = range;

                size++;
                return true;
            }
        }
        return false;
    }

    __device__ inline void writeMask(bool v, int pos)
    {
        u_char uc = 1 << (pos & 7);
        if (v)
            mask[pos >> 3] |= uc;
        else
            mask[pos >> 3] &= ~uc;
    }

    //if( tid > jnk.list1->minId && tid < jnk.list1->maxId && !readMask(jnk.list1->mask, tid) )
    __device__ inline bool readMask(int pos)
    {
        u_char uc = mask[pos >> 3];
        return (bool)((uc >> (pos & 7)) & 1);
    }

};

struct	Reljunk
{
    Relist *list1, *list2;
    int	starttype;
    char32_t startchar;
};

__device__ inline bool isAlphaNumeric(char32_t c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}

__device__ inline void swaplist(Relist*& l1, Relist*& l2)
{
    Relist* t = l1;
    l1 = l2;
    l2 = t;
}

__device__ dreclass::dreclass(unsigned char* flags)
                    : builtins(0), count(0), chrs(0), uflags(flags) {}

__device__ bool dreclass::is_match(char32_t ch)
{
    int i=0, len = count;
    for( ; i < len; i += 2 )
    {
        if( (ch >= chrs[i]) && (ch <= chrs[i+1]) )
            return true;
    }
    if( !builtins )
        return false;
    unsigned int uni = u82u(ch);
    if( uni > 0x00FFFF )
        return false;
    unsigned char fl = uflags[uni];
    if( (builtins & 1) && ((ch=='_') || IS_ALPHANUM(fl)) ) // \w
        return true;
    if( (builtins & 2) && IS_SPACE(fl) ) // \s
        return true;
    if( (builtins & 4) && IS_DIGIT(fl) ) // \d
        return true;
    if( (builtins & 8) && ((ch != '\n') && (ch != '_') && !IS_ALPHANUM(fl)) ) // \W
        return true;
    if( (builtins & 16) && !IS_SPACE(fl) )  // \S
        return true;
    if( (builtins & 32) && ((ch != '\n') && !IS_DIGIT(fl)) ) // \D
        return true;
    //
    return false;
}

dreprog::dreprog() {}
dreprog::~dreprog() {}

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

__host__ __device__ Reinst* dreprog::get_inst(int idx)
{
    if( idx < 0 || idx >= insts_count )
        return 0;
    u_char* buffer = (u_char*)this;
    Reinst* insts = (Reinst*)(buffer + sizeof(dreprog));
    return insts + idx;
}

//__device__ char32_t* dreprog::get_class(int idx, int& len)
//{
//	if( idx < 0 || idx >= classes_count )
//		return 0;
//	u_char* buffer = (u_char*)this;
//	buffer += sizeof(dreprog) + (insts_count * sizeof(Reinst));
//	int* offsets = (int*)buffer;
//	buffer += classes_count * sizeof(int);
//	char32_t* classes = (char32_t*)buffer;
//	int offset = offsets[idx];
//	len = offset;
//	if( idx==0 )
//		return classes;
//	offset = offsets[idx-1];
//	len -= offset;
//	classes += offset;
//	return classes;
//}

__device__ int dreprog::get_class(int idx, dreclass& cls)
{
    if( idx < 0 || idx >= classes_count )
        return 0;
    u_char* buffer = (u_char*)this;
    buffer += sizeof(dreprog) + (insts_count * sizeof(Reinst));
    int* offsets = (int*)buffer;
    buffer += classes_count * sizeof(int);
    char32_t* classes = (char32_t*)buffer;
    int offset = offsets[idx];
    int builtins, len = offset -1;
    if( idx > 0 )
    {
        offset = offsets[idx-1];
        len -= offset;
        classes += offset;
    }
    memcpy( &builtins, classes++, sizeof(int) );
    cls.builtins = builtins;
    cls.count = len;
    cls.chrs = classes;
    return len;
}


// execute compiled expression for each character in the provided string
__device__ int dreprog::regexec(custring_view* dstr, Reljunk &jnk, int& begin, int& end, int groupId)
{
    int match = 0;
    int checkstart = jnk.starttype;

    int txtlen = dstr->chars_count();

    int pos = begin;
    int eos = end;
    char32_t c = 0; // lc = 0;
    custring_view::iterator itr = custring_view::iterator(*dstr,pos);

    jnk.list1->reset();
    do
    {
        /* fast check for first char */
        if (checkstart)
        {
            switch (jnk.starttype)
            {
                case CHAR:
                {
                    int fidx = dstr->find((Char)jnk.startchar,pos);
                    if( fidx < 0 )
                        return match;
                    pos = fidx;
                    break;
                }
                case BOL:
                {
                    if( pos==0 )
                        break;
                    if( jnk.startchar != '^' )
                        return match;
                    --pos;
                    int fidx = dstr->find((Char)'\n',pos);
                    if( fidx < 0 )
                        return match;  // update begin/end values?
                    pos = fidx + 1;
                    break;
                }
            }
            //if( pos > 0 )
            //{
            //    itr = custring_view::iterator(*dstr,pos-1);
            //    lc = *itr;
            //    ++itr;
            //}
            //else
            //{
            //    itr = dstr->begin();
            //    lc = 0;
            //}
            itr = custring_view::iterator(*dstr,pos);
        }

        if (pos < eos && match == 0)
            jnk.list1->activate(startinst_id, pos, 0);

        //c = (char32_t)(pos >= txtlen ? 0 : dstr->at(pos) );
        c = (char32_t)(pos >= txtlen ? 0 : *itr); // iterator is many times faster than at()

        // expand LBRA, RBRA, BOL, EOL, BOW, NBOW, and OR
        bool expanded;
        do
        {
            jnk.list2->reset();
            expanded = false;

            for (short i = 0; i < jnk.list1->size; i++)
            {
                int inst_id = (int)jnk.list1->inst_ids[i];
                int2 &range = jnk.list1->ranges[i];
                const Reinst* inst = get_inst(inst_id);
                int id_activate = -1;

                switch (inst->type)
                {
                    case CHAR:
                    case ANY:
                    case ANYNL:
                    case CCLASS:
                    case NCCLASS:
                    case END:
                        id_activate = inst_id;
                        break;
                    case LBRA:
                        if (inst->u1.subid == groupId)
                            range.x = pos;
                        id_activate = inst->u2.next_id;
                        expanded = true;
                        break;
                    case RBRA:
                        if (inst->u1.subid == groupId)
                            range.y = pos;
                        id_activate = inst->u2.next_id;
                        expanded = true;
                        break;
                    case BOL:
                        if( (pos==0) || ((inst->u1.c=='^') && (dstr->at(pos-1)==(Char)'\n')) )
                        {
                            id_activate = inst->u2.next_id;
                            expanded = true;
                        }
                        break;
                    case EOL:
                        if( (c==0) || (inst->u1.c == '$' && c == '\n'))
                        {
                            id_activate = inst->u2.next_id;
                            expanded = true;
                        }
                        break;
                    case BOW:
                    {
                        unsigned int uni = u82u(c);
                        char32_t lc = (char32_t)(pos ? dstr->at(pos-1) : 0);
                        unsigned int luni = u82u(lc);
                        //bool cur_alphaNumeric = isAlphaNumeric(c);
                        //bool last_alphaNumeric = ( (pos==0) ? false : isAlphaNumeric((char32_t)dstr->at(pos-1)) );
                        bool cur_alphaNumeric = (uni < 0x010000) && IS_ALPHANUM(unicode_flags[uni]);
                        bool last_alphaNumeric = (luni < 0x010000) && IS_ALPHANUM(unicode_flags[luni]);
                        if( cur_alphaNumeric != last_alphaNumeric )
                        {
                            id_activate = inst->u2.next_id;
                            expanded = true;
                        }
                        break;
                    }
                    case NBOW:
                    {
                        unsigned int uni = u82u(c);
                        char32_t lc = (char32_t)(pos ? dstr->at(pos-1) : 0);
                        unsigned int luni = u82u(lc);
                        //bool cur_alphaNumeric = isAlphaNumeric(c);
                        //bool last_alphaNumeric = ( (pos==0) ? false : isAlphaNumeric((char32_t)dstr->at(pos-1)) );
                        bool cur_alphaNumeric = (uni < 0x010000) && IS_ALPHANUM(unicode_flags[uni]);
                        bool last_alphaNumeric = (luni < 0x010000) && IS_ALPHANUM(unicode_flags[luni]);
                        if( cur_alphaNumeric == last_alphaNumeric )
                        {
                            id_activate = inst->u2.next_id;
                            expanded = true;
                        }
                        break;
                    }
                    case OR:
                        jnk.list2->activate(inst->u1.right_id, range.x, range.y);
                        id_activate = inst->u2.left_id;
                        expanded = true;
                        break;
                }
                if (id_activate >= 0)
                    jnk.list2->activate(id_activate, range.x, range.y);

            }
            swaplist(jnk.list1, jnk.list2);

        } while (expanded);

        // execute, only CHAR, ANY, ANYNL, CCLASS, NCCLASS, END left now
        jnk.list2->reset();
        for (short i = 0; i < jnk.list1->size; i++)
        {
            int inst_id = (int)jnk.list1->inst_ids[i];
            int2 &range = jnk.list1->ranges[i];
            const Reinst* inst = get_inst(inst_id);
            int id_activate = -1;

            switch (inst->type)
            {
            case CHAR:
                if (inst->u1.c == c)
                    id_activate = inst->u2.next_id;
                break;
            case ANY:
                if (c != '\n')
                    id_activate = inst->u2.next_id;
                break;
            case ANYNL:
                id_activate = inst->u2.next_id;
                break;
            case CCLASS:
            {
                dreclass cls(unicode_flags);
                get_class(inst->u1.cls_id,cls);
                if( cls.is_match(c) )
                    id_activate = inst->u2.next_id;

                //int numCls = 0;
                //char32_t* cls = get_class(inst->u1.cls_id,numCls);
                //for( int i=0; i < numCls; i += 2 )
                //{
                //	if( (c >= cls[i]) && (c <= cls[i+1]) )
                //	{
                //		id_activate = inst->u2.next_id;
                //		break;
                //	}
                //}
                break;
            }
            case NCCLASS:
            {
                dreclass cls(unicode_flags);
                get_class(inst->u1.cls_id,cls);
                if( !cls.is_match(c) )
                    id_activate = inst->u2.next_id;

                //int numCls = 0;
                //char32_t* cls = get_class(inst->u1.cls_id,numCls);
                //int i=0;
                //for( ; i < numCls; i += 2 )
                //	if( c >= cls[i] && c <= cls[i+1] )
                //		break;
                //if( i == numCls )
                //	id_activate = inst->u2.next_id;
                break;
            }

            case END:
                match = 1;
                begin = range.x;
                end = groupId==0? pos : range.y;
                goto BreakFor;
            }
            if (id_activate >= 0)
                jnk.list2->activate(id_activate, range.x, range.y);

        }

    BreakFor:
        ++pos;
        ++itr;
        swaplist(jnk.list1, jnk.list2);
        checkstart = jnk.list1->size > 0 ? 0 : 1;
    }
    while (c && (jnk.list1->size>0 || match == 0));
    return match;
}

//
__device__ int dreprog::contains( custring_view* dstr )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( type == CHAR || type == BOL )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }
    Relist relist[2];
    jnk.list1 = relist;
    jnk.list2 = relist + 1;

    int begin=0, end=dstr->chars_count();
    int rtn = regexec(dstr,jnk,begin,end);
    return rtn;
}

__device__ int dreprog::contains( unsigned int idx, custring_view* dstr )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( type == CHAR || type == BOL )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }

    int begin=0, end=dstr->chars_count();
    if( relists_mem==0 )
    {
        Relist relist[2];
        jnk.list1 = relist;
        jnk.list2 = relist + 1;
        return regexec(dstr,jnk,begin,end);
    }
    int relsz = Relist::size_for(insts_count);
    char* drel = (char*)relists_mem; // beginning of Relist buffer
    drel += (idx * relsz * 2);       // two Relist ptrs in Reljunk
    jnk.list1 = (Relist*)drel;           // first one
    jnk.list2 = (Relist*)(drel + relsz); // second one
    jnk.list1->set_listsize((short)insts_count); // essentially this is
    jnk.list2->set_listsize((short)insts_count); // substitute ctor call
    return regexec(dstr,jnk,begin,end);
}

__device__ int dreprog::match( custring_view* dstr )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( type == CHAR || type == BOL )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }
    Relist relist[2];
    jnk.list1 = relist;
    jnk.list2 = relist + 1;

    int begin=0, end=1;
    int rtn = regexec(dstr,jnk,begin,end);
    return rtn;
}

__device__ int dreprog::match( unsigned int idx, custring_view* dstr )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( type == CHAR || type == BOL )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }

    int begin=0, end=1;
    if( relists_mem==0 )
    {
        Relist relist[2];
        jnk.list1 = relist;
        jnk.list2 = relist + 1;
        return regexec(dstr,jnk,begin,end);
    }
    int relsz = Relist::size_for(insts_count);
    char* drel = (char*)relists_mem; // beginning of Relist buffer
    drel += (idx * relsz * 2);       // two Relist ptrs in Reljunk
    jnk.list1 = (Relist*)drel;           // first one
    jnk.list2 = (Relist*)(drel + relsz); // second one
    jnk.list1->set_listsize((short)insts_count); // essentially this is
    jnk.list2->set_listsize((short)insts_count); // substitute ctor call
    return regexec(dstr,jnk,begin,end);
}

__device__ int dreprog::find( custring_view* dstr, int& begin, int& end )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( (type == CHAR) || (type == BOL) )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }
    Relist relist[2];
    jnk.list1 = relist;
    jnk.list2 = relist + 1;

    int rtn = regexec(dstr,jnk,begin,end);
    if( rtn <=0 )
        begin = end = -1;
    return rtn;
}

__device__ int dreprog::find( unsigned int idx, custring_view* dstr, int& begin, int& end )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( (type == CHAR) || (type == BOL) )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }
    int rtn = 0;
    if( relists_mem==0 )
    {
        Relist relist[2];
        jnk.list1 = relist;
        jnk.list2 = relist + 1;
        rtn = regexec(dstr,jnk,begin,end);
    }
    else
    {
        int relsz = Relist::size_for(insts_count);
        char* drel = (char*)relists_mem; // beginning of Relist buffer
        drel += (idx * relsz * 2);       // two Relist ptrs in Reljunk
        jnk.list1 = (Relist*)drel;           // first one
        jnk.list2 = (Relist*)(drel + relsz); // second one
        jnk.list1->set_listsize((short)insts_count); // essentially this is
        jnk.list2->set_listsize((short)insts_count); // substitute ctor call
        rtn = regexec(dstr,jnk,begin,end);
    }
    if( rtn <=0 )
        begin = end = -1;
    return rtn;
}

//
__device__ int dreprog::extract( custring_view* str, int& begin, int& end, int col )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( (type == CHAR) || (type == BOL) )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }
    Relist relist[2];

    jnk.list1 = relist;
    jnk.list2 = relist + 1;

    end = begin + 1;
    int rtn = regexec(str,jnk,begin,end, col +1);
    return rtn;
}

__device__ int dreprog::extract( unsigned int idx, custring_view* dstr, int& begin, int& end, int col )
{
    Reljunk jnk;
    jnk.starttype = 0;
    jnk.startchar = 0;
    int type = get_inst(startinst_id)->type;
    if( (type == CHAR) || (type == BOL) )
    {
        jnk.starttype = type;
        jnk.startchar = get_inst(startinst_id)->u1.c;
    }
    end = begin + 1;
    if( relists_mem==0 )
    {
        Relist relist[2];
        jnk.list1 = relist;
        jnk.list2 = relist + 1;
        return regexec(dstr,jnk,begin,end,col+1);
    }
    int relsz = Relist::size_for(insts_count);
    char* drel = (char*)relists_mem; // beginning of Relist buffer
    drel += (idx * relsz * 2);       // two Relist ptrs in Reljunk
    jnk.list1 = (Relist*)drel;           // first one
    jnk.list2 = (Relist*)(drel + relsz); // second one
    jnk.list1->set_listsize((short)insts_count); // essentially this is
    jnk.list2->set_listsize((short)insts_count); // substitute ctor call
    return regexec(dstr,jnk,begin,end,col+1);
}
