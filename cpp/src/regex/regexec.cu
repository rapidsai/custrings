
#include <memory.h>
#include <cuda_runtime.h>
#include "regex.cuh"
#include "regcomp.h"
#include "../custring_view.cuh"

dreprog::dreprog() {}
dreprog::~dreprog() {}
	
dreprog* dreprog::create_from(const char32_t* pattern)	
{
	// compile pattern
	Reprog* prog = Reprog::create_from(pattern);
	// compute size to hold prog
	int insts_count = (int)prog->inst_count();
	int classes_count = (int)prog->classes_count();
	int insts_size = insts_count * sizeof(Reinst);
	int classes_size = classes_count * sizeof(int); // offsets
	for( int idx=0; idx < classes_count; ++idx )
		classes_size += (int)(prog->class_at(idx).size())*sizeof(char32_t);
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
		int len = (int)cls.size();
		memcpy( classes, cls.c_str(), len*sizeof(char32_t) );
		offset += len;
		offsets[idx] = offset;
		classes += len;
	}
	// initialize the rest of the elements
	rtn->startinst_id = prog->get_start_inst();
	rtn->num_capturing_groups = prog->groups_count();
	rtn->insts_count = insts_count;
	rtn->classes_count = classes_count;
	// compiled prog copied into flat memory
	delete prog;

    // copy flat prog to device memory
	dreprog* d_rtn = 0;
	cudaMalloc(&d_rtn,memsize);
	cudaMemcpy(d_rtn,rtn,memsize,cudaMemcpyHostToDevice);
	free(rtn);
	return d_rtn;
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

__device__ char32_t* dreprog::get_class(int idx, int& len)
{
	if( idx < 0 || idx >= classes_count )
		return 0;
	u_char* buffer = (u_char*)this;
	buffer += sizeof(dreprog) + (insts_count * sizeof(Reinst));
	int* offsets = (int*)buffer;
	buffer += classes_count * sizeof(int);
	char32_t* classes = (char32_t*)buffer;
	int offset = offsets[idx];
	len = offset;
	if( idx==0 )
		return classes;
	offset = offsets[idx-1];
	len -= offset;
	classes += offset;
	return classes;
}

#define LISTBYTES 12
#define LISTSIZE (LISTBYTES<<3)
//
struct Relist
{
	u_char mask[LISTBYTES];
	u_char inst_ids[LISTSIZE];
	int2 ranges[LISTSIZE];
	short size;

	__host__ __device__ Relist()
	{
		reset();
	}

	__host__ __device__ inline void reset()
	{
		memset(mask, 0, LISTBYTES);
		size = 0;
	}

	__device__ inline bool activate(int i, int begin, int end)
	{
		if (i < LISTSIZE)
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

// execute compiled expression for each character in the provided string
__device__ int dreprog::regexec(custring_view* dstr, Reljunk &jnk, int& begin, int& end, int groupId)
{
	int match = 0;
	int checkstart = jnk.starttype;

	int txtlen = dstr->chars_count();
	
	int pos = begin;
	int eos = end;
	char32_t c = 0;

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
		}

		if (pos < eos && match == 0)
			jnk.list1->activate(startinst_id, pos, 0);

		c = (char32_t)(pos >= txtlen ? 0 : dstr->at(pos) );

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
						bool cur_alphaNumeric = isAlphaNumeric(c);
						bool last_alphaNumeric = ( (pos==0) ? false : isAlphaNumeric((char32_t)dstr->at(pos-1)) );
						if( cur_alphaNumeric != last_alphaNumeric )
						{
							id_activate = inst->u2.next_id;
							expanded = true;
						}
						break;
					}
					case NBOW:
					{
						bool cur_alphaNumeric = isAlphaNumeric(c);
						bool last_alphaNumeric = ( (pos == 0) ? false : isAlphaNumeric((char32_t)dstr->at(pos-1)) );
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
				int numCls = 0;
				char32_t* cls = get_class(inst->u1.cls_id,numCls);
				for( int i=0; i < numCls; i += 2 )
				{
					if( (c >= cls[i]) && (c <= cls[i+1]) )
					{
						id_activate = inst->u2.next_id;
						break;
					}
				}
				break;
			}
			case NCCLASS:
			{
				int numCls = 0;
				char32_t* cls = get_class(inst->u1.cls_id,numCls);
				int i=0;
				for( ; i < numCls; i += 2 )
					if( c >= cls[i] && c <= cls[i+1] )
						break;
				if( i == numCls )
					id_activate = inst->u2.next_id;
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
		swaplist(jnk.list1, jnk.list2);

		checkstart = jnk.list1->size > 0 ? 0 : 1;
	}
	while (c && (jnk.list1->size>0 || match == 0));	
	return match;

}

void dreprog::destroy(dreprog* prog)
{
	cudaFree(prog);
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
