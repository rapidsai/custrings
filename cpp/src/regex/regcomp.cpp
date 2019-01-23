
#include <string.h>
#include "regcomp.h"

static Reclass ccls_w = { 'a', 'z','A','Z','0','9','_','_' };
static Reclass ccls_W = { '\n','\n','a', 'z', 'A', 'Z', '0', '9', '_', '_' };
static Reclass ccls_s = { '\t', '\t', '\n', '\n', '\r', '\r', '\f', '\f', '\v', '\v', ' ', ' ' };
// ccls_S is the same as ccls_s
static Reclass ccls_d = { '0', '9' };
static Reclass ccls_D = { '\n', '\n', '0', '9' };
static Reclass ccls_W_Neg = { 0, 0x9, 0xB, 0x2F, 0x3A, 0x40, 0x5B, 0x5E, 0x60, 0x60, 0x7B, 0xFFFFFFFF };
static Reclass ccls_S_Neg = { 0, 0x8, 0xE, 0x1F, 0x21, 0xFFFFFFFF };
static Reclass ccls_D_Neg = { 0, 0x9, 0xB, 0x2F, 0x3A, 0xFFFFFFFF };

int Reprog::add_inst(int t)
{
	Reinst inst;
	inst.type = t;
	inst.u2.left_id = 0;
	inst.u1.right_id = 0;
	return add_inst(inst);
}

int Reprog::add_inst(Reinst inst)
{
	insts.push_back(inst);
	return (int)insts.size() - 1;
}

int Reprog::add_class(Reclass cls)
{
	classes.push_back(cls);
	return (int)classes.size()-1;
}

Reinst& Reprog::inst_at(int id)
{
	return insts[id];
}

Reclass& Reprog::class_at(int id)
{
	return classes[id];
}

void Reprog::set_start_inst(int id)
{
	startinst_id = id;
}

int Reprog::get_start_inst() const
{
	return startinst_id;
}

int Reprog::inst_count() const
{
	return (int)insts.size();
}

int Reprog::classes_count() const
{
	return (int)classes.size();
}

void Reprog::set_groups_count(int groups)
{
	num_capturing_groups = groups;
}

int Reprog::groups_count() const
{
	return num_capturing_groups;
}

const Reinst* Reprog::insts_data() const
{
	return insts.data();
}

class RegCompiler
{
	int id_ccls_w = -1, id_ccls_W = -1, id_ccls_s = -1, id_ccls_d = -1, id_ccls_D = -1;
	struct Node
	{
		int id_first;
		int id_last;
	};
	int cursubid;
	int pushsubid;
	std::vector<Node> andstack;
	std::vector<int> atorstack;
	std::vector<int> subidstack;
	const char32_t* exprp;
	bool lexdone;
	char32_t yy; /* last lex'd Char */
	int yyclass_id;	/* last lex'd class */
	bool lastwasand;
	int	nbra;

	inline void pushand(int f, int l)
	{
		andstack.push_back({ f, l });
	}

	inline Node popand(int op)
	{
		if( andstack.size() < 1 )
		{
			//missing operand for op
			int inst_id = m_prog.add_inst(NOP);
			pushand(inst_id, inst_id);
		}
		Node node = andstack[andstack.size() - 1];
		andstack.pop_back();
		return node;
	}

	inline void pushator(int t)
	{
		atorstack.push_back(t);
		subidstack.push_back(pushsubid);
	}

	inline int popator(int& subid)
	{
		int ret = atorstack[atorstack.size() - 1];
		subid = subidstack[subidstack.size() - 1];
		atorstack.pop_back();
		subidstack.pop_back();
		return ret;
	}

	bool nextc(char32_t& c) // return "quoted" == backslash-escape prefix
	{
		if(lexdone)
		{
			c = 0;
			return true;
		}
		c = *exprp++;
		if(c == '\\')
		{
			c = *exprp++;
			return true;
		}
		if(c == 0)
			lexdone = true;
		return false;
	}

	int	bldcclass()
	{
		int type = CCLASS;
		std::vector<char32_t> cls;
		/* look ahead for negation */
		/* SPECIAL CASE!!! negated classes don't match \n */
		char32_t c;
		int quoted = nextc(c);
		if(!quoted && c == '^')
		{
			type = NCCLASS;
			quoted = nextc(c);
			cls.push_back('\n');
			cls.push_back('\n');
		}
		/* parse class into a set of spans */
		while(true)
		{
			if(c == 0)
			{
				// malformed '[]'
				return 0;
			}
			if(quoted)
			{
				switch(c)
				{
				case 'n':
					c = '\n';
					break;
				case 'r':
					c = '\r';
					break;
				case 't':
					c = '\t';
					break;
				case 'a':
					c = 0x07;
					break;
				case 'b':
					c = 0x08;
					break;
				case 'f':
					c = 0x0C;
					break;
				case 'w':
					for (int i = 0; i < ccls_w.size(); i++)
						cls.push_back(ccls_w[i]);
					quoted = nextc(c);
					continue;
				case 's':
					for (int i = 0; i < ccls_s.size(); i++)
						cls.push_back(ccls_s[i]);
					quoted = nextc(c);
					continue;
				case 'd':
					for (int i = 0; i < ccls_d.size(); i++)
						cls.push_back(ccls_d[i]);
					quoted = nextc(c);
					continue;
				case 'W':
					for (int i = 0; i < ccls_W_Neg.size(); i++)
						cls.push_back(ccls_W_Neg[i]);
					quoted = nextc(c);
					continue;
				case 'S':
					for (int i = 0; i < ccls_S_Neg.size(); i++)
						cls.push_back(ccls_S_Neg[i]);
					quoted = nextc(c);
					continue;
				case 'D':
					for (int i = 0; i < ccls_D_Neg.size(); i++)
						cls.push_back(ccls_D_Neg[i]);
					quoted = nextc(c);
					continue;
				}
			}
			if(!quoted && c == ']')
				break;
			if(!quoted && c == '-')
			{
				if (cls.size() < 1)
				{
					// malformed '[]'
					return 0;
				}
				quoted = nextc(c);
				if ((!quoted && c == ']') || c == 0)
				{
					// malformed '[]'
					return 0;
				}
				cls[cls.size() - 1] = c;
			}
			else
			{
				cls.push_back(c);
				cls.push_back(c);
			}
			quoted = nextc(c);
		}
		
		/* sort on span start */
		for (int p = 0; p < cls.size(); p += 2)
			for (int np = p + 2; np < cls.size(); np+=2)
				if (cls[np] < cls[p])
				{
					c = cls[np];
					cls[np] = cls[p];
					cls[p] = c;
					c = cls[np+1];
					cls[np+1] = cls[p+1];
					cls[p+1] = c;
					
				}
		/* merge spans */
		Reclass yycls;
		int p = 0;
		if( cls.size()>=2 )
		{
			int np = 0;
			yycls += cls[p++];
			yycls += cls[p++];
			for (; p < cls.size(); p += 2)
			{
				/* overlapping or adjacent ranges? */
				if (cls[p] <= yycls[np + 1] + 1)
				{
					if (cls[p + 1] >= yycls[np + 1])
						yycls.replace(np + 1, 1, 1, cls[p + 1]); /* coalesce */
				}
				else
				{
					np += 2;
					yycls += cls[p];
					yycls += cls[p+1];
				}
			}
		}
		yyclass_id = m_prog.add_class(yycls);
		return type;
	}

	int lex(int dot_type)
	{
		int quoted;
		quoted = nextc(yy);
		if(quoted)
		{
			if (yy == 0)
				return END;
			// treating all quoted numbers as Octal, since we are not supporting backreferences
			if (yy >= '0' && yy <= '7')
			{
				yy = yy - '0';
				char32_t c = *exprp++;
				while( c >= '0' && c <= '7' )
				{
					yy = (yy << 3) | (c - '0');
					c = *exprp++;
				}
				return CHAR;
			}
			else
			{
				switch (yy)
				{
				case 't':
					yy = '\t';
					break;
				case 'n':
					yy = '\n';
					break;
				case 'r':
					yy = '\r';
					break;
				case 'a':
					yy = 0x07;
					break;
				case 'f':
					yy = 0x0C;
					break;
				case '0':
					yy = 0;
					break;
				case 'x':
				{
					char32_t a = *exprp++;
					char32_t b = *exprp++;
					yy = 0;
					if (a >= '0' && a <= '9') yy += (a - '0') << 4;
					else if (a > 'a' && a <= 'f') yy += (a - 'a' + 10) << 4;
					else if (a > 'A' && a <= 'F') yy += (a - 'A' + 10) << 4;
					if (b >= '0' && b <= '9') yy += b - '0';
					else if (b > 'a' && b <= 'f') yy += b - 'a' + 10;
					else if (b > 'A' && b <= 'F') yy += b - 'A' + 10;
					break;
				}
				// TODO : 'b' & 'B'
				case 'w':
				{
					if (id_ccls_w < 0)
					{
						yyclass_id = m_prog.add_class(ccls_w);//newclass();
						//m_prog.classes[yyclass_id] = ccls_w;
						id_ccls_w = yyclass_id;
					}
					else yyclass_id = id_ccls_w;
					return CCLASS;
				}
				case 'W':
				{
					if (id_ccls_W < 0)
					{
						yyclass_id = m_prog.add_class(ccls_W); //newclass();
						//m_prog.classes[yyclass_id] = ccls_W;
						id_ccls_W = yyclass_id;
					}
					else yyclass_id = id_ccls_W;
					return NCCLASS;
				}
				case 's':
				{
					if (id_ccls_s < 0)
					{
						yyclass_id = m_prog.add_class(ccls_s);// newclass();
						//m_prog.classes[yyclass_id] = ccls_s;
						id_ccls_s = yyclass_id;
					}
					else yyclass_id = id_ccls_s;
					return CCLASS;
				}
				case 'S':
				{
					if (id_ccls_s < 0)
					{
						yyclass_id = m_prog.add_class(ccls_s);// newclass();
						//m_prog.classes[yyclass_id] = ccls_s;
						id_ccls_s = yyclass_id;
					}
					else yyclass_id = id_ccls_s;
					return NCCLASS;
				}
				case 'd':
				{
					if (id_ccls_d < 0)
					{
						yyclass_id = m_prog.add_class(ccls_d);//newclass();
						//m_prog.classes[yyclass_id] = ccls_d;
						id_ccls_d = yyclass_id;
					}
					else yyclass_id = id_ccls_d;
					return CCLASS;
				}
				case 'D':
				{
					if (id_ccls_D < 0)
					{
						yyclass_id = m_prog.add_class(ccls_D);// newclass();
						//m_prog.classes[yyclass_id] = ccls_D;
						id_ccls_D = yyclass_id;
					}
					else yyclass_id = id_ccls_D;
					return NCCLASS;
				}
				case 'b':
					return BOW;
				case 'B':
					return NBOW;
				case 'A':
					return BOL;
				case 'Z':
					return EOL;
				}
				return CHAR;
			}
		}
		switch(yy)
		{
		case 0:
			return END;
		case '*':
			if (*exprp == '?')
			{
				exprp++;
				return STAR_LAZY;
			}
			return STAR;
		case '?':
			if (*exprp == '?')
			{
				exprp++;
				return QUEST_LAZY;
			}
			return QUEST;
		case '+':
			if (*exprp == '?')
			{
				exprp++;
				return PLUS_LAZY;
			}
			return PLUS;
		case '|':
			return OR;
		case '.':
			return dot_type;
		case '(':
			if (*exprp == '?' && *(exprp + 1) == ':')  // non-capturing group
			{
				exprp += 2;
				pushsubid = 0;
			}
			else
			{
				++cursubid;
				pushsubid = cursubid;
			}
			return LBRA;
		case ')':
			return RBRA;
		case '^':
			return BOL;
		case '$':
			return EOL;
		case '[':
			return bldcclass();
		}
		return CHAR;
	}

	void evaluntil(int pri)
	{
		Node op1, op2;
		int id_inst1, id_inst2;
		while( pri == RBRA || atorstack[atorstack.size() - 1] >= pri )
		{
			int subid;
			switch(popator(subid))
			{
			default:
				// unknown operator in evaluntil
				break;
			case LBRA:		/* must have been RBRA */
				op1 = popand('(');
				id_inst2 = m_prog.add_inst(RBRA);
				m_prog.inst_at(id_inst2).u1.subid = subid;//subidstack[subidstack.size()-1];
				m_prog.inst_at(op1.id_last).u2.next_id = id_inst2;
				id_inst1 = m_prog.add_inst(LBRA);
				m_prog.inst_at(id_inst1).u1.subid = subid;//subidstack[subidstack.size() - 1];
				m_prog.inst_at(id_inst1).u2.next_id = op1.id_first;
				pushand(id_inst1, id_inst2);
				return;
			case OR:
				op2 = popand('|');
				op1 = popand('|');
				id_inst2 = m_prog.add_inst(NOP);
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst2;
				m_prog.inst_at(op1.id_last).u2.next_id = id_inst2;
				id_inst1 = m_prog.add_inst(OR);
				m_prog.inst_at(id_inst1).u1.right_id = op1.id_first;
				m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
				pushand(id_inst1, id_inst2);
				break;
			case CAT:
				op2 = popand(0);
				op1 = popand(0);
				m_prog.inst_at(op1.id_last).u2.next_id = op2.id_first;
				pushand(op1.id_first, op2.id_last);
				break;
			case STAR:
				op2 = popand('*');
				id_inst1 = m_prog.add_inst(OR);
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
				m_prog.inst_at(id_inst1).u1.right_id = op2.id_first;
				pushand(id_inst1, id_inst1);
				break;
			case STAR_LAZY:
				op2 = popand('*');
				id_inst1 = m_prog.add_inst(OR);
				id_inst2 = m_prog.add_inst(NOP);
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
				m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
				m_prog.inst_at(id_inst1).u1.right_id = id_inst2;
				pushand(id_inst1, id_inst2);
				break;
			case PLUS:
				op2 = popand('+');
				id_inst1 = m_prog.add_inst(OR);
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
				m_prog.inst_at(id_inst1).u1.right_id = op2.id_first;
				pushand(op2.id_first, id_inst1);
				break;
			case PLUS_LAZY:
				op2 = popand('+');
				id_inst1 = m_prog.add_inst(OR);
				id_inst2 = m_prog.add_inst(NOP);
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
				m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
				m_prog.inst_at(id_inst1).u1.right_id = id_inst2;
				pushand(op2.id_first, id_inst2);
				break;
			case QUEST:
				op2 = popand('?');
				id_inst1 = m_prog.add_inst(OR);
				id_inst2 = m_prog.add_inst(NOP);
				m_prog.inst_at(id_inst1).u2.left_id = id_inst2;
				m_prog.inst_at(id_inst1).u1.right_id = op2.id_first;
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst2;
				pushand(id_inst1, id_inst2);
				break;
			case QUEST_LAZY:
				op2 = popand('?');
				id_inst1 = m_prog.add_inst(OR);
				id_inst2 = m_prog.add_inst(NOP);
				m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
				m_prog.inst_at(id_inst1).u1.right_id = id_inst2; 
				m_prog.inst_at(op2.id_last).u2.next_id = id_inst2;
				pushand(id_inst1, id_inst2);
				break;
			}
		}
	}

	void Operator(int t)
	{
		if (t == RBRA && --nbra < 0)
			//unmatched right paren
			return;
		if (t == LBRA)
		{
			nbra++;
			if (lastwasand)
				Operator(CAT);
		}
		else
			evaluntil(t);
		if (t != RBRA)
			pushator(t);
		lastwasand = (
			t == STAR || t == QUEST || t == PLUS || 
			t == STAR_LAZY || t == QUEST_LAZY || t == PLUS_LAZY ||
			t == RBRA);
	}

	void Operand(int t)
	{
		if (lastwasand)
			Operator(CAT);	/* catenate is implicit */
		int inst_id = m_prog.add_inst(t);
		if (t == CCLASS || t == NCCLASS)
			m_prog.inst_at(inst_id).u1.cls_id = yyclass_id;
		else if (t == CHAR || t==BOL || t==EOL)
			m_prog.inst_at(inst_id).u1.c = yy;
		pushand(inst_id, inst_id);
		lastwasand = true;
	}

public:
	Reprog m_prog;
	RegCompiler(const char32_t* pattern, int dot_type)
	{
		cursubid = 0;
		pushsubid = 0;
		exprp = pattern;
		lexdone = false;
		/* Start with a low priority operator to prime parser */
		pushator(START - 1);
					
		lastwasand = false;
		nbra = 0;
		int token;
		while ((token = lex(dot_type)) != END)
		{
			if ((token & 0300) == OPERATOR)
				Operator(token);
			else
				Operand(token);
		}
		/* Close with a low priority operator */
		evaluntil(START);
		/* Force END */
		Operand(END);
		evaluntil(START);
		if (nbra)
			; // "unmatched left paren";
		/* points to first and only operand */
		m_prog.set_start_inst(andstack[andstack.size() - 1].id_first);
		m_prog.optimize();
		m_prog.set_groups_count(cursubid);
	}
};

Reprog* Reprog::create_from(const char32_t* pattern)
{
	RegCompiler compiler(pattern, ANY);
	Reprog* rtn = new Reprog(compiler.m_prog);
	//rtn->print();
	return rtn;
}

//Reprog regcompnl(const char32_t* pattern)
//{
//	RegCompiler compiler(pattern, ANYNL);
//	return compiler.m_prog;
//}

void Reprog::optimize()
{
	// Treat non-capturing LBRAs/RBRAs as NOOP
	for (int i = 0; i < (int)insts.size(); i++)
	{
		if (insts[i].type == LBRA || insts[i].type == RBRA)
		{
			if (insts[i].u1.subid < 1)
			{
				insts[i].type = NOP;
			}
		}
	}

	// get rid of NOP chains 
	for (int i=0; i < inst_count(); i++)
	{
		if( insts[i].type != NOP )
		{
			{
				int target_id = insts[i].u2.next_id;
				while(insts[target_id].type == NOP)
					target_id = insts[target_id].u2.next_id;
				insts[i].u2.next_id = target_id;
			}
			if( insts[i].type == OR )
			{
				int target_id = insts[i].u1.right_id;
				while(insts[target_id].type == NOP)
					target_id = insts[target_id].u2.next_id;
				insts[i].u1.right_id = target_id;
			}
		}
	}
	// skip NOPs from the beginning
	{
		int target_id = startinst_id;
		while( insts[target_id].type == NOP)
			target_id = insts[target_id].u2.next_id;
		startinst_id = target_id;
	}
	// actually remove the no-ops
	std::vector<int> id_map(inst_count());
	int j = 0; // compact the ops (non no-ops)
	for( int i = 0; i < inst_count(); i++)
	{
		id_map[i] = j;
		if( insts[i].type != NOP )
		{
			insts[j] = insts[i];
			j++;
		}
	}
	insts.resize(j);
	// fix up the ORs
	for( int i=0; i < inst_count(); i++)
	{
		{
			int target_id = insts[i].u2.next_id;
			insts[i].u2.next_id = id_map[target_id];
		}
		if( insts[i].type == OR )
		{
			int target_id = insts[i].u1.right_id;
			insts[i].u1.right_id = id_map[target_id];
		}
	}
	// set the new start id
	startinst_id = id_map[startinst_id];
}

void Reprog::print()
{
	printf("Instructions:\n");
	for(int i = 0; i < insts.size(); i++)
	{
		const Reinst& inst = insts[i];
		printf("%d :", i);
		switch (inst.type)
		{
		default:
			printf("Unknown instruction: %d, nextid= %d", inst.type, inst.u2.next_id);
			break;
		case CHAR:
			if( inst.u1.c <=32 || inst.u1.c >=127 )
				printf("CHAR, c = '0x%02x', nextid= %d", (unsigned)inst.u1.c, inst.u2.next_id);
			else
				printf("CHAR, c = '%c', nextid= %d", inst.u1.c, inst.u2.next_id);
			break;
		case RBRA:
			printf("RBRA, subid= %d, nextid= %d", inst.u1.subid, inst.u2.next_id);
			break;
		case LBRA:
			printf("LBRA, subid= %d, nextid= %d", inst.u1.subid, inst.u2.next_id);
			break;
		case OR:
			printf("OR, rightid=%d, leftid=%d, nextid=%d", inst.u1.right_id, inst.u2.left_id, inst.u2.next_id);
			break;
		case STAR:
			printf("STAR, nextid= %d", inst.u2.next_id);
			break;
		case PLUS:
			printf("PLUS, nextid= %d", inst.u2.next_id);
			break;
		case QUEST:
			printf("QUEST, nextid= %d", inst.u2.next_id);
			break;
		case ANY:
			printf("ANY, nextid= %d", inst.u2.next_id);
			break;
		case ANYNL:
			printf("ANYNL, nextid= %d", inst.u2.next_id);
			break;
		case NOP:
			printf("NOP, nextid= %d", inst.u2.next_id);
			break;
		case BOL:
			printf("BOL, c = '%c', nextid= %d", inst.u1.c, inst.u2.next_id);
			break;
		case EOL:
			printf("EOL, c = '%c', nextid= %d", inst.u1.c, inst.u2.next_id);
			break;
		case CCLASS:
			printf("CCLASS, cls_id=%d , nextid= %d", inst.u1.cls_id, inst.u2.next_id);
			break;
		case NCCLASS:
			printf("NCCLASS, cls_id=%d , nextid= %d", inst.u1.cls_id, inst.u2.next_id);
			break;
		case BOW:
			printf("BOW, nextid= %d", inst.u2.next_id);
			break;
		case NBOW:
			printf("NBOW, nextid= %d", inst.u2.next_id);
			break;
		case END:
			printf("END");
			break;
		}
		printf("\n");
	}
	printf("startinst_id=%d\n", startinst_id);
	int count = (int)classes.size();
	printf("\nClasses %d\n",count);
	for( int i = 0; i < count; i++ )
	{
		const Reclass& cls = classes[i];
		int len = (int)cls.size();
		printf("%2d: ", i);
		for( int j=0; j < len; j += 2 )
		{
			char32_t c1 = cls[j];
			char32_t c2 = cls[j+1];
			if( c1 <= 32 || c1 >= 127 || c2 <= 32 || c2 >= 127 )
				printf("0x%02x-0x%02x",(unsigned)c1,(unsigned)c2);
			else
				printf("%c-%c",(char)c1,(char)c2);
			if( (j+2) < len )
				printf(", ");
		}
		printf("\n");
	}
	if( num_capturing_groups )
		printf("Number of capturing groups: %d\n", num_capturing_groups);
}
