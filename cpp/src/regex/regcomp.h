
// internal declarations for regex compilier and executor
#include <string>
#include <vector>

/*
* Actions and Tokens (Reinst types)
*	02xx are operators, value == precedence
*	03xx are tokens, i.e. operands for operators
*/
#define CHAR        0177
#define	OPERATOR    0200  /* Bitmask of all operators */
#define	START       0200  /* Start, used for marker on stack */
#define	RBRA        0201  /* Right bracket, ) */
#define	LBRA        0202  /* Left bracket, ( */
#define	OR          0203  /* Alternation, | */
#define	CAT         0204  /* Concatentation, implicit operator */
#define	STAR        0205  /* Closure, * */
#define	STAR_LAZY   0206
#define	PLUS        0207  /* a+ == aa* */
#define	PLUS_LAZY   0210
#define	QUEST       0211  /* a? == a|nothing, i.e. 0 or 1 a's */
#define	QUEST_LAZY  0212
#define	ANY         0300  /* Any character except newline, . */
#define	ANYNL       0301  /* Any character including newline, . */
#define	NOP         0302  /* No operation, internal use only */
#define	BOL         0303  /* Beginning of line, ^ */
#define	EOL         0304  /* End of line, $ */
#define	CCLASS      0305  /* Character class, [] */
#define	NCCLASS     0306  /* Negated character class, [] */
#define BOW         0307  /* Boundary of word, /b */
#define NBOW        0310  /* Not boundary of word, /b */
#define	END         0377  /* Terminate: match found */

//typedef std::u32string Reclass; // .length should be multiple of 2
struct Reclass
{
    int builtins;        // bit mask identifying builtin classes
    std::u32string chrs; // ranges as pairs of chars
    Reclass() : builtins(0) {}
    Reclass(int m) : builtins(m) {}
};

struct Reinst
{
    int	type;
    union	{
        int	cls_id;		/* class pointer */
        char32_t	c;		/* character */
        int	subid;		/* sub-expression id for RBRA and LBRA */
        int	right_id;		/* right child of OR */
    } u1;
    union {	/* regexp relies on these two being in the same union */
        int left_id;		/* left child of OR */
        int next_id;		/* next instruction for CAT & LBRA */
    } u2;
    int pad4; // extra 4 bytes to make this align on 8-byte boundary
};

//
class Reprog
{
    std::vector<Reinst> insts;
    std::vector<Reclass> classes;
    int startinst_id;
    int num_capturing_groups;

public:

    static Reprog* create_from(const char32_t* pattern);

    int add_inst(int type);
    int add_inst(Reinst inst);
    int add_class(Reclass cls);

    int inst_count() const;
    int classes_count() const;
    void set_groups_count(int groups);
    int groups_count() const;

    const Reinst* insts_data() const;

    Reinst& inst_at(int id);
    Reclass& class_at(int id);

    void set_start_inst(int id);
    int get_start_inst() const;

    void optimize();
    void print(); // for debugging
};

