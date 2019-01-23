
//
struct Reinst;
struct Reljunk;
struct Reljunk_Sub;
class custring_view;

//
class dreprog
{
	int startinst_id, num_capturing_groups;
	int insts_count, classes_count;

	dreprog();
	~dreprog();

	//
	__device__ int regexec( custring_view* dstr, Reljunk& jnk, int& begin, int& end, int groupid=0 );

public:
	//
	static dreprog* create_from(const char32_t* pattern);
	static void destroy(dreprog* ptr);

	int inst_counts();
	int group_counts();

	__host__ __device__ Reinst* get_inst(int idx);
	__device__ char32_t* get_class(int idx, int& len);

	//
	__device__ int contains( custring_view* dstr );
	__device__ int match( custring_view* dstr );
	__device__ int find( custring_view* dstr, int& begin, int& end );
	__device__ int extract( custring_view* str, int& begin, int& end, int col );
};