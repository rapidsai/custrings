
#include <cstdio>
#include <vector>
#include <random>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "../include/NVStrings.h"


//
// cd ../build
// g++ -std=c++11 ../tests/test_replace.cpp -I/usr/local/cuda/include -L. -lNVStrings -o test_replace -Wl,-rpath,.:
//

double GetTime()
{
	timeval tv;
	gettimeofday( &tv, NULL );
	return (double)(tv.tv_sec*1000000+tv.tv_usec)/1000000.0;
}

const char* hstrs[] = { "hello there, good friend!", "hi there!", nullptr, "", "!accénted" };
int count = 5;
const char* htgts[] = { ",","!","e" };
int tcount = 3;
const char* hresult[] = { "h_llo th_r__ good fri_nd_", "hi th_r__", nullptr, "", "_accént_d"};

void test1()
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);
    NVStrings* tgts = NVStrings::create_from_array(htgts,tcount);

    printf("strings: (%ld bytes)\n", strs->memsize());
    strs->print();
    printf("targets: (%ld bytes)\n", tgts->memsize());
    tgts->print();

    NVStrings* result = strs->replace(*tgts,"_");
    printf("result: (%ld bytes)\n", result->memsize());
    result->print();

    // verify result against hresult
    // can use NVStrings:match_strings to compare two instances
    // add up boolean values to check against count

    NVStrings::destroy(result);
    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}


void test2()
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(32,126);
    std::vector<std::string> data;
    std::vector<const char*> data_ptrs;
    for( int idx=0; idx < 1000000; ++idx )
    {
        std::string str;
        for( int jdx=0; jdx < 20; ++jdx )
        {
            char ch = (char)dist(mt);
            str.append(1,ch);
        }
        data.push_back(str);
        data_ptrs.push_back(data[data.size()-1].c_str());
    }

    NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
    NVStrings* tgts = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size()/10);

    printf("strings: (%ld bytes)\n", strs->memsize());
    strs->print(0,10);
    printf("targets: (%ld bytes)\n", tgts->memsize());
    tgts->print(0,10);

    {
        double st = GetTime();
        NVStrings* result = strs->replace(*tgts,"_");
        double et = GetTime() - st;
        printf("result: (%ld bytes)\n", result->memsize());
        result->print(0,10);
        printf("%g seconds\n",et);
        NVStrings::destroy(result);
    }

    // verify that the first size()/10 strings are all just "_"
    // - can use NVStrings:match_strings to compare two instances
    //   add up boolean values to check against count
    // - need to build NVStrings instance with "_" strings in it


    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}

int main( int argc, char** argv )
{
    test1();
    test2();
    return 0;
}
