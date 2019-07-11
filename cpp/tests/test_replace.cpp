
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


void test1()
{
    const char* hstrs[] = { "hello there, good friend!", "hi there!", nullptr, "", "!accénted" };
    int count = 5;
    const char* htgts[] = { "," , "!", "e" };
    int tcount = 3;
    const char* hresult[] = { "h_llo th_r__ good fri_nd_", "hi th_r__", nullptr, "", "_accént_d"};

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


void test_regex1()
{
    const char* hstrs[] = {"hello @abc @def world", "the quick brown @fox jumps over the",
                           "lazy @dog", "hello http://www.world.com were there @home"};
    int count = 4;
    const char* hptns[] = {"@\\S+", "\\bthe\\b"};
    int tcount = 2;
    const char* hrpls[] = {"***", ""};
    int rcount = 1;

    NVStrings* strs = NVStrings::create_from_array(hstrs,count);
    NVStrings* rpls = NVStrings::create_from_array(hrpls,rcount);

    printf("strings: (%ld bytes)\n", strs->memsize());
    strs->print();
    printf("patterns: (cpu)\n");
    std::vector<const char*> ptns;
    for( int i=0; i < tcount; ++i )
    {
        printf("%d:[%s]\n",i,hptns[i]);
        ptns.push_back(hptns[i]);
    }
    printf("repls: (%ld bytes)\n", rpls->memsize());
    rpls->print();

    NVStrings* result = strs->replace_re(ptns, *rpls);
    printf("result: (%ld bytes)\n", result->memsize());
    result->print();

    // verify result against hresult
    // can use NVStrings:match_strings to compare two instances
    // add up boolean values to check against count

    NVStrings::destroy(result);
    NVStrings::destroy(rpls);
    NVStrings::destroy(strs);
}

void test_regex2()
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0,4);
    const char* hstrs[] = { "the quick brown fox jumps over the lazy dog",
                            "the fat cat lays next to the other cat",
                            "a slow moving turtle cannot catch the bird",
                            "which can be composed together to form a more complete",
                            "the result does not include the value in the sum" };

    std::vector<std::string> data;
    std::vector<const char*> data_ptrs;
    for( int idx=0; idx < 1000000; ++idx )
        data_ptrs.push_back(hstrs[dist(mt)]);
    NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
    printf("strings: (%ld bytes)\n", strs->memsize());
    strs->print(0,10);

    const char* hptns[] = {"\\bi\\b","\\bme\\b","\\bmy\\b","\\bmyself\\b","\\bwe\\b","\\bour\\b","\\bours\\b","\\bourselves\\b","\\byou\\b","\\byour\\b","\\byours\\b","\\byourself\\b",
            "\\byourselves\\b","\\bhe\\b","\\bhim\\b","\\bhis\\b","\\bhimself\\b","\\bshe\\b","\\bher\\b","\\bhers\\b","\\bherself\\b","\\bit\\b","\\bits\\b","\\bitself\\b",
            "\\bthey\\b","\\bthem\\b","\\btheir\\b","\\btheirs\\b","\\bthemselves\\b","\\bwhat\\b","\\bwhich\\b","\\bwho\\b","\\bwhom\\b","\\bthis\\b","\\bthat\\b",
            "\\bthese\\b","\\bthose\\b","\\bam\\b","\\bis\\b","\\bare\\b","\\bwas\\b","\\bwere\\b","\\bbe\\b","\\bbeen\\b","\\bbeing\\b","\\bhave\\b","\\bhas\\b","\\bhad\\b",
            "\\bhaving\\b","\\bdo\\b","\\bdoes\\b","\\bdid\\b","\\bdoing\\b","\\ba\\b","\\ban\\b","\\bthe\\b","\\band\\b","\\bbut\\b","\\bif\\b","\\bor\\b","\\bbecause\\b","\\bas\\b",
            "\\buntil\\b","\\bwhile\\b","\\bof\\b","\\bat\\b","\\bby\\b","\\bfor\\b","\\bwith\\b","\\babout\\b","\\bagainst\\b","\\bbetween\\b","\\binto\\b","\\bthrough\\b",
            "\\bduring\\b","\\bbefore\\b","\\bafter\\b","\\babove\\b","\\bbelow\\b","\\bto\\b","\\bfrom\\b","\\bup\\b","\\bdown\\b","\\bin\\b","\\bout\\b","\\bon\\b","\\boff\\b",
            "\\bover\\b","\\bunder\\b","\\bagain\\b","\\bfurther\\b","\\bthen\\b","\\bonce\\b","\\bhere\\b","\\bthere\\b","\\bwhen\\b","\\bwhere\\b","\\bwhy\\b","\\bhow\\b",
            "\\ball\\b","\\bany\\b","\\bboth\\b","\\beach\\b","\\bfew\\b","\\bmore\\b","\\bmost\\b","\\bother\\b","\\bsome\\b","\\bsuch\\b","\\bno\\b","\\bnor\\b","\\bnot\\b",
            "\\bonly\\b","\\bown\\b","\\bsame\\b","\\bso\\b","\\bthan\\b","\\btoo\\b","\\bvery\\b","\\bs\\b","\\bt\\b","\\bcan\\b","\\bwill\\b","\\bjust\\b","\\bdon\\b","\\bshould\\b",
            "\\bnow\\b","\\buses\\b","\\buse\\b","\\busing\\b","\\bused\\b","\\bone\\b","\\balso\\b"};
    unsigned int tcount = 133;
    printf("patterns: (cpu)\n");
    std::vector<const char*> ptns;
    for( int idx=0; idx < tcount; ++idx )
    {
        if( idx < 10 )
            printf("%d:[%s]\n",idx,hptns[idx]);
        ptns.push_back(hptns[idx]);
    }

    const char* hrpls[] = {""};
    unsigned int rcount = 1;
    NVStrings* rpls = NVStrings::create_from_array(hrpls,rcount);
    printf("repls: (%ld bytes)\n", rpls->memsize());
    rpls->print();

    {
        double st = GetTime();
        NVStrings* result = strs->replace_re(ptns,*rpls);
        double et = GetTime() - st;
        printf("result: (%ld bytes)\n", result->memsize());
        result->print(0,10);
        printf("%g seconds\n",et);
        NVStrings::destroy(result);
    }


    NVStrings::destroy(rpls);
    NVStrings::destroy(strs);
}

int main( int argc, char** argv )
{
    test1();
    //test2();
    test_regex1();
    test_regex2();
    return 0;
}
