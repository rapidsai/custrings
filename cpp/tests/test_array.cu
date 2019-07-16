
#include <gtest/gtest.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "../include/NVStrings.h"

// utility to verify strings results
bool verify_strings( NVStrings* d_strs, const char** h_strs )
{
    unsigned int count = d_strs->size();
    std::vector<int> bytes(count);
    d_strs->byte_count(bytes.data(),false);
    std::vector<char*> ptrs(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int size = bytes[idx];
        if( size < 0 )
            continue;
        char* str = (char*)malloc(size+1);
        str[size] = 0;
        ptrs[idx] = str;
    }
    d_strs->to_host( ptrs.data(), 0, (int)count );
    bool bmatched = true;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        char* str1 = ptrs[idx];
        const char* str2 = h_strs[idx];
        if( str1 )
        {
            if( !str2 || (strcmp(str1,str2)!=0) )
                bmatched = false;
            //free(str1);
        }
        else if( str2 )
            bmatched = false;
        if( !bmatched )
            printf("%d:[%s]!=[%s]\n",idx,str1,str2);
        if( str1 )
            free(str1);
    }
    return bmatched;
}

const char* hstrs[] = { "John Smith", "Joe Blow", "Jane Smith", nullptr, "" };
int count = 5;

TEST(TestArray, Sublist)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    NVStrings* got = strs->sublist(1,4);

    const char* expected[] = {"Joe Blow", "Jane Smith", nullptr };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestArray, Gather)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    thrust::device_vector<int> indexes;
    indexes.push_back(1);
    indexes.push_back(3);
    indexes.push_back(2);

    NVStrings* got = strs->gather(indexes.data().get(), indexes.size());
    const char* expected[] = {"Joe Blow", nullptr, "Jane Smith" };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestArray, GatherBool)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    thrust::device_vector<bool> indexes;
    indexes.push_back(true);
    indexes.push_back(false);
    indexes.push_back(false);
    indexes.push_back(false);
    indexes.push_back(true);

    NVStrings* got = strs->gather(indexes.data().get());
    const char* expected[] = {"John Smith", "" };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestArray, Scatter)
{
    NVStrings* strs1 = NVStrings::create_from_array(hstrs,count);
    const char* h2[] = { "", "Joe Schmoe" };
    NVStrings* strs2 = NVStrings::create_from_array(h2,2);

    thrust::device_vector<int> indexes;
    indexes.push_back(1);
    indexes.push_back(3);

    NVStrings* got = strs1->scatter(*strs2, indexes.data().get() );
    const char* expected[] = {"John Smith", "", "Jane Smith", "Joe Schmoe", "" };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs1);
    NVStrings::destroy(strs2);
}

TEST(TestArray, RemoveStrings)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    thrust::device_vector<int> indexes;
    indexes.push_back(0);
    indexes.push_back(3);
    indexes.push_back(2);

    NVStrings* got = strs->remove_strings(indexes.data().get(), indexes.size());
    const char* expected[] = { "Joe Blow", "" };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestArray, SortLength)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    NVStrings* got = strs->sort(NVStrings::length);
    const char* expected[] = { nullptr, "", "Joe Blow", "John Smith", "Jane Smith" };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestArray, SortName)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    NVStrings* got = strs->sort(NVStrings::name);
    const char* expected[] = { nullptr, "",  "Jane Smith", "Joe Blow", "John Smith" };
    ASSERT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestArray, OrderLength)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    thrust::device_vector<unsigned int> indexes(strs->size());

    strs->order(NVStrings::length, false, indexes.data().get() );
    unsigned int expected[] = { 3,0,2,1,4 };
    for( int idx=0; idx<5; ++idx )
        ASSERT_EQ(indexes[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST(TestArray, OrderName)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs,count);

    thrust::device_vector<unsigned int> indexes(strs->size());

    strs->order(NVStrings::name, false, indexes.data().get(), false );
    unsigned int expected[] = { 0,1,2,4,3 };
    for( int idx=0; idx<5; ++idx )
        ASSERT_EQ(indexes[idx],expected[idx]);

    NVStrings::destroy(strs);
}

int main( int argc, char** argv )
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}