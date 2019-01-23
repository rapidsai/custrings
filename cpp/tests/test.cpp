
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "NVStrings.h"

inline char RandomChar()
{
	int index = (int)((unsigned long long)rand() * 36 / ((unsigned long long)RAND_MAX + 1));
	if (index < 26) return 'A' + index;
	else return '0' + index - 26;
}

inline char RandomDecimalDigit()
{
	int index = (int)((unsigned long long)rand() * 10 / ((unsigned long long)RAND_MAX + 1));
	return '0' + index;
}

inline int RandomLen(int minLen, int maxLen)
{
	int ranges = maxLen - minLen + 1;
	int rand_range = (int)((unsigned long long)rand()*ranges / ((unsigned long long)RAND_MAX + 1));
	return minLen + rand_range;
}


inline int createRandomStrings(int minLen, int maxLen, int count, std::vector<std::string>& strlist)
{
	std::vector<char> str;
	str.resize(maxLen + 1); // +1 to store /0

	for (int i = 0; i < count; i++)
	{
		int len = RandomLen(minLen, maxLen);
		for (int j = 0; j < len; j++)
			str[j] = RandomChar();

		str[len] = 0;
		strlist.push_back(&str[0]);
	}
	return (int)strlist.size();
}

inline int createRandomStrings_Float(int minLen, int maxLen, int count, std::vector<std::string>& strlist)
{
	std::vector<char> str;
	str.resize(maxLen + 1); // +1 to store /0
	for (int i = 0; i < count; i++)
	{
		int len = RandomLen(minLen, maxLen);
		int dec_pos = RandomLen(0, len);
		for (int j = 0; j < len; j++)
			if (j == dec_pos)
				str[j] = '.';
			else
				str[j] = RandomDecimalDigit();
		str[len] = 0;
		strlist.push_back(&str[0]);
	}
	return (int)strlist.size();
}


int main()
{
	srand(100);
	NVStrings::initLibrary();

	std::vector<std::string> strings1, strings2;
	createRandomStrings(2, 20, 500000, strings1);
	createRandomStrings_Float(3, 20, 500000, strings2);

	std::vector<const char*> v_ptrs1(strings1.size());
	for (int i = 0; i < (int)strings1.size(); i++)
		v_ptrs1[i] = strings1[i].c_str();

	std::vector<const char*> v_ptrs2(strings2.size());
	for (int i = 0; i < (int)strings2.size(); i++)
		v_ptrs2[i] = strings2[i].c_str();

	NVStrings d_strings1(v_ptrs1.data(), v_ptrs1.size());
	NVStrings d_strings2(v_ptrs2.data(), v_ptrs2.size());

	NVStrings* d_add_result = d_strings1.cat(&d_strings2, "");

	size_t size = d_add_result->size();

	char** list = new char*[size];
	d_add_result->to_host(list, 0, size);

	delete d_add_result;

#if 0
	{
		std::fstream f("result.txt", std::ios::out);
		for (int i = 0; i < size; i++)
		{
			f << list[i] << std::endl;
		}
		f.close();
	}
#endif

	std::vector<int> find_result(strings1.size());
	const char* toFind = "G";
	d_strings1.find(toFind, 0, -1, find_result.data(), strings1.size());

#if 0
	{
		// check find result
		std::fstream f("result.txt", std::ios::out);
		for (int i = 0; i < find_result.size(); i++)
		{
			f << find_result[i] << std::endl;
		}
		f.close();
	}
#endif

	std::vector<float> float_result(strings2.size());
	d_strings2.stof(float_result.data(), float_result.size());

#if 0
	{
		// check stof result
		std::fstream f("result.txt", std::ios::out);
		for (int i = 0; i < float_result.size(); i++)
		{
			f << float_result[i] << std::endl;
		}
		f.close();
	}
#endif


	for (int i = 0; i < v_ptrs1.size(); i++)
		delete list[i];

	delete list;
	return 0;
}
