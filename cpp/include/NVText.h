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
#pragma once

/**
 * @file NVStrings.h
 * @brief Class definition for NVStrings.
 */

class NVStrings;

/**
 * @brief This class is a collection of utilities for operating on words or tokens.
 * It uses methods on the NVStrings class to access its character arrays.
 */
class NVText
{

public:

    /**
     * @brief Tokenize all the strings into a single instance. Delimiter is whitespace.
     * @param strs Strings to tokenize.
     * @return Just the tokens. No empty or null strings.
     */
    static NVStrings* tokenize(NVStrings& strs);

    /**
     * @brief Tokenize all the strings into a single instance.
     * @param strs Strings to tokenize.
     * @param delimiter String or character used to identify tokens.
     * @return Just the tokens. No empty or null strings.
     */
    static NVStrings* tokenize(NVStrings& strs, const char* delimiter);

    /**
     * @brief Tokenize all the strings into a single instance using multiple delimiters.
     * @param strs Strings to tokenize.
     * @param delimiters These are used to identify and separate the tokens.
     * @return Just the tokens. No empty or null strings.
     */
    static NVStrings* tokenize(NVStrings& strs, NVStrings& delimiters);

    /**
     * @brief Tokenize all the strings into a single instance of unique tokens only.
     * @param strs Strings to tokenize
     * @param delimiter String or character used to identify tokens.
     * @return Unique tokens only. These are also sorted.
     */
    static NVStrings* unique_tokens(NVStrings& strs, const char* delimiter = " ");

    /**
     * @brief Computes the number of tokens in each string.
     * @param strs Strings to tokenize.
     * @param delimiter String or character used to identify tokens.
     * @param[in,out] results Array of counts, one per string.
     * @param devmem True if results in device memory.
     * @return 0 if successful.
     */
    static unsigned int token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool devmem=true );

    /**
     * @brief Fills a matrix of boolean values indicating the corresponding token appears in that string.
     * @param strs Strings to process.
     * @param tokens Strings to search within each string.
     * @param[in,out] results Matrix of booleans, one array per string. Must be able to hold strs.size()*tgts.size() bool values.
     * @param devmem True if results in device memory.
     * @return 0 if successful.
     */
    static unsigned int contains_strings( NVStrings& strs, NVStrings& tokens, bool* results, bool devmem=true );

    /**
     * @brief Fills a matrix of int values indicating how many times the corresponding token appears in that string.
     * @param strs Strings to process.
     * @param tokens Strings to search within each string.
     * @param[in,out] results Matrix of ints, one array per string. Must be able to hold strs.size()*tgts.size() int values.
     * @param devmem True if results in device memory.
     * @return 0 if successful.
     */
    static unsigned int strings_counts( NVStrings& strs, NVStrings& tokens, unsigned int* results, bool devmem=true );

    /**
     * @brief Fills a matrix of int values indicating how many times the corresponding token appears in that string.
     * @param strs Strings to process.
     * @param tokens Strings to check within each string.
     * @param delimiter String or character used to identify tokens.
     * @param[in,out] results Matrix of ints, one array per string. Must be able to hold strs.size()*tgts.size() int values.
     * @param devmem True if results in device memory.
     * @return 0 if successful.
     */
    static unsigned int tokens_counts( NVStrings& strs, NVStrings& tokens, const char* delimiter, unsigned int* results, bool devmem=true );

    /**
     * @brief Edit distance algorithms
     */
    enum distance_type {
        levenshtein ///< Use the levenshtein algorithm
    };

    /**
     * @brief Compute the edit distance between each string and the target string.
     * @param algo The edit distance algorithm to use for the computation.
     * @param strs Strings to process.
     * @param str Compute distances to this string.
     * @param[in,out] results Array of distances, one per string.
     * @param devmem True if results in device memory.
     * @return 0 if successful.
     */
    static unsigned int edit_distance( distance_type algo, NVStrings& strs, const char* str, unsigned int* results, bool devmem=true );
    /**
     * @brief Compute the edit distance between each string and the target strings.
     * @param algo The edit distance algorithm to use for the computation.
     * @param strs1 Strings to process.
     * @param strs2 Compute distances to each corresponding string.
     * @param[in,out] results Array of distances, one per string.
     * @param devmem True if results in device memory.
     * @return 0 if successful.
     */
    static unsigned int edit_distance( distance_type algo, NVStrings& strs1, NVStrings& strs2, unsigned int* results, bool devmem=true );

    /**
     * @brief Converts tokenized list of strings into instance with ngrams.
     * @param strs Tokens to make into ngrams.
     * @param ngrams The 'n' in ngrams. Example, use 2 for bigrams, 3 for trigrams, etc.
     * @param separator String used to join tokens.
     * @return The tokens as ngrams.
     */
    static NVStrings* create_ngrams(NVStrings& strs, unsigned int ngrams, const char* separator );
};
