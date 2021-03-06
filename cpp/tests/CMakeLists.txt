

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(NVSTRINGS_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

#add_library(cudftestutil STATIC
#            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/test_utils.cu"
#            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/valid_vectors.cpp")

#target_link_libraries(rmm cudart cuda NVCategory NVStrings)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread
                          rmm cudart cuda NVCategory NVStrings NVText)
    #if(USE_NVTX)
    #    target_link_libraries(${CMAKE_TEST_NAME} ${NVTX_LIBRARY})
    #endif(USE_NVTX)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${GTEST_INCLUDE_DIR}"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - create tests ----------------------------------------------------------------------------------

ConfigureTest(ARRAY_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_array.cu")
ConfigureTest(ATTRS_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_attrs.cu")
ConfigureTest(CASE_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_case.cpp")
ConfigureTest(COMBINE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_combine.cpp")
ConfigureTest(CONVERT_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_convert.cu")
ConfigureTest(COUNT_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_count.cu")
ConfigureTest(DATE_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_datetime.cu")
ConfigureTest(EXTRACT_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_extract.cpp")
ConfigureTest(FIND_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_find.cu")
ConfigureTest(MODIFY_TEST  "${CMAKE_CURRENT_SOURCE_DIR}/test_modify.cu")
ConfigureTest(PAD_TEST     "${CMAKE_CURRENT_SOURCE_DIR}/test_pad.cpp")
ConfigureTest(STRIP_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_strip.cpp")
ConfigureTest(SPLIT_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_split.cpp")
ConfigureTest(REPLACE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_replace.cpp")
ConfigureTest(TEXT_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_text.cu")
ConfigureTest(URL_TEST     "${CMAKE_CURRENT_SOURCE_DIR}/test_url.cpp")
