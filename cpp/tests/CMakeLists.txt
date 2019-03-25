cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUSTRINGS_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
            ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} rmm NVStrings NVCategory)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -rdc=true")

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        "${PYTHON_INCLUDE_DIRS}"
        "${CMAKE_BINARY_DIR}/include"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/../thirdparty/rmm/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
        "${CMAKE_BINARY_DIR}/lib")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - example test ----------------------------------------------------------------------------------
set(CATCOPY_TEST_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/catcopy.cu")

ConfigureTest(CATCOPY_TEST "${CATCOPY_TEST_SRC}")
