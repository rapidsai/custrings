cmake_minimum_required (VERSION 3.12)

project(NVStrings VERSION 0.2.0 LANGUAGES C CXX CUDA)
find_package(PythonLibs 3 REQUIRED)

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
option(BUILD_TESTS "Configure CMake to build tests" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\$ORIGIN")

if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)
endif()

add_subdirectory(../thirdparty/rmm "${CMAKE_BINARY_DIR}/rmm")

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${PYTHON_INCLUDE_DIRS}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/../thirdparty/rmm/include")

add_definitions(${DEFINES})

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

if(NOT CMAKE_CXX11_ABI)
    message(STATUS "NVStrings: Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0)
endif(NOT CMAKE_CXX11_ABI)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -rdc=true")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

add_library(NVStrings SHARED 
            src/NVStrings.cu
            src/NVStringsImpl.cu
            src/NVStrings_array.cu
            src/NVStrings_attrs.cu
            src/NVStrings_combine.cu
            src/NVStrings_split.cu
            src/NVStrings_pad.cu
            src/NVStrings_substr.cu
            src/NVStrings_modify.cu
            src/NVStrings_strip.cu
            src/NVStrings_case.cu
            src/NVStrings_find.cu
            src/NVStrings_convert.cu
            src/custring_view.cu
            src/custring.cu
            src/util.cu
            src/regex/regexec.cu
            src/regex/regcomp.cpp)

add_library(NVCategory SHARED
            src/NVCategory.cu
            src/custring_view.cu
            src/custring.cu)

add_library(NVText SHARED
            src/NVText.cu
            src/custring_view.cu
            src/custring.cu)

target_link_libraries(NVStrings rmm)
target_link_libraries(NVCategory NVStrings rmm)
target_link_libraries(NVText NVStrings rmm)

add_library(pyniNVStrings SHARED python/pystrings.cpp)
add_library(pyniNVCategory SHARED python/pycategory.cpp)
add_library(pyniNVText SHARED python/pytext.cpp)

if (WIN32) 
set_target_properties(pyniNVStrings PROPERTIES SUFFIX ".pyd")
set_target_properties(pyniNVCategory PROPERTIES SUFFIX ".pyd")
set_target_properties(pyniNVText PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(pyniNVStrings PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVStrings PROPERTIES PREFIX "")
set_target_properties(pyniNVCategory PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVCategory PROPERTIES PREFIX "")
set_target_properties(pyniNVText PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVText PROPERTIES PREFIX "")
endif()

target_link_libraries(pyniNVStrings ${PYTHON_LIBRARIES} NVStrings)
target_link_libraries(pyniNVCategory ${PYTHON_LIBRARIES} NVStrings NVCategory)
target_link_libraries(pyniNVText ${PYTHON_LIBRARIES} NVStrings NVText)

install(TARGETS NVStrings LIBRARY DESTINATION lib)
install(TARGETS NVCategory LIBRARY DESTINATION lib)
install(TARGETS NVText LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/nvstrings PATTERN "*.h")

# Doxygen documentation
add_custom_command(OUTPUT DOCS_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(doc DEPENDS DOCS_DOXYGEN)