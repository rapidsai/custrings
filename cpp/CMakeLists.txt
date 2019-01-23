cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)
find_package(CUDA REQUIRED)

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" OFF)

set(CUDA_SEPARABLE_COMPILATION ON)

set(LIB_STRINGS_SOURCES
src/custring_view.cu
src/custring.cu
src/NVStrings.cu
src/util.cu
src/regex/regexec.cu
src/regex/regcomp.cpp
)

set(LIB_STRINGS_HEADERS 
src/custring_view.cuh
src/custring.cuh
include/NVStrings.h
src/util.h
src/regex/regex.cuh
src/regex/regcomp.h
)

set(LIB_CATEGORY_SOURCES
src/custring_view.cu
src/custring.cu
src/NVCategory.cu
)

set(LIB_CATEGORY_HEADERS 
src/custring_view.cuh
src/custring.cuh
include/NVCategory.h 
)

set(LIB_RAVE_SOURCES
src/custring_view.cu
src/custring.cu
src/Rave.cu
)

set(LIB_RAVE_HEADERS 
src/custring_view.cuh
src/custring.cuh
include/Rave.h 
)

set (INCLUDE_DIR
${PYTHON_INCLUDE_DIRS}
include
src
)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\$ORIGIN")

if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
)
else()
add_definitions(-std=c++0x)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

if(NOT CMAKE_CXX11_ABI)
    message(STATUS "NVStrings: Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0)
endif(NOT CMAKE_CXX11_ABI)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 --expt-extended-lambda -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -rdc=true -Xcompiler -fPIC)

cuda_add_library (NVStrings SHARED ${LIB_STRINGS_SOURCES} ${LIB_STRINGS_HEADERS})
cuda_add_library (NVCategory SHARED ${LIB_CATEGORY_SOURCES} ${LIB_CATEGORY_HEADERS})
cuda_add_library (Rave SHARED ${LIB_RAVE_SOURCES} ${LIB_RAVE_HEADERS})

target_link_libraries(NVCategory NVStrings)
target_link_libraries(Rave NVStrings)

cuda_add_library (pyniNVStrings SHARED python/pystrings.cpp)
cuda_add_library (pyniNVCategory SHARED python/pycategory.cpp)
cuda_add_library (pyniRave SHARED python/pyrave.cpp)

if (WIN32) 
set_target_properties(pyniNVStrings PROPERTIES SUFFIX ".pyd")
set_target_properties(pyniNVCategory PROPERTIES SUFFIX ".pyd")
set_target_properties(pyniRave PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(pyniNVStrings PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVStrings PROPERTIES PREFIX "")
set_target_properties(pyniNVCategory PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVCategory PROPERTIES PREFIX "")
set_target_properties(pyniRave PROPERTIES SUFFIX ".so")
set_target_properties(pyniRave PROPERTIES PREFIX "")
endif()

target_link_libraries(pyniNVStrings ${PYTHON_LIBRARIES} NVStrings)
target_link_libraries(pyniNVCategory ${PYTHON_LIBRARIES} NVStrings NVCategory)
target_link_libraries(pyniRave ${PYTHON_LIBRARIES} NVStrings Rave)

install(TARGETS NVStrings LIBRARY DESTINATION lib)
