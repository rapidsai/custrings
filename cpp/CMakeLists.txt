#=============================================================================
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required (VERSION 3.12)

project(NVStrings VERSION 0.7.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\$ORIGIN")

if (WIN32) 
    set (DEFINES ${DEFINES}
         -D"_CRT_SECURE_NO_DEPRECATE"  
         -D"_SCL_SECURE_NO_DEPRECATE" 
    )
endif()
add_definitions(${DEFINES})

if(CMAKE_COMPILER_IS_GNUCXX)
    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(NOT CMAKE_CXX11_ABI)
        message(STATUS "NVStrings: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(NOT CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70")

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

option(BUILD_TESTS "Configure CMake to build tests" OFF)

###################################################################################################
# - rmm -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/rmm"
                "$ENV{CONDA_PREFIX}/include")

find_library(RMM_LIBRARY "rmm"
             HINTS "$ENV{RMM_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                 "${CMAKE_BINARY_DIR}/lib"
                 "${RMM_LIBRARY}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

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
            src/util.cu
            src/regex/regexec.cpp
            src/regex/regcomp.cpp)

add_library(NVCategory SHARED
            src/NVCategory.cu)

add_library(NVText SHARED
            src/NVText.cu)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(NVStrings rmm cudart cuda)
target_link_libraries(NVCategory NVStrings rmm cudart cuda)
target_link_libraries(NVText NVStrings rmm cudart cuda)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS NVStrings LIBRARY DESTINATION lib)
install(TARGETS NVCategory LIBRARY DESTINATION lib)
install(TARGETS NVText LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/nvstrings PATTERN "*.h")

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

add_custom_command(OUTPUT DOCS_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(doc DEPENDS DOCS_DOXYGEN)
