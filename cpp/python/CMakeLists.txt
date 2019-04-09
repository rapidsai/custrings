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

project(pyniNVStrings VERSION 0.7.0 LANGUAGES C CXX)

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
    endif(NOT CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

option(BUILD_TESTS "Configure CMake to build tests" OFF)

###################################################################################################
# - nvstrings -------------------------------------------------------------------------------------

find_file(NVSTRINGS_INCLUDE "NVStrings.h"
          HINTS "$ENV{NVSTRINGS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/nvstrings"
                "$ENV{CONDA_PREFIX}/include")

find_file(NVCATEGORY_INCLUDE "NVCategory.h"
          HINTS "$ENV{NVSTRINGS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/nvstrings"
                "$ENV{CONDA_PREFIX}/include")

find_file(NVTEXT_INCLUDE "NVText.h"
          HINTS "$ENV{NVSTRINGS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/nvstrings"
                "$ENV{CONDA_PREFIX}/include")

find_library(NVSTRINGS_LIBRARY "NVStrings"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

find_library(NVCATEGORY_LIBRARY "NVCategory"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

find_library(NVText_LIBRARY "NVText"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

add_library(nvstrings SHARED IMPORTED ${NVSTRINGS_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)
    set_target_properties(nvstrings PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)

add_library(nvcategory SHARED IMPORTED ${NVCATEGORY_LIBRARY})
if (NVCATEGORY_INCLUDE AND NVCATEGORY_LIBRARY)
    set_target_properties(nvcategory PROPERTIES IMPORTED_LOCATION ${NVCATEGORY_LIBRARY})
endif (NVCATEGORY_INCLUDE AND NVCATEGORY_LIBRARY)

add_library(nvtext SHARED IMPORTED ${NVTEXT_LIBRARY})
if (NVTEXT_INCLUDE AND NVTEXT_LIBRARY)
    set_target_properties(nvtext PROPERTIES IMPORTED_LOCATION ${NVTEXT_LIBRARY})
endif (NVTEXT_INCLUDE AND NVTEXT_LIBRARY)

###################################################################################################
# - python ----------------------------------------------------------------------------------------

find_package(Python3 COMPONENTS Development REQUIRED)

message(STATUS "PYTHON: Python3_LIBRARIES set to ${Python3_LIBRARIES}")
message(STATUS "PYTHON: Python3_INCLUDE_DIRS set to ${Python3_INCLUDE_DIRS}")
message(STATUS "PYTHON: Python3_VERSION set to ${Python3_VERSION}")

add_library(python3 SHARED IMPORTED ${Python3_LIBRARIES})
if (Python3_INCLUDE_DIRS AND Python3_LIBRARIES)
    set_target_properties(python3 PROPERTIES IMPORTED_LOCATION ${Python3_LIBRARIES})
endif (Python3_INCLUDE_DIRS AND Python3_LIBRARIES)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${Python3_INCLUDE_DIRS}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_BINARY_DIR}/lib"
                 "${Python3_LIBRARIES}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(pyniNVStrings SHARED pystrings.cpp)
add_library(pyniNVCategory SHARED pycategory.cpp)
add_library(pyniNVText SHARED pytext.cpp)

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

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(pyniNVStrings python3 NVStrings)
target_link_libraries(pyniNVCategory python3 NVStrings NVCategory)
target_link_libraries(pyniNVText python3 NVStrings NVText)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

add_custom_target(python_cpp DEPENDS pyniNVStrings pyniNVCategory pyniNVText)

add_custom_command(OUTPUT install_python_command
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/pyniNVStrings.so ${CMAKE_SOURCE_DIR}/../python/pyniNVStrings.so
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/pyniNVCategory.so ${CMAKE_SOURCE_DIR}/../python/pyniNVCategory.so
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/pyniNVText.so ${CMAKE_SOURCE_DIR}/../python/pyniNVText.so
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/../../python pip install .
                   VERBATIM)

add_custom_target(install_python DEPENDS install_python_command python_cpp)

add_custom_command(OUTPUT develop_python_command
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/pyniNVStrings.so ${CMAKE_SOURCE_DIR}/../python/pyniNVStrings.so
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/pyniNVCategory.so ${CMAKE_SOURCE_DIR}/../python/pyniNVCategory.so
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/pyniNVText.so ${CMAKE_SOURCE_DIR}/../python/pyniNVText.so
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/../../python pip install -e .
                   VERBATIM)

add_custom_target(develop_python DEPENDS develop_python_command python_cpp)
