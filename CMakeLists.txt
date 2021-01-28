#  Project ParallelBatchRCM
#  https://www.tugraz.at/institute/icg/research/team-steinberger/
#
#  Copyright (C) 2021 Institute for Computer Graphics and Vision,
#                     Graz University of Technology
#
#  Author(s):  Daniel Mlakar - daniel.mlakar ( at ) icg.tugraz.at
#              Martin Winter - martin.winter ( at ) icg.tugraz.at
#              Mathias Parger - mathias.parger ( at ) icg.tugraz.at
#              Markus Steinberger - steinberger ( at ) icg.tugraz.at
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#

cmake_minimum_required(VERSION 3.16)
project(CuthillMcKee LANGUAGES CXX CUDA)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)
include_directories(include/device)

find_package(CUDAToolkit REQUIRED)

##########################################################################
# Choose for which CC to build and if to enable Debug
option(CUDA_BUILD_INFO "Verbose CUDA build" FALSE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)


if (NOT CUDA_BUILD_CC)
	set(SUPPORTED_CUDA_BUILD_CC 61 70 75)

	list(JOIN SUPPORTED_CUDA_BUILD_CC " " SUPPORTED_CC_STRING)
	message(WARNING "No compute capability (CUDA_BUILD_CC) specified."
		"Supported values are: " ${SUPPORTED_CC_STRING}.
		" Building for all supported architectures.")
	
	set(CUDA_BUILD_CC ${SUPPORTED_CUDA_BUILD_CC})
endif()

if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()
if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

##########################################################################
# CXX Flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if (WIN32)

else()
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17")
	string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g")
	string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
endif()


##########################################################################
# Setup executable
file(GLOB CUDHDR include/device/*.cuh)
file(GLOB CUDSRC src/device/*.cu)
file(GLOB HDR include/*.h)
file(GLOB SRC src/*.cpp)
add_executable(CuthillMcKee
    ${CUDHDR}
    ${CUDSRC}
	${HDR}
	${SRC}
)


set_property(TARGET CuthillMcKee PROPERTY CXX_STANDARD 17)
set_property(TARGET CuthillMcKee PROPERTY CUDA_ARCHITECTURES ${CUDA_BUILD_CC})

##########################################################################
# Link libraries
target_link_libraries(CuthillMcKee CUDA::cusolver)
