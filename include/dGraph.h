//  Project ParallelBatchRCM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2021 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Daniel Mlakar - daniel.mlakar ( at ) icg.tugraz.at
//              Martin Winter - martin.winter ( at ) icg.tugraz.at
//              Mathias Parger - mathias.parger ( at ) icg.tugraz.at
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#pragma once
#include <cuda_runtime.h>
#include <memory>


#include "dCSR.h"
#include "Graph.h"
#include "handle_error.h"

template<typename T>
struct dGraph
{
	size_t size;

	unsigned* level;
	unsigned* parents;

	unsigned* children_offsets;
	unsigned* children_counts;
	unsigned* children_;

	dCSR<T> csr;

	__device__ __forceinline__
	unsigned* neighbors(const unsigned node) const
	{
		return &csr.col_ids[csr.row_offsets[node]];
	}

	__device__ __forceinline__
	unsigned neighbor_count(const unsigned node) const
	{
		return csr.row_offsets[node + 1] - csr.row_offsets[node];
	}

	__device__ __forceinline__
	unsigned* children(const unsigned node) const
	{
		return &children_[children_offsets[node]];
	}

	__device__ __forceinline__
	unsigned children_count(const unsigned node) const
	{
		return children_offsets[node + 1] - children_offsets[node];
	}

	void alloc(size_t size)
	{
		this->size = size;
		HANDLE_ERROR(cudaMalloc(&level, size * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc(&parents, size * sizeof(unsigned int)));

		HANDLE_ERROR(cudaMemset(parents, 0xff, size * sizeof(unsigned int)));

		HANDLE_ERROR(cudaMalloc(&children_, size * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc(&children_offsets, (size + 1) * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc(&children_counts, size * sizeof(unsigned int)));

		HANDLE_ERROR(cudaMemset(children_counts, 0, size * sizeof(unsigned int)));
	}

	void reset()
	{
		HANDLE_ERROR(cudaMemset(parents, 0xff, this->size * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMemset(level, 0xff, this->size * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMemset(children_counts, 0, this->size * sizeof(unsigned int)));
	}
};

template <typename T>
void convert(dGraph<T>& dst, Graph<T>& src)
{
	convert(dst.csr, src.csr);

	dst.alloc(src.size);
}

template <typename T>
void convert(Graph<T>& dst, dGraph<T>& src)
{
	convert(dst.csr, src.csr);

	dst.alloc(src.size);
} 