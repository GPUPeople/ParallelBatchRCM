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

#include <cassert>
#include <chrono>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "cub/cub.cuh"
#include "../dGraph.h"
#include "cusolverSp.h"

void checkCuSolverError(cusolverStatus_t const & e)
{
	if(e == CUSOLVER_STATUS_SUCCESS)
		return;
		
	switch(e)
	{
		case CUSOLVER_STATUS_NOT_INITIALIZED:
		{
			printf("Error: CUSOLVER_STATUS_NOT_INITIALIZED\n");
			break;
		}
		case CUSOLVER_STATUS_ALLOC_FAILED:
		{
			printf("Error: CUSOLVER_STATUS_ALLOC_FAILED\n");
			break;
		}
		case CUSOLVER_STATUS_INVALID_VALUE:
		{
			printf("Error: CUSOLVER_STATUS_INVALID_VALUE\n");
			break;
		}
		case CUSOLVER_STATUS_ARCH_MISMATCH:
		{
			printf("Error: CUSOLVER_STATUS_ARCH_MISMATCH\n");
			break;
		}
		case CUSOLVER_STATUS_EXECUTION_FAILED:
		{
			printf("Error: CUSOLVER_STATUS_EXECUTION_FAILED\n");
			break;
		}
		case CUSOLVER_STATUS_INTERNAL_ERROR:
		{
			printf("Error: CUSOLVER_STATUS_INTERNAL_ERROR\n");
			break;
		}
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		{
			printf("Error: CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n");
			break;
		}
		default:
			printf("Error: Unknown cuSolver Error\n");
	}
}

namespace CuSolverRCM
{
	using uint = unsigned int;
	using DataType = float;

	void rcm(Graph<DataType>& graph, std::ofstream& csv_file, std::vector<uint>& permutation)
	{
		static constexpr bool test_cuSolver{true};
		static constexpr int test_iter{test_cuSolver ? 20 : 1};
		cusolverSpHandle_t handle;
		auto solver_status = cusolverSpCreate(&handle);
		assert(solver_status == CUSOLVER_STATUS_SUCCESS);

		cusparseMatDescr_t descrA = nullptr;
		auto sparse_status = cusparseCreateMatDescr(&descrA);
		assert(sparse_status == CUSPARSE_STATUS_SUCCESS);

		sparse_status = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		assert(sparse_status == CUSPARSE_STATUS_SUCCESS);
		sparse_status = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		assert(sparse_status == CUSPARSE_STATUS_SUCCESS);

		permutation.resize(graph.size);
		float timing{0.0f};
		for(auto iter = 0; iter < test_iter; ++iter)
		{
			auto start = std::chrono::high_resolution_clock::now();
			solver_status = cusolverSpXcsrsymrcmHost(handle, graph.size, graph.csr.nnz, descrA, (int*)graph.csr.row_offsets.get(), (int*)graph.csr.col_ids.get(), (int*)permutation.data());
			assert(solver_status == CUSOLVER_STATUS_SUCCESS);
			auto end = std::chrono::high_resolution_clock::now();
			using float_millis = std::chrono::duration<float, std::milli>;
			timing += std::chrono::duration_cast<float_millis>(end - start).count();
		}
		timing /= test_iter;

		std::cout << std::setw(27) << std::right << "CuSolver-RCM" << " duration: " << timing << " ms" << std::endl;
		csv_file << timing;
		csv_file << std::flush;
	}

	template <typename T>
	void apply_permutation_in_place(T* const vec, const std::vector<int>& p, int begin, int end)
	{
		std::vector<bool> done(end - begin, false);
		for (std::size_t i = 0; i < done.size(); ++i)
		{
			if (done[i])
				continue;
			done[i] = true;
			std::size_t prev_j = i;
			std::size_t j = p[i];
			while (i != j)
			{
				std::swap(vec[begin + prev_j], vec[begin + j]);
				done[j] = true;
				prev_j = j;
				j = p[j];
			}
		}
	}

	void reorder(const Graph<DataType>& graph, const std::vector<uint>& permutation)
	{
		cusolverSpHandle_t handle;
		auto solver_status = cusolverSpCreate(&handle);
		assert(solver_status == CUSOLVER_STATUS_SUCCESS);

		cusparseMatDescr_t descrA = nullptr;
		auto sparse_status = cusparseCreateMatDescr(&descrA);
		assert(sparse_status == CUSPARSE_STATUS_SUCCESS);

		sparse_status = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		assert(sparse_status == CUSPARSE_STATUS_SUCCESS);
		sparse_status = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		assert(sparse_status == CUSPARSE_STATUS_SUCCESS);

		size_t buffer_size;
		solver_status = cusolverSpXcsrperm_bufferSizeHost(handle, graph.size, graph.size, graph.csr.nnz, descrA, (int*)graph.csr.row_offsets.get(),
			(int*)graph.csr.col_ids.get(), (int*)permutation.data(), (int*)permutation.data(), &buffer_size);
		assert(solver_status == CUSOLVER_STATUS_SUCCESS);

		void* buffer = malloc(buffer_size);
		std::vector<int> map(graph.csr.nnz);
		std::iota(std::begin(map), std::end(map), 0);

		solver_status = cusolverSpXcsrpermHost(handle, graph.size, graph.size, graph.csr.nnz, descrA,
			(int*)graph.csr.row_offsets.get(), (int*)graph.csr.col_ids.get(), (int*)permutation.data(), (int*)permutation.data(), map.data(), buffer);
		assert(solver_status == CUSOLVER_STATUS_SUCCESS);

		free(buffer);
		
		apply_permutation_in_place(graph.csr.data.get(), map, 0, graph.csr.nnz);
	}
}
