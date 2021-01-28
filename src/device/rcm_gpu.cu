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

#include "rcm_gpu.h"
#include "handle_error.h"
#include <iomanip>
#include <iostream>
#ifdef _WIN32
#include <intrin.h>
#endif

#include <cub/cub.cuh>

namespace GPU_SIMPLE
{
	static constexpr bool WarpBased{false};
	static constexpr int WARPSIZE{32};
	static constexpr bool PrintDebug{false};

	__device__ bool workFinished{false};

	constexpr __host__ __device__ __forceinline__ unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1) / b;
	}

	struct DeviceGraph
	{
		unsigned int num_nodes;
		unsigned int* offset;
		unsigned int* col_ids;
	};

	struct Helper
	{
		unsigned int* amountOfWork_;

		// Queue
		unsigned int front_;
		unsigned int count_;
		unsigned int* visited_;

		// Permutation
		unsigned int* permutation_;
	};

	struct BitEncoding
	{
		unsigned int NumBitsValence{0U};
		unsigned int NumBitsNodes{0U};

		__forceinline__ __device__ unsigned int childValenceShift()
		{
			return NumBitsNodes;
		}

		__forceinline__ __device__ unsigned int parentPosShift()
		{
			return NumBitsNodes + NumBitsValence;
		}

		static int getNextPow2Pow(unsigned int n)
		{
			#ifndef _WIN32
			if ((n & (n - 1)) == 0)
				return 32 - __builtin_clz(n) - 1;
			else
				return 32 - __builtin_clz(n);
			#else
			if ((n & (n - 1)) == 0)
				return 32 - __lzcnt(n) - 1;
			else
				return 32 - __lzcnt(n);
			#endif
		}

		static unsigned int numBits(unsigned int n)
		{
			return getNextPow2Pow(n);
		}
	};

	template <typename SortDataType>
	struct SortHelper;

	template <>
	struct SortHelper<uint32_t>
	{
		static constexpr uint32_t max_value = 0xFFFFFFFF;
		static constexpr __device__ uint32_t getChildValue(uint32_t key, unsigned int NumBitsNodes)
		{
			return key & ((1U << NumBitsNodes) - 1U);
		}
	};

	template <>
	struct SortHelper<uint64_t>
	{
		static constexpr uint64_t max_value = 0xFFFFFFFFFFFFFFFF;
		static constexpr __device__ uint64_t getChildValue(uint64_t key, unsigned int NumBitsNodes)
		{
			return key & ((1ULL << NumBitsNodes) - 1ULL);
		}
	};

	// Compute how much work is to be done for each node in the queue currently
	__global__ void amountofWork(const DeviceGraph d_graph, const Helper* helper)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid < helper->count_)
		{
			auto parent_node = helper->permutation_[helper->front_ + tid];
			auto offset = d_graph.offset[parent_node];
			auto neighbours = d_graph.offset[parent_node + 1] - offset;
			helper->amountOfWork_[tid] = neighbours;
		}
		else
		{
			if(tid < d_graph.num_nodes)
				helper->amountOfWork_[tid] = 0;
		}
	}

	// Write the sort keys (consisting of parent pos | child valence | child ID), set visited status for each node if atomicMin goes through
	template <typename SortDataType>
	__global__ void writeKeys(const DeviceGraph d_graph, Helper* helper, SortDataType* __restrict sort_keys, BitEncoding* bit_encoding)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= d_graph.num_nodes)
			return;
		
		auto key_offset = helper->amountOfWork_[tid];
		auto numEntries = helper->amountOfWork_[tid + 1] - key_offset;
		auto queue_pos = helper->front_ + tid;
		auto parent_node = helper->permutation_[queue_pos];
		auto node_offset = d_graph.offset[parent_node];
		// Go over child nodes for given parent node
		for(auto i = 0; i < numEntries; ++i)
		{
			auto child_node = d_graph.col_ids[node_offset + i];
			
			auto res = atomicMin(&helper->visited_[child_node], queue_pos);
			if(queue_pos >= res)
			{
				// Someone else already added this key
				sort_keys[key_offset + i] = SortHelper<SortDataType>::max_value;
				continue;
			}
			
			// We want to include this node, key consists of parent ID (here we simply take tid), child valency and the child node info
			auto child_num_neighbours = d_graph.offset[child_node + 1] - d_graph.offset[child_node];
			sort_keys[key_offset + i] = (tid << bit_encoding->parentPosShift()) + (child_num_neighbours << bit_encoding->childValenceShift()) + child_node;
		}
	}

	// Write the sort keys (consisting of parent pos | child valence | child ID), set visited status for each node if atomicMin goes through
	template <typename SortDataType>
	__global__ void writeKeysWarp(const DeviceGraph d_graph, Helper* helper, SortDataType* __restrict sort_keys, BitEncoding* bit_encoding)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		tid /= WARPSIZE;
		if(tid >= d_graph.num_nodes)
			return;
		
		auto key_offset = helper->amountOfWork_[tid];
		auto numEntries = helper->amountOfWork_[tid + 1] - key_offset;
		auto queue_pos = helper->front_ + tid;
		auto parent_node = helper->permutation_[queue_pos];
		auto node_offset = d_graph.offset[parent_node];
		// Go over child nodes for given parent node
		for(auto i = threadIdx.x % WARPSIZE; i < numEntries; i += WARPSIZE)
		{
			auto child_node = d_graph.col_ids[node_offset + i];
			
			auto res = atomicMin(&helper->visited_[child_node], queue_pos);
			if(queue_pos >= res)
			{
				// Someone else already added this key
				sort_keys[key_offset + i] = SortHelper<SortDataType>::max_value;
				continue;
			}
			
			// We want to include this node, key consists of parent ID (here we simply take tid), child valency and the child node info
			auto child_num_neighbours = d_graph.offset[child_node + 1] - d_graph.offset[child_node];
			sort_keys[key_offset + i] = (tid << bit_encoding->parentPosShift()) + (child_num_neighbours << bit_encoding->childValenceShift()) + child_node;
		}
	}

	// The atomicMin might still have let through a few nodes which are already handled at this level, delete them
	template <typename SortDataType>
	__global__ void verifyKeys(const DeviceGraph d_graph, Helper* helper, SortDataType* __restrict sort_keys, BitEncoding* bit_encoding)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= d_graph.num_nodes)
			return;
		
		auto key_offset = helper->amountOfWork_[tid];
		auto numEntries = helper->amountOfWork_[tid + 1] - key_offset;
		for(auto i = 0; i < numEntries; ++i)
		{
			auto key = sort_keys[key_offset + i];
			if(key == SortHelper<SortDataType>::max_value)
				continue;
			auto child_node = SortHelper<SortDataType>::getChildValue(key, bit_encoding->NumBitsNodes);
			if(helper->visited_[child_node] != helper->front_ + tid)
			{
				// Delete this key, as someone else should add this which comes before in the insertion order
				sort_keys[key_offset + i] = SortHelper<SortDataType>::max_value;
			}
		}
	}

	// The atomicMin might still have let through a few nodes which are already handled at this level, delete them
	template <typename SortDataType>
	__global__ void verifyKeysWarp(const DeviceGraph d_graph, Helper* helper, SortDataType* __restrict sort_keys, BitEncoding* bit_encoding)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		tid /= WARPSIZE;
		if(tid >= d_graph.num_nodes)
			return;
		
		auto key_offset = helper->amountOfWork_[tid];
		auto numEntries = helper->amountOfWork_[tid + 1] - key_offset;
		for(auto i = threadIdx.x % WARPSIZE; i < numEntries; i += WARPSIZE)
		{
			auto key = sort_keys[key_offset + i];
			if(key == SortHelper<SortDataType>::max_value)
				continue;
			auto child_node = SortHelper<SortDataType>::getChildValue(key, bit_encoding->NumBitsNodes);
			if(helper->visited_[child_node] != helper->front_ + tid)
			{
				// Delete this key, as someone else should add this which comes before in the insertion order
				sort_keys[key_offset + i] = SortHelper<SortDataType>::max_value;
			}
		}
	}

	// Write the new nodes to the queue and permutation, if the first threads does not get any work, we are done!
	template <typename SortDataType>
	__global__ void writePermutationAndQueue(const DeviceGraph d_graph, Helper* helper, SortDataType* __restrict sort_keys, BitEncoding* bit_encoding, const unsigned int num_new_nodes)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_new_nodes)
			return;

		auto key = sort_keys[tid];
		if(key == SortHelper<SortDataType>::max_value)
		{
			// If even the first did not get anything, then we are done!
			if(tid == 0)
			{
				if(PrintDebug)
					printf("We are done!\n");
				workFinished = true;
			}
				
			return;
		}

		if(PrintDebug)
		{
			if(sort_keys[tid + 1] == SortHelper<SortDataType>::max_value)
				printf("We actually integrated %u new nodes!\n", tid + 1);
		}
		
		// Increase queue count
		atomicAdd(&helper->count_, 1);
		auto child_node = SortHelper<SortDataType>::getChildValue(key, bit_encoding->NumBitsNodes);
		helper->permutation_[helper->front_ + tid] = child_node;
	}

	// Update the queue params and clean the visited array
	__global__ void updateQueue(const DeviceGraph d_graph, Helper* helper)
	{
		// Update Queue params
		if(threadIdx.x + blockIdx.x * blockDim.x == 0)
		{
			helper->front_ += helper->count_;
			helper->count_ = 0U;
		}
	}

	template <typename SortDataType>
	void run(DeviceGraph d_graph, unsigned int start_node, Helper* helper, unsigned int* amount_of_work, BitEncoding* bit_encoding)
	{
		void *d_temp_storage{nullptr};
		size_t   temp_storage_bytes{0};
		SortDataType* sort_keys_in{nullptr};
		SortDataType* sort_keys_out{nullptr};

		unsigned int blockSize{256};
		unsigned int gridSize;

		bool workDone{false};
		HANDLE_ERROR(cudaMemcpyToSymbol(workFinished, &workDone, sizeof(bool)));
		auto bfs_level{0U};
		while(!workDone)
		{
			if(PrintDebug)
				std::cout << "--------------------------------------\nBFS-Level: " << bfs_level++ << std::endl;
			// Compute how much work we have to do on this level
			gridSize = divup(d_graph.num_nodes, blockSize);
			amountofWork <<<gridSize, blockSize>>> (d_graph, helper);

			// ------------
			// ------------ CUB Exclusive SUM
			// ------------
			HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, amount_of_work, amount_of_work, d_graph.num_nodes + 1));
			HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, amount_of_work, amount_of_work, d_graph.num_nodes + 1));
			HANDLE_ERROR(cudaFree(d_temp_storage)); d_temp_storage = nullptr; temp_storage_bytes = 0;
			// ------------
			// ------------ CUB Exclusive SUM
			// ------------

			// How much new nodes are there
			auto num_new_nodes{1U};
			cudaMemcpy(&num_new_nodes, &amount_of_work[d_graph.num_nodes], sizeof(unsigned int), cudaMemcpyDeviceToHost);
			if(PrintDebug)
				std::cout << "Number of new nodes discovered: " << num_new_nodes << std::endl;

			// Allocate sort keys
			cudaMalloc(&sort_keys_in, sizeof(SortDataType) * num_new_nodes);
			cudaMalloc(&sort_keys_out, sizeof(SortDataType) * num_new_nodes);

			// Write sort keys
			if(WarpBased)
				writeKeysWarp<SortDataType><<<gridSize * WARPSIZE, blockSize>>>(d_graph, helper, sort_keys_in, bit_encoding);
			else
				writeKeys<SortDataType><<<gridSize, blockSize>>>(d_graph, helper, sort_keys_in, bit_encoding);

			// Delete duplicates in sort keys
			if(WarpBased)
				verifyKeysWarp<SortDataType><<<gridSize * WARPSIZE, blockSize>>>(d_graph, helper, sort_keys_in, bit_encoding);
			else
				verifyKeys<SortDataType><<<gridSize, blockSize>>>(d_graph, helper, sort_keys_in, bit_encoding);

			// ------------
			// ------------ CUB SORT
			// ------------
			HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, sort_keys_in, sort_keys_out, num_new_nodes));
			HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, sort_keys_in, sort_keys_out, num_new_nodes));
			HANDLE_ERROR(cudaFree(d_temp_storage)); d_temp_storage = nullptr; temp_storage_bytes = 0;
			// ------------
			// ------------ CUB SORT
			// ------------

			// Update queue parameters and clean the visitied status (all non inf and non 0 -> 0)
			updateQueue/*AndCleanVisited*/ <<<1, 1>>>(d_graph, helper);

			// Write new nodes to queue and permutation
			gridSize = divup(num_new_nodes, blockSize);
			writePermutationAndQueue<SortDataType><<<gridSize, blockSize>>>(d_graph, helper, sort_keys_out, bit_encoding, num_new_nodes);

			// Free memory
			HANDLE_ERROR(cudaFree(sort_keys_in));
			HANDLE_ERROR(cudaFree(sort_keys_out));

			// Check if there is work left
			HANDLE_ERROR(cudaMemcpyFromSymbol(&workDone, workFinished, sizeof(bool)));
		}
	}

	__global__ void initArrays(unsigned int* __restrict visited, unsigned int num_nodes, unsigned int start_node)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_nodes)
			return;
		visited[tid] = (tid == start_node) ? 0U : 0xFFFFFFFF;
	}

	template<typename T>
	void rcm(Graph<T>& graph, std::ofstream& csv_result, std::vector<unsigned>& permutation, unsigned start_node)
	{
		uint32_t warmups = 5;
		uint32_t runs = 20;

		uint32_t num_nodes = graph.csr.rows;

		cudaEvent_t events[2];
		for (auto& e : events)
			cudaEventCreate(&e);

		// Allocate simple graph on device
		DeviceGraph d_graph;
		d_graph.num_nodes = num_nodes;
		HANDLE_ERROR(cudaMalloc(&(d_graph.offset), sizeof(uint32_t) * (num_nodes + 1)));
		HANDLE_ERROR(cudaMalloc(&(d_graph.col_ids), sizeof(uint32_t) * graph.csr.nnz));
		HANDLE_ERROR(cudaMemcpy(d_graph.offset, graph.csr.row_offsets.get(), (num_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_graph.col_ids, graph.csr.col_ids.get(), graph.csr.nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));

		uint32_t maxval = 0;
		for (uint32_t r = 0; r < graph.csr.rows; ++r)
			maxval = std::max(maxval, graph.csr.row_offsets[r + 1] - graph.csr.row_offsets[r]);

		// Setup up bitencoding for sort key
		auto numBitsValence = BitEncoding::numBits(maxval);
		auto numBitsNodes = BitEncoding::numBits(num_nodes);
		BitEncoding bit_encoding{numBitsValence, numBitsNodes};
		BitEncoding* d_bit_encoding{nullptr};
		HANDLE_ERROR(cudaMalloc(&d_bit_encoding, sizeof(BitEncoding)));
		HANDLE_ERROR(cudaMemcpy(d_bit_encoding, &bit_encoding, sizeof(BitEncoding), cudaMemcpyHostToDevice));
		auto numBitsRequired = numBitsValence + 2*numBitsNodes;

		// Setup bfs queue with the maximum size possible per iteration, which the number of nodes -> round of to next power of 2 for correct wrap around (probably not needed)
		Helper bfs_queue;		
		HANDLE_ERROR(cudaMalloc(&bfs_queue.visited_, sizeof(uint32_t) * num_nodes));
		HANDLE_ERROR(cudaMalloc(&bfs_queue.permutation_, sizeof(uint32_t) * num_nodes));
		HANDLE_ERROR(cudaMalloc(&bfs_queue.amountOfWork_, sizeof(uint32_t) * (num_nodes + 1)));
		Helper* d_helper{nullptr};
		HANDLE_ERROR(cudaMalloc(&d_helper, sizeof(Helper)));

		unsigned int blockSize{256};
		unsigned int gridSize{divup(d_graph.num_nodes, blockSize)};

		double sumt = 0;
		for (int r = 0; r < warmups + runs; ++r)
		{
			cudaEventRecord(events[0]);
			bfs_queue.front_ = 0U;
			bfs_queue.count_ = 1U;
			HANDLE_ERROR(cudaMemcpy(d_helper, &bfs_queue, sizeof(Helper), cudaMemcpyHostToDevice));

			// Place first element in permutation and init visited
			HANDLE_ERROR(cudaMemcpy(bfs_queue.permutation_, &start_node, sizeof(start_node), cudaMemcpyHostToDevice));
			initArrays<<<gridSize, blockSize>>>(bfs_queue.visited_, num_nodes, start_node);
			
			if(numBitsRequired < 32)
				run<uint32_t>(d_graph, start_node, d_helper, bfs_queue.amountOfWork_, d_bit_encoding);
			else if(numBitsRequired < 64)
				run<uint64_t>(d_graph, start_node, d_helper, bfs_queue.amountOfWork_, d_bit_encoding);
			else
			{
				std::cout << "Do not have enough bits for sorting as of now! Abort!" << std::endl; 
				exit(-1);
			}
			cudaEventRecord(events[1]);
			cudaError_t err = cudaEventSynchronize(events[1]);
			if (err != cudaSuccess)
				throw std::runtime_error("CUDA ERROR");
			if (r >= warmups)
			{
				float t;
				cudaEventElapsedTime(&t, events[0], events[1]);
				sumt += t;
			}
		}
		sumt /= runs;
		std::cout << std::setw(27) << std::right << "GPUSimple-RCM" << " duration: " << sumt << " ms" << std::endl;
		csv_result << sumt;
		csv_result << std::flush;

		permutation.resize(num_nodes);
		HANDLE_ERROR(cudaMemcpy(permutation.data(), bfs_queue.permutation_, sizeof(uint32_t) * num_nodes, cudaMemcpyDeviceToHost));

		for (auto& e : events)
			cudaEventDestroy(e);

		HANDLE_ERROR(cudaFree(d_graph.offset));
		HANDLE_ERROR(cudaFree(d_graph.col_ids));
		HANDLE_ERROR(cudaFree(bfs_queue.permutation_));
		HANDLE_ERROR(cudaFree(bfs_queue.visited_));
		HANDLE_ERROR(cudaFree(bfs_queue.amountOfWork_));
		HANDLE_ERROR(cudaFree(d_helper));
		HANDLE_ERROR(cudaFree(d_bit_encoding));
	}

	
	template void rcm(Graph<float>& graph, std::ofstream& csv_result, std::vector<unsigned>& permutation, unsigned start_node);
	template void rcm(Graph<double>& graph, std::ofstream& csv_result, std::vector<unsigned>& permutation, unsigned start_node);
}