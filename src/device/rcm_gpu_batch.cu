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

#include "rcm_gpu_batch.h"

#include <iomanip>
#include <unordered_map>
#include <iostream>

#include "batchWorker.cuh"

namespace GPU_BATCH
{

	template<typename T>
	void rcm(Graph<T>& graph, std::ofstream& csv_file, std::vector<unsigned>& permutation, unsigned start_node)
	{
		using namespace RCM_DYNAMIC;

		uint32_t warmups = 20;
		uint32_t runs = 20;

		uint32_t* drow_offsets, * dcol_ids, * dpermutation, * dvisited;
		uint32_t num_nodes = graph.csr.rows;

		cudaEvent_t events[2];
		for (auto& e : events)
			cudaEventCreate(&e);

		cudaMalloc(&drow_offsets, sizeof(uint32_t) * (num_nodes + 1));
		cudaMalloc(&dcol_ids, sizeof(uint32_t) * graph.csr.nnz);
		cudaMalloc(&dpermutation, sizeof(uint32_t) * num_nodes);
		cudaMalloc(&dvisited, sizeof(uint32_t) * num_nodes);
		cudaMemset(dpermutation, 0xFFFFFFFF, sizeof(uint32_t) * num_nodes);

		cudaMemcpy(drow_offsets, graph.csr.row_offsets.get(), (num_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dcol_ids, graph.csr.col_ids.get(), graph.csr.nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);

		size_t dynsmem = sizeof(TempData) * MaxTemps;
		int dev;
		cudaGetDevice(&dev);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev);
		int numBlocks;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, run, BlockSize, dynsmem);

		numBlocks *= prop.multiProcessorCount;

		double sumt = 0;
		for (int r = 0; r < warmups + runs; ++r)
		{
			init <<<128, BlockSize >>> (num_nodes, drow_offsets, dcol_ids, dpermutation, dvisited, start_node);

			cudaEventRecord(events[0]);
			run <<<numBlocks, BlockSize, dynsmem, 0>>> (numBlocks);
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
		std::cout << std::setw(27) << std::right << "GPU-DYN-RCM" << " duration: " << sumt << " ms" << std::endl;
		csv_file << sumt;
		csv_file << std::flush;

		permutation.resize(num_nodes);
		cudaMemcpy(&permutation[0], dpermutation, sizeof(uint32_t) * num_nodes, cudaMemcpyDeviceToHost);


		for (auto& e : events)
			cudaEventDestroy(e);

		cudaFree(drow_offsets);
		cudaFree(dcol_ids);
		cudaFree(dpermutation);
		cudaFree(dvisited);
	}

	unsigned int findMaxDistance(uint32_t num_nodes, uint32_t* drow_offsets, uint32_t* row_offsets, uint32_t* dcol_ids, unsigned int& start_node, uint32_t* dvisited, uint32_t* dpermutation, uint32_t numBlocks, uint32_t BlockSize, uint32_t dynsmem)
	{
		using namespace RCM_DYNAMIC;

		init <<<128, BlockSize >>> (num_nodes, drow_offsets, dcol_ids, dpermutation, dvisited, start_node);

		run <<<numBlocks, BlockSize, dynsmem, 0>>> (numBlocks);

		std::vector<int> distances(num_nodes);
		cudaMemcpy(distances.data(), dvisited, sizeof(unsigned int) * num_nodes, cudaMemcpyDeviceToHost);
		auto it = std::max_element(distances.begin(), distances.end());
		std::vector<unsigned int> possible_nodes;
		std::vector<unsigned int> corresponding_valence;
		for(auto i = 0; i < distances.size(); ++i)
		{
			if(distances[i] == *it)
			{
				possible_nodes.push_back(i);
				corresponding_valence.push_back(row_offsets[i + 1] - row_offsets[i]);
			}
		}
		auto min_val = std::min_element(corresponding_valence.begin(), corresponding_valence.end());
		auto index = min_val - corresponding_valence.begin();
		start_node = possible_nodes[index];
		return *it;
	}

	template<typename T>
	void findPeripheral(Graph<T>& graph, std::ofstream& csv_file, bool writeOutput)
	{
		static constexpr bool test_gpu_peripheral{false};
		static constexpr int test_iter{(test_gpu_peripheral) ? 20 : 1};
		using namespace RCM_DYNAMIC;

		uint32_t* drow_offsets, * dcol_ids, * dpermutation, * dvisited;
		uint32_t num_nodes = graph.csr.rows;

		cudaEvent_t events[2];
		for (auto& e : events)
			cudaEventCreate(&e);

		cudaMalloc(&drow_offsets, sizeof(uint32_t) * (num_nodes + 1));
		cudaMalloc(&dcol_ids, sizeof(uint32_t) * graph.csr.nnz);
		cudaMalloc(&dpermutation, sizeof(uint32_t) * num_nodes);
		cudaMalloc(&dvisited, sizeof(uint32_t) * num_nodes);

		cudaMemcpy(drow_offsets, graph.csr.row_offsets.get(), (num_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dcol_ids, graph.csr.col_ids.get(), graph.csr.nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);

		size_t dynsmem = sizeof(TempData) * MaxTemps;
		int dev;
		cudaGetDevice(&dev);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev);
		int numBlocks;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, run, BlockSize, dynsmem);

		numBlocks *= prop.multiProcessorCount;

		
		uint32_t start_node = graph.csr.rows / 2;
		auto max_distance{0U};
		float timing{0.0f};
		int num_iter{0};
		for(auto iter = 0; iter < test_iter; ++iter)
		{
			cudaEventRecord(events[0]);
			cudaMemset(dpermutation, 0xFFFFFFFF, sizeof(uint32_t) * num_nodes);
			for(auto i = 1; ; ++i)
			{
				auto new_distance = findMaxDistance(num_nodes, drow_offsets, graph.csr.row_offsets.get(), dcol_ids, start_node, dvisited, dpermutation, numBlocks, BlockSize, dynsmem);
				std::cout << "Iter " << i << " with startnode " << start_node << " and max distance " << new_distance << std::endl;
				if (new_distance == max_distance)
				{
					cudaEventRecord(events[1]);
					cudaEventSynchronize(events[1]);
					float t;
					cudaEventElapsedTime(&t, events[0], events[1]);
					timing += t;
					num_iter = i;
					
					break;
				}
				max_distance = new_distance;
			}
		}
		timing /= test_iter;
		std::string approach(std::string("Opt-GPU-Peripheral<") + std::to_string(num_iter) + std::string(", ") + std::to_string(start_node) + std::string(", ") + std::to_string(max_distance) + std::string(">"));
		std::cout << std::setw(27) << std::right << approach << " duration: " << timing << " ms" << std::endl;
		if(writeOutput)
		{
			csv_file << "," << timing;
			csv_file << std::flush;
		}

		for (auto& e : events)
			cudaEventDestroy(e);

		cudaFree(drow_offsets);
		cudaFree(dcol_ids);
		cudaFree(dpermutation);
		cudaFree(dvisited);
	}

	

	template void rcm(Graph<float>& graph, std::ofstream& csv_file, std::vector<unsigned>& permutation, unsigned start_node);
	template void rcm(Graph<double>& graph, std::ofstream& csv_file, std::vector<unsigned>& permutation, unsigned start_node);
	template void findPeripheral(Graph<float>& graph, std::ofstream& csv_file, bool writeOutput);
	template void findPeripheral(Graph<double>& graph, std::ofstream& csv_file, bool writeOutput);
}