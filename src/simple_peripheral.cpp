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

#include "simple_peripheral.h"

#include <queue>
#include <tuple>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>

#include "workqueue.h"

namespace SimplePeripheral
{
	template<typename T>
	void findMaxDistance(const Graph<T>& graph, unsigned start_node, std::vector<uint32_t>& nodes, std::vector<uint32_t>& distances, uint32_t outputnodes)
	{
		thread_local std::vector<uint32_t> visited;
		visited.resize(graph.csr.rows);
		std::fill_n(begin(visited), visited.size(), 0xFFFFFFFF);
		size_t to_visit = graph.csr.rows - 1;

		std::queue<std::tuple<uint32_t, uint32_t>> q;
		q.emplace(start_node, 0U);
		visited[start_node] = 0U;

		distances.clear();
		nodes.clear();
		distances.reserve(outputnodes);
		nodes.reserve(outputnodes);

		while (!q.empty())
		{
			auto [v, dist] = q.front();
			q.pop();

			const auto* p = &graph.csr.col_ids[graph.csr.row_offsets[v]];
			const auto* pend = &graph.csr.col_ids[graph.csr.row_offsets[v + 1]];
			uint32_t n_dist = dist + 1;

			for (; p != pend; ++p)
			{
				uint32_t n = *p;
				if (visited[n] == 0xFFFFFFFF)
				{
					visited[n] = n_dist;
					q.emplace(n, n_dist);
					if (--to_visit < outputnodes)
					{
						nodes.push_back(n);
						distances.push_back(n_dist);
					}
				}
			}
		}
		if (to_visit > 0)
		{
			std::cout << "not fully connected graph " << graph.csr.rows - to_visit << " nodes in first cluster" << std::endl;
			throw std::runtime_error("not fully connected graph");
		}
	}

	template<typename T>
	uint32_t findPeripheral(const Graph<T>& graph, std::ostream& csv_file, bool writeOutput, uint32_t outputnodes)
	{
		static constexpr bool test_cpu_peripheral{ false };
		static constexpr int test_iter{ (test_cpu_peripheral) ? 20 : 1 };
		std::vector<uint32_t> nodes, distances;
		uint32_t lastdist = 0;
		uint32_t n = graph.csr.rows / 2;
		float timing{ 0.0f };
		int num_iter{ 0 };
		uint32_t end_node;
		for (auto iter = 0; iter < test_iter; ++iter)
		{
			auto t0 = std::chrono::high_resolution_clock::now();
			for (size_t i = 1; ; ++i)
			{
				findMaxDistance(graph, n, nodes, distances, outputnodes);
				if (distances.back() == lastdist)
				{
					auto t1 = std::chrono::high_resolution_clock::now();
					timing += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000000.0;
					num_iter = i;
					end_node = nodes.back();
					break;
				}
				// Take the element with largest distance but smallest valence
				auto it = std::max_element(distances.begin(), distances.end()); // Points to first of the max distances
				auto index = it - distances.begin(); // Get index of this
				int minvalence{ std::numeric_limits<int>::max() };
				for (size_t i = index; i < nodes.size(); ++i)
				{
					auto node = nodes[i];
					int valence = static_cast<int>(graph.neighbour_count(node));
					if ((minvalence = std::min(minvalence, valence)) == valence)
						n = node;
				}
				lastdist = distances.back();
			}
		}
		timing /= test_iter;
		std::string approach(std::string("Peripheral<") + std::to_string(num_iter) + std::string(", ") + std::to_string(n) + std::string(", ") + std::to_string(end_node) + std::string(", ") + std::to_string(lastdist) + std::string(">"));
		std::cout << std::setw(27) << std::right << approach << " duration: " << timing << " ms" << std::endl;
		if (writeOutput)
		{
			csv_file << "," << timing;
			csv_file << std::flush;
		}

		return n;
	}

	template uint32_t findPeripheral(const Graph<float>& graph, std::ostream& csv_file, bool writeOutput, uint32_t outputnodes);
	template uint32_t findPeripheral(const Graph<double>& graph, std::ostream& csv_file, bool writeOutput, uint32_t outputnodes);

}