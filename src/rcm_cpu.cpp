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

#include "rcm_cpu.h"

#include <queue>
#include <memory>
#include <tuple>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "workqueue.h"
#include "batchWorker.h"

namespace RealCPU
{
	template<typename A, typename B>
	struct KeyValue
	{
		A key;
		B value;
		bool operator < (const KeyValue<A, B>& other) const
		{
			return key < other.key;
		}
	};
	template<bool STABLE, typename T>
	void rcm_seq(Graph<T>& graph, std::vector<unsigned>& permutation, unsigned start_node)
	{
		permutation.clear();
		permutation.reserve(graph.csr.rows);

		std::vector<uint32_t> visited(graph.csr.rows, 0xFFFFFFFF);
		visited[start_node] = 0;

		std::queue<uint32_t> q;
		q.emplace(start_node);
		permutation.push_back(start_node);

		std::vector<KeyValue<uint32_t, uint32_t>> temp;
		temp.reserve(graph.csr.rows);

		while (!q.empty())
		{
			uint32_t v = q.front();
			q.pop();

			temp.clear();

			const auto* p = &graph.csr.col_ids[graph.csr.row_offsets[v]];
			const auto* pend = &graph.csr.col_ids[graph.csr.row_offsets[v + 1]];
			
			for (; p != pend; ++p)
			{
				uint32_t n = *p;
				if (visited[n] == 0xFFFFFFFF)
				{
					visited[n] = 1;
					uint32_t valence = graph.neighbour_count(n);
					temp.push_back({ valence, n });
				}
			}

			if(STABLE)
				std::stable_sort(begin(temp), end(temp));
			else
				std::sort(begin(temp), end(temp));
			for (auto& n : temp)
			{
				permutation.emplace_back(n.value);
				q.push(n.value);
			}
		}
	}





	template<bool STABLE_SORT, typename T>
	void rcm(Graph<T>& graph, std::ostream& csv_file, std::vector<unsigned>& permutation, unsigned start_node, unsigned num_threads, unsigned batchsize, const std::string& writequeue)
	{
		static constexpr bool test_realcpu{true};
		static constexpr int test_iter{test_realcpu ? 25 : 1};
		std::vector<float> timings(test_iter);
		decltype(std::chrono::high_resolution_clock::now()) t0, t1;
		if (num_threads == 1 && batchsize == 1)
		{
			float timing{0.0f};
			for(auto iter = 0; iter < test_iter; ++iter)
			{
				t0 = std::chrono::high_resolution_clock::now();
				rcm_seq<STABLE_SORT>(graph, permutation, start_node);
				t1 = std::chrono::high_resolution_clock::now();
				timings[iter] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000000.0;
				timing += timings[iter];
			}

			// if std::nth_element was a thing already...
			std::sort(std::begin(timings), std::end(timings), std::less<float>());
			float min_timing = *std::begin(timings);
			float max_timing = *std::rbegin(timings);
			static_assert(test_iter % 2 == 1, "Median computation not correct for even iteration counts\n");
			float med_timing = timings[test_iter / 2];
			

			timing /= test_iter;
			std::string approach(std::string("RealCPU<") + std::string((STABLE_SORT ? "STABLE" : "NON_STABLE")) + std::string(", single>"));
			std::cout << std::setw(27) << std::right << approach << " duration: " << timing << " ms" 
			<< " (" << min_timing << " / " << med_timing << " / " << max_timing << ")" << std::endl;
			csv_file << timing;
			csv_file << std::flush;
		}
		else
		{
			using BatchQueue = ::BatchQueue<>;
			auto queue = std::make_unique<BatchQueue>();
			float timing{0.0f};
			for(auto iter = 0; iter < test_iter; ++iter)
			{
				queue->init();

				permutation.resize(graph.csr.rows);
				AlgorithmState state(queue.get(), graph.csr, &permutation[0]);
				state.MaxInputNodes = batchsize;
				state.MaxWorkSize = 32 * batchsize;

				std::vector<std::thread> threads;
				threads.reserve(num_threads);
				for (uint32_t i = 0; i < num_threads; ++i)
					threads.emplace_back(std::thread(BatchWorker::run<STABLE_SORT, BatchQueue::Size>, &state, queue.get()));

				while (state.workerCount.load() != num_threads)
					std::this_thread::yield();
				t0 = std::chrono::high_resolution_clock::now();
				BatchWorker::start(&state, queue.get(), start_node);
				{
					std::unique_lock<std::mutex> l(state.mutex);
					state.finishWaiter.wait(l, [&]() { return !state.running; });
				}

				t1 = std::chrono::high_resolution_clock::now();
				timings[iter] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000000.0;
				timing += timings[iter];

				// wait for workers to return
				for (auto& t : threads)
					t.join();
			}

			// if std::nth_element was a thing already...
			std::sort(std::begin(timings), std::end(timings), std::less<float>());
			float min_timing = *std::begin(timings);
			float max_timing = *std::rbegin(timings);
			static_assert(test_iter % 2 == 1, "Median computation not correct for even iteration counts\n");
			float med_timing = timings[test_iter / 2];

			timing /= test_iter;
			std::string approach(std::string("RealCPU<") + std::string((STABLE_SORT ? "STABLE" : "NON_STABLE")) + std::string(", ") +  std::to_string(num_threads) + std::string(", ") +  std::to_string(batchsize) + std::string(">"));
			std::cout << std::setw(27) << std::right << approach << " duration: " << timing << " ms" 
			<< " (" << min_timing << " / " << med_timing << " / " << max_timing << ")" << std::endl;
			csv_file << timing;
			csv_file << std::flush;
			if (writequeue != "")
			{
				std::ofstream qf(writequeue.c_str());
				for(size_t i = 0; i < BatchQueue::Size; ++i)
				{
					auto& b = queue->at(i);
					if (b.consumestate != 0 || b.state != 0)
					{
						qf << i << "\n" 
						   << "  state: " << b.state << "\n"
						   << "  consumestate: " << b.consumestate << "\n"
						   << "  input_start: " << b.input_start << "\n"
						   << "  input_end: " << b.input_end << "\n"
						   << "  offset: " << b.offset << "\n"
						   << "  followupStartBatchId: " << b.followupStartBatchId << "\n"
						   << "  combine: " << b.combine << "\n"
						   << "  combineOutputs: " << b.combineOutputs << "\n";
					}
				}
			}
		}
	}

	template void rcm<true>(Graph<float>& graph, std::ostream& csv_file, std::vector<unsigned>& permutation, unsigned start_node, unsigned threads, unsigned batchsize, const std::string& writequeue);

	template void rcm<false>(Graph<float>& graph,std::ostream& csv_file, std::vector<unsigned>& permutation, unsigned start_node, unsigned threads, unsigned batchsize, const std::string& writequeue);
}
