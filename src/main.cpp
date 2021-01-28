
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

#include <vector>
#include <numeric>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "args.h"

#include "CSR.h"
#include "Graph.h"
#include "dGraph.h"


#include "simple_peripheral.h"
#include "rcm_cpu.h"
#include "rcm_cusolver.h"

#include "rcm_gpu.h"
#include "rcm_gpu_batch.h"


namespace
{

	struct voidStream : std::ostream
	{
		struct voidBuff : std::streambuf
		{
			int overflow(int) override { return -1; };
		} buffer;

		voidStream()
			: std::ostream(&buffer)
		{}
	};
}

using DataType = float;

std::string join(const std::vector<std::string>& pieces)
{
	std::string joined = "{";
	for (const auto& piece : pieces)
		joined += piece + ", ";

	joined.pop_back();
	joined.pop_back();

	joined += "}";
	return joined;
}

struct BandwidthMetric
{
	unsigned int min_row_bw{0U};
	unsigned int max_row_bw{0U};
	unsigned int avg_row_bw{0U};
	unsigned int median_row_bw{0U};
};

BandwidthMetric computeBandwidthMetric(const Graph<DataType>& graph)
{
	BandwidthMetric metric;
	auto& matrix = graph.csr;
	std::vector<unsigned int> row_bw;
	row_bw.reserve(matrix.rows);

	for(size_t i = 0; i < matrix.rows; ++i)
	{
		row_bw.push_back(matrix.col_ids[matrix.row_offsets[i + 1]- 1] - matrix.col_ids[matrix.row_offsets[i]]);
	}
	std::sort(row_bw.begin(), row_bw.end());
	metric.min_row_bw = *std::min_element(row_bw.begin(), row_bw.end());
	metric.max_row_bw = *std::max_element(row_bw.begin(), row_bw.end());
	metric.median_row_bw = row_bw[row_bw.size()/2];
	metric.avg_row_bw = std::accumulate(row_bw.begin(), row_bw.end(), 0u) / row_bw.size();

	return metric;
}

int checkBandwidth(const Graph<DataType>& graph)
{
	auto& matrix = graph.csr;
	int bandwidth{0};
	for(int i = 0; i < static_cast<int>(matrix.rows); ++i)
	{
		int min{i - static_cast<int>(matrix.col_ids[matrix.row_offsets[i]])};
		int max{static_cast<int>(matrix.col_ids[matrix.row_offsets[i + 1]- 1]) - i};
		
		max = std::max(min, max);
		
		if(bandwidth < max)
			bandwidth = max;
	}
	return bandwidth;
}

int maxValency(const Graph<DataType>& graph)
{
	auto& matrix = graph.csr;
	int max_valency = 0;
	for (size_t i = 0; i < matrix.rows; ++i)
		max_valency = std::max(static_cast<int>(matrix.row_offsets[i + 1])
			- static_cast<int>(matrix.row_offsets[i]), max_valency);

	return max_valency;
}


int computeBandwidth(const Graph<DataType>& graph, std::vector<uint32_t>& permutation)
{	
	if (permutation.empty())
		return -1;

	auto& matrix = graph.csr;
	int bandwidth{ 0 };

	std::vector<uint32_t> invperm(permutation.size());
	for (size_t i = 0; i < permutation.size(); ++i)
		invperm[permutation[i]] = i;

	std::vector<unsigned int> this_row(20);
	for (size_t i = 0; i < matrix.rows; ++i)
	{
		auto row_start = matrix.row_offsets[i];
		auto row_end = matrix.row_offsets[i + 1];
		auto this_row_len = row_end - row_start;

		if (this_row_len > this_row.size())
			this_row.resize(this_row_len * 1.2);

		std::transform(matrix.col_ids.get() + row_start, matrix.col_ids.get() + row_end, std::begin(this_row),
			[&invperm](const auto& x)
			{
				return invperm[x];
			});

		const auto [min_cid, max_cid] = std::minmax_element(std::begin(this_row), std::begin(this_row) + this_row_len);
		int this_rid = static_cast<int>(invperm[i]);

		int lower{ this_rid - static_cast<int>(*min_cid) };
		int upper{ static_cast<int>(*max_cid) - this_rid };

		auto max_bw = std::max(lower, upper);

		if (bandwidth < max_bw)
			bandwidth = max_bw;

	}
	return bandwidth;
}

int checkPermutation(const std::vector<uint32_t>& permutation)
{
	bool result = true;
	uint32_t i = 0;
	for (auto element : permutation)
	{
		if (element > permutation.size())
		{
			std::cout << "ERROR: element " << i << " in permutation element too big (" << element << ")" << std::endl;
			result = false;
		}
		++i;
	}

	if (!result)
		return -1;

	std::vector<uint32_t> present(permutation.size());


	std::fill(present.begin(), present.end(), 0);

	for (auto element : permutation)
	{
		present[element] = 1;
	}

	uint32_t sum = std::accumulate(present.begin(), present.end(), 0u);
	if (sum != permutation.size())
	{
		std::cout << "ERROR: " << permutation.size() - sum << " elements not present in permutation array" << std::endl;
		for (auto i = 0U; i < present.size(); ++i)
		{
			if (present[i] != 1)
				std::cout << "Node: " << i << " is missing!\n";
		}
		return -1;
	}

	return 0;
}

enum class Impl : int
{
	NONE = -1,
	ALL = 0,
	cuSolverRCM = 1,
	CPU = 2,
	CPU_BATCH = 3,
	GPU = 4,
	GPU_BATCH = 5
};


int main(int argc, char** argv)
{
	const std::vector<std::string> implementations = { "ALL" , "cuSolverRCM", "CPU", "CPU_BATCH", "GPU", "GPU_BATCH"};
	const std::string implementations_string = join(implementations);

	args::ArgumentParser parser("Computes the Cuthill-McKee reordering of a sparse quadratic matrix given in the CSR format.", "");
	args::HelpFlag arg_help(parser, "help", "Display this help menu", { 'h', "help" });
	args::Positional<std::string> arg_input(parser, "input", "quadratic matrix in CSR format to reorder", args::Options::Required);
	args::ValueFlag<std::string> arg_implemenation(parser, "implementation", "Select an RCM implementation. Available values are: " + implementations_string, 
		{'i', "implementation"}, "None");
	args::Flag arg_reverse(parser, "reverse", "Revert CM permutation (RCM).", { 'r', "reverse" });
	args::ValueFlag<int> arg_start(parser, "start", "Manually pick start node. If not specified, a pseudo-peripheral node is selected automatically.", { 's', "start" });
	args::ValueFlag<int> arg_stablesort(parser, "stable", "Select whether sorting should be stable.", { 'b', "stable" });
	args::Flag arg_testbandwidth(parser, "bandwidth", "Select whether to compute the bandwidth of reordered matrix.", { 'w', "bandwidth" });
	args::ValueFlag<int> arg_threads(parser, "threads", "Select the number of threads to run for CPU_BATCH.", { 't', "threads" });
	args::ValueFlag<std::string> arg_perf_file(parser, "perffile", "File to store the performance data.", { "f", "perffile" }, "perf.csv");
	args::ValueFlag<std::string> arg_output(parser, "output", "File to store the reorder matrix.", { 'o', "output" });

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help&)
	{
		std::cout << parser;
	}
	catch (args::Error& e)
	{
		std::cerr << e.what() << std::endl << parser;
		return -1;
	}

	if (arg_help)
		return 0;

	int impl = std::distance(implementations.begin(), std::find(implementations.begin(), implementations.end(), arg_implemenation.Get()));
	if (impl == static_cast<int>(implementations.size()))
	{
		impl = static_cast<int>(Impl::NONE);
	}

	bool testBandwidth{ arg_testbandwidth };

	Graph<DataType> graph;

	std::string mat_path = arg_input.Get();
	int status = graph.load(mat_path);
	if (status != 0)
		return status;

	size_t off = mat_path.find_last_of("\\/") + 1;
	size_t length = mat_path.find_first_of(".", off) - off;
	std::string mat_name = mat_path.substr(off, length);

	std::cout << "\nMatrix " + mat_name << " " << graph.size << " x " << graph.size << " with " << graph.csr.nnz << " entries\n" << std::endl;

	// Create profiling output file
	bool perf_file_exists = std::ifstream(arg_perf_file.Get()).good();

	std::ofstream csv_result;
	csv_result.open(arg_perf_file.Get(), std::ios_base::app);
	

	if(!perf_file_exists)
	{
		if (testBandwidth)
			csv_result << ", Initial Bandw." << std::flush;

		csv_result << ", max_valency" << std::flush;
		csv_result << ", start_node" << std::flush;

		if (impl == static_cast<int>(Impl::cuSolverRCM) || impl == static_cast<int>(Impl::ALL))
		{
			csv_result << ", cuSolver" << std::flush;
			if (testBandwidth)
				csv_result << ", Bandw." << std::flush;
		}

		// write csv header
		if (impl == static_cast<int>(Impl::CPU))
		{
			int stable = 2;
			if (arg_stablesort)
				stable = arg_stablesort.Get();

			if (stable == 0 || stable == 2)
				csv_result << ", CPU<single | NON_STABLE>" << std::flush;
			if (stable == 1 || stable == 2)
				csv_result << ", CPU<single | STABLE>" << std::flush;
			if (testBandwidth)
				csv_result << ", Bandw." << std::flush;
		}

		if (impl == static_cast<int>(Impl::CPU_BATCH) || impl == static_cast<int>(Impl::ALL))
		{
			int stable = 2;
			if (arg_stablesort)
				stable = arg_stablesort.Get();

			for (int st = stable == 2 ? 0 : stable; st < 2; st += stable == 2 ? 1 : 2)
			{
				if (st == 0)
					csv_result << ", CPU<single | NON_STABLE>" << std::flush;
				else
					csv_result << ", CPU<single | STABLE>" << std::flush;
				if (testBandwidth)
					csv_result << ", Bandw." << std::flush;

				for (int t = 1; t <= arg_threads.Get(); ++t)
				{
					if (st == 0)
						csv_result << ", CPU<" << t << " | NON_STABLE>" << std::flush;
					else
						csv_result << ", CPU<" << t << " | STABLE>" << std::flush;
					
					if (testBandwidth)
						csv_result << ", Bandw." << std::flush;
				}
			}
		}
		if (impl == static_cast<int>(Impl::GPU) || impl == static_cast<int>(Impl::ALL))
		{
			csv_result << ", GPU" << std::flush;

			if (testBandwidth)
				csv_result << ", Bandw." << std::flush;
		}

		if (impl == static_cast<int>(Impl::GPU_BATCH) || impl == static_cast<int>(Impl::ALL))
		{
			csv_result << ", GPU_BATCH" << std::flush;

			if (testBandwidth)
				csv_result << ", Bandw." << std::flush;
		}
		csv_result << std::endl;
	}

	csv_result << mat_name << std::flush;

	std::vector<uint32_t> permutation;
	permutation.reserve(graph.csr.rows);
	
	if (testBandwidth)
		csv_result << ", " << checkBandwidth(graph) << std::flush;

	csv_result << ", " << maxValency(graph) << std::flush;

	unsigned int start;
	if (arg_start)
		start = arg_start.Get();
	else
	{
		voidStream no_out;
		start = SimplePeripheral::findPeripheral(graph, no_out, (impl == static_cast<int>(Impl::ALL)));
	}
	csv_result << ", " << start << std::flush;

	if (impl == static_cast<int>(Impl::cuSolverRCM) || impl == static_cast<int>(Impl::ALL))
	{
		csv_result << ","; csv_result << std::flush;
		permutation.clear();

		CuSolverRCM::rcm(graph, csv_result, permutation);

		checkPermutation(permutation);
		if (testBandwidth)
		{
			auto bandwidth = computeBandwidth(graph, permutation);
			csv_result << ","; csv_result << bandwidth; csv_result << std::flush;
		}
	}

	if (impl == static_cast<int>(Impl::CPU))
	{
		csv_result << ","; csv_result << std::flush;
		permutation.clear();

		int stable = 2;
		if (arg_stablesort)
			stable = arg_stablesort.Get();
		if (stable == 0 || stable == 2)
			RealCPU::rcm<false>(graph, csv_result, permutation, start, 1, 1);
		if (stable == 1 || stable == 2)
			RealCPU::rcm<true>(graph, csv_result, permutation, start, 1, 1);

		checkPermutation(permutation);
		if (testBandwidth)
		{
			auto bandwidth = computeBandwidth(graph, permutation);
			csv_result << ","; csv_result << bandwidth; csv_result << std::flush;
		}
	}

	if (impl == static_cast<int>(Impl::CPU_BATCH) || impl == static_cast<int>(Impl::ALL))
	{
		permutation.clear();

		int stable = 2;
		if (arg_stablesort)
			stable = arg_stablesort.Get();

		for (int st = stable == 2 ? 0 : stable; st < 2; st += stable == 2 ? 1 : 2)
		{
			csv_result << ","; csv_result << std::flush;
			auto p2_single = permutation;
			if (st == 0)
				RealCPU::rcm<false>(graph, csv_result, p2_single, start, 1, 1);
			else
				RealCPU::rcm<true>(graph, csv_result, p2_single, start, 1, 1);
			if (testBandwidth)
			{
				auto bandwidth = computeBandwidth(graph, p2_single);
				csv_result << ","; csv_result << bandwidth; csv_result << std::flush;
			}

			for (int t = 1; t <= arg_threads.Get(); ++t)
			{
				csv_result << ","; csv_result << std::flush;
				permutation.clear();
				bool match = true;
				if (st == 0)
					RealCPU::rcm<false>(graph, csv_result, permutation, start, t, 512);
				else
					RealCPU::rcm<true>(graph, csv_result, permutation, start, t, 512);
				if (st != 0)
					for (size_t i = 0; i < p2_single.size(); ++i)
					{
						if (permutation[i] != p2_single[i])
						{
							if (match)
							{
								std::cout << "permutation does not match: \n";
								match = false;
							}
							std::cout << "@" << i << ": " << p2_single[i] << " != " << permutation[i] << std::endl;
						}
					}
				if (testBandwidth)
				{
					auto bandwidth = computeBandwidth(graph, permutation);
					csv_result << ","; csv_result << bandwidth; csv_result << std::flush;
				}
			}
		}

		checkPermutation(permutation);
	}

	if (impl == static_cast<int>(Impl::GPU) || impl == static_cast<int>(Impl::ALL))
	{
		csv_result << ","; csv_result << std::flush;
		permutation.clear();

		GPU_SIMPLE::rcm(graph, csv_result, permutation, start);
		checkPermutation(permutation);
		if (testBandwidth)
		{
			auto bandwidth = computeBandwidth(graph, permutation);
			csv_result << ","; csv_result << bandwidth; csv_result << std::flush;
		}
	}

	if (impl == static_cast<int>(Impl::GPU_BATCH) || impl == static_cast<int>(Impl::ALL))
	{
		csv_result << ","; csv_result << std::flush;
		permutation.clear();

		GPU_BATCH::rcm(graph, csv_result, permutation, start);
		checkPermutation(permutation);
		if (testBandwidth)
		{
			auto bandwidth = computeBandwidth(graph, permutation);
			csv_result << ","; csv_result << bandwidth; csv_result << std::flush;
		}
	}

	
	if (arg_output)
	{
		permutation.clear();
		voidStream no_out;
		RealCPU::rcm<true>(graph, no_out, permutation, start, 1, 1);
		CuSolverRCM::reorder(graph, permutation);

		COO<DataType> coo_mat;
		convert(coo_mat, graph.csr);

		std::string csr_name = arg_output.Get();
		std::cout << "writing output matrix \"" << csr_name << "\"\n";
		storeMTX(coo_mat, csr_name.c_str());
	}

	csv_result << std::endl;
	csv_result.close();

	return 0;
}