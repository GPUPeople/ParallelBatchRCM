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

#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "batch.h"
#include "batchQueue.h"
#include "Graph.h"

template<typename T>
struct CSR;

struct AlgorithmState
{
	static constexpr bool Simple = false;

	uint32_t MaxInputNodes = 128; 
	uint32_t MaxWorkSize = 32 * 128;
	uint32_t UseMaxTemps = 2;


	uint32_t MaxTemps = Simple ? 1 : UseMaxTemps;

	bool running = true;
	uint32_t num_nodes = 0;
	uint32_t* row_offsets;
	uint32_t* col_ids;
	uint32_t* permutation;
	std::unique_ptr<std::atomic<uint32_t>[]> visited;

	std::mutex mutex;
	std::condition_variable finishWaiter;

	std::atomic<uint32_t> workerCount = 0;


	IBatchQueue* queue;

	AlgorithmState() = default;
	template<typename T>
	AlgorithmState(IBatchQueue* queue, CSR<T>& csr, uint32_t* permutations);

	void reset()
	{
		running = true;
		workerCount = 0;
		for (uint32_t i = 0; i < num_nodes; ++i)
			visited[i] = 0xFFFFFFFF;
	}
};

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

class BatchWorker
{
	using TempVector = std::vector<KeyValue<uint32_t, uint32_t>>;
	struct TempData
	{
		TempVector temptosort;
		std::vector<uint32_t> sortoffsets;
	};

	static thread_local std::vector<TempData> temps;
	
	struct TempInfo
	{
		uint32_t tempId;
		uint32_t batchId;
		uint32_t state;
		bool earlyDiscovered;
		bool combine;
		uint32_t nodes;
		uint32_t outputs;
	};
	std::vector<TempInfo> active_temps;
	uint32_t hasTemps;
	uint32_t nextTemp;

	AlgorithmState* state;

	void init(AlgorithmState* state);

	template<bool STABLE>
	uint32_t processNewBatch(uint32_t batchId, uint32_t tempId, Batch* batches, uint32_t ringBufferMod);
	template<bool SORT, bool STABLE>
	uint32_t updateTemp(TempInfo& tempInfo, Batch* batch, Batch* batches, uint32_t ringBufferMod, uint32_t& endnode);

	uint32_t updateTemps(Batch* batches, uint32_t ringBufferMod);
	uint32_t recheckDiscovered(std::vector<KeyValue<uint32_t, uint32_t>>& temp, uint32_t batchid, uint32_t& outputs);
	void updateNext(uint32_t batchId, Batch* batch, Batch* nextBatch, uint32_t lastState, uint32_t currentState, uint32_t nextNodes, uint32_t overallNextTemps, bool& combine);

	void writeNodesBlind(TempVector::const_iterator& begin, TempVector::const_iterator end, uint32_t* nextNodes, bool earlyDiscovered);
	uint32_t writeNodes(TempVector::const_iterator& begin, TempVector::const_iterator end, uint32_t* nextNodes, bool earlyDiscovered, uint32_t remBatches, uint32_t& remnodes, uint32_t nodesWritten = 0, uint32_t sumOutputs = 0);

	template<bool STABLE_SORT, uint32_t QUEUE_SIZE>
	void run_internal(AlgorithmState* state, BatchQueue<QUEUE_SIZE>* queue);
public:
	template<bool STABLE_SORT, uint32_t QUEUE_SIZE>
	static void run(AlgorithmState* state, BatchQueue<QUEUE_SIZE>* queue);

	static void start(AlgorithmState* state, IBatchQueue* queue, uint32_t startnode);

};