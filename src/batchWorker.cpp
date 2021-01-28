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

#include "batchWorker.h"
#include "batchQueue.h"
#include "CSR.h"

#include <cassert>
#include <stdio.h>
#include <thread>

thread_local std::vector<BatchWorker::TempData> BatchWorker::temps;

namespace
{
	template<class T>
	T atomic_min(std::atomic<T>& a, T val, T expected = 0)
	{
		if (expected == 0)
			expected = a.load();
		while (val < expected)
		{
			if (a.compare_exchange_strong(expected, val))
				break;
		}
		return expected;
	}

	template<typename T>
	T divup(T a, T b)
	{
		return (a + b - 1) / b;
	}
}

template<typename T>
AlgorithmState::AlgorithmState(IBatchQueue* queue, CSR<T>& csr, uint32_t* permutation)
	: num_nodes(csr.rows), row_offsets(csr.row_offsets.get()), col_ids(csr.col_ids.get()), permutation(permutation), queue(queue)
{
	visited = std::make_unique<std::atomic<uint32_t>[]>(num_nodes);
	for (uint32_t i = 0; i < num_nodes; ++i)
		visited[i] = 0xFFFFFFFF;
}

void BatchWorker::init(AlgorithmState* state)
{
	temps.resize(state->MaxTemps);
	for (auto& t : temps)
	{
		t.temptosort.clear();
		t.temptosort.reserve(state->MaxWorkSize);
		t.sortoffsets.clear();
		t.sortoffsets.reserve(state->MaxInputNodes + 1);
	}

	this->state = state;
	hasTemps = 0;
	nextTemp = 0;
	active_temps.clear();
	active_temps.reserve(state->MaxTemps);
	state->workerCount.fetch_add(1);
}

template<bool STABLE>
uint32_t BatchWorker::processNewBatch(uint32_t batchId, uint32_t tempId, Batch* batches, uint32_t ringBufferMod)
{
	Batch* batch = &batches[batchId % ringBufferMod];
	TempData& temp = temps[tempId];
	temp.temptosort.clear();
	temp.sortoffsets.clear();

	uint32_t earlystate = batch->state.load();
	uint32_t input_start = batch->input_start;
	uint32_t input_end = batch->input_end;
	
	uint32_t nextValencies = 0;
	temp.sortoffsets.push_back(0);
	for (uint32_t i = input_start; i < input_end; ++i)
	{
		auto& v = state->permutation[i];
		const auto* p = &state->col_ids[state->row_offsets[v]];
		const auto* pend = &state->col_ids[state->row_offsets[v + 1]];

		temp.temptosort.reserve(temp.temptosort.size() + pend-p);
		

		for (; p != pend; ++p)
		{
			uint32_t n = *p;
			uint32_t prev = atomic_min(state->visited[n], batchId, 0u);
			if (prev > batchId)
			{
				uint32_t valence = state->row_offsets[n + 1] - state->row_offsets[n];
				nextValencies += valence;
				temp.temptosort.push_back({ valence, n });
			}
		}
		uint32_t toffset = temp.temptosort.size();
		if(toffset != temp.sortoffsets.back())
			temp.sortoffsets.push_back(toffset);
	}

	uint32_t numnodes = temp.temptosort.size();

	TempInfo stateInfo{ tempId, batchId, Empty, earlystate >= Discovered, false, numnodes, nextValencies};
	uint32_t endnode = 0;

	uint32_t newState = updateTemp<true, STABLE>(stateInfo, batch, batches, ringBufferMod, endnode);

	if (newState != Finished)
	{
		nextTemp = (nextTemp + 1) % state->MaxTemps;
		active_temps.emplace_back(std::move(stateInfo));
		++hasTemps;
		return 0;
	}
	return endnode;
}

template<bool SORT, bool STABLE>
uint32_t BatchWorker::updateTemp(TempInfo& tempInfo, Batch* batch, Batch* batches, uint32_t ringBufferMod, uint32_t& endnode)
{

	if (AlgorithmState::Simple && !SORT)
	{
		while (batch->state.load() < Discovered)
			std::this_thread::yield();
	}

	uint32_t currentState = batch->state.load();
	bool earlyexit = currentState <= tempInfo.state;
	if(earlyexit && !SORT)
		return tempInfo.state;

	Batch* nextBatch = &batches[(tempInfo.batchId + 1) % ringBufferMod];
	TempData& temp = temps[tempInfo.tempId];

	if(!earlyexit)
	{ 
		if (currentState >= Discovered && tempInfo.state < ReadyNext)
		{
			if (tempInfo.state < Discovered)
			{
				nextBatch->state.store(Discovered);
				if (!tempInfo.earlyDiscovered)
					tempInfo.nodes = recheckDiscovered(temp.temptosort, tempInfo.batchId, tempInfo.outputs);
					
			}
			if(tempInfo.state < ReadyNext && !AlgorithmState::Simple )
				updateNext(tempInfo.batchId, batch, nextBatch, tempInfo.state, currentState, tempInfo.nodes, tempInfo.outputs, tempInfo.combine);
		}
	}

	if (SORT)
	{
		auto startsort = begin(temp.temptosort);
		for (uint32_t i = 1; i < temp.sortoffsets.size(); ++i)
		{

			auto endsort = begin(temp.temptosort) + temp.sortoffsets[i];
			if(STABLE)
				std::stable_sort(startsort, endsort);
			else
				std::sort(startsort, endsort);
			startsort = endsort;
		}
		
		if (AlgorithmState::Simple && tempInfo.earlyDiscovered)
		{
			recheckDiscovered(temp.temptosort, tempInfo.batchId, tempInfo.outputs);
		}

		tempInfo.state = currentState;
		if(currentState < ReadyNext)
			return updateTemp<false, STABLE>(tempInfo, batch, batches, ringBufferMod, endnode);
	}

	if (AlgorithmState::Simple)
		updateNext(tempInfo.batchId, batch, nextBatch, Discovered, currentState, tempInfo.nodes, tempInfo.outputs, tempInfo.combine);

	if (earlyexit)
		return tempInfo.state;

	uint32_t offset = batch->offset;
	if (currentState == CombNext)
	{
		uint32_t bcombine = batch->combine;
		uint32_t bcombineOutputs = batch->combineOutputs;
		uint32_t nodes = bcombine + tempInfo.nodes;
		uint32_t outputs = bcombineOutputs + tempInfo.outputs;

		endnode = offset + tempInfo.nodes;

		if (nodes != 0)
		{
			uint32_t followupStartBatchId = batch->followupStartBatchId;
			Batch* followupBatch = &batches[followupStartBatchId % ringBufferMod];

			uint32_t writeoffset = offset;
			uint32_t* output_nodes = state->permutation + writeoffset;

			if (tempInfo.combine)
			{
				if (!state->queue->tryfill(followupStartBatchId))
					throw std::runtime_error("out of batchqueue space");

				//ensure batch is available
				auto c_iter = std::cbegin(temp.temptosort);
				writeNodesBlind(c_iter, end(temp.temptosort), output_nodes, tempInfo.earlyDiscovered);
				nextBatch->state.store(CombNext);
			}
			else if (tempInfo.nodes != 0)
			{
				if (!state->queue->tryfill(followupStartBatchId))
					throw std::runtime_error("out of batchqueue space");
				uint32_t numbatches = std::max(divup(nodes, state->MaxInputNodes), divup(outputs, state->MaxWorkSize));

				TempVector::const_iterator it = begin(temp.temptosort);
				uint32_t remnodes = tempInfo.nodes;
				uint32_t writtennodes = writeNodes(it, end(temp.temptosort), output_nodes, tempInfo.earlyDiscovered, numbatches, remnodes, bcombine, bcombineOutputs);
				
				followupBatch->input_start = (offset - bcombine);
				followupBatch->input_end = (offset + writtennodes);
				state->queue->ready(followupStartBatchId);


				writeoffset += writtennodes;
				output_nodes += writtennodes;

				for (uint32_t b = 1; b < numbatches; ++b)
				{
					followupBatch = &batches[(followupStartBatchId + b) % ringBufferMod];
					if (!state->queue->startfill(followupStartBatchId + b))
						throw std::runtime_error("out of batchqueue space");
					writtennodes = writeNodes(it, end(temp.temptosort), output_nodes, tempInfo.earlyDiscovered, numbatches - b, remnodes);

					followupBatch->input_start = (writeoffset);
					followupBatch->input_end = (writeoffset + writtennodes);

					state->queue->ready(followupStartBatchId + b);

					writeoffset += writtennodes;
					output_nodes += writtennodes;
				}
			}
			else
			{
				followupBatch->input_start = (offset - bcombine);
				followupBatch->input_end = (offset);
				state->queue->ready(followupStartBatchId);
			}

		}
		else if (tempInfo.combine)
		{
			nextBatch->state.store(CombNext);
		}

		// free this one
		batch->reset();
		state->queue->completed(tempInfo.batchId);
		tempInfo.state = currentState = Finished;
	}
	else
		tempInfo.state = currentState;
	return currentState;
}

void BatchWorker::writeNodesBlind(TempVector::const_iterator& begin, TempVector::const_iterator end, uint32_t* nextNodes, bool earlyDiscovered)
{
	if (earlyDiscovered)
	{
		for (; begin != end; ++begin, ++nextNodes)
			*nextNodes = begin->value;
	}
	else
	{
		for (; begin != end; ++begin)
		{
			if (begin->key != 0xFFFFFFFF)
			{
				*nextNodes = begin->value;
				++nextNodes;
			}
		}
	}
}
uint32_t BatchWorker::writeNodes(TempVector::const_iterator& begin, TempVector::const_iterator end,  uint32_t* nextNodes, bool earlyDiscovered, uint32_t remBatches, uint32_t& remnodes, uint32_t nodesWritten, uint32_t sumOutputs)
{
	// min nodes that need to be put
	uint32_t minWrite = 0;
	uint32_t maxOthernodes = (remBatches - 1) * state->MaxInputNodes;
	if (remnodes > maxOthernodes)
		minWrite = nodesWritten + remnodes - maxOthernodes;

	uint32_t thiswritten = 0;

	auto inneradvance = [&]()
	{
		*nextNodes = begin->value;
		sumOutputs += begin->key;

		++begin;
		++nextNodes;
		++thiswritten;
		++nodesWritten;

		if (nodesWritten >= state->MaxInputNodes)
			return false;

		if (sumOutputs >= state->MaxWorkSize && thiswritten > minWrite)
			return false;
		return true;
	};

	if (earlyDiscovered)
	{
		while (begin != end && inneradvance());
	}
	else
	{
		while (begin != end)
		{
			if (begin->key != 0xFFFFFFFF)
			{
				if (!inneradvance())
					break;
			}
			else
				++begin;
		}
	}
	remnodes -= thiswritten;
	return thiswritten;
}



uint32_t BatchWorker::updateTemps(Batch* batches, uint32_t ringBufferMod)
{
	uint32_t endnode = 0;
	if (hasTemps > 0)
	{
		uint32_t newStart = 0;
		uint32_t count = 0;
		for (auto& t : active_temps)
		{
			uint32_t laststate = t.state;
			Batch& b = batches[t.batchId % ringBufferMod];
			uint32_t tendnode;
			uint32_t newstate = laststate;

			if(laststate != Finished)
			{
				newstate = updateTemp<false, false>(t, &b, batches, ringBufferMod, tendnode);
				if(newstate == Finished)
					endnode = tendnode;
			}

			if (newstate == Finished && count == newStart)
			{
				++newStart;
			}
			else
				t.state = newstate;
			

			if (newstate < Discovered)
			{
				// no need to check others, if this one is not yet discovered...
				break;
			}
			++count;
		}
		if (newStart > 0)
		{
			hasTemps -= newStart;
			for (uint32_t i = 0; i < hasTemps; ++i)
			{
				active_temps[i] = active_temps[newStart + i];
			}
			active_temps.resize(hasTemps);
		}
	}
	return endnode;
}

uint32_t BatchWorker::recheckDiscovered(std::vector<KeyValue<uint32_t, uint32_t>>& temp, uint32_t batchid, uint32_t& outputs)
{
	uint32_t count = temp.size();
	for (auto& n : temp)
	{
		if (state->visited[n.value].load() < batchid)
		{
			outputs -= n.key;
			n.key = 0xFFFFFFFF;
			n.value = 0xFFFFFFFF;
			--count;
		}
	}
	return count;
}

void BatchWorker::updateNext(uint32_t batchId, Batch* batch, Batch* nextBatch, uint32_t lastState, uint32_t cstate, uint32_t nextNodes, uint32_t overallNextTemps, bool& combine)
{
	if (cstate <= lastState)
		return;

	uint32_t nextState;
	if (cstate >= CountSet)
	{
		if(lastState < CountSet)
			nextBatch->offset = (batch->offset + nextNodes);
		if (lastState < ReadyNext && cstate >= ReadyNext)
		{
			uint32_t nodes = batch->combine + nextNodes;
			uint32_t outputs = batch->combineOutputs+ overallNextTemps;
			uint32_t childrenBatchId = batch->followupStartBatchId;
			combine = (2 * nodes < state->MaxInputNodes) && (2 * outputs < state->MaxWorkSize) && (childrenBatchId > batchId + 1);
			if (AlgorithmState::Simple)
				combine = false;
			if (combine)
			{
				nextBatch->combine = (nodes);
				nextBatch->combineOutputs = (outputs);
				nextBatch->followupStartBatchId = (childrenBatchId);
				nextState = ReadyNext;
			}
			else
			{
				uint32_t batches = std::max(divup(nodes, state->MaxInputNodes), divup(outputs, state->MaxWorkSize));
				if (nodes == 0)
				{
					assert(childrenBatchId > batchId + 1);
					batches = 0;
				}
				nextBatch->combine = (0);
				nextBatch->combineOutputs = (0);
				nextBatch->followupStartBatchId = (childrenBatchId + batches);
				nextState = CombNext;
			}
			
		}
		else
			nextState = CountSet;
	}
	else
		nextState = Discovered;

	nextBatch->state.store(nextState);
}


template<bool STABLE, uint32_t QUEUE_SIZE>
void BatchWorker::run(AlgorithmState* state, BatchQueue<QUEUE_SIZE>* queue)
{
	BatchWorker temp;
	temp.template run_internal<STABLE, QUEUE_SIZE>(state, queue);
}

void BatchWorker::start(AlgorithmState* state, IBatchQueue* queue, uint32_t startnode)
{
	state->permutation[0] = startnode;
	state->visited[startnode] = 0;
	Batch& startbatch = queue->at(0);
	startbatch.input_start = 0;
	startbatch.input_end = 1;
	startbatch.offset = 1;
	startbatch.followupStartBatchId = 1;
	startbatch.combine = 0;
	startbatch.combineOutputs = 0;
	startbatch.state = CombNext;
	queue->ready(0);
}

template<bool STABLE, uint32_t QUEUE_SIZE>
void BatchWorker::run_internal(AlgorithmState* state, BatchQueue<QUEUE_SIZE>* queue)
{
	init(state);
	uint32_t finishedNode = 0;
	while (true)
	{
		
		if(hasTemps < state->MaxTemps)
		{ 
			uint32_t batch = queue->get(hasTemps != 0);
			if (batch == 0xFFFFFFFF)
			{
				state->workerCount.fetch_sub(1);
				return;
			}

			if (batch != 0xFFFFFFFE)
			{
				// work on new batch (as far as we get)
				uint32_t tempId = nextTemp;
				finishedNode = processNewBatch<STABLE>(batch, tempId, &queue->at(0), QUEUE_SIZE);
			}
		}

		// are there temps left to work on?
		uint32_t thisfinished = 0;
		if ((thisfinished = updateTemps(&queue->at(0), QUEUE_SIZE)) != 0)
			finishedNode = std::max(finishedNode, thisfinished);
		

		// if still temps left -> yield, then continue
		if (hasTemps > 0)
		{
			std::this_thread::yield();
			continue;
		}


		// did we finish working on the last??
		if(finishedNode == state->num_nodes)
		{ 
			queue->end();
			state->workerCount.fetch_sub(1);
			{
				std::unique_lock<std::mutex> l(state->mutex);
				state->running = false;
			}
			state->finishWaiter.notify_all();
			std::this_thread::yield();
			return;
		}
	}
}

template AlgorithmState::AlgorithmState(IBatchQueue* queue, CSR<float>& csr, uint32_t* permutations);
template AlgorithmState::AlgorithmState(IBatchQueue* queue, CSR<double>& csr, uint32_t* permutations);

template void BatchWorker::run<true, 128 * 1024>(AlgorithmState* state, BatchQueue<128 * 1024>* queue);
template void BatchWorker::run<false, 128 * 1024>(AlgorithmState* state, BatchQueue<128 * 1024>* queue);