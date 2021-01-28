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

#include <stdint.h>
#include <type_traits>
#include "batchQueue.cuh"
#include "rcm_gpu_batch_params.h"

#include "cub/cub.cuh"
#include "stdio.h"


#if 0
#define DPRINTF(...) printf(__VA_ARGS__)
#define CONDDPRINTF(COND, ...) if(COND) printf(__VA_ARGS__)
#else
#define DPRINTF(...) 
#define CONDDPRINTF(COND, ...) 
#endif

#define ADPRINTF(...) printf(__VA_ARGS__)
#define ACONDDPRINTF(COND, ...) if(COND) printf(__VA_ARGS__)

namespace {
	template<typename T>
	__host__ __device__ constexpr T divup(T a, T b)
	{
		return (a + b - 1) / b;
	}
}

namespace RCM_DYNAMIC
{
	constexpr uint32_t BlockSize = 1u << BlockSizePow;
	constexpr uint32_t MaxElementsPerThreads = 1u << MaxElementsPerThreadPow;

	constexpr uint32_t MaxInputNodes = BlockSize;
	constexpr uint32_t MaxWorkSize = MaxElementsPerThreads * BlockSize;
	constexpr uint32_t VisitedShift = BlockSizePow + MaxElementsPerThreadPow;
	constexpr uint32_t ScanOffsetShift = 31 - VisitedShift;

	constexpr uint32_t HistogramBins = 32;
	constexpr uint32_t HistogramCollectionBits = 4;
	constexpr uint32_t HistogramCounters = divup(HistogramBins * HistogramCollectionBits, 64u);
	using HistogramResultType = typename std::conditional<HistogramBins <= 32, uint32_t, uint64_t>::type;

	struct AlgorithmState
	{
		uint32_t num_nodes;
		const uint32_t* __restrict__ row_offsets;
		const uint32_t* __restrict__ col_ids;
		uint32_t* permutation;
		uint32_t* visited;



		__device__ void init(uint32_t num_nodes, const uint32_t* row_offsets, const uint32_t* col_ids, uint32_t* permutation, uint32_t* visited)
		{
			uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
			for (uint32_t i = tid; i < num_nodes; i += gridDim.x + blockDim.x)
			{
				visited[i] = 0xFFFFFFFF;
			}
			if (tid == 0)
			{
				this->num_nodes = num_nodes;
				this->row_offsets = row_offsets;
				this->col_ids = col_ids;
				this->permutation = permutation;
				this->visited = visited;

			}
		}
	};

	struct TempData
	{
		uint32_t batchId;
		uint32_t lastState;
		uint32_t earlyState;
		uint32_t sortperthreads;
		uint32_t ownednodes;
		uint32_t nextoutputs;
		uint32_t overlongoutputs;
		uint32_t combine_long;

		uint32_t minVal, maxVal;

		uint32_t valencies_offsets[MaxWorkSize + 1];
		uint32_t nodes[MaxWorkSize];
	};


	struct Temporaries
	{
		union
		{
			struct
			{
				union
				{
					typename cub::BlockScan<uint32_t, BlockSize>::TempStorage BlockScanTempStorage;
					uint32_t nodeOffsets[BlockSize + 1];
					typename cub::WarpReduce<uint32_t>::TempStorage WarpReduceTemp;
				};
				uint32_t maxNodes;
			};
			typename cub::BlockRadixSort<uint32_t, BlockSize, MaxElementsPerThreads, uint32_t>::TempStorage RadixSortTempStorage;
			uint32_t outputs[BlockSize + 1];
			struct
			{
				uint32_t histogram[HistogramBins];
				typename cub::WarpScan<int>::TempStorage WarpScanTemp;
			};
		};
	};

	struct StateTemp
	{
		uint32_t runstate;
		uint32_t nextTemp;
		uint32_t temps;
		uint32_t finishedNode;
		uint32_t reservedBatch;
	};


	__device__ void newBatch(StateTemp& state, TempData& temp);
	__device__ uint32_t updateTemps(StateTemp& state, TempData* temps);
	__device__ uint32_t updateTemp(StateTemp& state, TempData& temp);
	__device__ void updateNext(TempData& temp, uint32_t newState);

	__device__ void discover(StateTemp& state, TempData& temp, uint32_t startOffset, uint32_t endOffset);
	__device__ void discoverLong(StateTemp& state, TempData& temp, uint32_t startOffset, uint32_t endOffset);
	__device__ void discoverInner(StateTemp& state, TempData& temp, uint32_t startOffset, uint32_t endOffset, uint32_t threads_per_node_shift, uint32_t threads_per_node);
	__device__ void rediscover(TempData& temp);
	__device__ void rediscoverLong(TempData& temp);
	__device__ void sort(TempData& temp);
	__device__ void constructHist(TempData& temp, const uint32_t& rBegin, const uint32_t& rEnd, HistogramResultType& partResets, HistogramResultType& overflow);
	template<bool CHECKVALENCY, bool AVOIDOVERFLOW>
	__device__ uint32_t compactReader(TempData& temp, const uint32_t& rBegin, const uint32_t& rEnd, uint32_t valBegin, uint32_t valEnd, uint32_t& scounter, uint32_t offset = 0);

	template<bool FINISHNODE>
	__device__ void write(TempData& temp, uint32_t& offset, uint32_t& writtenNodes, uint32_t& writtenTemps, uint32_t& followupStartBatchId, uint32_t& batches, const uint32_t& tnodes);



	__device__ AlgorithmState algorithmState;
	__device__ BatchQueue<QueueSize, QueueCheckEnd> queue;


	__device__ Temporaries& getTemporaries()
	{
		__shared__ Temporaries t;
		return t;
	}

	__global__ void init(uint32_t num_nodes, const uint32_t* row_offsets, const uint32_t* col_ids, uint32_t* permutation, uint32_t* visited, uint32_t start_node)
	{
		algorithmState.init(num_nodes, row_offsets, col_ids, permutation, visited);
		queue.init(start_node);

		if (threadIdx.x == 0)
		{
			permutation[0] = start_node;
			visited[start_node] = 0;
		}
	}

	__global__ void run(int blocks)
	{
		static_assert(MaxTemps <= 32, "cannot use more than 32 temps");

		__shared__ StateTemp state;
		extern __shared__ TempData temps[];
		if (threadIdx.x == 0)
		{
			state.temps = 0;
			state.nextTemp = 0;
			state.finishedNode = 0;
			state.reservedBatch = 0xFFFFFFFF;
		}

		while (true)
		{
			__syncthreads();
			if (threadIdx.x == 0)
			{
				if (state.temps < MaxTemps)
				{
					if (state.reservedBatch == 0xFFFFFFFF)
					{
						if (state.temps == 0)
							state.runstate = queue.get(-blocks);
						else
							state.runstate = queue.getNonblocking(state.reservedBatch);
					}
					else
					{
						uint32_t res;
						if (state.temps == 0)
							res = queue.getReservedSpot<true>(state.reservedBatch, -blocks);
						else
							res = queue.getReservedSpot<false>(state.reservedBatch, 0);
						if (res == state.reservedBatch || res == 0xFFFFFFFF)
							state.reservedBatch = 0xFFFFFFFF;
						state.runstate = res;
					}
					DPRINTF("%d %d got %d(%x)\n", blockIdx.x, threadIdx.x, state.runstate, state.runstate);
				}
				else
					state.runstate = 0xFFFFFFFE;
			}
			__syncthreads();
			if (state.runstate < 0xFFFFFFFE)
			{
				newBatch(state, temps[state.nextTemp]);
				__syncthreads();
				CONDDPRINTF(threadIdx.x == 0, "%d %d newBatch %d done with all, now has %d temps and nextTemp %d\n", blockIdx.x, threadIdx.x, state.runstate, state.temps, state.nextTemp);
			}
			if (state.temps != 0)
			{
				uint32_t moved = updateTemps(state, &temps[0]);
				CONDDPRINTF(threadIdx.x == 0 && moved > 0, "%d %d (%d) completed => %d/%d\n", blockIdx.x, threadIdx.x, temps[0].batchId, state.finishedNode, algorithmState.num_nodes);
				__syncthreads();
				if (state.runstate == 0xFFFFFFFE && moved == 0)
				{
#if __CUDA_ARCH__ >= 700
					__nanosleep(20);
#else
					__threadfence();
#endif
				}
			}
			CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) iteration done has temps %d  last done node = %d / %d\n", blockIdx.x, threadIdx.x, temps[0].batchId, state.temps, state.finishedNode, algorithmState.num_nodes);
			// did we finish working on the last??
			if (!QueueCheckEnd && threadIdx.x == 0 && state.finishedNode == algorithmState.num_nodes)
			{
				queue.end();
				state.runstate = 0xFFFFFFFF;
				DPRINTF("%d %d all done (%d == %d) , we are ending\n", blockIdx.x, threadIdx.x, state.finishedNode, algorithmState.num_nodes);
			}

			__syncthreads();
			if ((state.temps == 0 && state.runstate == 0xFFFFFFFF && state.reservedBatch == 0xFFFFFFFF) || (state.temps == 0 && queue.ended()))
			{
				CONDDPRINTF(threadIdx.x == 0, "%d %d block ending due to runstate, temps: %d\n", blockIdx.x, threadIdx.x, state.temps);
				return;
			}
		}
	}

	__device__ void newBatch(StateTemp& state, TempData& temp)
	{
		__shared__ uint32_t startOffset;
		__shared__ uint32_t endOffset;

		// single thread load info
		if (threadIdx.x == 0)
		{
			Batch& batch = queue[state.runstate];
			uint2 input_start_end = __ldcg(&batch.input_start_end);
			startOffset = input_start_end.x;
			endOffset = input_start_end.y;
			temp.batchId = state.runstate;
			temp.earlyState = __ldcg(&batch.state);
			temp.lastState = Empty;
			temp.combine_long = input_start_end.y - input_start_end.x == 1 ? 2 : 0;

			CONDDPRINTF(temp.batchId >= 1940, "%d %d starting with batch %d: input %d-%d earlystate: %d long: %d\n", blockIdx.x, threadIdx.x, temp.batchId, startOffset, endOffset, temp.earlyState, temp.combine_long);
		}
		__syncthreads();

		// call discover
		if (temp.combine_long == 2)
			discoverLong(state, temp, startOffset, endOffset);
		else
			discover(state, temp, startOffset, endOffset);
		__syncthreads();

		// update next early
		if (threadIdx.x == 0)
		{
			DPRINTF("%d %d discover done, owning %d nodes with %d next temporaries\n", blockIdx.x, threadIdx.x, temp.ownednodes, temp.nextoutputs);

			Batch& batch = queue[state.runstate];
			temp.lastState = Empty;
			uint32_t nextState = __ldcg(&batch.state);
			if (nextState >= Discovered)
			{
				nextState = temp.earlyState >= Discovered ? nextState : Discovered;

				DPRINTF("%d %d early update for next: early state %d, this state %d\n", blockIdx.x, threadIdx.x, temp.earlyState, nextState);
				updateNext(temp, nextState);
			}
			temp.lastState = nextState;
		}

		__syncthreads();

		if (temp.earlyState < Discovered && temp.lastState >= Discovered)
		{
			if (temp.ownednodes != 0)
			{
				if (temp.combine_long > 1)
					rediscoverLong(temp);
				else
					rediscover(temp);
			}

			__syncthreads();
			CONDDPRINTF(threadIdx.x == 0 && temp.batchId == 842 || temp.batchId == 841, "%d %d rediscover newBatch done, now owning %d nodes with %d next temporaries\n", blockIdx.x, threadIdx.x, temp.ownednodes, temp.nextoutputs);

			if (threadIdx.x == 0)
			{
				Batch& batch = queue[state.runstate];
				uint32_t nextState = __ldcg(&batch.state);
				if (nextState >= Discovered)
					updateNext(temp, nextState);
				temp.lastState = nextState;
				temp.earlyState = Discovered;
			}
		}

		// sort
		if (temp.ownednodes != 0 && temp.combine_long <= 1)
			sort(temp);

		CONDDPRINTF(threadIdx.x == 0, "%d %d sort done\n", blockIdx.x, threadIdx.x);

		// update temp
		uint32_t tstate = updateTemp(state, temp);

		CONDDPRINTF(threadIdx.x == 0, "%d %d updateTemp done with state %d\n", blockIdx.x, threadIdx.x, tstate);
		// increase nextTemp and numtemps if not done
		if (tstate != Finished && threadIdx.x == 0)
		{
			++state.temps;
			state.nextTemp = (state.nextTemp + 1) % MaxTemps;
		}
	}


	__device__ uint32_t updateTemps(StateTemp& state, TempData* temps)
	{
		uint32_t movestart = 0;
		if (state.temps > 0)
		{
			uint32_t checktempsstart = (state.nextTemp - state.temps + MaxTemps) % MaxTemps;

			for (uint32_t i = 0; i < state.temps; ++i)
			{

				TempData* thistemp = temps + (checktempsstart + i) % MaxTemps;
				if (thistemp->lastState != Finished)
				{
					updateTemp(state, *thistemp);
					if (thistemp->lastState == Finished && movestart == i)
						++movestart;
				}
				else if (movestart == i)
					++movestart;


			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				state.temps -= movestart;
				CONDDPRINTF(movestart != 0, "%d %d updateTemps removed %d temps \n", blockIdx.x, threadIdx.x, movestart);
			}
		}
		return movestart;
	}

	__device__ void updateNext(TempData& temp, uint32_t newState)
	{
		// single threaded write the info for the next
		if (temp.lastState >= newState)
			return;

		Batch& nextbatch = queue[temp.batchId + 1];
		if (temp.lastState < Discovered && newState >= Discovered)
		{
			__stcg(&nextbatch.state, Discovered);
			CONDDPRINTF(temp.batchId == 842 || temp.batchId == 841, "%d %d updateNext setting state for next (%d) to %d\n", blockIdx.x, threadIdx.x, temp.batchId + 1, Discovered);
		}

		// ensure this is not reordered, not sure whether we need full threadfence here.. unified cache etc
		__threadfence_block();

		if (newState >= CountSet)
		{
			uint32_t nextState;
			Batch& batch = queue[temp.batchId];
			uint32_t nextNodes = temp.ownednodes;
			uint32_t overallNextTemps = temp.nextoutputs;

			uint4 offset_followupStartBatchId_combine_combineOutputs = __ldcg(&batch.offset_followupStartBatchId_combine_combineOutputs);

			if (temp.lastState < CountSet && newState == CountSet)
			{
				__stcg(&nextbatch.offset_followupStartBatchId_combine_combineOutputs.x, offset_followupStartBatchId_combine_combineOutputs.x + nextNodes);
				CONDDPRINTF(temp.batchId == 842 || temp.batchId == 841, "%d %d updateNext setting offset for (%d) to %d\n", blockIdx.x, threadIdx.x, temp.batchId + 1, offset_followupStartBatchId_combine_combineOutputs.x + nextNodes);
			}

			if (temp.lastState < ReadyNext && newState >= ReadyNext)
			{

				uint32_t offset = offset_followupStartBatchId_combine_combineOutputs.x + nextNodes;
				uint32_t childrenBatchId = offset_followupStartBatchId_combine_combineOutputs.y;
				uint32_t nodes = offset_followupStartBatchId_combine_combineOutputs.z + nextNodes;
				uint32_t outputs = offset_followupStartBatchId_combine_combineOutputs.w + overallNextTemps;

				bool combine = nodes != 0 && (2 * nodes < MaxInputNodes) && (2 * outputs < MaxWorkSize) && (childrenBatchId > temp.batchId + 1);

				if (combine == 1)
				{
					temp.combine_long = temp.combine_long | 1;

					uint4 store{ offset, childrenBatchId, nodes, outputs };
					__stcg(&nextbatch.offset_followupStartBatchId_combine_combineOutputs, store);
					nextState = ReadyNext;
					CONDDPRINTF(temp.batchId == 842 || temp.batchId == 841, "%d %d updateNext - combine - nodes: %d setting combentry for (%d) to %d %d %d %d\n", blockIdx.x, threadIdx.x, nodes, temp.batchId + 1, store.x, store.y, store.z, store.w);

				}
				else
				{
					uint32_t batches = min(nextNodes + 1, max(divup(nodes, MaxInputNodes), divup((outputs - temp.overlongoutputs) * TempNodeOverallocNom / TempNodeOverallocDenom, MaxWorkSize)));
					if (nodes == 0)
					{
						//std::cout << "no batches for " << batchId << std::endl;
						batches = 0;
					}

					uint4 store{ offset, childrenBatchId + batches, 0, 0 };
					__stcg(&nextbatch.offset_followupStartBatchId_combine_combineOutputs, store);
					nextState = CombNext;
					CONDDPRINTF(temp.batchId == 842 || temp.batchId == 841, "%d %d updateNext - no combine - setting combentry for (%d) to %d (%d+%d) %d (%d+%d) %d %d\n",
						blockIdx.x, threadIdx.x, temp.batchId + 1, store.x, offset_followupStartBatchId_combine_combineOutputs.x, nextNodes, store.y, childrenBatchId, batches, store.z, store.w);
				}
			}
			else
				nextState = CountSet;
			__threadfence();
			__stcg(&nextbatch.state, nextState);
			CONDDPRINTF(temp.batchId == 841 || temp.batchId == 841, "%d %d updateNext setting state for (%d) to %d\n", blockIdx.x, threadIdx.x, temp.batchId + 1, nextState);
		}
	}

	__device__ void discover(StateTemp& state, TempData& temp, uint32_t startOffset, uint32_t endOffset)
	{
		// every thread loads row_offset for its node
		uint32_t count[1] = { 0 };
		uint32_t offset = startOffset + threadIdx.x;
		uint32_t node;
		if (offset < endOffset)
		{
			node = __ldcg(&algorithmState.permutation[offset]);
			//CONDDPRINTF(node == 0xFFFFFFFF, "%d %d (%d) got dead node: %x at %d .. %x\n", blockIdx.x, threadIdx.x, temp.batchId, node, offset, algorithmState.permutation[offset]);
			count[0] = algorithmState.row_offsets[node + 1] - algorithmState.row_offsets[node];
		}

		// prefix sum to get offsets for each adjacency
		uint32_t tcount = count[0];
		uint32_t tosortnodes;
		using BlockScan = cub::BlockScan<uint32_t, BlockSize>;
		BlockScan(getTemporaries().BlockScanTempStorage).ExclusiveSum(count, count, tosortnodes);
		getTemporaries().maxNodes = 0;
		__syncthreads();

		getTemporaries().nodeOffsets[threadIdx.x] = count[0];
		getTemporaries().nodeOffsets[BlockSize] = tosortnodes;

		// based on mean number of nodes, choose number of threads per adjacency
		uint32_t res = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Reduce(tcount, cub::Max());
		if (threadIdx.x % 32 == 0)
			atomicMax(&getTemporaries().maxNodes, res);
		__syncthreads();



		uint32_t maxNodes = getTemporaries().maxNodes;
		uint32_t avg = tosortnodes / (endOffset - startOffset) + 1;
		uint32_t threads_per_node_shift = maxNodes > BlockSize ? BlockSizePow :
			maxNodes > BlockSize / 2 ? BlockSizePow - 1 :
			maxNodes > BlockSize / 4 || avg > 64 ? 6 :
			(31 - __clz(avg));
		uint32_t threads_per_node = 1u << threads_per_node_shift;

		// initialzie adjacency to 0xFFFFFFFF
		uint32_t elperthread = divup(tosortnodes, BlockSize);
		temp.sortperthreads = elperthread;
		temp.nextoutputs = 0;
		temp.ownednodes = 0;
		temp.overlongoutputs = 0;


		CONDDPRINTF(threadIdx.x + startOffset < endOffset&& threadIdx.x + startOffset >= 46282 && threadIdx.x + startOffset < 46341, "%d %d discover: %d = %d -> %d\n",
			blockIdx.x, threadIdx.x, startOffset + threadIdx.x, node, tcount);

		CONDDPRINTF(threadIdx.x % 32 == 0 && temp.sortperthreads > MaxElementsPerThreads, "%d %d (%d) will discover %d child nodes from (%d-%d) with a avg/max of %d/%d using %d threads per parent and %d elements to sort per thread\n",
			blockIdx.x, threadIdx.x, temp.batchId, tosortnodes, startOffset, endOffset, avg, maxNodes, threads_per_node, elperthread);

		discoverInner(state, temp, startOffset, endOffset, threads_per_node_shift, threads_per_node);
	}

	__device__ void discoverInner(StateTemp& state, TempData& temp, uint32_t startOffset, uint32_t endOffset, uint32_t threads_per_node_shift, uint32_t threads_per_node)
	{
		if (temp.sortperthreads > MaxElementsPerThreads)
		{
			ADPRINTF("%d %d (%d) sort per thread too high: %d  (%d->%d)\n", blockIdx.x, threadIdx.x, temp.batchId, temp.sortperthreads, startOffset, endOffset);
			__trap();
		}

		for (uint32_t i = 0; i < temp.sortperthreads; ++i)
		{
			uint32_t id = threadIdx.x + i * BlockSize;
			temp.valencies_offsets[id] = 0xFFFFFFFF;
			temp.nodes[id] = 0xFFFFFFFF;
		}

		__syncthreads();

		uint32_t groupId = threadIdx.x >> threads_per_node_shift;
		uint32_t ingroupId = threadIdx.x & (threads_per_node - 1);


		uint32_t ownednodes = 0;
		uint32_t nexttemps = 0;
		uint32_t overlongoutputs = 0;
		for (uint32_t lnode = groupId; lnode < endOffset - startOffset; lnode += (BlockSize >> threads_per_node_shift))
		{
			uint32_t offset = startOffset + lnode;

			CONDDPRINTF(endOffset - startOffset > MaxInputNodes, "%d %d (%d) exceeded max input: %d", blockIdx.x, threadIdx.x, temp.batchId, endOffset - startOffset);
			uint32_t node = __ldcg(&algorithmState.permutation[offset]);

			CONDDPRINTF(node == 0xFFFFFFFF, "%d %d (%d) got bad input node: 0xFFFFFFFF at %d", blockIdx.x, threadIdx.x, temp.batchId, offset);


			uint32_t count = getTemporaries().nodeOffsets[lnode + 1] - getTemporaries().nodeOffsets[lnode];

			uint32_t childrenoffset = algorithmState.row_offsets[node];
			uint32_t discovercode = (state.runstate << VisitedShift);
			for (uint32_t r = ingroupId; r < count; r += threads_per_node)
			{
				uint32_t childnode = algorithmState.col_ids[childrenoffset + r];

				// discover with atomicMin
				uint32_t thismemoffset = getTemporaries().nodeOffsets[lnode] + r;

				temp.nodes[thismemoffset] = childnode;
				__threadfence_block();

				uint32_t thisdiscovercode = discovercode | thismemoffset;
				uint32_t old = atomicMin(&algorithmState.visited[childnode], thisdiscovercode);

				//  if earlier discovered-> set to 0xFFFFFFFF
				if (old <= thisdiscovercode)
				{
					temp.nodes[thismemoffset] = 0xFFFFFFFF;
				}
				else
				{
					uint32_t child_valency = algorithmState.row_offsets[childnode + 1] - algorithmState.row_offsets[childnode];
					uint32_t sortkey = (lnode << ScanOffsetShift) | child_valency;
					if (old < ((state.runstate + 1) << VisitedShift))
					{
						//  if this is earlier but discovered in this block -> remove the node from the other 
						uint32_t othermemoffset = old & ((1u << VisitedShift) - 1);
						temp.nodes[othermemoffset] = 0xFFFFFFFF;
					}
					else
					{
						//  if this is earlier but discovered in other block -> we count the adjacency
						nexttemps += child_valency;
						overlongoutputs += max(static_cast<int>(child_valency) - static_cast<int>(MaxWorkSize), 0);
						++ownednodes;
					}
					temp.valencies_offsets[thismemoffset] = sortkey;
				}
			}
		}

		// sum over next temps
		nexttemps = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(nexttemps);
		ownednodes = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(ownednodes);
		overlongoutputs = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(overlongoutputs);
		if (threadIdx.x % 32 == 0)
		{
			atomicAdd(&temp.nextoutputs, nexttemps);
			atomicAdd(&temp.ownednodes, ownednodes);
			atomicAdd(&temp.overlongoutputs, overlongoutputs);
		}

	}
	__device__ void discoverLong(StateTemp& state, TempData& temp, uint32_t startOffset, uint32_t endOffset)
	{

		if (threadIdx.x == 0)
		{
			uint32_t node = __ldcg(&algorithmState.permutation[startOffset]);
			uint32_t rw0 = algorithmState.row_offsets[node];
			uint32_t rw1 = algorithmState.row_offsets[node + 1];
			temp.ownednodes = rw1 - rw0;
			temp.nextoutputs = 0;
			temp.overlongoutputs = 0;



			if (rw1 - rw0 <= MaxWorkSize)
			{
				temp.ownednodes = 0;
				getTemporaries().nodeOffsets[0] = 0;
				getTemporaries().nodeOffsets[1] = rw1 - rw0;
				temp.sortperthreads = divup(rw1 - rw0, BlockSize);

				DPRINTF("%d %d (%d) discoverLong for [%d]:%d switches to normal as owned %d <= %d nodes\n", blockIdx.x, threadIdx.x, temp.batchId, startOffset, node, rw1 - rw0, MaxWorkSize);
			}
			else
			{
				if (temp.earlyState < Discovered)
				{
					temp.minVal = rw0;
					temp.maxVal = rw1;
				}
				else
				{
					temp.minVal = 0xFFFFFFFF;
					temp.maxVal = 0;
				}
				temp.valencies_offsets[0] = rw0;
				temp.valencies_offsets[1] = rw1;
				getTemporaries().nodeOffsets[0] = rw0;
				getTemporaries().nodeOffsets[1] = rw1;
				temp.sortperthreads = divup(rw1 - rw0, BlockSize);

				DPRINTF("%d %d (%d) discoverLong for [%d]:%d needs full long as %d > %d nodes\n", blockIdx.x, threadIdx.x, temp.batchId, startOffset, node, rw1 - rw0, MaxWorkSize);
			}
		}

		__syncthreads();

		if (temp.ownednodes == 0)
		{
			temp.combine_long = 0;
			uint32_t threads_per_node_shift = BlockSizePow;
			uint32_t threads_per_node = 1u << threads_per_node_shift;
			discoverInner(state, temp, startOffset, endOffset, threads_per_node_shift, threads_per_node);
			return;
		}

		// else determine min/max for histogram and discover

		uint32_t endNode = getTemporaries().nodeOffsets[1];
		uint32_t r = getTemporaries().nodeOffsets[0] + threadIdx.x;
		uint32_t minVal = 0xFFFFFFFF;
		uint32_t maxVal = 0;
		uint32_t discovercode = (state.runstate << VisitedShift);
		uint32_t notownednodes = 0;
		uint32_t nextoutputs = 0;
		uint32_t overlongoutputs = 0;
		for (; r < endNode; r += BlockSize)
		{

			uint32_t childnode = algorithmState.col_ids[r];
			uint32_t old = atomicMin(&algorithmState.visited[childnode], discovercode);

			if (old > discovercode)
			{
				// we own it
				uint32_t child_valency = algorithmState.row_offsets[childnode + 1] - algorithmState.row_offsets[childnode];
				nextoutputs += child_valency;
				overlongoutputs += max(static_cast<int>(child_valency) - static_cast<int>(MaxWorkSize), 0);
				minVal = min(minVal, child_valency);
				maxVal = max(maxVal, child_valency);
			}
			else
				++notownednodes;
		}


		// sum over next temps
		nextoutputs = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(nextoutputs);
		notownednodes = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(notownednodes);
		overlongoutputs = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(overlongoutputs);

		if (temp.earlyState >= Discovered)
		{
			minVal = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Reduce(minVal, cub::Min());
			maxVal = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Reduce(maxVal, cub::Max());
		}

		if (threadIdx.x % 32 == 0)
		{
			atomicAdd(&temp.nextoutputs, nextoutputs);
			atomicAdd(&temp.overlongoutputs, overlongoutputs);
			atomicSub(&temp.ownednodes, notownednodes);
			if (temp.earlyState >= Discovered)
			{
				atomicMin(&temp.minVal, minVal);
				atomicMax(&temp.maxVal, maxVal);
			}
		}
	}
	__device__ void rediscover(TempData& temp)
	{
		uint32_t nextoutputsReduce = 0;
		uint32_t ownednodesReduce = 0;
		uint32_t overlongoutputReduce = 0;
		uint32_t batchcode = (temp.batchId << VisitedShift);

		for (uint32_t i = 0; i < temp.sortperthreads; ++i)
		{
			uint32_t id = threadIdx.x + i * BlockSize;
			uint32_t childnode = temp.nodes[id];

			if (childnode != 0xFFFFFFFF)
			{
				if (__ldcg(&algorithmState.visited[childnode]) < batchcode)
				{
					uint32_t entry = temp.valencies_offsets[id];
					uint32_t child_valency = entry & ((1u << ScanOffsetShift) - 1);
					nextoutputsReduce += child_valency;
					++ownednodesReduce;
					overlongoutputReduce += max(static_cast<int>(child_valency) - static_cast<int>(MaxWorkSize), 0);
					temp.valencies_offsets[id] = 0xFFFFFFFF;
					temp.nodes[id] = 0xFFFFFFFF;
				}
			}
		}

		nextoutputsReduce = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(nextoutputsReduce);
		ownednodesReduce = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(ownednodesReduce);
		overlongoutputReduce = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(overlongoutputReduce);
		if (threadIdx.x % 32 == 0)
		{
			atomicSub(&temp.nextoutputs, nextoutputsReduce);
			atomicSub(&temp.ownednodes, ownednodesReduce);
			atomicSub(&temp.overlongoutputs, overlongoutputReduce);
		}
	}

	__device__ void rediscoverLong(TempData& temp)
	{
		uint32_t nextoutputs = 0;
		uint32_t ownednodes = 0;
		uint32_t overlongoutputs = 0;
		uint32_t batchcode = (temp.batchId << VisitedShift);

		uint32_t minVal = 0xFFFFFFFF;
		uint32_t maxVal = 0;


		for (uint32_t id = temp.minVal + threadIdx.x; id < temp.maxVal; id += BlockSize)
		{
			uint32_t childnode = algorithmState.col_ids[id];
			if (__ldcg(&algorithmState.visited[childnode]) == batchcode)
			{
				uint32_t child_valency = algorithmState.row_offsets[childnode + 1] - algorithmState.row_offsets[childnode];
				overlongoutputs += max(static_cast<int>(child_valency) - static_cast<int>(MaxWorkSize), 0);
				nextoutputs += child_valency;
				minVal = min(minVal, child_valency);
				maxVal = max(maxVal, child_valency);
				++ownednodes;
			}
		}

		__syncthreads();

		if (threadIdx.x == 0)
		{
			temp.nextoutputs = 0;
			temp.overlongoutputs = 0;
			temp.ownednodes = 0;
			temp.minVal = 0xFFFFFFFF;
			temp.maxVal = 0;
		}

		__syncthreads();
		nextoutputs = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(nextoutputs);
		ownednodes = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(ownednodes);
		overlongoutputs = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Sum(overlongoutputs);
		minVal = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Reduce(minVal, cub::Min());
		maxVal = cub::WarpReduce<uint32_t>(getTemporaries().WarpReduceTemp).Reduce(maxVal, cub::Max());



		if (threadIdx.x % 32 == 0)
		{
			atomicAdd(&temp.nextoutputs, nextoutputs);
			atomicAdd(&temp.ownednodes, ownednodes);
			atomicAdd(&temp.overlongoutputs, overlongoutputs);

			atomicMin(&temp.minVal, minVal);
			atomicMax(&temp.maxVal, maxVal);
		}
	}

	template<uint32_t ElementsPerThread, uint32_t MaxPerThread, bool Done = false>
	struct MySortIterator;

	template<uint32_t ElementsPerThread, uint32_t MaxPerThread>
	struct MySortIterator< ElementsPerThread, MaxPerThread, false>
	{

		__device__ static void sort(TempData& temp)
		{
			if (temp.sortperthreads > ElementsPerThread)
				MySortIterator<ElementsPerThread + 1, MaxPerThread, ElementsPerThread == MaxPerThread>::sort(temp);
			else
			{
				uint32_t keys[ElementsPerThread];
				uint32_t values[ElementsPerThread];

#pragma unroll
				for (uint32_t i = 0; i < ElementsPerThread; ++i)
				{
					uint32_t id = threadIdx.x + i * BlockSize;
					keys[i] = temp.valencies_offsets[id];
					values[i] = temp.nodes[id];
					if (values[i] == 0xFFFFFFFF)
					{
						// this one has been invalidated, so remove it
						keys[i] = 0xFFFFFFFF;
					}
				}

				using Sorter = cub::BlockRadixSort<uint32_t, BlockSize, ElementsPerThread, uint32_t>;
				Sorter(*reinterpret_cast<typename Sorter::TempStorage*>(&getTemporaries().RadixSortTempStorage)).SortBlockedToStriped(keys, values);

#pragma unroll
				for (uint32_t i = 0; i < ElementsPerThread; ++i)
				{
					uint32_t id = threadIdx.x + i * BlockSize;
					temp.valencies_offsets[id] = keys[i];
					temp.nodes[id] = values[i];
					//	CONDDPRINTF((keys[i] != 0xFFFFFFFF || values[i] != 0xFFFFFFFF), "%d %d sort[%d] %d[%d,%d]->%d\n", blockIdx.x, threadIdx.x, id, keys[i], keys[i] >> ScanOffsetShift, keys[i] & ((1u << ScanOffsetShift)-1), values[i]);
				}
			}
		}
	};

	template<uint32_t ElementsPerThread, uint32_t MaxPerThread>
	struct MySortIterator< ElementsPerThread, MaxPerThread, true>
	{
		static __device__ void sort(TempData& temp)
		{
			// never called
		}
	};

	__device__ void sort(TempData& temp)
	{
		MySortIterator<1, MaxElementsPerThreads>::sort(temp);
	}


	__device__ uint32_t updateTemp(StateTemp& state, TempData& temp)
	{
		// single threaded check state and then move on
		__shared__ uint32_t currentState;
		if (threadIdx.x == 0)
		{
			Batch& batch = queue[temp.batchId];
			currentState = __ldcg(&batch.state);
			if (currentState > temp.lastState && temp.lastState < ReadyNext)
			{
				uint32_t usestate = temp.lastState >= Discovered ? currentState : Discovered;
				DPRINTF("%d %d updateTemp calling update Next with %d\n", blockIdx.x, threadIdx.x, usestate);
				updateNext(temp, usestate);
			}

			DPRINTF("%d %d updateTemp for %d has state %d and prev %d \n", blockIdx.x, threadIdx.x, temp.batchId, currentState, temp.lastState);
		}

		__syncthreads();

		if (currentState <= temp.lastState && currentState < CombNext)
			return temp.lastState;


		if (temp.lastState < Discovered)
		{
			CONDDPRINTF(threadIdx.x == 0, "%d %d updateTemp calling rediscover\n", blockIdx.x, threadIdx.x);
			//do we need to rediscover
			if (temp.ownednodes != 0)
			{
				if (temp.combine_long > 1)
					rediscoverLong(temp);
				else
					rediscover(temp);
			}
			__syncthreads();
			CONDDPRINTF(threadIdx.x == 0 && (temp.batchId == 842 || temp.batchId == 841), "%d %d rediscover updateTemp done, now owning %d nodes with %d next temporaries\n", blockIdx.x, threadIdx.x, temp.ownednodes, temp.nextoutputs);

		}

		if (threadIdx.x == 0)
		{
			if (currentState >= Discovered && temp.lastState < Discovered)
			{
				DPRINTF("%d %d updateTemp calling update Next with %d\n", blockIdx.x, threadIdx.x, currentState);
				updateNext(temp, currentState);
			}
			temp.lastState = currentState;
		}

		if (currentState == CombNext)
		{
			// finish this batch
			__shared__ uint32_t boffset, followupStartBatchId, bcombine, bcombineOutputs, batches, rBegin, rEnd, scounter;

			if (threadIdx.x == 0)
			{
				// compute all offsets
				Batch& batch = queue[temp.batchId];
				uint4 offset_followupStartBatchId_combine_combineOutputs = __ldcg(&batch.offset_followupStartBatchId_combine_combineOutputs);

				boffset = offset_followupStartBatchId_combine_combineOutputs.x;
				followupStartBatchId = offset_followupStartBatchId_combine_combineOutputs.y;
				bcombine = offset_followupStartBatchId_combine_combineOutputs.z;
				bcombineOutputs = offset_followupStartBatchId_combine_combineOutputs.w;

				uint32_t nextNodes = temp.ownednodes;
				uint32_t overallNextTemps = temp.nextoutputs;

				uint32_t nodes = offset_followupStartBatchId_combine_combineOutputs.z + nextNodes;
				uint32_t outputs = offset_followupStartBatchId_combine_combineOutputs.w + overallNextTemps;

				if ((temp.combine_long & 0x1) == 0x1 || nodes == 0)
					batches = 0;
				else
					batches = min(nextNodes + 1, max(divup(nodes, MaxInputNodes), divup((outputs - temp.overlongoutputs) * TempNodeOverallocNom / TempNodeOverallocDenom, MaxWorkSize)));

				DPRINTF("%d %d batches: %d = min(%d + 1, max(divup(%d, %d), divup((%d - %d) * %d / %d, %d)));\n", blockIdx.x, threadIdx.x, batches, nextNodes, nodes, MaxInputNodes, outputs, temp.overlongoutputs, TempNodeOverallocNom, TempNodeOverallocDenom, MaxWorkSize);

				// set the final node for state
				state.finishedNode = max(state.finishedNode, offset_followupStartBatchId_combine_combineOutputs.x + nextNodes);
			}
			__syncthreads();

			// write outputs and activate followups
			if (temp.combine_long > 1 && temp.ownednodes > 0)
			{
				// set early Discovery for simpler writeout
				temp.earlyState = Discovered;
				rBegin = temp.valencies_offsets[0];
				rEnd = temp.valencies_offsets[1];
				temp.sortperthreads = divup(temp.ownednodes, BlockSize);
				scounter = 0;

				__syncthreads();


				if (temp.ownednodes < MaxWorkSize)
				{
					// do not create histogram, but call compact reader and sorter directly
					compactReader<false, false>(temp, rBegin, rEnd, 0, 0xFFFFFFFF, scounter);
					__syncthreads();

					CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) calling sort with %d\n", blockIdx.x, threadIdx.x, temp.batchId, temp.sortperthreads);

					sort(temp);

					__syncthreads();

					CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) simplified long case in last part collected %d nodes (%d promised)\n", blockIdx.x, threadIdx.x, temp.batchId, scounter, temp.ownednodes);
				}
				else
				{
					CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) sort with hist case: owned: %d, node range %d %d - val range %d %d \n",
						blockIdx.x, threadIdx.x, temp.batchId, temp.ownednodes, rBegin, rEnd, temp.minVal, temp.maxVal);

					// create histogram considering minVal and maxVal
					__shared__ HistogramResultType partResets, overflows;
					__shared__ uint32_t histstart, histend, toverflow;

					histend = 0;

					constructHist(temp, rBegin, rEnd, partResets, overflows);


					CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) construct hist return %x reset code \n", blockIdx.x, threadIdx.x, temp.batchId, partResets);

					// work on one part after the other
					do
					{
						__syncthreads();
						if (threadIdx.x == 0)
						{

							histstart = histend;
							//uint32_t histMul = divup(1u + temp.maxVal - temp.minVal, HistogramBins);


							uint32_t mean = (temp.nextoutputs + temp.ownednodes / 2) / temp.ownednodes;
							uint32_t nextEnd = __ffs(partResets);
							if (nextEnd == 0)
								histend = 0xFFFFFFFF;
							else if (nextEnd - 1 < HistogramBins / 2)
							{
								uint32_t divlower = divup(1u + mean - temp.minVal, HistogramBins / 2);
								histend = temp.minVal + divlower * (nextEnd - 1);
							}
							else
							{
								uint32_t divupper = divup(1u + temp.maxVal - mean, HistogramBins / 2);
								histend = mean + divupper * (nextEnd - 1 - HistogramBins / 2);
							}


							//histend = nextEnd == 0 ? 0xFFFFFFFF : temp.minVal + histMul * (nextEnd - 1);
							uint32_t selector = (static_cast<HistogramResultType>(1) << (nextEnd - 1));

							if ((selector & overflows) != 0)
							{
								DPRINTF("%d %d (%d) overflow at %d   %x  %x\n", blockIdx.x, threadIdx.x, temp.batchId, nextEnd - 1, overflows, partResets);
								toverflow = 0;
							}
							else
								toverflow = 0xFFFFFFFF;

							partResets = partResets & (~selector);
							scounter = 0;
						}
						__syncthreads();

						CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) long case call compact for %u - %u \n", blockIdx.x, threadIdx.x, temp.batchId, histstart, histend);


						do
						{
							// call compact reader and sorter
							if (toverflow == 0xFFFFFFFF)
							{
								compactReader<true, false>(temp, rBegin, rEnd, histstart, histend, scounter);
							}
							else
							{
								__syncthreads();
								scounter = 0;
								uint32_t doneoffset = compactReader<true, true>(temp, rBegin, rEnd, histstart, histend, scounter, toverflow);
								if (threadIdx.x == 0)
								{

									DPRINTF("%d %d (%d) overflow compact reader from offset %d has written %d => new is %d .. will write batches %d..%d\n",
										blockIdx.x, threadIdx.x, temp.batchId, toverflow, scounter, doneoffset, followupStartBatchId, followupStartBatchId + batches);
									toverflow = doneoffset;

								}
							}

							__syncthreads();

							sort(temp);
							__syncthreads();

							CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) long case for %u - %u collected %d nodes (of %d overall)\n", blockIdx.x, threadIdx.x, temp.batchId, histstart, histend, scounter, temp.ownednodes);


							// write the part (specialized write that does not fill up empty or last and store overhead next)
							if (histend != 0xFFFFFFFF || toverflow != 0xFFFFFFFF)
								write<false>(temp, boffset, bcombine, bcombineOutputs, followupStartBatchId, batches, scounter);
							else
							{
								//	called outside
								CONDDPRINTF(threadIdx.x == 0 && temp.batchId > 1700 && temp.batchId < 1705,
									"%d %d (%d) final write prepared for outside for %d %d %d %d %d %d",
									blockIdx.x, threadIdx.x, temp.batchId,
									boffset, bcombine, bcombineOutputs, followupStartBatchId, batches, scounter);
								if (threadIdx.x == 0)
									temp.ownednodes = scounter;
								__syncthreads();
							}
						} while (toverflow != 0xFFFFFFFF);
					} while (histend != 0xFFFFFFFF);
				}
			}

			if (temp.ownednodes > 0 || batches > 0)
				write<true>(temp, boffset, bcombine, bcombineOutputs, followupStartBatchId, batches, temp.ownednodes);
			__syncthreads();


			// set next one if we combined
			if (threadIdx.x == 0)
			{
				if ((temp.combine_long & 0x1) == 0x1)
				{
					Batch& nextbatch = queue[temp.batchId + 1];
					__threadfence_block();
					__stcg(&nextbatch.state, CombNext);
				}
				// free this one
				queue.completed(temp.batchId);
				temp.lastState = currentState = Finished;
				DPRINTF("%d %d updateTemp completed batch %d\n", blockIdx.x, threadIdx.x, temp.batchId);
			}

		}
		__syncthreads();
		return currentState;
	}


	template<uint32_t ElementsPerThread, uint32_t MaxPerThread, bool Done = false >
	struct ScanIterator;

	template<uint32_t ElementsPerThread, uint32_t MaxPerThread>
	struct ScanIterator<ElementsPerThread, MaxPerThread, false>
	{
		template<bool OFFSETS>
		__device__ static void scan(TempData& temp, uint32_t shift = ScanOffsetShift)
		{
			if (temp.sortperthreads > ElementsPerThread)
				ScanIterator<ElementsPerThread + 1, MaxPerThread, ElementsPerThread == MaxPerThread>:: template scan<OFFSETS>(temp, shift);
			else
			{
				uint32_t data[ElementsPerThread];

				CONDDPRINTF(threadIdx.x == 0 && temp.ownednodes >= (1 << (32 - shift)) && temp.ownednodes < MaxWorkSize,
					"%d %d (%d) scan overflow?: %d >= %d (useshift: %d) ... %d > %d?\n",
					blockIdx.x, threadIdx.x, temp.batchId,
					temp.nextoutputs, (1 << shift), shift, temp.ownednodes, (1 << (32 - shift)));
#pragma unroll
				for (uint32_t i = 0; i < ElementsPerThread; ++i)
				{
					uint32_t id = threadIdx.x * ElementsPerThread + i;

					uint32_t tdata = temp.valencies_offsets[id];
					if (tdata == 0xFFFFFFFF)
					{
						tdata = 0;
					}
					else
					{
						CONDDPRINTF(temp.batchId == 11,
							"%d %d (%d) scan input %d: %d %d (%d -> %d) => %x  (node: %d)\n",
							blockIdx.x, threadIdx.x, temp.batchId, i,
							tdata, tdata & ((1u << ScanOffsetShift) - 1), OFFSETS, (1u << ScanOffsetShift),
							(tdata & ((1u << ScanOffsetShift) - 1)) | (OFFSETS ? (1u << ScanOffsetShift) : 0),
							temp.nodes[id]);

						tdata = tdata & ((1u << ScanOffsetShift) - 1);
						if (OFFSETS)
							tdata = tdata | (1u << shift);

					}
					data[i] = tdata;
				}

				using Scanner = cub::BlockScan<uint32_t, BlockSize>;
				uint32_t aggregate;
				Scanner(getTemporaries().BlockScanTempStorage).ExclusiveSum(data, data, aggregate);

				__syncthreads();

#pragma unroll
				for (uint32_t i = 0; i < ElementsPerThread; ++i)
				{
					uint32_t id = threadIdx.x * ElementsPerThread + i;
					temp.valencies_offsets[id] = data[i];

					CONDDPRINTF(temp.batchId == 11,
						"%d %d (%d) scan output[%d/%d]: %x\n",
						blockIdx.x, threadIdx.x, temp.batchId, i, id, data[i]);
				}
				if (threadIdx.x == 0)
					temp.valencies_offsets[BlockSize * ElementsPerThread] = aggregate;
			}
		}
	};

	template<uint32_t ElementsPerThread, uint32_t MaxPerThread>
	struct ScanIterator<ElementsPerThread, MaxPerThread, true>
	{
		template<bool OFFSETS>
		__device__ static void scan(TempData& temp, uint32_t shift = 0)
		{
			// never called
		}
	};

	template<bool FINISHNODE>
	__device__ void write(TempData& temp, uint32_t& offset, uint32_t& writtenNodes, uint32_t& writtenTemps, uint32_t& followupStartBatchId, uint32_t& batches, const uint32_t& tnodes)
	{

		CONDDPRINTF(threadIdx.x == 0 && followupStartBatchId < 1960 && followupStartBatchId + batches > 1940, "%d %d (%d) write<%d> called with  offset %d  writtenNodes %d  writtenTemps %d  followupStartBatchId %d  batches %d ... long %d\n",
			blockIdx.x, threadIdx.x, temp.batchId, FINISHNODE, offset, writtenNodes, writtenTemps, followupStartBatchId, batches, temp.combine_long);

		uint32_t shift = ScanOffsetShift;
		if (temp.earlyState >= Discovered)
		{
			// directly write out
			for (uint32_t i = 0; i < temp.sortperthreads; ++i)
			{
				uint32_t id = threadIdx.x + i * BlockSize;
				uint32_t entry = temp.valencies_offsets[id];
				if (entry != 0xFFFFFFFF)
				{
					uint32_t childnode = temp.nodes[id];
					__stcg(&algorithmState.permutation[offset + id], childnode);
					CONDDPRINTF((offset + id >= 1350 && offset + id < 1500), "%d %d (%d) wrote directly [%d+%d=%d] = %d\n",
						blockIdx.x, threadIdx.x, temp.batchId, offset, id, offset + id, childnode);
				}
			}
			// run prefix sum - only for next batches if EarlyDiscovered
			ScanIterator<1, MaxElementsPerThreads>::template scan<false>(temp);
			__syncthreads();
			CONDDPRINTF(threadIdx.x == 0 && temp.combine_long > 1, "%d %d (%d) wrote nodes %d -> %d <%d\n",
				blockIdx.x, threadIdx.x, temp.batchId, offset, offset + tnodes, offset + temp.sortperthreads * BlockSize);
		}
		else
		{
			CONDDPRINTF(threadIdx.x == 0 && temp.nextoutputs > ((1u << ScanOffsetShift) - 1), "%d %d (%d) masked scan overflow!! %d > %d\n",
				blockIdx.x, threadIdx.x, temp.batchId, temp.nextoutputs, ((1u << ScanOffsetShift) - 1));

			// run prefix sum - for offset and next batches if !EarlyDiscovered
			if (temp.nextoutputs >= (1u << ScanOffsetShift))
				shift = 32 - __clz(temp.nextoutputs);
			ScanIterator<1, MaxElementsPerThreads>::template scan<true>(temp, shift);
			__syncthreads();

			// write out
			for (uint32_t i = 0; i < temp.sortperthreads; ++i)
			{
				uint32_t id = threadIdx.x + i * BlockSize;
				uint32_t outoffset = temp.valencies_offsets[id] >> shift;
				uint32_t childnode = temp.nodes[id];

				CONDDPRINTF(threadIdx.x == 0 && temp.batchId == 11,
					"%d %d (%d) data before write: %d %x  %d and %d\n",
					blockIdx.x, threadIdx.x, temp.batchId, id, temp.valencies_offsets[id], outoffset, childnode);

				if (childnode != 0xFFFFFFFF)
				{
					__stcg(&algorithmState.permutation[offset + outoffset], childnode);
					CONDDPRINTF((offset + id >= 1350 && offset + id < 1500), "%d %d (%d) wrote after scan [%d] = %d\n",
						blockIdx.x, threadIdx.x, temp.batchId, offset + outoffset, childnode);
				}

			}
			__syncthreads();
		}

		// warp wide new batch finding
		if ((temp.combine_long & 0x1) == 0 && threadIdx.x < 32)
		{

			getTemporaries().outputs[0] = offset - writtenNodes;

			int subnodes = -writtenNodes;
			int subtemps = -writtenTemps;
			uint32_t mask = temp.earlyState >= Discovered ? 0xFFFFFFFF : ((1u << shift) - 1u);
			uint32_t usedBatches = 0;
			uint32_t checkentries = temp.sortperthreads * BlockSize;
			if (temp.earlyState >= Discovered)
				checkentries = tnodes;
			for (int i = 0; i < checkentries; i += 32)
			{
				int off = i + threadIdx.x + 1;
				uint32_t val = temp.valencies_offsets[off];
				int ttemps = (val & mask);

				if (temp.earlyState < Discovered)
					off = val >> shift;
				else
					off = min(off, static_cast<int>(tnodes));

				while (true)
				{
					int tnodes = off - subnodes;
					int exttemps = ttemps - subtemps;
					bool end = (tnodes > 1 || subnodes < 0) && (tnodes > (static_cast<int>(MaxInputNodes)) || exttemps > (static_cast<int>(MaxWorkSize)));

					CONDDPRINTF(i == 0 && followupStartBatchId > 1220 && followupStartBatchId < 1300,
						"%d %d need to end batch[%d] before? %d with %d (%d - %d) > %d || %d > %d\n",
						blockIdx.x, threadIdx.x, followupStartBatchId + usedBatches, offset + off, tnodes, off, subnodes, MaxInputNodes, exttemps, MaxWorkSize);
					uint32_t ballotres = __ballot_sync(0xFFFFFFFF, end);

					if (ballotres != 0)
					{
						uint32_t twrite = (usedBatches % 32) + 1;
						++usedBatches;
						int endthread = __ffs(ballotres) - 1; // we would subtract one here but add one too and then subtract one again
						// write new batch offsets to smem
						uint32_t endoffset = i + endthread;
						if (temp.earlyState < Discovered)
							endoffset = temp.valencies_offsets[endoffset] >> shift;

						getTemporaries().outputs[twrite] = offset + endoffset;

						if (twrite == 32)
						{
							__syncwarp();
							// write them out
							uint32_t followBatchId = followupStartBatchId + usedBatches - 32 + threadIdx.x;
							Batch& nextBatch = queue[followBatchId];
							uint2 followInput{ getTemporaries().outputs[threadIdx.x], getTemporaries().outputs[threadIdx.x + 1] };
							__stcg(&nextBatch.input_start_end, followInput);
							CONDDPRINTF(temp.batchId == 842 || temp.batchId == 841,
								"%d %d activating batch %d  (%d/%d) (%d -> %d)\n", blockIdx.x, threadIdx.x, followBatchId, usedBatches - 32 + threadIdx.x, batches, followInput.x, followInput.y);

							__threadfence();
							CONDDPRINTF(followBatchId == 1922, "%d %d (%d) activating batch %d\n", blockIdx.x, threadIdx.x, temp.batchId, followBatchId);
							queue.ready(followBatchId);
							__syncwarp();
							// reset info
							getTemporaries().outputs[0] = offset + endoffset;
						}
						subnodes = endoffset;
						subtemps = temp.valencies_offsets[i + endthread] & mask;
					}
					else
						break;
				}
			}

			// finalize with last batch - check whether the last one contains any nodes
			int remnodes = tnodes - subnodes;
			int remtemps = (temp.valencies_offsets[BlockSize * temp.sortperthreads] & mask) - subtemps;
			uint32_t twrite = (usedBatches % 32);
			if ((FINISHNODE && remnodes > 0) ||
				(!FINISHNODE && (remnodes > (static_cast<int>(MaxInputNodes)) || remtemps > (static_cast<int>(MaxWorkSize)))))
			{
				CONDDPRINTF(threadIdx.x == 0 && temp.nextoutputs - subtemps > MaxWorkSize,
					"%d %d adding too large batch!!! [%d] %d   %d > %d ... nodes: %d at %d \n",
					blockIdx.x, threadIdx.x,
					twrite, offset + temp.ownednodes, temp.nextoutputs - subtemps, MaxWorkSize, remnodes, followupStartBatchId + usedBatches - twrite);
				twrite = twrite + 1;
				++usedBatches;
				getTemporaries().outputs[twrite] = offset + tnodes;

				subnodes = tnodes;
				subtemps = (temp.valencies_offsets[BlockSize * temp.sortperthreads] & mask);
				CONDDPRINTF(threadIdx.x == 0 && temp.batchId >= 841 && temp.batchId <= 843, "%d %d adding final batch [%d] %d\n",
					blockIdx.x, threadIdx.x, twrite, offset + temp.ownednodes);
			}
			__syncwarp();
			// write the rest from smem
			if (threadIdx.x < twrite)
			{

				uint32_t followBatchId = followupStartBatchId + usedBatches - twrite + threadIdx.x;
				Batch& nextBatch = queue[followBatchId];
				uint2 followInput{ getTemporaries().outputs[threadIdx.x], getTemporaries().outputs[threadIdx.x + 1] };
				__stcg(&nextBatch.input_start_end, followInput);
				CONDDPRINTF(temp.batchId == 842 || temp.batchId == 841,
					"%d %d activating batch outside %d  (%d/%d) (%d -> %d)  - early: %d\n", blockIdx.x, threadIdx.x, followBatchId, usedBatches - twrite + threadIdx.x, batches, followInput.x, followInput.y, temp.earlyState >= Discovered ? 1 : 0);

				__threadfence();
				CONDDPRINTF(followBatchId == 1922, "%d %d (%d) activating batch outside %d\n", blockIdx.x, threadIdx.x, temp.batchId, followBatchId);

				queue.ready(followBatchId);
			}

			// add empty batches to match number of previously computed batches

			DPRINTF("%d %d activating empty usedBatches %d  batches %d  ownednodes %d \n", blockIdx.x, threadIdx.x, usedBatches, batches, temp.ownednodes);

			if constexpr (FINISHNODE)
			{
				while (usedBatches < batches)
				{
					uint32_t endnode = offset + tnodes;
					if (threadIdx.x < batches - usedBatches)
					{
						uint32_t followBatchId = followupStartBatchId + usedBatches + threadIdx.x;
						Batch& nextBatch = queue[followBatchId];
						uint2 followInput{ endnode, endnode };
						__stcg(&nextBatch.input_start_end, followInput);
						DPRINTF("%d %d activating empty batch %d (%d -> %d) due to %d < %d\n", blockIdx.x, threadIdx.x, followBatchId, followInput.x, followInput.y, usedBatches, batches);
						__threadfence_block();
						queue.ready(followBatchId);
					}
					usedBatches += 32;
				}
			}
			else
			{
				__syncwarp();
				if (threadIdx.x == 0)
				{
					followupStartBatchId += usedBatches;
					batches -= usedBatches;

					offset += tnodes;
					writtenNodes = tnodes - subnodes;
					writtenTemps = temp.valencies_offsets[temp.sortperthreads * BlockSize] - subtemps;

					DPRINTF("%d %d (%d) seeting up for next round followupStartBatchId %d, batches %d, offset %d, writtenNodes %d, writtenTemps %d\n",
						blockIdx.x, threadIdx.x, temp.batchId, followupStartBatchId, batches, offset, writtenNodes, writtenTemps);
				}
			}
		}
	}


	__device__ void constructHist(TempData& temp, const uint32_t& rBegin, const uint32_t& rEnd, HistogramResultType& partResets, HistogramResultType& overflow)
	{
		if (threadIdx.x < HistogramBins)
			getTemporaries().histogram[threadIdx.x] = 0;
		__syncthreads();

		uint32_t mean = (temp.nextoutputs + temp.ownednodes / 2) / temp.ownednodes;
		uint32_t divlower = divup(1u + mean - temp.minVal, HistogramBins / 2);
		uint32_t divupper = divup(1u + temp.maxVal - mean, HistogramBins / 2);

		uint64_t histogramCounters[HistogramCounters];
		constexpr uint32_t HistMaxCounter = (1u << HistogramCollectionBits) - 1u;
		for (uint32_t i = 0; i < HistogramCounters; ++i)
			histogramCounters[i] = 0;

		CONDDPRINTF(threadIdx.x == 0, "%d %d (%d) minVal %d, maxVal %d, meanVal %d -> divlower %d  divupper %d\n",
			blockIdx.x, threadIdx.x, temp.batchId, temp.minVal, temp.maxVal, mean, divlower, divupper);


		// value collection
		uint32_t batchcode = (temp.batchId << VisitedShift);
		for (uint32_t id = rBegin + threadIdx.x; id < rEnd; id += BlockSize)
		{
			uint32_t childnode = algorithmState.col_ids[id];
			if (__ldcg(&algorithmState.visited[childnode]) == batchcode)
			{
				uint32_t valency = algorithmState.row_offsets[childnode + 1] - algorithmState.row_offsets[childnode];

				uint32_t bin;
				if (valency < mean)
					bin = (valency - temp.minVal) / divlower;
				else
					bin = HistogramBins / 2 + (valency - mean) / divupper;

				uint32_t offset = bin * HistogramCollectionBits;
				uint32_t word = offset / 64u;
				uint32_t lshift = offset % 64u;
				uint64_t val = 1ULL << lshift;

#pragma unroll
				for (uint32_t i = 0; i < HistogramCounters; ++i)
				{
					if (word == i)
					{


						// overflow protection
						if (((histogramCounters[i] >> lshift) & HistMaxCounter) == HistMaxCounter)
						{
							atomicAdd(&getTemporaries().histogram[bin], HistMaxCounter + 1);
							histogramCounters[i] = histogramCounters[i] & (~(static_cast<uint64_t>(HistMaxCounter) << lshift));
						}
						else
							histogramCounters[i] += val;


					}
				}
			}
		}

		DPRINTF("%d %d histogramCounters %llx %llx  (%d -> %d / %d)\n", blockIdx.x, threadIdx.x, histogramCounters[0], histogramCounters[1], temp.minVal, temp.maxVal, div);

		// warp vote hist collection
		uint32_t offset = 0;
		uint32_t mycount = 0;
#pragma unroll
		for (uint32_t i = 0; i < HistogramCounters; ++i)
		{
			for (uint32_t lshift = 0; lshift < 64u; lshift += HistogramCollectionBits)
			{
				for (uint32_t bit = 0; bit < HistogramCollectionBits; ++bit)
				{
					uint32_t res = __ballot_sync(0xFFFFFFFF, ((histogramCounters[i] >> (lshift + bit)) & 0x1LL) == 0x1LL);
					if ((threadIdx.x % 32) == (offset % 32))
						mycount += __popc(res) * (1u << bit);
				}
				++offset;
				if (offset % 32 == 0)
				{
					atomicAdd(&getTemporaries().histogram[offset - 32 + (threadIdx.x % 32)], mycount);
					mycount = 0;
				}
			}
		}
		if (offset % 32 != 0)
			atomicAdd(&getTemporaries().histogram[offset / 32 * 32 + (threadIdx.x % 32)], mycount);

		__syncthreads();

		CONDDPRINTF(threadIdx.x < HistogramBins, "%d %d (%d) histogram[%d] = %d\n", blockIdx.x, threadIdx.x, temp.batchId, threadIdx.x, getTemporaries().histogram[threadIdx.x]);

		// compute resets within warp

		if (threadIdx.x < 32)
		{
			HistogramResultType lpartResets = 0;
			HistogramResultType loverflow = 0;
			int overhang = 0;
			for (int i = 0; i < HistogramBins; i += 32)
			{
				int id = i + threadIdx.x;
				int input = 0;
				if (HistogramBins % 32 != 0 || id < HistogramBins)
					input = getTemporaries().histogram[i + threadIdx.x];

				int output;
				cub::WarpScan<int>(getTemporaries().WarpScanTemp).InclusiveSum(input, output);
				output += overhang;

				while (true)
				{
					uint32_t needreset = __ballot_sync(0xFFFFFFFF, output > static_cast<int>(MaxWorkSize));
					if (needreset != 0)
					{
						uint32_t first = __ffs(needreset) - 1;
						lpartResets = lpartResets | (static_cast<HistogramResultType>(1u) << (first + i));
						int red = __shfl_sync(0xFFFFFFFF, output - input, first);
						bool toverflow = __shfl_sync(0xFFFFFFFF, input > MaxWorkSize, first);

						output -= red;
						int overflowit = 1;
						while (toverflow)
						{
							lpartResets = lpartResets | (static_cast<HistogramResultType>(1u) << (first + i + overflowit));
							loverflow = loverflow | (static_cast<HistogramResultType>(1u) << (first + i + overflowit));
							int red = __shfl_sync(0xFFFFFFFF, output - input, first + overflowit);
							output -= red;
							toverflow = __shfl_sync(0xFFFFFFFF, input > MaxWorkSize, first + overflowit);
							++overflowit;
						}

						DPRINTF("%d %d (%d) hist reset %x (%d): %d > %d: %d -> red %d -> next %d  overflow: %d \n",
							blockIdx.x, threadIdx.x, temp.batchId, needreset, first, output, MaxWorkSize, (int)(output > static_cast<int>(MaxWorkSize)), red, output - red, toverflow);


					}
					else
						break;
				}
				overhang = __shfl_sync(0xFFFFFFFF, output, 31);
			}

			partResets = lpartResets;
			overflow = loverflow;

			CONDDPRINTF(threadIdx.x == 0 && loverflow != 0, "%d %d (%d) hist %x overflow %x min %d max %d mean %d divs: %d %d\n",
				blockIdx.x, threadIdx.x, temp.batchId, lpartResets, loverflow, temp.minVal, temp.maxVal, mean, divlower, divupper);
		}
	}


	template<bool CHECKVALENCY, bool AVOIDOVERFLOW>
	__device__ uint32_t compactReader(TempData& temp, const uint32_t& rBegin, const uint32_t& rEnd, uint32_t valBegin, uint32_t valEnd, uint32_t& scounter, uint32_t offset)
	{
		uint32_t batchcode = (temp.batchId << VisitedShift);
		uint32_t returnv = 0xFFFFFFFF;
		if constexpr (AVOIDOVERFLOW)
		{
			int localcounter = 0;
			for (uint32_t id = rBegin + offset; id < rEnd; id += BlockSize)
			{
				uint32_t childnode = 0xFFFFFFFF;
				uint32_t valency = 0xFFFFFFFF;
				if (threadIdx.x + id < rEnd)
				{
					childnode = algorithmState.col_ids[id + threadIdx.x];
					if (__ldcg(&algorithmState.visited[childnode]) == batchcode)
					{
						uint32_t v = algorithmState.row_offsets[childnode + 1] - algorithmState.row_offsets[childnode];
						if (!CHECKVALENCY || (v >= valBegin && v < valEnd))
							valency = v;
					}
				}
				int n = __syncthreads_count(valency != 0xFFFFFFFF);
				if (n + localcounter < static_cast<int>(MaxWorkSize))
				{
					localcounter += n;
					if (valency != 0xFFFFFFFF)
					{
						uint32_t p = atomicAdd(&scounter, 1);
						temp.nodes[p] = childnode;
						temp.valencies_offsets[p] = valency;
					}
				}
				else
				{
					returnv = id - rBegin;
					break;
				}
			}
		}
		else
		{
			for (uint32_t id = rBegin + offset + threadIdx.x; id < rEnd; id += BlockSize)
			{
				uint32_t childnode = algorithmState.col_ids[id];
				if (__ldcg(&algorithmState.visited[childnode]) == batchcode)
				{
					uint32_t valency = algorithmState.row_offsets[childnode + 1] - algorithmState.row_offsets[childnode];
					if (!CHECKVALENCY || (valency >= valBegin && valency < valEnd))
					{
						// TODO: combine atomics?
						uint32_t p = atomicAdd(&scounter, 1);
						temp.nodes[p] = childnode;
						temp.valencies_offsets[p] = valency;
					}
				}
			}

		}
		__syncthreads();
		uint32_t sortperthreads = divup(scounter, BlockSize);
		for (uint32_t i = scounter + threadIdx.x; i < BlockSize * sortperthreads; i += BlockSize)
		{
			CONDDPRINTF(temp.batchId == 841, "%d %d (%d) clearing %d\n", blockIdx.x, threadIdx.x, temp.batchId, i);
			temp.valencies_offsets[i] = 0xFFFFFFFF;
			temp.nodes[i] = 0xFFFFFFFF;
		}
		temp.sortperthreads = sortperthreads;
		return returnv;
	}
}