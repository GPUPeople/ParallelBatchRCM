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

#include "batch.cuh"



template<size_t QUEUE_SIZE = 128 * 1024, bool TRACK_ACTIVE_AND_END = false>
class BatchQueue
{
	uint32_t work;
	uint2 pointer_elements;
	Batch ringbuffer[QUEUE_SIZE];
	int32_t active;

public:
	static constexpr size_t Size = QUEUE_SIZE;

	__device__ void init(uint32_t start_node)
	{
		uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		for (uint32_t i = tid; i < QUEUE_SIZE; i += gridDim.x + blockDim.x)
		{
			ringbuffer[i].init();
			ringbuffer[i].consumestate = 0;
		}
		
		
		if (tid == 0)
		{
			active = 1;

			work = 1;
			pointer_elements = uint2{0,1};

			ringbuffer[0].state = CombNext;
			ringbuffer[0].consumestate = 2;
			ringbuffer[0].input_start_end = uint2{ 0,1 };
			ringbuffer[0].offset_followupStartBatchId_combine_combineOutputs = uint4{ 1,1,0,0 };
		}

	}
	__device__ Batch& operator[] (uint32_t b)
	{
		return ringbuffer[b % QUEUE_SIZE];
	}
	__device__ Batch& at(uint32_t b)
	{
		return ringbuffer[b % QUEUE_SIZE];
	}
	__device__ bool startfill(uint32_t b)
	{
		uint32_t empty = 4 * (b / QUEUE_SIZE);
		uint32_t filled = empty + 1;
		return atomicCAS(&ringbuffer[b % QUEUE_SIZE].consumestate, empty, filled) == empty;
	}
	__device__ bool tryfill(uint32_t b)
	{
		uint32_t empty = 4 * (b / QUEUE_SIZE);
		uint32_t filled = empty + 1;
		uint32_t old = atomicCAS(&ringbuffer[b % QUEUE_SIZE].consumestate, empty, filled);
		return old == empty || old == filled;
	}
	__device__ void ready(uint32_t b)
	{
		
		uint32_t r = 4 * (b / QUEUE_SIZE) + 2;
		uint32_t oldstate = atomicExch(&ringbuffer[b % QUEUE_SIZE].consumestate, r);
		if (TRACK_ACTIVE_AND_END && (oldstate == r - 1 || oldstate == r - 2))
		{
			atomicAdd(&active, 1);
		}

		atomicMax(&pointer_elements.y, b);
	}
	__device__ void completed(uint32_t b)
	{
		uint32_t r = 4 * ((b + QUEUE_SIZE) / QUEUE_SIZE);
		uint32_t slot = b % QUEUE_SIZE;
		ringbuffer[slot].reset();
		atomicExch(&ringbuffer[slot].consumestate, r);
	}
	__device__ void end()
	{
		__stcg(&work, false);
	}

	__device__ bool ended() const
	{
		return __ldcg(&work) == false;
	}
	template<bool BLOCK>
	__device__ uint32_t getReservedSpot(uint32_t spot, int32_t endstate)
	{
		uint32_t target = 4 * (spot / QUEUE_SIZE) + 2;
		uint32_t location = spot % QUEUE_SIZE;

		if (TRACK_ACTIVE_AND_END && BLOCK && __ldcg(&active) == endstate)
		{
			end();
			return 0xFFFFFFFF;
		}

		if constexpr (!BLOCK)
		{
			if (atomicCAS(&ringbuffer[location].consumestate, target, target + 1) != target)
			{
				if (!__ldcg(&work) && spot >= __ldcg(&pointer_elements.y))
					return 0xFFFFFFFF;
				return 0xFFFFFFFE;
			}
			return spot;
		}
		else
		{
			uint32_t nano = 8;
			while (atomicCAS(&ringbuffer[location].consumestate, target, target + 1) != target)
			{
				if (!__ldcg(&work) && spot >= __ldcg(&pointer_elements.y))
					return 0xFFFFFFFF;
				#if __CUDA_ARCH__ >= 700
				__nanosleep(nano);
				#else
				__threadfence();
				#endif
				if (nano < 512)
					nano *= 2;
			}
			return spot;
		}

		// to silence spurious nvcc warning about missing return in non-void function - unreachable
		return 0;
	}
	__device__ uint32_t getNonblocking(uint32_t& slot)
	{
		if constexpr(TRACK_ACTIVE_AND_END)
			atomicSub(&active, 1);
		slot = atomicAdd(&pointer_elements.x, 1);

		uint32_t res = getReservedSpot<false>(slot, 0);
		if (res == slot)
			slot = 0xFFFFFFFF;
		return res;
	}

	__device__ uint32_t get(int endstate)
	{
		if constexpr (TRACK_ACTIVE_AND_END)
		{ 
			int32_t nstate = atomicSub(&active, 1) - 1;
			if (nstate == endstate)
			{
				end();
				return 0xFFFFFFFF;
			}
		}
		uint32_t slot = atomicAdd(&pointer_elements.x, 1);

		return getReservedSpot<true>(slot, endstate);
	}

};
