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

#include "batch.h"

#include <mutex>
#include <condition_variable>
#include <array>


class IBatchQueue
{
public:
	virtual bool startfill(uint32_t b) = 0;
	virtual bool tryfill(uint32_t b) = 0;
	virtual void ready(uint32_t b) = 0;
	virtual void completed(uint32_t b) = 0;
	virtual void end() = 0;
	virtual uint32_t get(bool nonblock) = 0;
	virtual Batch& at(uint32_t b) = 0;
	
};

template<size_t QUEUE_SIZE = 128*1024>
class BatchQueue : public IBatchQueue
{
	std::mutex mutex;
	std::condition_variable waiters;
	bool work;
	std::atomic<uint32_t> pointer;
	std::array<Batch, QUEUE_SIZE> ringbuffer;

public:
	static constexpr size_t Size = QUEUE_SIZE;
	void init()
	{
		work = true;
		pointer = 0;
		for (auto& b : ringbuffer)
		{
			b.init();
			b.consumestate = 0;
		}
	}
	Batch& operator[] (uint32_t b)
	{
		return ringbuffer[b % QUEUE_SIZE];
	}
	Batch& at(uint32_t b)
	{
		return ringbuffer[b % QUEUE_SIZE];
	}
	bool startfill(uint32_t b)
	{
		uint32_t empty = 4 * (b / QUEUE_SIZE);
		uint32_t filled = empty + 1;
		if (ringbuffer[b % QUEUE_SIZE].consumestate.compare_exchange_strong(empty, filled))
			return true;
		return false;
	}
	bool tryfill(uint32_t b)
	{
		uint32_t empty = 4 * (b / QUEUE_SIZE);
		uint32_t filled = empty + 1;
		if (ringbuffer[b % QUEUE_SIZE].consumestate.compare_exchange_strong(empty, filled))
			return true;
		return empty == filled;
	}
	void ready(uint32_t b)
	{
		uint32_t r = 4 * (b / QUEUE_SIZE) + 2;
		ringbuffer[b % QUEUE_SIZE].consumestate.store(r);
		{
			std::unique_lock<std::mutex> l(mutex);
		}
		waiters.notify_all();
	}
	void completed(uint32_t b)
	{
		uint32_t r = 4 * ((b + QUEUE_SIZE) / QUEUE_SIZE);
		// uint32_t slot = b % QUEUE_SIZE;
		ringbuffer[b % QUEUE_SIZE].consumestate.store(r);
	}
	void end()
	{
		{
			std::unique_lock<std::mutex> l(mutex);
			work = false;
		}
		waiters.notify_all();
	}
	uint32_t get(bool nonblock)
	{
		uint32_t op = pointer.load(std::memory_order_relaxed);
		uint32_t p = op;
		uint32_t l = p % QUEUE_SIZE;

		while (true)
		{
			uint32_t last = 4 * (p / QUEUE_SIZE) + 2;
			uint32_t tlast = last;
			uint32_t next = last + 1;
			if (ringbuffer[l].consumestate.compare_exchange_strong(tlast, next))
				break;

			else if(tlast < last)
			{
				// recheck
				std::unique_lock<std::mutex> lock(mutex);
				tlast = last;
				if (ringbuffer[l].consumestate.compare_exchange_strong(tlast, next))
				{
					break;
				}
				else if (tlast < last)
				{
					if (nonblock)
						return 0xFFFFFFFE;
					if (!work)
						return 0xFFFFFFFF;

					waiters.wait(lock);
					lock.unlock();
					if (!work)
						return 0xFFFFFFFF;
					op = pointer.load(std::memory_order_relaxed);
					p = op;
					l = p % QUEUE_SIZE;
					continue;
				}
			}

			// last > BatchState::Ready -> move on
			++p;
			l = p % QUEUE_SIZE;
		}

		pointer.compare_exchange_weak(op, p + 1, std::memory_order_relaxed);
		return p;
	}

};