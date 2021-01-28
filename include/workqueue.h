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
#include <atomic>
#include <thread>


template<typename T, T MAGIC, uint32_t SIZE = 512*1024>
class WorkQueue
{
	std::atomic<T> ringbuffer[SIZE];
	std::atomic<uint32_t> front, back;
	std::atomic<int32_t> count;
public:
	WorkQueue() : front(0), back(0), count(0)
	{
		for (size_t i = 0; i < SIZE; ++i)
		{
			ringbuffer[i] = MAGIC;
		}
	}
	void push(T val)
	{
		count.fetch_add(1);
		uint32_t pos = back.fetch_add(1) % SIZE;
		while ((val = ringbuffer[pos].exchange(val)) != MAGIC)
			std::this_thread::yield();
	}
	bool get(T& val)
	{
		int32_t cnt = count.load();
		while (cnt > 0)
		{
			cnt = count.fetch_sub(1);
			if (cnt <= 0)
			{
				cnt = count.fetch_add(1) + 1;
			}
			else
			{
				uint32_t pos = front.fetch_add(1) % SIZE;
				while ((val = ringbuffer[pos].exchange(MAGIC)) == MAGIC)
					std::this_thread::yield();
				return true;
			}
		}
		return false;
	}
};
