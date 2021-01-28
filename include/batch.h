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

#include <memory>
#include <atomic>
#include <stdint.h>


class BatchWorker;

enum BatchState
{
	Empty = 0,
	Filling = 1,
	Ready = 2,
	Taken = 3,

	Discovered = 4,
	CountSet = 5,
	ReadyNext = 6,
	CombNext = 7,
	Finished = 8
};

struct Batch
{
	friend class BatchWorker;
	template<size_t S>
	friend class BatchQueue;

	std::atomic<uint32_t> state;
	std::atomic<uint32_t> consumestate;
	uint32_t input_start, input_end;
	uint32_t offset;
	uint32_t followupStartBatchId;
	uint32_t combine;
	uint32_t combineOutputs;

public:
	void init()
	{
		combine = 0;
		state = BatchState::Empty;
	}

	void reset()
	{
		//combine = 0;
		state = BatchState::Empty;
	}

};