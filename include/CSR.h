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
#include <algorithm>
#include <math.h>
#include <cstring>
#include <string>

template<typename T>
struct COO;

template<typename T>
struct CSR
{
	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_offsets;
	std::unique_ptr<unsigned int[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t rows, size_t cols, size_t nnz, bool allocData=true);

	CSR(const CSR& csr) : rows{csr.rows}, cols{csr.cols}, nnz{csr.nnz}
	{
		row_offsets = std::make_unique<unsigned int[]>(rows + 1);
		col_ids = std::make_unique<unsigned int[]>(nnz);
		data = std::make_unique<T[]>(nnz);

		memcpy(row_offsets.get(), csr.row_offsets.get(), sizeof(unsigned int) * (rows + 1));
		memcpy(col_ids.get(), csr.col_ids.get(), sizeof(unsigned int) * nnz);
		memcpy(data.get(), csr.data.get(), sizeof(T) * nnz);
	}
	
	CSR& operator=(CSR other)
	{
		std::swap(rows, other.rows);
		std::swap(cols, other.cols);
		std::swap(nnz, other.nnz);
		std::swap(data, other.data);
		std::swap(row_offsets, other.row_offsets);
		std::swap(col_ids, other.col_ids);
		return *this;
	}
};


template<typename T>
CSR<T> loadCSR(const char* file);
template<typename T>
void storeCSR(const CSR<T>& mat, const char* file);

template<typename T>
void convert(CSR<T>& res, const COO<T>& coo);

template<typename T>
void convert(COO<T>& dst, const CSR<T>& src);

template<typename T>
std::string typeext();