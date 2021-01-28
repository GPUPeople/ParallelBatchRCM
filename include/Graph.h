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
#include <iostream>


#include "CSR.h"
#include "COO.h"

#include <unordered_map>
#include <queue>

template<typename T>
struct Graph
{
    size_t size;
    bool csr_ready = false;

    CSR<T> csr;

	Graph(){}
	Graph(const Graph& graph) : size{graph.size}, csr_ready{graph.csr_ready}, csr{graph.csr} {}

	unsigned int* neighbours(unsigned int node)
	{
		return &csr.col_ids[csr.row_offsets[node]];
	}

    unsigned int neighbour_count(unsigned int node) const
    {
        return csr.row_offsets[node + 1] - csr.row_offsets[node];
    }

	void alloc(size_t size)
	{
		this->size = size;
	}

	void load(CSR<T> csr)
	{
		alloc(csr.cols);
		this->csr = csr;
	}

    void prepareRCM(bool check_sym = true)
    {
        // figure out whether this is symmetric
        if (check_sym)
        {
            std::cout << "making sure matrix is symmetric" << std::endl;
            std::unordered_map<uint64_t, uint32_t> entries;
            entries.reserve(csr.nnz / 2);
            size_t diags = 0;
            for (uint32_t r = 0; r < csr.rows; ++r)
            {
                for (uint32_t i = csr.row_offsets[r]; i < csr.row_offsets[r + 1]; ++i)
                {
                    uint32_t c = csr.col_ids[i];
                    uint64_t comb;
                    if (r == c)
                    {
                        ++diags;
                        continue;
                    }
                    if (r > c)
                        comb = (static_cast<uint64_t>(c) << 32ull) | r;
                    else
                        comb = (static_cast<uint64_t>(r) << 32ull) | c;
                    auto found = entries.find(comb);
                    if (found == end(entries))
                        entries[comb] = 1;
                    else
                        entries.erase(found);
                }
            }
            bool sym = true;
            size_t missing = 0;
            for (auto& entry : entries)
            {
                if (entry.second != 2)
                {
                    sym = false;
                    ++missing;
                }
            }


            // make symmetric
            if (!sym)
            {
                std::cout << "matrix is not symmetric, creating missing entries..." << std::endl;
                uint32_t s = std::max(csr.rows, csr.cols);
                COO<T> coo;
                coo.alloc(s, s, csr.nnz + missing);
                size_t nnz = 0;
                for (uint32_t r = 0; r < csr.rows; ++r)
                {
                    for (uint32_t i = csr.row_offsets[r]; i < csr.row_offsets[r + 1]; ++i)
                    {
                        uint32_t c = csr.col_ids[i];
                        T val = csr.data[i];
                        coo.row_ids[nnz] = r;
                        coo.col_ids[nnz] = c;
                        coo.data[nnz] = val;
                        ++nnz;

                        if (r == c)
                            continue;

                        uint64_t comb;
                        if (r > c)
                            comb = (static_cast<uint64_t>(c) << 32ull) | r;
                        else
                            comb = (static_cast<uint64_t>(r) << 32ull) | c;

                        if (entries[comb] == 1)
                        {
                            coo.row_ids[nnz] = c;
                            coo.col_ids[nnz] = r;
                            coo.data[nnz] = val;
                            ++nnz;
                        }
                    }
                }

                std::cout << "created " << missing << " entries, recreating CSR format" << std::endl;
                convert(csr, coo);
            }
        }

        std::cout << "checking whether the entire matrix is one connected component" << std::endl;

        // is it all connected?
        size_t num_visited = 0;
        std::vector<uint32_t> visited(csr.rows, 0);
        std::queue<uint32_t> q;
        uint32_t minn = csr.rows;

        // find first existing and mark all non existing
        for (uint32_t r = 0; r < csr.rows; ++r)
        {
            uint32_t n = csr.row_offsets[r + 1] - csr.row_offsets[r];
            if (n != 0 && minn == csr.rows)
                minn = r;
        }
        
        
        q.push(minn);
        visited[minn] = 1;
        ++num_visited;
        std::vector<std::pair<uint32_t, uint32_t>> addconnections;
        while (num_visited != csr.rows)
        {
            while (!q.empty())
            {
                uint32_t qn = q.front();
                q.pop();
                for (uint32_t i = csr.row_offsets[qn]; i < csr.row_offsets[qn + 1]; ++i)
                {
                    uint32_t n = csr.col_ids[i];
                    if (visited[n] == 0)
                    {
                        visited[n] = 1;
                        q.push(n);
                        ++num_visited;
                    }
                }
            }
            if (num_visited != csr.rows)
            {
                std::cout << "found a non-connected component" << std::endl;
                for (size_t i = 0; i < visited.size(); ++i)
                {
                    if (visited[i] == 0)
                    {
                        bool connected = false;
                        for (uint32_t j = i; j > 0; --j)
                        {
                            if (visited[j - 1] == 1)
                            {
                                addconnections.push_back(std::make_pair(j - 1, i));
                                connected = true;
                                break;
                            }
                        }
                        if (!connected)
                        {
                            for (uint32_t j = i + 1; j < visited.size(); ++j)
                            {
                                if (visited[j] == 1)
                                {
                                    addconnections.push_back(std::make_pair(i, j));
                                    connected = true;
                                    break;
                                }
                            }
                        }
                        q.push(i);
                        std::cout << "adding connection from " << addconnections.back().first << " to " << addconnections.back().second << std::endl;
                        break;
                    }
                }

            }
        }
        if (!addconnections.empty())
        {
            // otherwise connect
            COO<T> coo;
            coo.alloc(csr.rows, csr.cols, csr.nnz + 2 * addconnections.size());
            size_t c = 0;
            for (uint32_t r = 0; r < csr.rows; ++r)
            {
                for (uint32_t i = csr.row_offsets[r]; i < csr.row_offsets[r + 1]; ++i)
                {
                    coo.col_ids[c] = csr.col_ids[i];
                    coo.row_ids[c] = r;
                    coo.data[c] = csr.data[i];
                    ++c;
                }
            }
            for (auto& con : addconnections)
            {
                coo.col_ids[c] = con.first;
                coo.row_ids[c] = con.second;
                coo.data[c] = 1;
                ++c;
                coo.col_ids[c] = con.second;
                coo.row_ids[c] = con.first;
                coo.data[c] = 1;
                ++c;
            }
            convert(csr, coo);
        }
    }

    int load(const std::string& file)
    {
        std::string rcmcsr_name = file + typeext<T>() + ".rcm.hicsr";
        std::string csr_name = file + typeext<T>() + ".hicsr";

        try
        {
            std::cout << "trying to load rcm.csr file \"" << csr_name << "\"\n";
            this->csr = loadCSR<T>(rcmcsr_name.c_str());
            std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
        }
        catch (std::exception& ex)
        {
            bool symmetric = false;
            try
            { 
                std::cout << "trying to load csr file \"" << csr_name << "\"\n";
                this->csr = loadCSR<T>(csr_name.c_str());
                std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
            }
            catch (std::exception& ex)
            {
                try
                {
                    std::cout << "trying to load mtx file \"" << file << "\"\n";
                    bool pattern, complex, hermitian;
                    auto coo_mat = loadMTX<T>(file.c_str(), pattern, complex, symmetric, hermitian);
                    convert(this->csr, coo_mat);
                    std::cout << "succesfully loaded and converted: \"" << file << "\"\n";
                }
                catch (std::exception& ex)
                {
                    std::cout << ex.what() << std::endl;
                    return -1;
                }
                try
                {
                    std::cout << "write csr file for future use\n";
                    storeCSR(this->csr, csr_name.c_str());
                }
                catch (std::exception& ex)
                {
                    std::cout << ex.what() << std::endl;
                }
            }
            prepareRCM(!symmetric);
            try
            {
                std::cout << "write rcm.csr file for future use\n";
                storeCSR(this->csr, rcmcsr_name.c_str());
            }
            catch (std::exception& ex)
            {
                std::cout << ex.what() << std::endl;
            }
        }

        if (this->csr.rows != this->csr.cols)
        {
            std::cout << "Matrix must be symmetric!" << std::endl;
            return 1;
        }

        alloc(this->csr.cols);

        return 0;
    }
};
