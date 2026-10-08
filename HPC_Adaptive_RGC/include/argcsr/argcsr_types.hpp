#pragma once

#include "csr/csr_matrix.hpp"

#include <vector>
#include <chrono>

namespace argcsr {
// Each group has set a 4 tuple information
struct ArgCSRGroupInfo { 
    int firstRow;
    int size;
    int offset; // It denotes first non zero element of the group in the values array
    int chunkSize; // It denotes the number of non zero elements in the group
};

struct ArgCSRMatrix {
    int rows;
    int cols;
    int nnz;
    int blockSize; // Number of rows in each block or group
    int desiredChunkSize; // How much workload for each group - no : of non zeros in each group
    std::vector<ArgCSRGroupInfo> groups;
    std::vector<double> values; // Non zero values of the matrix
    std::vector<int> columns; // Stores column indices, so that value[i] is in column columns[i]
    std::vector<int> threadsMapping; // Mapping of threads to groups
};

struct ConversionTimings {
    std::chrono::steady_clock::duration grouping{};
    std::chrono::steady_clock::duration chunking{};
    std::chrono::steady_clock::duration mapping{};
    std::chrono::steady_clock::duration storage{};
};

std::vector<ArgCSRGroupInfo> buildGroups(const CSRMatrix& csr, int desiredChunkSize, int blockSize);
std::vector<int> exclusivePrefixSum(const std::vector<int>& values);
ArgCSRMatrix convertToArgCSR(const CSRMatrix& csr, int desiredChunkSize, int blockSize,
                             ConversionTimings* timings = nullptr);
void printArgCSR(const ArgCSRMatrix& matrix);

}  // namespace argcsr
