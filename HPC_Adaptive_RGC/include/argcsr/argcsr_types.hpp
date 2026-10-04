#pragma once

#include "csr/csr_matrix.hpp"

#include <vector>

namespace argcsr {

struct ArgCSRGroupInfo {
    int firstRow;
    int size;
    int offset;
    int chunkSize;
};

struct ArgCSRMatrix {
    int rows;
    int cols;
    int nnz;
    int blockSize;
    int desiredChunkSize;
    std::vector<ArgCSRGroupInfo> groups;
    std::vector<double> values;
    std::vector<int> columns;
    std::vector<int> threadsMapping;
};

std::vector<ArgCSRGroupInfo> buildGroups(const CSRMatrix& csr, int desiredChunkSize, int blockSize);
std::vector<int> exclusivePrefixSum(const std::vector<int>& values);
ArgCSRMatrix convertToArgCSR(const CSRMatrix& csr, int desiredChunkSize, int blockSize);
void printArgCSR(const ArgCSRMatrix& matrix);

}  // namespace argcsr
