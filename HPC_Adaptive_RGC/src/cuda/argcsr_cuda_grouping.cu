#include "argcsr_cuda_pipeline_internal.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace argcsr {
namespace cuda_detail {
namespace {
constexpr int kThreads = 256;

__global__ void rowStatisticsKernel(const int* rowPointers, int rows, int* rowLengths) {
    const std::size_t rowIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (rowIndex < static_cast<std::size_t>(rows)) {
        const int row = static_cast<int>(rowIndex);
        rowLengths[row] = rowPointers[row + 1] - rowPointers[row]; // Calculate the number of non-zero elements in the row
    }
}
} // namespace

// Calculate the number of non-zero elements in each row of a CSR matrix on the GPU.
void launchRowStatistics(const int* rowPointers, int rows, int* rowLengths) {
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<std::size_t>(rows) + kThreads - 1) / kThreads);
    rowStatisticsKernel<<<blocks, kThreads>>>(rowPointers, rows, rowLengths);
}    

// CPU function to group rows based on their lengths
std::vector<ArgCSRGroupInfo> makeGroupsFromRowLengths(const std::vector<int>& rowLengths,
                                                      int desiredChunkSize, int blockSize) {
    if (desiredChunkSize <= 0 || blockSize <= 0)
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    
    const long long nnzLimit = static_cast<long long>(desiredChunkSize) * blockSize; // Maximum number of non-zero elements allowed in a group
    
    std::vector<ArgCSRGroupInfo> groups;
    int first = 0;
    int size = 0; // Count of rows in the current group
    long long count = 0; // Count of non-zero elements in the current group
    
    for (int row = 0; row < static_cast<int>(rowLengths.size()); ++row) {
        const int rowNnz = rowLengths[static_cast<std::size_t>(row)];
        
        if (rowNnz < 0) throw std::logic_error("GPU produced a negative row length");
        
        // Checks if the current row can fit to this group. If not, finalize the current group and start a new one.
        if (size > 0 && (count + rowNnz > nnzLimit || size + 1 > blockSize)) {
            groups.push_back({first, size, 0, 0}); // New group
            first = row;
            size = 0;
            count = 0;
        }
        ++size;
        count += rowNnz;
    }
    if (size > 0) groups.push_back({first, size, 0, 0});
    return groups;
}

} // namespace cuda_detail
} // namespace argcsr
