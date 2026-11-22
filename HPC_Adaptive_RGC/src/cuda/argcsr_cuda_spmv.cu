#include "argcsr_cuda_pipeline_internal.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace argcsr {
namespace cuda_detail {
namespace {
constexpr int kThreads = 256;

// One thread owns each output row, preserving the CPU lane/chunk accumulation order.
__global__ void spmvKernel(ArgCSRDeviceView matrix, const int* rowGroup,
                           const int* rowPrefix, const double* x, double* y) {
    const std::size_t rowIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (rowIndex >= static_cast<std::size_t>(matrix.rows)) return;
    const int row = static_cast<int>(rowIndex); // Convert to int for indexing
    const ArgCSRGroupInfo group = matrix.groups[rowGroup[row]]; // Get the group metadata for this row
    const int laneCount = matrix.threadsMapping[row]; // Number of threads assigned to this row
    const int firstLane = rowPrefix[row]; // The first lane assigned to this row
    double rowSum = 0.0;
    // Outer loop visits all lanes, and inner loop visits all chunks assigned to the lane.
    for (int assigned = 0; assigned < laneCount; ++assigned) {
        const int lane = firstLane + assigned;
        double laneSum = 0.0;
        for (int chunk = 0; chunk < group.chunkSize; ++chunk) {
            const std::size_t position = static_cast<std::size_t>(group.offset) +
                static_cast<std::size_t>(chunk) * static_cast<std::size_t>(matrix.blockSize) +
                static_cast<std::size_t>(lane);
            const int column = matrix.columns[position];
            if (column == -1) break;
            laneSum += matrix.values[position] * x[column];
        }
        rowSum += laneSum;
    }
    y[row] = rowSum;
}
} // namespace

void launchAdaptiveSpmv(ArgCSRDeviceView matrix, const int* rowGroup,
                        const int* rowPrefix, const double* x, double* y) {
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<std::size_t>(matrix.rows) + kThreads - 1) / kThreads);
    spmvKernel<<<blocks, kThreads>>>(matrix, rowGroup, rowPrefix, x, y);
}

} // namespace cuda_detail
} // namespace argcsr
