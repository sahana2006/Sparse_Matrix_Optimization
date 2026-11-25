#include "argcsr_hybrid.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace argcsr_hybrid {
namespace {

using Clock = std::chrono::steady_clock;

void checkCuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() noexcept { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        if (ptr_) throw std::logic_error("DeviceBuffer already allocated");
        if (count == 0) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("CUDA device buffer byte size overflow");
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "cudaMalloc");
        count_ = count;
    }

    void release() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

template <class T>
double copyH2D(T* dst, const T* src, std::size_t count, const char* label) {
    if (count == 0) return 0.0;
    const auto start = Clock::now();
    checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), label);
    return elapsedMs(start, Clock::now());
}

template <class T>
double copyD2H(T* dst, const T* src, std::size_t count, const char* label) {
    if (count == 0) return 0.0;
    const auto start = Clock::now();
    checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), label);
    return elapsedMs(start, Clock::now());
}

// Row-owned ArgCSR SpMV kernel matching the exact accumulation semantics
// of the frozen baseline: one thread per output row iterates lanes in order
// and chunks within each lane in order, stopping at column == -1.
__global__ void hybridArgCsrSpmvKernel(int rows,
                                       int blockSize,
                                       const argcsr::ArgCSRGroupInfo* __restrict__ groups,
                                       const double* __restrict__ values,
                                       const int* __restrict__ columns,
                                       const int* __restrict__ threadsMapping,
                                       const int* __restrict__ rowGroup,
                                       const int* __restrict__ rowPrefix,
                                       const double* __restrict__ x,
                                       double* __restrict__ y) {
    const std::size_t rowIdx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (rowIdx >= static_cast<std::size_t>(rows)) return;

    const int row = static_cast<int>(rowIdx);
    const argcsr::ArgCSRGroupInfo group = groups[rowGroup[row]];
    const int laneCount = threadsMapping[row];
    const int firstLane = rowPrefix[row];

    double rowSum = 0.0;
    for (int assigned = 0; assigned < laneCount; ++assigned) {
        const int lane = firstLane + assigned;
        double laneSum = 0.0;
        for (int chunk = 0; chunk < group.chunkSize; ++chunk) {
            const std::size_t pos = static_cast<std::size_t>(group.offset) +
                                    static_cast<std::size_t>(chunk) * static_cast<std::size_t>(blockSize) +
                                    static_cast<std::size_t>(lane);
            const int col = columns[pos];
            if (col == -1) break;
            laneSum += values[pos] * x[col];
        }
        rowSum += laneSum;
    }
    y[row] = rowSum;
}

double launchAndTimeSpmvKernel(int rows,
                               int blockSize,
                               int cudaBlockSize,
                               const argcsr::ArgCSRGroupInfo* dGroups,
                               const double* dValues,
                               const int* dColumns,
                               const int* dMapping,
                               const int* dRowGroup,
                               const int* dRowPrefix,
                               const double* dX,
                               double* dY) {
    if (rows <= 0) return 0.0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    try {
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        const unsigned int gridSize = static_cast<unsigned int>(
            (static_cast<std::size_t>(rows) + static_cast<std::size_t>(cudaBlockSize) - 1) /
            static_cast<std::size_t>(cudaBlockSize));

        checkCuda(cudaEventRecord(start), "cudaEventRecord start");
        hybridArgCsrSpmvKernel<<<gridSize, static_cast<unsigned int>(cudaBlockSize)>>>(
            rows, blockSize, dGroups, dValues, dColumns, dMapping, dRowGroup, dRowPrefix, dX, dY);
        checkCuda(cudaGetLastError(), "launch hybridArgCsrSpmvKernel");
        checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return static_cast<double>(ms);
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

void validateHybridArgs(const argcsr::CSRMatrix& csr,
                        const std::vector<double>& x,
                        int desiredChunkSize,
                        int blockSize,
                        int cudaBlockSize) {
    if (desiredChunkSize <= 0 || blockSize <= 0) {
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    }
    if (cudaBlockSize <= 0 || cudaBlockSize > 1024) {
        throw std::invalid_argument("cudaBlockSize must be in [1, 1024]");
    }
    if (x.size() != static_cast<std::size_t>(csr.cols)) {
        throw std::invalid_argument("Hybrid SpMV input vector length must equal matrix columns");
    }
}

}  // namespace

HybridPipelineResult runArgCsrHybridPipeline(const argcsr::CSRMatrix& csr,
                                             const std::vector<double>& x,
                                             int desiredChunkSize,
                                             int blockSize,
                                             const HybridConfig& config,
                                             bool captureRepresentation) {
    validateHybridArgs(csr, x, desiredChunkSize, blockSize, config.cudaBlockSize);

    HybridPipelineResult result;
    const auto totalStart = Clock::now();

    // 1. CPU Side: OpenMP ArgCSR Conversion
    auto converted = argcsr_omp::convertToArgCSROpenMPDetailed(
        csr, desiredChunkSize, blockSize, config.ompThreads, config.runPostValidation);
    result.timings.ompConversion = converted.timings;
    result.timings.ompTotalMs = converted.timings.totalMs;

    const auto& mat = converted.matrix;
    const std::size_t groupCount = mat.groups.size();
    const std::size_t storageSize = mat.values.size();
    const std::size_t rowCount = static_cast<std::size_t>(mat.rows);
    const std::size_t colCount = static_cast<std::size_t>(mat.cols);

    // 2. GPU Device Allocation
    DeviceBuffer<argcsr::ArgCSRGroupInfo> dGroups;
    DeviceBuffer<double> dValues;
    DeviceBuffer<int> dColumns;
    DeviceBuffer<int> dMapping;
    DeviceBuffer<int> dRowGroup;
    DeviceBuffer<int> dRowPrefix;
    DeviceBuffer<double> dX;
    DeviceBuffer<double> dY;

    const auto allocStart = Clock::now();
    dGroups.allocate(groupCount);
    dValues.allocate(storageSize);
    dColumns.allocate(storageSize);
    dMapping.allocate(rowCount);
    dRowGroup.allocate(rowCount);
    dRowPrefix.allocate(rowCount);
    dX.allocate(colCount);
    dY.allocate(rowCount);
    result.timings.deviceAllocMs = elapsedMs(allocStart, Clock::now());

    // 3. Host-to-Device Transfer (ArgCSR representation + row metadata + input vector x)
    double h2dRepr = 0.0;
    h2dRepr += copyH2D(dGroups.get(), mat.groups.data(), groupCount, "H2D groups");
    h2dRepr += copyH2D(dValues.get(), mat.values.data(), storageSize, "H2D values");
    h2dRepr += copyH2D(dColumns.get(), mat.columns.data(), storageSize, "H2D columns");
    h2dRepr += copyH2D(dMapping.get(), mat.threadsMapping.data(), rowCount, "H2D threadsMapping");
    h2dRepr += copyH2D(dRowGroup.get(), converted.rowGroup.data(), rowCount, "H2D rowGroup");
    h2dRepr += copyH2D(dRowPrefix.get(), converted.rowPrefix.data(), rowCount, "H2D rowPrefix");
    result.timings.h2dArgCsrMs = h2dRepr;

    result.timings.h2dInputVectorMs = copyH2D(dX.get(), x.data(), colCount, "H2D input vector x");
    result.timings.h2dTotalMs = result.timings.h2dArgCsrMs + result.timings.h2dInputVectorMs;

    // 4. GPU Side: CUDA ArgCSR SpMV
    result.timings.cudaSpmvMs = launchAndTimeSpmvKernel(
        mat.rows, mat.blockSize, config.cudaBlockSize,
        dGroups.get(), dValues.get(), dColumns.get(), dMapping.get(),
        dRowGroup.get(), dRowPrefix.get(), dX.get(), dY.get());

    // 5. Device-to-Host Transfer (Output vector y)
    result.output.resize(rowCount);
    result.timings.d2hOutputMs = copyD2H(result.output.data(), dY.get(), rowCount, "D2H output vector y");

    // 6. Explicit Device Deallocation
    const auto freeStart = Clock::now();
    dY.release();
    dX.release();
    dRowPrefix.release();
    dRowGroup.release();
    dMapping.release();
    dColumns.release();
    dValues.release();
    dGroups.release();
    result.timings.deviceFreeMs = elapsedMs(freeStart, Clock::now());

    result.timings.totalEndToEndMs = elapsedMs(totalStart, Clock::now());

    if (captureRepresentation) {
        result.representation = std::move(converted.matrix);
    }
    return result;
}

double benchmarkHybridSpmvOnly(const argcsr_omp::OpenMPArgCSRResult& converted,
                               const std::vector<double>& x,
                               int cudaBlockSize,
                               std::vector<double>& output) {
    if (cudaBlockSize <= 0 || cudaBlockSize > 1024) {
        throw std::invalid_argument("cudaBlockSize must be in [1, 1024]");
    }
    const auto& mat = converted.matrix;
    if (x.size() != static_cast<std::size_t>(mat.cols)) {
        throw std::invalid_argument("Hybrid SpMV input vector length must equal matrix columns");
    }

    const std::size_t rowCount = static_cast<std::size_t>(mat.rows);
    DeviceBuffer<argcsr::ArgCSRGroupInfo> dGroups;
    DeviceBuffer<double> dValues;
    DeviceBuffer<int> dColumns;
    DeviceBuffer<int> dMapping;
    DeviceBuffer<int> dRowGroup;
    DeviceBuffer<int> dRowPrefix;
    DeviceBuffer<double> dX;
    DeviceBuffer<double> dY;

    dGroups.allocate(mat.groups.size());
    dValues.allocate(mat.values.size());
    dColumns.allocate(mat.columns.size());
    dMapping.allocate(rowCount);
    dRowGroup.allocate(rowCount);
    dRowPrefix.allocate(rowCount);
    dX.allocate(static_cast<std::size_t>(mat.cols));
    dY.allocate(rowCount);

    copyH2D(dGroups.get(), mat.groups.data(), mat.groups.size(), "H2D groups");
    copyH2D(dValues.get(), mat.values.data(), mat.values.size(), "H2D values");
    copyH2D(dColumns.get(), mat.columns.data(), mat.columns.size(), "H2D columns");
    copyH2D(dMapping.get(), mat.threadsMapping.data(), rowCount, "H2D threadsMapping");
    copyH2D(dRowGroup.get(), converted.rowGroup.data(), rowCount, "H2D rowGroup");
    copyH2D(dRowPrefix.get(), converted.rowPrefix.data(), rowCount, "H2D rowPrefix");
    copyH2D(dX.get(), x.data(), static_cast<std::size_t>(mat.cols), "H2D x");

    // Warmup launch before timed kernel.
    (void)launchAndTimeSpmvKernel(
        mat.rows, mat.blockSize, cudaBlockSize,
        dGroups.get(), dValues.get(), dColumns.get(), dMapping.get(),
        dRowGroup.get(), dRowPrefix.get(), dX.get(), dY.get());

    const double kernelMs = launchAndTimeSpmvKernel(
        mat.rows, mat.blockSize, cudaBlockSize,
        dGroups.get(), dValues.get(), dColumns.get(), dMapping.get(),
        dRowGroup.get(), dRowPrefix.get(), dX.get(), dY.get());

    output.resize(rowCount);
    copyD2H(output.data(), dY.get(), rowCount, "D2H y");
    return kernelMs;
}

}  // namespace argcsr_hybrid
