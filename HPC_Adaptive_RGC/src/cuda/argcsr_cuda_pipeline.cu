#include "cuda/argcsr_cuda_pipeline.hpp"
#include "cuda/argcsr_device.hpp"
#include "argcsr_cuda_pipeline_internal.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace argcsr {
namespace {

using Clock = std::chrono::steady_clock;
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <class Function>
double timeKernel(Function&& function) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "create CUDA timing event");
    try {
        checkCuda(cudaEventCreate(&stop), "create CUDA timing event");
        checkCuda(cudaEventRecord(start), "record CUDA timing event");
        function();
        checkCuda(cudaGetLastError(), "launch CUDA pipeline kernel");
        checkCuda(cudaEventRecord(stop), "record CUDA timing event");
        checkCuda(cudaEventSynchronize(stop), "wait for CUDA pipeline kernel");
        float milliseconds = 0.0f;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "read CUDA event time");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return milliseconds;
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

template <class T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t count) { allocate(count); }
    ~DeviceArray() noexcept { if (data_) cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void allocate(std::size_t count) {
        if (data_) throw std::logic_error("CUDA pipeline array already allocated");
        if (count == 0) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("CUDA pipeline allocation size overflow");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                  "allocate CUDA pipeline array");
        size_ = count;
    }

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

template <class T>
double copyHostToDevice(T* destination, const T* source, std::size_t count) {
    if (count == 0) return 0.0;
    const auto start = Clock::now();
    checkCuda(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice),
              "copy input to CUDA device");
    return elapsedMs(start, Clock::now());
}

template <class T>
double copyDeviceToHost(T* destination, const T* source, std::size_t count) {
    if (count == 0) return 0.0;
    const auto start = Clock::now();
    checkCuda(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost),
              "copy CUDA result to host");
    return elapsedMs(start, Clock::now());
}

} // namespace

double CudaPipelineTimings::cpuPreprocessingMs() const noexcept {
    return hostValidationMs + cpuGroupingMs + cpuGroupOffsetsMs;
}

double CudaPipelineTimings::gpuKernelMs() const noexcept {
    return gpuRowStatisticsMs + gpuMappingMs + gpuChunkSizingMs +
           gpuStorageConversionMs + gpuSpmvMs;
}

double CudaPipelineTimings::transferMs() const noexcept {
    return h2dCsrMs + d2hRowStatisticsMs + h2dGroupsMs + d2hChunkMetadataMs + d2hStorageStatusMs +
           h2dFinalMetadataMs + h2dInputVectorMs + d2hOutputMs + d2hRepresentationMs;
}

CudaPipelineResult runArgCsrCudaPipeline(const CSRMatrix& csr,
                                         const std::vector<double>& x,
                                         int desiredChunkSize,
                                         int blockSize,
                                         bool captureRepresentation) {
    CudaPipelineResult result;
    const auto totalStart = Clock::now();

    auto start = Clock::now();
    csr.validate();
    if (x.size() != static_cast<std::size_t>(csr.cols))
        throw std::invalid_argument("CUDA Adaptive RGC SpMV input length must equal columns");
    if (desiredChunkSize <= 0 || blockSize <= 0)
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    result.timings.hostValidationMs = elapsedMs(start, Clock::now());

    DeviceArray<int> dRowPointers(static_cast<std::size_t>(csr.rows) + 1);
    DeviceArray<double> dCsrValues(static_cast<std::size_t>(csr.nnz));
    DeviceArray<int> dCsrColumns(static_cast<std::size_t>(csr.nnz));
    DeviceArray<int> dRowLengths(static_cast<std::size_t>(csr.rows));
    result.timings.h2dCsrMs += copyHostToDevice(dRowPointers.get(), csr.row_pointers.data(),
                                                csr.row_pointers.size());
    result.timings.h2dCsrMs += copyHostToDevice(dCsrValues.get(), csr.values.data(), csr.values.size());
    result.timings.h2dCsrMs += copyHostToDevice(dCsrColumns.get(), csr.column_indices.data(),
                                                csr.column_indices.size());

    if (csr.rows > 0) {
        result.timings.gpuRowStatisticsMs = timeKernel([&] {
            cuda_detail::launchRowStatistics(dRowPointers.get(), csr.rows, dRowLengths.get());
        });
    }
    std::vector<int> hostRowLengths(static_cast<std::size_t>(csr.rows));
    result.timings.d2hRowStatisticsMs = copyDeviceToHost(hostRowLengths.data(), dRowLengths.get(),
                                                         hostRowLengths.size());

    start = Clock::now();
    std::vector<ArgCSRGroupInfo> groups = cuda_detail::makeGroupsFromRowLengths(
        hostRowLengths, desiredChunkSize, blockSize);
    result.timings.cpuGroupingMs = elapsedMs(start, Clock::now());
    if (groups.size() > static_cast<std::size_t>(INT_MAX))
        throw std::overflow_error("too many Adaptive RGC groups for CUDA indexing");

    DeviceArray<ArgCSRGroupInfo> dGroups(groups.size());
    DeviceArray<int> dMapping(static_cast<std::size_t>(csr.rows));
    DeviceArray<int> dPrefix(static_cast<std::size_t>(csr.rows));
    DeviceArray<int> dRowGroup(static_cast<std::size_t>(csr.rows));
    result.timings.h2dGroupsMs = copyHostToDevice(dGroups.get(), groups.data(), groups.size());

    if (!groups.empty()) {
        result.timings.gpuMappingMs = timeKernel([&] {
            cuda_detail::launchLaneMapping(dRowLengths.get(), dGroups.get(),
                static_cast<int>(groups.size()), blockSize, dMapping.get());
        });
        result.timings.gpuChunkSizingMs = timeKernel([&] {
            cuda_detail::launchChunkSizing(dRowLengths.get(), dMapping.get(), dGroups.get(),
                static_cast<int>(groups.size()));
        });
    }

    std::vector<ArgCSRGroupInfo> sizedGroups(groups.size());
    result.timings.d2hChunkMetadataMs = copyDeviceToHost(sizedGroups.data(), dGroups.get(),
                                                         sizedGroups.size());
    start = Clock::now();
    long long totalStorage = 0;
    for (auto& group : sizedGroups) {
        if (totalStorage > INT_MAX)
            throw std::overflow_error("ArgCSR storage offset exceeds int range");
        group.offset = static_cast<int>(totalStorage);
        totalStorage += static_cast<long long>(group.chunkSize) * blockSize;
        if (totalStorage > INT_MAX)
            throw std::overflow_error("ArgCSR storage exceeds int range");
    }
    groups = std::move(sizedGroups);
    result.timings.cpuGroupOffsetsMs = elapsedMs(start, Clock::now());

    result.timings.h2dFinalMetadataMs = copyHostToDevice(dGroups.get(), groups.data(), groups.size());
    if (csr.rows > 0) {
        result.timings.gpuMappingMs += timeKernel([&] {
            if (!groups.empty()) {
                cuda_detail::launchMappingPrefixes(
                    dGroups.get(), static_cast<int>(groups.size()), dMapping.get(), dPrefix.get());
            }
            cuda_detail::launchRowGroups(dGroups.get(), static_cast<int>(groups.size()),
                                         csr.rows, dRowGroup.get());
        });
    }

    const std::size_t storageSize = static_cast<std::size_t>(totalStorage);
    DeviceArray<double> dAdaptiveValues(storageSize);
    DeviceArray<int> dAdaptiveColumns(storageSize);
    DeviceArray<int> dStorageError(1);
    result.timings.gpuStorageConversionMs = timeKernel([&] {
        checkCuda(cudaMemset(dStorageError.get(), 0, sizeof(int)), "initialize storage error flag");
        if (storageSize > 0) {
            checkCuda(cudaMemset(dAdaptiveValues.get(), 0, storageSize * sizeof(double)),
                      "initialize Adaptive RGC values");
            checkCuda(cudaMemset(dAdaptiveColumns.get(), 0xff, storageSize * sizeof(int)),
                      "initialize Adaptive RGC padding columns");
        }
        if (csr.nnz > 0) {
            cuda_detail::launchStorageScatter(
                dRowPointers.get(), dCsrValues.get(), dCsrColumns.get(), csr.rows, csr.nnz,
                dGroups.get(), dRowGroup.get(), dMapping.get(), dPrefix.get(), blockSize,
                dAdaptiveValues.get(), dAdaptiveColumns.get(), dStorageError.get());
        }
    });
    int storageError = 0;
    result.timings.d2hStorageStatusMs = copyDeviceToHost(&storageError, dStorageError.get(), 1);
    if (storageError != 0)
        throw std::logic_error("GPU storage mapping exceeded the row's assigned lane count");

    DeviceArray<double> dX(static_cast<std::size_t>(csr.cols));
    DeviceArray<double> dY(static_cast<std::size_t>(csr.rows));
    result.timings.h2dInputVectorMs = copyHostToDevice(dX.get(), x.data(), x.size());
    ArgCSRDeviceView deviceView{csr.rows, csr.cols, csr.nnz, blockSize, desiredChunkSize,
        groups.size(), storageSize, dGroups.get(), dAdaptiveValues.get(), dAdaptiveColumns.get(),
        dMapping.get()};
    if (csr.rows > 0) {
        result.timings.gpuSpmvMs = timeKernel([&] {
            cuda_detail::launchAdaptiveSpmv(deviceView, dRowGroup.get(), dPrefix.get(),
                                             dX.get(), dY.get());
        });
    }
    result.output.resize(static_cast<std::size_t>(csr.rows));
    result.timings.d2hOutputMs = copyDeviceToHost(result.output.data(), dY.get(), result.output.size());

    if (captureRepresentation) {
        result.representation = {csr.rows, csr.cols, csr.nnz, blockSize, desiredChunkSize,
                                 groups, {}, {}, {}};
        result.representation.values.resize(storageSize);
        result.representation.columns.resize(storageSize);
        result.representation.threadsMapping.resize(static_cast<std::size_t>(csr.rows));
        result.timings.d2hRepresentationMs += copyDeviceToHost(
            result.representation.values.data(), dAdaptiveValues.get(), storageSize);
        result.timings.d2hRepresentationMs += copyDeviceToHost(
            result.representation.columns.data(), dAdaptiveColumns.get(), storageSize);
        result.timings.d2hRepresentationMs += copyDeviceToHost(
            result.representation.threadsMapping.data(), dMapping.get(),
            result.representation.threadsMapping.size());
    }

    result.timings.totalPipelineMs = elapsedMs(totalStart, Clock::now());
    return result;
}

} // namespace argcsr
