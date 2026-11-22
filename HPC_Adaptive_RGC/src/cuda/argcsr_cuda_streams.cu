#include "cuda/argcsr_cuda_streams.h"
#include "argcsr_cuda_pipeline_internal.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace argcsr {
namespace {
using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char* where) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

class StreamOwner {
public:
    StreamOwner() { check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create Streams stream"); }
    ~StreamOwner() { if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); } }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { check(cudaStreamSynchronize(stream_), "synchronize Streams stream"); }
private:
    cudaStream_t stream_ = nullptr;
};

template<class T> class DeviceBuffer {
public:
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(std::size_t n) {
        if (!n) return;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("Streams device allocation size overflow");
        check(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)), "allocate Streams device buffer");
    }
    T* get() const noexcept { return ptr_; }
private:
    T* ptr_ = nullptr;
};

template<class T> class PinnedBuffer {
public:
    ~PinnedBuffer() { if (ptr_) cudaFreeHost(ptr_); }
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void allocate(std::size_t n) {
        if (!n) return;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("Streams pinned allocation size overflow");
        check(cudaMallocHost(reinterpret_cast<void**>(&ptr_), n * sizeof(T)), "allocate pinned Streams buffer");
    }
    T* get() const noexcept { return ptr_; }
private:
    T* ptr_ = nullptr;
};

template<class T> void enqueueH2D(T* dst, const T* src, std::size_t count,
                                 cudaStream_t stream) {
    if (!count) return;
    check(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream), "enqueue Streams H2D");
}

template<class T> void enqueueD2H(T* dst, const T* src, std::size_t count,
                                 cudaStream_t stream) {
    if (!count) return;
    check(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "enqueue Streams D2H");
}

template<class F> double timeDefaultStream(F&& work) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "create Streams timing event");
    try {
        check(cudaEventCreate(&stop), "create Streams timing event");
        check(cudaEventRecord(start, nullptr), "record Streams timing event");
        work();
        check(cudaGetLastError(), "launch Phase 6C kernel from Streams path");
        check(cudaEventRecord(stop, nullptr), "record Streams timing event");
        check(cudaEventSynchronize(stop), "wait for Phase 6C kernel");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "read Streams kernel time");
        cudaEventDestroy(stop); cudaEventDestroy(start);
        return ms;
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}
} // namespace

CudaPipelineResult runArgCsrCudaStreamsPipeline(const CSRMatrix& csr,
                                                const std::vector<double>& x,
                                                int desiredChunkSize, int blockSize) {
    CudaPipelineResult result;
    const auto totalStart = Clock::now();
    auto hostStart = Clock::now();
    csr.validate();
    if (x.size() != static_cast<std::size_t>(csr.cols))
        throw std::invalid_argument("Streams SpMV input length must equal matrix columns");
    if (desiredChunkSize <= 0 || blockSize <= 0)
        throw std::invalid_argument("desiredChunkSize and blockSize must be positive");
    result.timings.hostValidationMs = elapsed(hostStart, Clock::now());

    const std::size_t rows = static_cast<std::size_t>(csr.rows);
    const std::size_t nnz = static_cast<std::size_t>(csr.nnz);
    DeviceBuffer<int> dRowPointers, dCsrColumns, dRowLengths, dMapping, dPrefix, dRowGroup, dStorageError;
    DeviceBuffer<double> dCsrValues, dAdaptiveValues, dX, dY;
    DeviceBuffer<ArgCSRGroupInfo> dGroups;
    PinnedBuffer<int> hRowPointers, hCsrColumns, hRowLengths, hMapping, hStorageError;
    PinnedBuffer<double> hCsrValues, hX, hOutput;
    PinnedBuffer<ArgCSRGroupInfo> hGroups;

    dRowPointers.allocate(rows + 1); dCsrValues.allocate(nnz); dCsrColumns.allocate(nnz);
    dRowLengths.allocate(rows); dMapping.allocate(rows); dPrefix.allocate(rows); dRowGroup.allocate(rows);
    dStorageError.allocate(1); dX.allocate(static_cast<std::size_t>(csr.cols)); dY.allocate(rows);
    hRowPointers.allocate(rows + 1); hCsrValues.allocate(nnz); hCsrColumns.allocate(nnz);
    hRowLengths.allocate(rows); hMapping.allocate(rows); hStorageError.allocate(1); hOutput.allocate(rows);
    hX.allocate(static_cast<std::size_t>(csr.cols));
    // Declared after buffers so it is synchronized/destroyed before host or
    // device storage is released during normal return or exception unwinding.
    StreamOwner stream;
    std::copy(csr.row_pointers.begin(), csr.row_pointers.end(), hRowPointers.get());
    if (nnz) {
        std::copy(csr.values.begin(), csr.values.end(), hCsrValues.get());
        std::copy(csr.column_indices.begin(), csr.column_indices.end(), hCsrColumns.get());
    }
    if (csr.cols) std::copy(x.begin(), x.end(), hX.get());

    auto transferStart = Clock::now();
    enqueueH2D(dRowPointers.get(), hRowPointers.get(), rows + 1, stream.get());
    enqueueH2D(dCsrValues.get(), hCsrValues.get(), nnz, stream.get());
    enqueueH2D(dCsrColumns.get(), hCsrColumns.get(), nnz, stream.get());
    stream.synchronize();
    result.timings.h2dCsrMs = elapsed(transferStart, Clock::now());
    if (csr.rows > 0) result.timings.gpuRowStatisticsMs = timeDefaultStream([&] {
        cuda_detail::launchRowStatistics(dRowPointers.get(), csr.rows, dRowLengths.get());
    });
    transferStart = Clock::now();
    enqueueD2H(hRowLengths.get(), dRowLengths.get(), rows, stream.get());
    stream.synchronize();
    result.timings.d2hRowStatisticsMs = elapsed(transferStart, Clock::now());

    hostStart = Clock::now();
    std::vector<int> rowLengths;
    if (rows) rowLengths.assign(hRowLengths.get(), hRowLengths.get() + rows);
    auto groups = cuda_detail::makeGroupsFromRowLengths(rowLengths, desiredChunkSize, blockSize);
    result.timings.cpuGroupingMs = elapsed(hostStart, Clock::now());
    if (groups.size() > static_cast<std::size_t>(INT_MAX))
        throw std::overflow_error("too many groups for CUDA indexing");
    hGroups.allocate(groups.size());
    if (!groups.empty()) std::copy(groups.begin(), groups.end(), hGroups.get());
    dGroups.allocate(groups.size());
    const int groupCount = static_cast<int>(groups.size());
    transferStart = Clock::now();
    enqueueH2D(dGroups.get(), hGroups.get(), groups.size(), stream.get());
    stream.synchronize();
    result.timings.h2dGroupsMs = elapsed(transferStart, Clock::now());

    if (groupCount) {
        result.timings.gpuMappingMs = timeDefaultStream([&] {
            cuda_detail::launchLaneMapping(dRowLengths.get(), dGroups.get(), groupCount, blockSize, dMapping.get());
        });
        result.timings.gpuChunkSizingMs = timeDefaultStream([&] {
            cuda_detail::launchChunkSizing(dRowLengths.get(), dMapping.get(), dGroups.get(), groupCount);
        });
    }
    // Kernel helpers belong to Phase 6C and launch on the default stream. Fence
    // before the explicit transfer stream reads their result.
    check(cudaDeviceSynchronize(), "wait for Streams chunk sizing");
    transferStart = Clock::now();
    enqueueD2H(hGroups.get(), dGroups.get(), groups.size(), stream.get());
    stream.synchronize();
    result.timings.d2hChunkMetadataMs = elapsed(transferStart, Clock::now());
    if (!groups.empty()) groups.assign(hGroups.get(), hGroups.get() + groups.size());
    hostStart = Clock::now();
    long long totalStorage = 0;
    for (auto& group : groups) {
        if (totalStorage > INT_MAX) throw std::overflow_error("Adaptive RGC offset exceeds int range");
        group.offset = static_cast<int>(totalStorage);
        totalStorage += static_cast<long long>(group.chunkSize) * blockSize;
        if (totalStorage > INT_MAX) throw std::overflow_error("Adaptive RGC storage exceeds int range");
    }
    result.timings.cpuGroupOffsetsMs = elapsed(hostStart, Clock::now());
    if (!groups.empty()) std::copy(groups.begin(), groups.end(), hGroups.get());
    transferStart = Clock::now();
    enqueueH2D(dGroups.get(), hGroups.get(), groups.size(), stream.get());
    stream.synchronize();
    result.timings.h2dFinalMetadataMs = elapsed(transferStart, Clock::now());

    if (csr.rows > 0) result.timings.gpuMappingMs += timeDefaultStream([&] {
        if (groupCount) cuda_detail::launchMappingPrefixes(dGroups.get(), groupCount, dMapping.get(), dPrefix.get());
        cuda_detail::launchRowGroups(dGroups.get(), groupCount, csr.rows, dRowGroup.get());
    });
    const std::size_t storageSize = static_cast<std::size_t>(totalStorage);
    dAdaptiveValues.allocate(storageSize);
    DeviceBuffer<int> dAdaptiveColumns; dAdaptiveColumns.allocate(storageSize);
    result.timings.gpuStorageConversionMs = timeDefaultStream([&] {
        check(cudaMemset(dStorageError.get(), 0, sizeof(int)), "clear Streams storage error");
        if (storageSize) {
            check(cudaMemset(dAdaptiveValues.get(), 0, storageSize * sizeof(double)), "clear Streams values");
            check(cudaMemset(dAdaptiveColumns.get(), 0xff, storageSize * sizeof(int)), "clear Streams padding");
        }
        if (csr.nnz) cuda_detail::launchStorageScatter(dRowPointers.get(), dCsrValues.get(), dCsrColumns.get(),
            csr.rows, csr.nnz, dGroups.get(), dRowGroup.get(), dMapping.get(), dPrefix.get(), blockSize,
            dAdaptiveValues.get(), dAdaptiveColumns.get(), dStorageError.get());
    });
    transferStart = Clock::now();
    enqueueD2H(hStorageError.get(), dStorageError.get(), 1, stream.get());
    stream.synchronize();
    result.timings.d2hStorageStatusMs = elapsed(transferStart, Clock::now());
    if (*hStorageError.get()) throw std::logic_error("Streams storage mapping exceeded assigned lane count");

    transferStart = Clock::now();
    enqueueH2D(dX.get(), hX.get(), static_cast<std::size_t>(csr.cols), stream.get());
    stream.synchronize();
    result.timings.h2dInputVectorMs = elapsed(transferStart, Clock::now());
    const ArgCSRDeviceView view{csr.rows, csr.cols, csr.nnz, blockSize, desiredChunkSize, groups.size(),
        storageSize, dGroups.get(), dAdaptiveValues.get(), dAdaptiveColumns.get(), dMapping.get()};
    if (csr.rows > 0) result.timings.gpuSpmvMs = timeDefaultStream([&] {
        cuda_detail::launchAdaptiveSpmv(view, dRowGroup.get(), dPrefix.get(), dX.get(), dY.get());
    });
    result.output.resize(rows);
    transferStart = Clock::now();
    enqueueD2H(hOutput.get(), dY.get(), rows, stream.get());
    stream.synchronize();
    result.timings.d2hOutputMs = elapsed(transferStart, Clock::now());
    if (rows) std::copy(hOutput.get(), hOutput.get() + rows, result.output.begin());
    result.timings.totalPipelineMs = elapsed(totalStart, Clock::now());
    return result;
}
} // namespace argcsr
