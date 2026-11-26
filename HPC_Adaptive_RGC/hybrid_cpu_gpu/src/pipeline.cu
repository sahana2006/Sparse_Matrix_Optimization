#include "hybrid_cpu_gpu/pipeline.hpp"

#include "hybrid_cpu_gpu/conversion_openmp.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hybrid_cpu_gpu {
namespace detail {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void checkCuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() noexcept { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    bool ensureSize(std::size_t count) {
        if (count == 0) {
            release();
            return false;
        }
        if (ptr_ && count_ == count) return false;
        release();
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "cudaMalloc");
        count_ = count;
        return true;
    }

    void release() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class CudaStream {
public:
    CudaStream() {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~CudaStream() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

template <class T>
void enqueueH2D(T* dst, const T* src, std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    checkCuda(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D async");
}

template <class T>
void enqueueD2H(T* dst, const T* src, std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    checkCuda(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H async");
}

template <typename F>
double elapsedStreamMs(cudaStream_t stream, F&& work) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate");
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    work();
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return static_cast<double>(ms);
}

__global__ void hybridSpmvKernel(int rows,
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

double launchTimedKernel(int rows,
                         int blockSize,
                         int cudaBlockSize,
                         cudaStream_t stream,
                         const argcsr::ArgCSRGroupInfo* dGroups,
                         const double* dValues,
                         const int* dColumns,
                         const int* dMapping,
                         const int* dRowGroup,
                         const int* dRowPrefix,
                         const double* dX,
                         double* dY) {
    if (rows <= 0) return 0.0;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<std::size_t>(rows) + cudaBlockSize - 1) / cudaBlockSize);
    return elapsedStreamMs(stream, [&] {
        hybridSpmvKernel<<<grid, static_cast<unsigned int>(cudaBlockSize), 0, stream>>>(
            rows, blockSize, dGroups, dValues, dColumns, dMapping, dRowGroup, dRowPrefix, dX, dY);
        checkCuda(cudaGetLastError(), "hybridSpmvKernel");
    });
}

std::size_t countH2DBytes(const OpenMPConversionResult& converted, const std::vector<double>& x) {
    const auto& m = converted.matrix;
    std::size_t bytes = 0;
    bytes += m.groups.size() * sizeof(argcsr::ArgCSRGroupInfo);
    bytes += m.values.size() * sizeof(double);
    bytes += m.columns.size() * sizeof(int);
    bytes += m.threadsMapping.size() * sizeof(int);
    bytes += converted.rowGroup.size() * sizeof(int);
    bytes += converted.rowPrefix.size() * sizeof(int);
    bytes += x.size() * sizeof(double);
    return bytes;
}

struct UploadedState {
    DeviceBuffer<argcsr::ArgCSRGroupInfo> groups;
    DeviceBuffer<double> values;
    DeviceBuffer<int> columns;
    DeviceBuffer<int> mapping;
    DeviceBuffer<int> rowGroup;
    DeviceBuffer<int> rowPrefix;
    DeviceBuffer<double> x;
    DeviceBuffer<double> y;
};

void ensureAndUpload(UploadedState& state,
                     const OpenMPConversionResult& converted,
                     const std::vector<double>& x,
                     cudaStream_t stream,
                     TransferMetrics& xfer,
                     double& allocMs) {
    const auto& m = converted.matrix;
    const auto allocStart = Clock::now();
    bool allocated = false;
    allocated |= state.groups.ensureSize(m.groups.size());
    allocated |= state.values.ensureSize(m.values.size());
    allocated |= state.columns.ensureSize(m.columns.size());
    allocated |= state.mapping.ensureSize(m.threadsMapping.size());
    allocated |= state.rowGroup.ensureSize(converted.rowGroup.size());
    allocated |= state.rowPrefix.ensureSize(converted.rowPrefix.size());
    allocated |= state.x.ensureSize(x.size());
    allocated |= state.y.ensureSize(static_cast<std::size_t>(m.rows));
    allocMs = allocated ? elapsedMs(allocStart, Clock::now()) : 0.0;

    xfer.h2dBytes = countH2DBytes(converted, x);
    xfer.h2dMs = elapsedStreamMs(stream, [&] {
        enqueueH2D(state.groups.get(), m.groups.data(), m.groups.size(), stream);
        enqueueH2D(state.values.get(), m.values.data(), m.values.size(), stream);
        enqueueH2D(state.columns.get(), m.columns.data(), m.columns.size(), stream);
        enqueueH2D(state.mapping.get(), m.threadsMapping.data(), m.threadsMapping.size(), stream);
        enqueueH2D(state.rowGroup.get(), converted.rowGroup.data(), converted.rowGroup.size(), stream);
        enqueueH2D(state.rowPrefix.get(), converted.rowPrefix.data(), converted.rowPrefix.size(), stream);
        enqueueH2D(state.x.get(), x.data(), x.size(), stream);
    });
    xfer.h2dBandwidthGbps = gigaBytesPerSecond(xfer.h2dBytes, xfer.h2dMs);
}

void runKernelAndD2h(UploadedState& device,
                     const argcsr::ArgCSRMatrix& mat,
                     const PipelineConfig& config,
                     cudaStream_t stream,
                     PipelineResult& result) {
    result.timings.kernelMs = launchTimedKernel(
        mat.rows, mat.blockSize, config.cudaBlockSize, stream, device.groups.get(), device.values.get(),
        device.columns.get(), device.mapping.get(), device.rowGroup.get(), device.rowPrefix.get(),
        device.x.get(), device.y.get());

    result.output.resize(static_cast<std::size_t>(mat.rows));
    result.timings.transfer.d2hBytes = result.output.size() * sizeof(double);
    result.timings.transfer.d2hMs = elapsedStreamMs(stream, [&] {
        enqueueD2H(result.output.data(), device.y.get(), result.output.size(), stream);
    });
    result.timings.transfer.d2hBandwidthGbps =
        gigaBytesPerSecond(result.timings.transfer.d2hBytes, result.timings.transfer.d2hMs);
}

}  // namespace detail

struct PipelineDeviceCache::Storage {
    detail::UploadedState device;
    detail::CudaStream stream;
};

PipelineDeviceCache::PipelineDeviceCache() = default;
PipelineDeviceCache::~PipelineDeviceCache() = default;
PipelineDeviceCache::PipelineDeviceCache(PipelineDeviceCache&&) noexcept = default;
PipelineDeviceCache& PipelineDeviceCache::operator=(PipelineDeviceCache&&) noexcept = default;

void PipelineDeviceCache::clear() {
    if (storage_) storage_->device = detail::UploadedState{};
}

bool cudaDeviceAvailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

PipelineResult runHybridPipeline(const argcsr::CSRMatrix& csr,
                                 const std::vector<double>& x,
                                 int desiredChunkSize,
                                 int blockSize,
                                 const PipelineConfig& config,
                                 bool keepRepresentation,
                                 PipelineDeviceCache* deviceCache) {
    if (!cudaDeviceAvailable()) {
        throw std::runtime_error("CUDA device required for hybrid pipeline");
    }
    if (x.size() != static_cast<std::size_t>(csr.cols)) {
        throw std::invalid_argument("input vector length must equal matrix columns");
    }
    if (config.cudaBlockSize <= 0 || config.cudaBlockSize > 1024) {
        throw std::invalid_argument("cudaBlockSize must be in [1, 1024]");
    }

    const auto wallStart = detail::Clock::now();
    PipelineResult result;

    OpenMPConversionResult converted =
        convertCsrToArgCsrOpenMP(csr, desiredChunkSize, blockSize, config.ompThreads);
    result.timings.ompConversion = converted.timings;

    const auto& mat = converted.matrix;
    double allocMs = 0.0;

    if (deviceCache) {
        if (!deviceCache->storage_)
            deviceCache->storage_ = std::make_unique<PipelineDeviceCache::Storage>();
        const cudaStream_t stream = deviceCache->storage_->stream.get();
        detail::ensureAndUpload(deviceCache->storage_->device, converted, x, stream, result.timings.transfer,
                                allocMs);
        result.timings.deviceAllocMs = allocMs;
        detail::runKernelAndD2h(deviceCache->storage_->device, mat, config, stream, result);
    } else {
        detail::UploadedState device;
        detail::CudaStream stream;
        detail::ensureAndUpload(device, converted, x, stream.get(), result.timings.transfer, allocMs);
        result.timings.deviceAllocMs = allocMs;
        detail::runKernelAndD2h(device, mat, config, stream.get(), result);
    }

    result.timings.totalMs = detail::elapsedMs(wallStart, detail::Clock::now());
    if (keepRepresentation) {
        result.representation = std::move(converted.matrix);
    }
    return result;
}

double benchmarkHybridKernelOnly(const OpenMPConversionResult& converted,
                                 const std::vector<double>& x,
                                 int cudaBlockSize,
                                 std::vector<double>& output,
                                 double& kernelMs,
                                 TransferMetrics* transfer) {
    TransferMetrics local{};
    TransferMetrics& xfer = transfer ? *transfer : local;
    detail::UploadedState device;
    detail::CudaStream stream;
    double allocMs = 0.0;
    detail::ensureAndUpload(device, converted, x, stream.get(), xfer, allocMs);
    (void)allocMs;
    const auto& mat = converted.matrix;
    kernelMs = detail::launchTimedKernel(mat.rows, mat.blockSize, cudaBlockSize, stream.get(),
                                         device.groups.get(), device.values.get(), device.columns.get(),
                                         device.mapping.get(), device.rowGroup.get(), device.rowPrefix.get(),
                                         device.x.get(), device.y.get());
    output.resize(static_cast<std::size_t>(mat.rows));
    xfer.d2hBytes = output.size() * sizeof(double);
    xfer.d2hMs = detail::elapsedStreamMs(stream.get(), [&] {
        detail::enqueueD2H(output.data(), device.y.get(), output.size(), stream.get());
    });
    xfer.d2hBandwidthGbps = gigaBytesPerSecond(xfer.d2hBytes, xfer.d2hMs);
    return kernelMs;
}

}  // namespace hybrid_cpu_gpu
