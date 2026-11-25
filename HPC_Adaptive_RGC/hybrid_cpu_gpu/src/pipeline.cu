#include "hybrid_cpu_gpu/pipeline.hpp"

#include "hybrid_cpu_gpu/conversion_openmp.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hybrid_cpu_gpu {
namespace {

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

    void allocate(std::size_t count) {
        if (ptr_) throw std::logic_error("device buffer already allocated");
        if (count == 0) return;
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
    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

template <class T>
double copyH2D(T* dst, const T* src, std::size_t count) {
    if (count == 0) return 0.0;
    const auto start = Clock::now();
    checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    return elapsedMs(start, Clock::now());
}

template <class T>
double copyD2H(T* dst, const T* src, std::size_t count) {
    if (count == 0) return 0.0;
    const auto start = Clock::now();
    checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    return elapsedMs(start, Clock::now());
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
    checkCuda(cudaEventCreate(&start), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate");
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<std::size_t>(rows) + cudaBlockSize - 1) / cudaBlockSize);
    checkCuda(cudaEventRecord(start), "cudaEventRecord");
    hybridSpmvKernel<<<grid, static_cast<unsigned int>(cudaBlockSize)>>>(
        rows, blockSize, dGroups, dValues, dColumns, dMapping, dRowGroup, dRowPrefix, dX, dY);
    checkCuda(cudaGetLastError(), "hybridSpmvKernel");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return static_cast<double>(ms);
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

UploadedState upload(const OpenMPConversionResult& converted, const std::vector<double>& x,
                     TransferMetrics& xfer) {
    const auto& m = converted.matrix;
    UploadedState state;
    const auto allocStart = Clock::now();
    state.groups.allocate(m.groups.size());
    state.values.allocate(m.values.size());
    state.columns.allocate(m.columns.size());
    state.mapping.allocate(static_cast<std::size_t>(m.rows));
    state.rowGroup.allocate(converted.rowGroup.size());
    state.rowPrefix.allocate(converted.rowPrefix.size());
    state.x.allocate(x.size());
    state.y.allocate(static_cast<std::size_t>(m.rows));
    (void)allocStart;

    double h2d = 0.0;
    h2d += copyH2D(state.groups.get(), m.groups.data(), m.groups.size());
    h2d += copyH2D(state.values.get(), m.values.data(), m.values.size());
    h2d += copyH2D(state.columns.get(), m.columns.data(), m.columns.size());
    h2d += copyH2D(state.mapping.get(), m.threadsMapping.data(), m.threadsMapping.size());
    h2d += copyH2D(state.rowGroup.get(), converted.rowGroup.data(), converted.rowGroup.size());
    h2d += copyH2D(state.rowPrefix.get(), converted.rowPrefix.data(), converted.rowPrefix.size());
    h2d += copyH2D(state.x.get(), x.data(), x.size());

    xfer.h2dBytes = countH2DBytes(converted, x);
    xfer.h2dMs = h2d;
    xfer.h2dBandwidthGbps = gigaBytesPerSecond(xfer.h2dBytes, xfer.h2dMs);
  return state;
}

}  // namespace

bool cudaDeviceAvailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

PipelineResult runHybridPipeline(const argcsr::CSRMatrix& csr,
                                 const std::vector<double>& x,
                                 int desiredChunkSize,
                                 int blockSize,
                                 const PipelineConfig& config,
                                 bool keepRepresentation) {
    if (!cudaDeviceAvailable()) {
        throw std::runtime_error("CUDA device required for hybrid pipeline");
    }
    if (x.size() != static_cast<std::size_t>(csr.cols)) {
        throw std::invalid_argument("input vector length must equal matrix columns");
    }
  if (config.cudaBlockSize <= 0 || config.cudaBlockSize > 1024) {
    throw std::invalid_argument("cudaBlockSize must be in [1, 1024]");
  }

    const auto wallStart = Clock::now();
    PipelineResult result;

    OpenMPConversionResult converted =
        convertCsrToArgCsrOpenMP(csr, desiredChunkSize, blockSize, config.ompThreads);
    result.timings.ompConversion = converted.timings;

    const auto& mat = converted.matrix;
    const auto allocStart = Clock::now();
    UploadedState device = upload(converted, x, result.timings.transfer);
    result.timings.deviceAllocMs = elapsedMs(allocStart, Clock::now());

    result.timings.kernelMs = launchTimedKernel(
        mat.rows, mat.blockSize, config.cudaBlockSize, device.groups.get(), device.values.get(),
        device.columns.get(), device.mapping.get(), device.rowGroup.get(), device.rowPrefix.get(),
        device.x.get(), device.y.get());

    result.output.resize(static_cast<std::size_t>(mat.rows));
    const auto d2hStart = Clock::now();
    const double d2h = copyD2H(result.output.data(), device.y.get(), result.output.size());
    result.timings.transfer.d2hMs = d2h;
    result.timings.transfer.d2hBytes = result.output.size() * sizeof(double);
    result.timings.transfer.d2hBandwidthGbps =
        gigaBytesPerSecond(result.timings.transfer.d2hBytes, d2h);
    (void)d2hStart;

    result.timings.totalMs = elapsedMs(wallStart, Clock::now());
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
    UploadedState device = upload(converted, x, xfer);
    const auto& mat = converted.matrix;
    (void)launchTimedKernel(mat.rows, mat.blockSize, cudaBlockSize, device.groups.get(),
                            device.values.get(), device.columns.get(), device.mapping.get(),
                            device.rowGroup.get(), device.rowPrefix.get(), device.x.get(),
                            device.y.get());
    kernelMs = launchTimedKernel(mat.rows, mat.blockSize, cudaBlockSize, device.groups.get(),
                                 device.values.get(), device.columns.get(), device.mapping.get(),
                                 device.rowGroup.get(), device.rowPrefix.get(), device.x.get(),
                                 device.y.get());
    output.resize(static_cast<std::size_t>(mat.rows));
    xfer.d2hMs = copyD2H(output.data(), device.y.get(), output.size());
    xfer.d2hBytes = output.size() * sizeof(double);
    xfer.d2hBandwidthGbps = gigaBytesPerSecond(xfer.d2hBytes, xfer.d2hMs);
    return kernelMs;
}

}  // namespace hybrid_cpu_gpu
