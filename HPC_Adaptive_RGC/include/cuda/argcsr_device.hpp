#pragma once

#include "argcsr/argcsr_types.hpp"

#include <cstddef>
#include <vector>

namespace argcsr {

// Non-owning POD view of the Phase 6B representation, suitable for kernel arguments.
struct ArgCSRDeviceView {
    int rows;
    int cols;
    int nnz;
    int blockSize;
    int desiredChunkSize;
    std::size_t groupCount;
    std::size_t storageSize;
    const ArgCSRGroupInfo* groups;
    const double* values;
    const int* columns;
    const int* threadsMapping;
};

// Owns a device-side copy of the existing adaptive group/lane layout.
// The arrays are read-only to CUDA kernels; this class owns their allocations.
class ArgCSRDeviceMatrix {
public:
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int blockSize = 0;
    int desiredChunkSize = 0;
    std::size_t groupCount = 0;
    std::size_t storageSize = 0;

    const ArgCSRGroupInfo* groups = nullptr;
    const double* values = nullptr;
    const int* columns = nullptr;
    const int* threadsMapping = nullptr;

    ArgCSRDeviceMatrix() noexcept = default;
    explicit ArgCSRDeviceMatrix(const ArgCSRMatrix& host); // creates a device-side copy of the host matrix
    ~ArgCSRDeviceMatrix() noexcept; // releases device-side allocations

    ArgCSRDeviceMatrix(const ArgCSRDeviceMatrix&) = delete;
    ArgCSRDeviceMatrix& operator=(const ArgCSRDeviceMatrix&) = delete;
    ArgCSRDeviceMatrix(ArgCSRDeviceMatrix&& other) noexcept;
    ArgCSRDeviceMatrix& operator=(ArgCSRDeviceMatrix&& other) noexcept;

    void upload(const ArgCSRMatrix& host);
    void reset() noexcept;
    ArgCSRMatrix copyToHost() const;
    ArgCSRDeviceView view() const noexcept;

private:
    void* groupsAllocation_ = nullptr;
    void* valuesAllocation_ = nullptr;
    void* columnsAllocation_ = nullptr;
    void* mappingAllocation_ = nullptr;
};

} // namespace argcsr
