#include "cuda/argcsr_device.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace argcsr {
namespace {

// Helper function to check cuda error status
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void allocateAndCopy(void** destination, const void* source, std::size_t count,
                     std::size_t elementSize, const char* name) {
    if (count == 0) return;
    if (count > std::numeric_limits<std::size_t>::max() / elementSize)
        throw std::overflow_error(std::string("ArgCSR device byte size overflow for ") + name);
    const std::size_t bytes = count * elementSize;
    checkCuda(cudaMalloc(destination, bytes), "cudaMalloc");
    checkCuda(cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice), name);
}

void validateHostRepresentation(const ArgCSRMatrix& host) {
    if (host.rows < 0 || host.cols < 0 || host.nnz < 0 || host.blockSize <= 0)
        throw std::invalid_argument("invalid ArgCSR dimensions, NNZ, or block size");
    if (host.values.size() != host.columns.size())
        throw std::invalid_argument("ArgCSR values and columns lengths differ");
    if (host.threadsMapping.size() != static_cast<std::size_t>(host.rows))
        throw std::invalid_argument("ArgCSR row mapping length does not equal rows");
    if (host.values.size() % static_cast<std::size_t>(host.blockSize) != 0)
        throw std::invalid_argument("ArgCSR storage length is not a whole number of blocks");

    std::size_t expectedOffset = 0; // tracks where the next group should start in the values array
    int expectedRow = 0; // tracks where the next group should start in the rows
    
    for (const auto& group : host.groups) {
        if (group.firstRow != expectedRow || group.size <= 0 || group.chunkSize <= 0 ||
            group.offset < 0 || static_cast<std::size_t>(group.offset) != expectedOffset ||
            group.size > host.rows - expectedRow)
            throw std::invalid_argument("invalid ArgCSR group metadata");
        
        const std::size_t chunkSize = static_cast<std::size_t>(group.chunkSize);
        if (chunkSize > (host.values.size() - expectedOffset) /
                            static_cast<std::size_t>(host.blockSize))
            throw std::invalid_argument("ArgCSR group exceeds storage arrays");
        int lanes = 0; // 
        for (int row = group.firstRow; row < group.firstRow + group.size; ++row) {
            const int count = host.threadsMapping[static_cast<std::size_t>(row)];
            if (count <= 0 || count > host.blockSize - lanes)
                throw std::invalid_argument("invalid ArgCSR row mapping");
            lanes += count;
        }
        expectedOffset += chunkSize * static_cast<std::size_t>(host.blockSize);
        expectedRow += group.size;
    }
    
    if (expectedRow != host.rows || expectedOffset != host.values.size())
        throw std::invalid_argument("ArgCSR groups do not cover rows and storage");
}

} // namespace

// Constructors, destructors, and assignment operators
ArgCSRDeviceMatrix::ArgCSRDeviceMatrix(const ArgCSRMatrix& host) { upload(host); }

ArgCSRDeviceMatrix::~ArgCSRDeviceMatrix() noexcept { reset(); }

ArgCSRDeviceMatrix::ArgCSRDeviceMatrix(ArgCSRDeviceMatrix&& other) noexcept {
    *this = std::move(other);
}

ArgCSRDeviceMatrix& ArgCSRDeviceMatrix::operator=(ArgCSRDeviceMatrix&& other) noexcept {
    if (this == &other) return *this;
    reset();
    rows = other.rows; cols = other.cols; nnz = other.nnz;
    blockSize = other.blockSize; desiredChunkSize = other.desiredChunkSize;
    groupCount = other.groupCount; storageSize = other.storageSize;
    groupsAllocation_ = std::exchange(other.groupsAllocation_, nullptr);
    valuesAllocation_ = std::exchange(other.valuesAllocation_, nullptr);
    columnsAllocation_ = std::exchange(other.columnsAllocation_, nullptr);
    mappingAllocation_ = std::exchange(other.mappingAllocation_, nullptr);
    groups = static_cast<const ArgCSRGroupInfo*>(groupsAllocation_);
    values = static_cast<const double*>(valuesAllocation_);
    columns = static_cast<const int*>(columnsAllocation_);
    threadsMapping = static_cast<const int*>(mappingAllocation_);
    other.groups = nullptr; other.values = nullptr; other.columns = nullptr;
    other.threadsMapping = nullptr;
    other.rows = other.cols = other.nnz = other.blockSize = other.desiredChunkSize = 0;
    other.groupCount = other.storageSize = 0;
    return *this;
}

// Uploadds the host representation of the matrix to the device, replacing any existing device representation
void ArgCSRDeviceMatrix::upload(const ArgCSRMatrix& host) {
    validateHostRepresentation(host);
    void* newGroups = nullptr;
    void* newValues = nullptr;
    void* newColumns = nullptr;
    void* newMapping = nullptr;
    try {
        allocateAndCopy(&newGroups, host.groups.data(), host.groups.size(),
                        sizeof(ArgCSRGroupInfo), "copy group metadata");
        allocateAndCopy(&newValues, host.values.data(), host.values.size(),
                        sizeof(double), "copy values");
        allocateAndCopy(&newColumns, host.columns.data(), host.columns.size(),
                        sizeof(int), "copy columns");
        allocateAndCopy(&newMapping, host.threadsMapping.data(), host.threadsMapping.size(),
                        sizeof(int), "copy row mapping");
    } catch (...) {
        if (newMapping) cudaFree(newMapping);
        if (newColumns) cudaFree(newColumns);
        if (newValues) cudaFree(newValues);
        if (newGroups) cudaFree(newGroups);
        throw;
    }
    reset(); // free any existing device allocations
    groupsAllocation_ = newGroups; valuesAllocation_ = newValues;
    columnsAllocation_ = newColumns; mappingAllocation_ = newMapping;
    groups = static_cast<const ArgCSRGroupInfo*>(groupsAllocation_);
    values = static_cast<const double*>(valuesAllocation_);
    columns = static_cast<const int*>(columnsAllocation_);
    threadsMapping = static_cast<const int*>(mappingAllocation_);
    rows = host.rows; cols = host.cols; nnz = host.nnz;
    blockSize = host.blockSize; desiredChunkSize = host.desiredChunkSize;
    groupCount = host.groups.size(); storageSize = host.values.size();
}

void ArgCSRDeviceMatrix::reset() noexcept {
    if (mappingAllocation_) cudaFree(mappingAllocation_);
    if (columnsAllocation_) cudaFree(columnsAllocation_);
    if (valuesAllocation_) cudaFree(valuesAllocation_);
    if (groupsAllocation_) cudaFree(groupsAllocation_);
    mappingAllocation_ = columnsAllocation_ = valuesAllocation_ = groupsAllocation_ = nullptr;
    groups = nullptr; values = nullptr; columns = nullptr; threadsMapping = nullptr;
    rows = cols = nnz = blockSize = desiredChunkSize = 0;
    groupCount = storageSize = 0;
}

ArgCSRMatrix ArgCSRDeviceMatrix::copyToHost() const {
    ArgCSRMatrix host{rows, cols, nnz, blockSize, desiredChunkSize, {}, {}, {}, {}};
    host.groups.resize(groupCount);
    host.values.resize(storageSize);
    host.columns.resize(storageSize);
    host.threadsMapping.resize(static_cast<std::size_t>(rows));
    if (groupCount) checkCuda(cudaMemcpy(host.groups.data(), groups,
        groupCount * sizeof(ArgCSRGroupInfo), cudaMemcpyDeviceToHost), "copy groups to host");
    if (storageSize) {
        checkCuda(cudaMemcpy(host.values.data(), values, storageSize * sizeof(double),
            cudaMemcpyDeviceToHost), "copy values to host");
        checkCuda(cudaMemcpy(host.columns.data(), columns, storageSize * sizeof(int),
            cudaMemcpyDeviceToHost), "copy columns to host");
    }
    if (rows) checkCuda(cudaMemcpy(host.threadsMapping.data(), threadsMapping,
        static_cast<std::size_t>(rows) * sizeof(int), cudaMemcpyDeviceToHost), "copy row mapping to host");
    return host;
}

ArgCSRDeviceView ArgCSRDeviceMatrix::view() const noexcept {
    return {rows, cols, nnz, blockSize, desiredChunkSize, groupCount, storageSize,
            groups, values, columns, threadsMapping};
}

} // namespace argcsr
