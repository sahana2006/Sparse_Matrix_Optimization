#include "cuda/argcsr_device.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void requireSame(const argcsr::ArgCSRMatrix& a, const argcsr::ArgCSRMatrix& b) {
    require(a.rows == b.rows && a.cols == b.cols && a.nnz == b.nnz &&
            a.blockSize == b.blockSize && a.desiredChunkSize == b.desiredChunkSize,
            "matrix scalar metadata differs after copy-back");
    require(a.values == b.values, "values differ after copy-back");
    require(a.columns == b.columns, "columns differ after copy-back");
    require(a.threadsMapping == b.threadsMapping, "row mapping differs after copy-back");
    require(a.groups.size() == b.groups.size(), "group count differs after copy-back");
    for (std::size_t i = 0; i < a.groups.size(); ++i) {
        const auto& x = a.groups[i];
        const auto& y = b.groups[i];
        require(x.firstRow == y.firstRow && x.size == y.size && x.offset == y.offset &&
                x.chunkSize == y.chunkSize, "group metadata differs after copy-back");
    }
}
}

int main() {
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        std::cout << "ArgCSR device test skipped: no usable CUDA device\n";
        (void)cudaGetLastError();
        return 0;
    }

    try {
        const argcsr::CSRMatrix csr(4, 8,
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            {0, 2, 1, 3, 4, 5, 6, 7, 0, 2, 7},
            {0, 2, 5, 7, 11});
        const auto host = argcsr::convertToArgCSR(csr, 4, 4);
        {
            argcsr::ArgCSRDeviceMatrix device(host);
            require(device.rows == host.rows && device.cols == host.cols && device.nnz == host.nnz,
                    "scalar metadata was not uploaded");
            require(device.groupCount == host.groups.size() && device.storageSize == host.values.size(),
                    "device array lengths were not preserved");
            require(device.groups != nullptr && device.values != nullptr && device.columns != nullptr &&
                    device.threadsMapping != nullptr, "non-empty device arrays have null pointers");
            requireSame(host, device.copyToHost());

            // Repeated upload replaces the old allocation set safely.
            device.upload(host);
            requireSame(host, device.copyToHost());
            argcsr::ArgCSRDeviceMatrix moved(std::move(device));
            requireSame(host, moved.copyToHost());
            moved.reset();
            moved.reset();
            require(moved.groups == nullptr && moved.values == nullptr && moved.columns == nullptr &&
                    moved.threadsMapping == nullptr, "reset did not clear device pointers");
        }

        const argcsr::CSRMatrix empty(0, 3, {}, {}, {0});
        const auto emptyHost = argcsr::convertToArgCSR(empty, 4, 4);
        argcsr::ArgCSRDeviceMatrix emptyDevice(emptyHost);
        require(emptyDevice.groupCount == 0 && emptyDevice.storageSize == 0,
                "empty representation has unexpected allocations");
        requireSame(emptyHost, emptyDevice.copyToHost());
        emptyDevice.reset();

        for (int iteration = 0; iteration < 8; ++iteration) {
            argcsr::ArgCSRDeviceMatrix repeated(host);
            requireSame(host, repeated.copyToHost());
        }
        std::cout << "ArgCSR device representation tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << "ArgCSR device representation tests failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
