# Phase 6B: GPU-Compatible Adaptive RGC Data Representation

## Scope

Phase 6B adds an owning CUDA copy of the existing Adaptive RGC representation. It does not alter grouping, lane assignment, chunk sizing, conversion, or CPU SpMV, and it does not add a CUDA SpMV kernel.

## Existing CPU representation

The input `CSRMatrix` stores `rows`, `cols`, and `nnz` as `int`, values as `double`, column indices as `int`, and `rows + 1` row pointers as `int`. Its vectors own their storage and live for the lifetime of the matrix.

Conversion produces `ArgCSRMatrix` with the following scalar metadata: dimensions, NNZ, `blockSize`, and `desiredChunkSize` (all `int`). Its owning vectors are:

| CPU member | Type | Meaning |
| --- | --- | --- |
| `groups` | `vector<ArgCSRGroupInfo>` | Ordered groups covering consecutive output rows. |
| `values` | `vector<double>` | Group storage slabs, including zero padding. |
| `columns` | `vector<int>` | Column for each stored value; `-1` denotes padding. |
| `threadsMapping` | `vector<int>` | One count per matrix row, giving the number of consecutive lanes assigned to that row in its group. |

Each group has four `int` fields: `firstRow`, `size`, `offset`, and `chunkSize`. `offset` is in elements, not bytes. The group's slab has `chunkSize * blockSize` entries. For group-local lane `t` and chunk index `k`, the storage position is `offset + k * blockSize + t`. Rows occupy consecutive lane ranges in order; the start lane of a row is the prefix sum of the preceding rows' `threadsMapping` counts. The mapping counts for a group sum to at most `blockSize`. The representation does not store CSR row pointers after conversion.

The CPU SpMV reads lanes from zero to the sum of the row mapping counts and reduces each row's consecutive assigned lanes into its output row. It treats `-1` as the end of valid entries in a lane. Groups are ordered, cover each output row once, and have contiguous storage offsets. These details, including padded entries and row ownership, are preserved by the device copy.

## Device representation and ownership

`ArgCSRDeviceMatrix` is a move-only RAII owner. It retains scalar matrix metadata on the host object and owns four separate CUDA allocations exposed through const pointers: groups, values, columns, and row mapping. Their element types remain `ArgCSRGroupInfo`, `double`, and `int`; no index narrowing or conversion occurs. The CUDA allocation API supplies device alignment, and each typed array starts at its own allocation boundary. Separate allocations keep each array contiguous without packing unrelated element types or adding alignment/padding logic.

The group array preserves the four fields for every group. `values` and `columns` have identical lengths and preserve the complete padded slabs. `threadsMapping` has exactly `rows` entries. CSR row pointers are not transferred: they are not part of the converted representation and are not needed to consume it. The input vector `x` and output vector `y` are also outside this transfer-only module.

All host vectors remain owned by the caller's `ArgCSRMatrix`. Device pointers remain owned by `ArgCSRDeviceMatrix`; they are valid until `reset`, destruction, or a successful replacement upload. Copy and copy assignment are disabled; move operations transfer ownership. Upload validates structural lengths and ranges, allocates and copies into temporary buffers, then replaces existing buffers only after all operations succeed. If allocation or transfer fails, temporary buffers are released and the previous device representation remains intact. Destruction and `reset` release every allocated buffer. Errors during allocation/copy are reported as exceptions containing the CUDA error text; `cudaFree` failures during no-throw cleanup cannot be reported.

`copyToHost()` is provided for representation validation and diagnostics. A future kernel should treat the exposed arrays as read-only and use the dimensions and layout rules above. The device struct is a host-side owner/view, not a kernel argument ABI; a kernel-facing POD view can be introduced with the SpMV implementation phase.

## Array mapping and transfer

| Host source | Device destination | Transfer count |
| --- | --- | --- |
| `groups` | `groups` | `groups.size()` group records |
| `values` | `values` | `values.size()` doubles |
| `columns` | `columns` | `columns.size()` ints |
| `threadsMapping` | `threadsMapping` | `rows` ints |

Empty arrays produce null device pointers and zero lengths; no zero-byte CUDA allocation or copy is attempted. The four arrays are copied host-to-device once per upload. Group metadata is small but needed to preserve all group boundaries and slab offsets. No input CSR arrays, vectors, CPU validation scratch, or timing data are transferred.

## Build configuration

CUDA support is opt-in (`ENABLE_CUDA=OFF` by default), so the existing C++ build and CPU targets remain available without a CUDA compiler. The CUDA target requires CMake 3.18 or newer, CUDA Toolkit 13, and a Windows host compiler supported by NVCC. On Windows, use a Visual Studio 2022 x64 Native Tools/Developer PowerShell with the C++ workload installed; MinGW is not selected as NVCC's host compiler.

From that Visual Studio developer environment, at the project root (Ninja is the build tool; MSVC is both the C++ compiler and NVCC host compiler):

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CUDA_COMPILER=nvcc.exe -DCMAKE_CUDA_HOST_COMPILER=cl.exe
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

The CUDA targets compile for compute capability 12.0 (`sm_120`), matching the project's RTX 5070 Laptop GPU. The CUDA language is kept optional so systems without the CUDA Toolkit do not affect the CPU build.

To build the existing CPU configuration without CUDA:

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

Using separate build directories avoids mixing compiler and language settings. The current environment has CUDA Toolkit 13.0 (`nvcc` 13.0.88), Visual Studio 2022 MSVC 19.44, and an RTX 5070 Laptop GPU. `cl.exe` is not on the default shell PATH, so enter the VS 2022 developer environment before running the CUDA commands above. A Visual Studio generator configure was also attempted, but this installation lacks Visual Studio's CUDA toolset integration; the Ninja generator with explicit CUDA/MSVC compilers works.

## Tests and results

`test_argcsr_device` constructs a converted matrix, uploads it, checks scalar and array lengths, copies every array back and compares values and all group fields, repeats upload, exercises move ownership, checks a zero-row representation, and repeats scoped allocation/destruction. If no CUDA device is available at runtime, the test reports that it was skipped and returns success; a successful CTest result in that situation does not mean device transfers were exercised.

On this workstation, the CPU suite passed all 4 existing CTest tests in a fresh MinGW build. The CUDA-enabled Ninja build using MSVC 19.44 and CUDA 13.0 compiled successfully, and all 5 CTest tests passed, including the device representation test on the available GPU. The repository's existing CPU test targets remain unchanged.

## Design decisions and limitations

- Existing `int` index types and `double` values are retained exactly. The CPU implementation already limits adaptive storage offsets to the `int` range; Phase 6B does not expand those limits.
- Four independent allocations provide contiguous storage per array and straightforward ownership. No packed arena or speculative alignment optimization is introduced.
- Upload is synchronous and performs a complete representation copy. Pinned memory, asynchronous streams, partial updates, and kernel execution are outside this phase.
- Allocation and H2D failures are checked and surfaced. Cleanup is no-throw to remain safe in destructors.
- No CUDA SpMV kernel, GPU output ownership, device-side grouping, or performance claim is included.
