# Phase 6C: Complete GPU Adaptive RGC Pipeline

## Implemented path

`runArgCsrCudaPipeline` executes the existing Adaptive RGC layout from CSR input to output:

```text
host CSR validation
  -> H2D CSR arrays
  -> GPU row nonzero counts
  -> D2H row counts
  -> CPU ordered group-boundary decisions
  -> H2D groups
  -> GPU per-group iterative lane mapping
  -> GPU per-group chunk-size reduction
  -> D2H chunk metadata and CPU contiguous-offset calculation
  -> H2D completed group metadata
  -> GPU row mapping/prefix metadata
  -> GPU padded storage initialization and CSR-to-ArgCSR scatter
  -> H2D x
  -> GPU Adaptive RGC SpMV
  -> D2H y
```

The GPU representation uses the Phase 6B array types and layout through `ArgCSRDeviceView`: group records, padded `double` values, `int` columns with `-1` padding, and one `int` lane count per row. The pipeline builds these arrays directly on the device; optional copy-back is available for stage correctness tests. Normal execution leaves the representation on the device through SpMV and copies back only the output.

## Source modules

The public entry point and allocation/timing orchestration are in `src/cuda/argcsr_cuda_pipeline.cu`. Stage implementations are split into:

- `src/cuda/argcsr_cuda_grouping.cu` — parallel row statistics and ordered host group decisions from those statistics.
- `src/cuda/argcsr_cuda_mapping.cu` — per-group iterative lane mapping, parallel chunk-size reduction, mapping prefixes, and row-to-group metadata.
- `src/cuda/argcsr_cuda_storage.cu` — padded storage initialization is orchestrated by the pipeline; this module scatters CSR nonzeros into their unique Adaptive RGC destinations.
- `src/cuda/argcsr_cuda_spmv.cu` — row-owned Adaptive RGC SpMV kernel.
- `src/cuda/argcsr_cuda_pipeline_internal.cuh` — private launch declarations shared by the orchestrator and stage modules.

These are linked into `argcsr_cuda`; they are not separate end-user executables. Run the pipeline through the test or benchmark targets below.

## Stage behavior and CPU work retained

- **Row statistics / grouping:** A GPU thread computes each CSR row's NNZ count. Those counts are copied to the host because the existing grouping decision consumes rows in order and carries an accumulated workload across each boundary. The host applies the same rule: close a nonempty group when adding the next row would exceed `desiredChunkSize * blockSize` or the row capacity. This prefix dependency stays sequential; each row's count is still computed independently on the GPU.
- **Mapping:** A CUDA block is assigned to each independent group. One thread in that block applies the existing iterative lane-allocation rule, including its first-row tie break and “only keep an assignment if it reduces the current required chunk” behavior. The iterations within a group are dependent, so they remain sequential. Separate groups execute concurrently. A second group-parallel kernel forms the ordered row-to-lane prefix; a row-parallel binary search builds row-to-group indices.
- **Chunk sizing:** One block handles each group. Threads calculate `ceil(rowNNZ / assignedLanes)` for separate rows, followed by a simple shared-memory max reduction. The result is clamped to at least one, matching the CPU conversion.
- **Offsets:** Group slab offsets are an exclusive prefix of `chunkSize * blockSize`. The small metadata list is copied to the host and offsets are assigned sequentially, preserving the CPU's contiguous layout and `int` overflow checks. The completed group records are copied back to the GPU.
- **Storage conversion:** The device initializes every value slot to zero and every column slot to `-1`, preserving CPU padding. One GPU thread owns each CSR nonzero. It finds the CSR row by binary search, then writes to `group.offset + (localNNZ % chunkSize) * blockSize + rowPrefix[row] + (localNNZ / chunkSize)`. Each input nonzero has one destination, so value/column writes do not race. A device error flag reports any unexpected lane-capacity mismatch.
- **SpMV:** One GPU thread owns one output row. It visits that row's assigned lanes in order, computes each lane's chunk sum in order, then adds lane partial sums in order. This follows the CPU reduction structure without warp shuffles or shared-memory reductions. Empty lanes stop at `-1`; empty rows write zero. Row ownership is unique, and groups may be multiple or irregular.
- **Host-only work:** CSR input validation, ordered group-boundary decisions, and group storage-offset prefixing remain on the CPU. They have sequential dependencies and small metadata volume. All intermediate transfer and CPU durations are exposed in `CudaPipelineTimings`.

## Timing fields

`CudaPipelineTimings` reports milliseconds for host validation, H2D CSR, GPU row statistics, D2H row statistics, CPU grouping, H2D groups, GPU mapping, GPU chunk sizing, D2H chunk metadata, CPU offsets, final metadata H2D, GPU storage construction, x H2D, GPU SpMV, output D2H, optional representation D2H, and end-to-end pipeline wall time. `cpuPreprocessingMs()`, `gpuKernelMs()`, and `transferMs()` provide component sums. The wall time includes allocation and synchronization overhead; component sums need not equal wall time because host setup, allocation, launches, and event handling also contribute.

GPU stage durations use CUDA events and synchronize at stage boundaries to obtain independent measurements. The benchmark reports medians, validates every GPU result against CSR SpMV, and also reports CPU Adaptive RGC conversion and SpMV time. Profiling runs add overhead and should be analyzed separately from the benchmark's unprofiled timings.

## Build and tests

Use a VS 2022 x64 Native Tools/Developer PowerShell with CUDA Toolkit 13.0 on `PATH`. The CMake CUDA build uses NVCC and MSVC as the CUDA host and C++ compilers; MinGW is not used as the CUDA host compiler. The project keeps the CPU-only configuration as its default.

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CUDA_COMPILER=nvcc.exe -DCMAKE_CUDA_HOST_COMPILER=cl.exe
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

After configuring, you can build only the CUDA library or the pipeline test/benchmark targets:

```powershell
cmake --build build-cuda --target argcsr_cuda
cmake --build build-cuda --target test_argcsr_cuda_pipeline
cmake --build build-cuda --target cuda_pipeline_benchmark
```

The added `test_argcsr_cuda_pipeline` checks the GPU-produced group boundaries, chunk sizes, offsets, row mapping, values, columns, and final output against the unchanged CPU conversion and CSR SpMV. It covers uniform and irregular rows, multiple groups, empty rows, and a zero-row matrix. `test_argcsr_device` from Phase 6B remains in CTest. CUDA tests print a skip message and return success if no usable GPU is visible, so run them directly when you need to confirm that GPU work actually ran:

```powershell
.\build-cuda\test_argcsr_cuda_pipeline.exe
.\build-cuda\test_argcsr_device.exe
```

Tests and builds were not run for Phase 6C in this implementation session, per the execution plan that you will compile and validate locally. No test or benchmark outcome is claimed here.

## Benchmark commands

`cuda_pipeline_benchmark` accepts `[matrix-directory] [runs] [desired-chunk-size] [block-size] [matrix-name ...]`. With no names it processes `U1 U2 U3 U4 I1 I2 I3 I4`, the existing uniform and irregular datasets. The matrix generator defines 10,000, 50,000, 100,000, and 500,000 row cases. You can also point it at a directory of additional `.csrbin` matrices and pass their names.

Small validation run:

```powershell
.\build-cuda\cuda_pipeline_benchmark.exe benchmarks\matrices 3 2 4 U1 I1
```

All current benchmark datasets:

```powershell
.\build-cuda\cuda_pipeline_benchmark.exe benchmarks\matrices 3 2 4
```

The benchmark writes `benchmarks/results/cuda_pipeline_results.csv`, including CPU group decisions and offset time, GPU time by kernel stage, H2D and D2H, component sums, complete pipeline wall time, and CPU Adaptive RGC conversion/SpMV reference times. It does not run during CTest.

## Profiling guide

Start with the CSV's per-stage CUDA-event and transfer timings and its end-to-end wall time. For CUDA API, kernel launch, transfer, and CPU/GPU timeline inspection, capture one representative matrix with Nsight Systems:

```powershell
nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o phase6c-U1 .\build-cuda\cuda_pipeline_benchmark.exe benchmarks\matrices 1 2 4 U1
```

Open `phase6c-U1.nsys-rep` in Nsight Systems. The CUDA timeline distinguishes the named row-statistics, mapping, chunk-sizing, row-prefix/group-index, storage-scatter, and SpMV kernels, as well as H2D/D2H copies and host gaps. The CSV provides named CPU grouping and offset durations and complete pipeline wall time; those CPU phases do not currently have NVTX ranges.

For per-kernel duration, register use, occupancy analysis, and memory throughput, use Nsight Compute on a single matrix:

```powershell
ncu --set full --force-overwrite -o phase6c-U1 .\build-cuda\cuda_pipeline_benchmark.exe benchmarks\matrices 1 2 4 U1
```

Open `phase6c-U1.ncu-rep` in Nsight Compute. Inspect each kernel by its name: `rowStatisticsKernel`, `mappingKernel`, `chunkSizingKernel`, `mappingPrefixKernel`, `rowGroupKernel`, `storageScatterKernel`, and `spmvKernel`. The Launch Statistics and Occupancy sections show registers per thread and occupancy limits; Memory Workload Analysis reports achieved memory throughput and cache/DRAM activity. For a CLI metric inventory on the installed Nsight Compute version:

```powershell
ncu --query-metrics | Select-String "dram__throughput|launch__registers_per_thread|occupancy"
```

For a coarse live utilization trace, run the benchmark in one terminal and this in another:

```powershell
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

`utilization.memory` is memory-controller activity, not a direct bandwidth measurement. Use Nsight Compute's DRAM throughput metrics for memory throughput. Profilers may replay kernels or serialize work; do not use profiled durations as the unprofiled benchmark result. Nsight Systems' CLI supports `nsys profile --trace=cuda,...`; Nsight Compute provides CLI collection and detailed per-kernel metrics in its report. See NVIDIA's [Nsight Systems CLI guide](https://docs.nvidia.com/nsight-systems/UserGuide/) and [Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/).

## Limitations and deferred work

- Group boundary selection and group offsets remain sequential CPU work by design; GPU row-count and chunk calculations are parallel.
- Mapping assignment is serialized within each group to preserve iterative decisions and tie behavior. Groups run concurrently.
- The baseline SpMV gives one thread to each row; rows with many chunks do more serial work and may underutilize the GPU.
- Storage construction uses a binary search through CSR row pointers per nonzero. This is simple and race-free but adds work for large inputs.
- Each invocation allocates fresh device buffers and uses synchronous copies/default-stream execution with stage synchronization. No persistent buffers, streams, pinned memory, CUDA Graphs, warp shuffles, aggressive shared-memory tuning, cache-policy tuning, or dynamic scheduling are introduced.
- Current CMake CUDA targets are compiled for `sm_120`, matching the project's RTX 5070 Laptop GPU. Other GPUs require changing the architecture setting.
- Group records and all existing index values remain `int`; existing CPU offset-range limits still apply.
- Performance and correctness results are intentionally unreported until you run the provided commands.
