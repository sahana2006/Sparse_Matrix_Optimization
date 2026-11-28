## 1. Project Overview

This project focuses on optimizing Sparse Matrix–Vector Multiplication (SpMV) using Adaptive Row-Grouped CSR (Adaptive RGC) and GPU acceleration.

The implementation evaluates sparse matrix computation across uniform and irregular matrix patterns using sequential CPU and CUDA-based implementations. It also includes an experimental CUDA Streams implementation to investigate data-transfer performance.

Key components include:
- Adaptive RGC construction and storage conversion.
- CUDA acceleration of GPU processing stages and SpMV.
- CUDA Streams experimentation.
- Performance benchmarking across different matrix sizes and sparsity patterns.

## 2. Hardware and Software Requirements

### Hardware
- NVIDIA CUDA-compatible GPU.
- Sufficient GPU memory for the selected matrix size.
- Multicore CPU.

### Software Dependencies
- C++17-compatible compiler.
- NVIDIA CUDA Toolkit 13.0, or a compatible configured version.
- NVIDIA GPU driver.
- CMake 4.4.4, or a compatible version.
- Microsoft Visual Studio 2022 with C++ development tools on Windows.
- Git (optional).

## 3. Installation and Configuration

### Step 1: Obtain the Project

Clone the repository:

```bash
git clone https://github.com/sahana2006/Sparse_Matrix_Optimization.git
cd HPC_Adaptive_RGC
```

Alternatively, extract the project ZIP and open a terminal in the project directory.

### Step 2: Verify Dependencies

Check that the NVIDIA driver and CUDA compiler are available:

```bash
nvidia-smi
nvcc --version
```

Check the CMake installation:

```bash
cmake --version
```

On Windows, use a Visual Studio 2022 x64 development environment with the C++ compiler available. CUDA compilation with NVCC requires a supported host compiler.

### Step 3: Configure the Build

From the project root, configure the CUDA Streams build:

```bash
cmake -S . -B build-cuda-streams
```

### Step 4: Build

```bash
cmake --build build-cuda-streams --config Release
```

Use a Release build for performance measurements.

## 4. Running the Benchmark

To run the CUDA Streams benchmark from the project root on Windows:

```powershell
.\\build-cuda-streams\\cuda_pipeline_streams_benchmark.exe
```

The benchmark reports CPU preprocessing and execution time, GPU kernel time, host-to-device (H2D) transfer time, device-to-host (D2H) transfer time, SpMV time, and total pipeline execution time.

Benchmark results are written to:

```text
benchmarks/results/cuda_pipeline_streams_results.csv
```

The executable name and build directory above correspond to the CUDA Streams build configuration. Use the target names defined in the project's `CMakeLists.txt` if building other configurations.

## 5. Benchmark Matrices

The experiments use eight sparse matrix configurations:

| Matrix | Dimensions | Non-zero elements |
|---|---:|---:|
| U1 | 10,000 × 10,000 | 100,000 |
| U2 | 50,000 × 50,000 | 500,000 |
| U3 | 100,000 × 100,000 | 1,000,000 |
| U4 | 500,000 × 500,000 | 5,000,000 |
| I1 | 10,000 × 10,000 | 100,000 |
| I2 | 50,000 × 50,000 | 500,000 |
| I3 | 100,000 × 100,000 | 1,000,000 |
| I4 | 500,000 × 500,000 | 5,000,000 |

U1–U4 represent uniform matrices, while I1–I4 represent irregular matrices.

## 6. Performance Metrics

The implementations are evaluated using the following metrics:

- **End-to-end execution time:** Measures the complete computation pipeline.
- **Speedup:** Compares CUDA execution against sequential CPU execution.
- **GPU kernel time:** Measures GPU computation time.
- **H2D transfer time:** Measures data transfer from host memory to device memory.
- **D2H transfer time:** Measures data transfer from device memory to host memory.
- **SpMV execution time:** Measures the sparse matrix–vector multiplication stage.
- **Scalability:** Examines performance as matrix dimensions and non-zero element counts increase.
