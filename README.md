# Adaptive Row-Grouped CSR for Sparse Matrix Optimization

## 1. Overview

This project implements the **Adaptive Row-Grouped CSR (Adaptive RGC)** format for sparse matrix-vector multiplication (SpMV), based on the paper *“Adaptive Row-grouped CSR format for Storing of Sparse Matrices on GPU”*. It groups rows and adaptively distributes non-zero elements to improve workload balance and parallel execution on GPUs.


---

## 2. Methodology

The implementation follows the following pipeline:

**CSR Matrix → Row Grouping → Adaptive Chunking → Thread Mapping → Storage Conversion → SpMV**

- **Grouping:** Consecutive rows are grouped according to their workload.
- **Chunking:** The workload of each group is divided into suitable chunks.
- **Mapping:** Chunks are assigned to rows based on their number of non-zero elements.
- **Storage Conversion:** CSR data is reorganized into a chunk-oriented Adaptive RGC representation.
- **SpMV:** Matrix-vector multiplication is performed using the generated chunks.

The sequential implementation is first profiled to identify computational hotspots and determine which stages are suitable for parallel execution.

---

## 3. Parallelization

The project investigates parallelization of Adaptive RGC using **CUDA** and **OpenMP**.

CUDA is selected as the primary platform because the Adaptive RGC representation provides fine-grained chunk-level parallelism, which maps naturally to GPU threads. SpMV also contains a large amount of independent computation, while storage conversion can be parallelized across independent rows and chunks.

OpenMP is considered as a CPU shared-memory implementation for comparison.
