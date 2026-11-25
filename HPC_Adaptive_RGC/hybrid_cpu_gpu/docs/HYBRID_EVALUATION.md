# Hybrid evaluation

## Implementations compared

1. **Sequential** — existing `convertToArgCSR` + `argcsr_spmv`
2. **CUDA pipeline** — existing `runArgCsrCudaPipeline` (unchanged)
3. **OpenMP conversion** — `hybrid_cpu_gpu::convertCsrToArgCsrOpenMP`
4. **Hybrid** — OpenMP conversion + H2D + CUDA SpMV + D2H (`runHybridPipeline`)

## Metrics

| Metric | Source |
|--------|--------|
| **Kernel time** | `PipelineTimingsMs::kernelMs` (`cudaEvent`) |
| **Memory bandwidth** | `TransferMetrics::{h2d,d2h}BandwidthGbps` in CSV |
| **GPU utilization** | Run `scripts/capture_gpu_utilization.sh` during benchmark |
| **Speedup** | `speedup_*` columns in `four_way_comparison.csv` |
| **Scalability** | Re-run benchmark with `omp_threads` argument; matrix sweep U1–I4 |

## GPU laptop run 

```bash
cd HPC_Adaptive_RGC
cmake -S . -B build -DENABLE_CUDA=ON -DENABLE_HYBRID_CPU_GPU=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
ctest --test-dir build --output-on-failure

# Optional: generate matrices if .csrbin missing
cmake --build build --target generate_matrices
./build/generate_matrices

# Utilization (separate terminal)
./hybrid_cpu_gpu/scripts/capture_gpu_utilization.sh 120 /tmp/gpu_util.log

# Benchmark
./build/hybrid_four_way_benchmark benchmarks/matrices 8
```

Results: `hybrid_cpu_gpu/benchmarks/results/four_way_comparison.csv`

## CPU-only (no GPU)

```bash
cmake -S . -B build -DENABLE_HYBRID_CPU_GPU=ON -DENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build -R hybrid_conversion -V
```

Uses built-in tiny matrices; **no** `.csrbin` required.
