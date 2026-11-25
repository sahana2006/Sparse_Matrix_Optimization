
# Hybrid CPU (OpenMP) + GPU (CUDA) module

Self-contained implementation: **OpenMP parallel Adaptive RGC conversion on CPU**, then **H2D → CUDA SpMV → D2H**.

## Layout

```
hybrid_cpu_gpu/
  include/hybrid_cpu_gpu/   Public API
  src/                    conversion_openmp.cpp, pipeline.cu
  tests/                  CPU tests (always); GPU tests when CUDA available
  benchmarks/             Four-way comparison CSV (Sequential / OpenMP / CUDA / Hybrid)
  scripts/                GPU utilization capture helper
  docs/HYBRID_EVALUATION.md
```

## Build

```bash
cmake -S . -B build -DENABLE_HYBRID_CPU_GPU=ON -DENABLE_CUDA=OFF   # CPU conversion tests only
cmake -S . -B build -DENABLE_HYBRID_CPU_GPU=ON -DENABLE_CUDA=ON    # full hybrid on GPU machine
cmake --build build -j
ctest --test-dir build --output-on-failure
```

See `docs/HYBRID_EVALUATION.md` for metrics and laptop run steps.
