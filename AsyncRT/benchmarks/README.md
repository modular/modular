# AsyncRT Benchmarks

This directory contains benchmarks for AsyncRT components.

## Benchmarks

### QueueBenchmark

Tests the performance of AsyncRT queue operations.

**Build and run:**

```bash
br //AsyncRT/benchmarks:QueueBenchmark
```

### bench_gpu_kernel_enqueue_cuda

Measures CUDA kernel enqueue latency using the driver API. This benchmark uses
the `Support/MicroBenchmark.h` framework to measure the time taken to enqueue
(not execute) CUDA kernels using `cuLaunchKernelEx`.

**Build and run:**

```bash
br //AsyncRT/benchmarks:bench_gpu_kernel_enqueue_cuda
```

**Requirements:** NVIDIA GPU
