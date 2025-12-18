# AsyncRT Benchmarks

This directory contains benchmarks for AsyncRT components.

## Benchmarks

### QueueBenchmark

Tests the performance of AsyncRT queue operations.

**Build and run:**

```bash
br //MLRT/benchmarks/AsyncRT:QueueBenchmark
```

### bench_gpu_kernel_enqueue_cuda

Measures CUDA kernel enqueue latency using the driver API. This benchmark uses
the `Support/MicroBenchmark.h` framework to measure the time taken to enqueue
(not execute) CUDA kernels using `cuLaunchKernelEx`.

**Build and run:**

```bash
br //MLRT/benchmarks/AsyncRT:bench_gpu_kernel_enqueue_cuda
```

**Requirements:** NVIDIA GPU

### bench_gpu_kernel_enqueue_devicecontext

Measures GPU kernel enqueue latency using the AsyncRT DeviceContext API. This
benchmark uses the `Support/MicroBenchmark.h` framework to measure the time
taken to enqueue (not execute) GPU kernels through the DeviceContext interface.
This provides a comparison point for the overhead of the AsyncRT abstraction
layer compared to the raw CUDA driver API.

**Build and run:**

```bash
br //MLRT/benchmarks/AsyncRT:bench_gpu_kernel_enqueue_devicecontext
```

**Requirements:** NVIDIA GPU
