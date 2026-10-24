# GPU kernel enqueue benchmarks

Benchmarks for GPU kernel *enqueue* latency (not execution), comparing the raw
CUDA driver API against the DeviceContext abstraction.

## Benchmarks

### bench_gpu_kernel_enqueue_cuda

Measures CUDA kernel enqueue latency using the driver API. This benchmark uses
the `Support/MicroBenchmark.h` framework to measure the time taken to enqueue
(not execute) CUDA kernels using `cuLaunchKernelEx`.

**Build and run:**

```bash
br //MLRT/benchmarks/Driver:bench_gpu_kernel_enqueue_cuda
```

**Requirements:** NVIDIA GPU

### bench_gpu_kernel_enqueue_devicecontext

Measures GPU kernel enqueue latency using the DeviceContext API. This benchmark
uses the `Support/MicroBenchmark.h` framework to measure the time taken to
enqueue (not execute) GPU kernels through the DeviceContext interface. This
provides a comparison point for the overhead of the DeviceContext abstraction
layer compared to the raw CUDA driver API.

**Build and run:**

```bash
br //MLRT/benchmarks/Driver:bench_gpu_kernel_enqueue_devicecontext
```

**Requirements:** NVIDIA GPU
