# cuBLAS Integration

This directory contains Mojo bindings for cuBLAS, NVIDIA's GPU BLAS library.

## Purpose

**cuBLAS is included _only_ for:**

- **Accuracy validation:** Cross-checking our custom GPU BLAS kernels.
- **Competitive benchmarking:** Comparing performance with vendor routines.

> cuBLAS is **not** used in production code paths or as a primary backend.

## Usage

- cuBLAS is invoked in test suites and benchmarks.
- Not required for normal GPU operation.

## Requirements

- NVIDIA GPU with CUDA support.
- Proper installation of CUDA and cuBLAS.

## References

- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
