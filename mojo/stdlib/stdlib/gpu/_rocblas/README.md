# ROCm rocBLAS Integration

This directory contains Mojo bindings and integration code for
[rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS), AMD's
high-performance BLAS (Basic Linear Algebra Subprograms) library for GPUs.

## Purpose

**rocBLAS is included in this project _solely_ for the following purposes:**

- **Accuracy Validation:** To cross-check the correctness of our custom GPU
  kernels and ensure numerical results match established vendor libraries.
- **Competitive Benchmarking:** To provide a fair performance comparison
  between our implementations and the official AMD BLAS routines.

> **Note:** rocBLAS is _not_ used in production code paths or as a primary
> backend for any core Mojo GPU operations.

## Usage

- The rocBLAS integration is only invoked in test suites and benchmarking
  scripts.
- If you are not running accuracy or performance validation, you do **not**
  need to install or configure rocBLAS.

## Requirements

- AMD GPU with ROCm support.
- Proper installation of the ROCm stack and rocBLAS library.

## References

- [rocBLAS Documentation](https://rocblas.readthedocs.io/en/latest/)
- [ROCm Platform](https://rocmdocs.amd.com/en/latest/)

---
