# cuDNN Integration

This directory contains Mojo bindings for cuDNN, NVIDIA's GPU deep learning
library.

## Purpose

**cuDNN is included _only_ for:**

- **Accuracy validation:** Cross-checking our custom GPU kernels.
- **Competitive benchmarking:** Comparing performance with vendor routines.

> cuDNN is **not** used in production code paths or as a primary backend.

## Usage

- cuDNN is invoked in test suites and benchmarks.
- Not required for normal GPU operation.

## Requirements

- NVIDIA GPU with CUDA support.
- Proper installation of CUDA and cuDNN.

## References

- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
