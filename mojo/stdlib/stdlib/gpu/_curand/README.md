# cuRAND Integration

This directory contains Mojo bindings for cuRAND, NVIDIA's GPU RNG library.

## Purpose

**cuRAND is included _only_ for:**

- **Accuracy validation:** Cross-checking our custom GPU RNGs.
- **Competitive benchmarking:** Comparing performance with vendor routines.

> cuRAND is **not** used in production code paths or as a primary backend.

## Usage

- cuRAND is invoked in test suites and benchmarks.
- Not required for normal GPU operation.

## Requirements

- NVIDIA GPU with CUDA support.
- Proper installation of CUDA and cuRAND.

## References

- [cuRAND Documentation](https://docs.nvidia.com/cuda/curand/index.html)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
