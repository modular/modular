# Chapter 7: Convolution

This chapter uses 2D convolution as a case study for memory optimization, specifically reducing global memory accesses by putting the filter in constant memory and using tiled shared memory for the input.

## Files

| File | Description |
|------|-------------|
| `fig7_7.mojo` | Basic 2D convolution with no optimizations; each thread loads all its required input elements from global memory |
| `fig7_9.mojo` | 2D convolution corresponding to the CUDA constant-memory version; see Mojo vs. CUDA note below |
| `fig7_12.mojo` | Tiled 2D convolution; threads cooperatively load an input tile into shared memory, filter in constant memory in CUDA |
| `fig7_15.mojo` | Cached tiled convolution; interior elements come from shared memory, halo elements fall back to global memory |

## Mojo vs. CUDA

CUDA's `__constant__` memory does not have a direct equivalent in the current Mojo API. `fig7_9.mojo` and `fig7_12.mojo` use global memory for the filter instead. The algorithmic structure is otherwise the same.

Read in order: `fig7_7.mojo` -> `fig7_9.mojo` -> `fig7_12.mojo` -> `fig7_15.mojo`.
