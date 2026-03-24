# Chapter 7: Convolution

2D convolution makes an ideal case study for memory optimization: the filter is
small and read-only (constant memory), while the input benefits from tiled
shared memory to reduce redundant global loads.

## Files

| File | Description |
|------|-------------|
| `fig7_7.mojo` | Basic 2D convolution with no optimizations; each thread loads all its required input elements from global memory |
| `fig7_9.mojo` | 2D convolution corresponding to the CUDA constant-memory version; Mojo has no `__constant__` equivalent, so the filter is stored in global memory instead |
| `fig7_12.mojo` | Tiled 2D convolution; threads cooperatively load an input tile into shared memory |
| `fig7_15.mojo` | Cached tiled convolution; interior elements come from shared memory, halo elements fall back to global memory |
