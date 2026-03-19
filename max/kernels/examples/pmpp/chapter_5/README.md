# Chapter 5: Memory architecture and data locality

This chapter introduces shared memory and tiling as tools for reducing global memory traffic. Matrix multiplication is the running example, progressively optimized from a naive implementation to one that handles arbitrary dimensions.

## Files

| File | Description |
|------|-------------|
| `fig5_1.mojo` | Matrix multiplication inner loop, a code snippet showing the basic computation before any GPU-specific optimization |
| `fig5_10.mojo` | Tiled matrix multiplication using shared memory; threads cooperatively load a tile into shared memory before computing |
| `fig5_14.mojo` | Tiled matrix multiplication with boundary checking; handles matrices whose dimensions are not evenly divisible by the tile width |
| `dynamic_smem.mojo` | Dynamic shared memory variant; tile width is a runtime parameter rather than a compile-time constant |

## Mojo vs. CUDA

Shared memory allocation uses `external_memory()` with an `AddressSpace` specifier. `FuncAttribute` controls shared memory size at launch, replacing `<<<..., smem_bytes>>>` in CUDA. `barrier()` maps directly to `__syncthreads()`.

Read in order: `fig5_1.mojo` -> `fig5_10.mojo` -> `fig5_14.mojo`.
