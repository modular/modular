# Chapter 5: Memory architecture and data locality

Global memory is slow. Shared memory and tiling fix that. Matrix multiplication
serves as the running example, progressively optimized from a naive
implementation to one that handles arbitrary dimensions.

## Files

| File                | Description                                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `fig5_1.mojo`       | Matrix multiplication inner loop, a code snippet showing the basic computation before any GPU-specific optimization              |
| `fig5_10.mojo`      | Tiled matrix multiplication using shared memory; threads cooperatively load a tile into shared memory before computing           |
| `fig5_14.mojo`      | Tiled matrix multiplication with boundary checking; handles matrices whose dimensions are not evenly divisible by the tile width |
| `dynamic_smem.mojo` | Dynamic shared memory variant; tile width is a runtime parameter rather than a compile-time constant                             |
