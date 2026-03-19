# Chapter 15: Performance optimizations

This chapter covers register blocking and software pipelining (double buffering) for matrix multiplication. Register blocking keeps partial results in registers across the k-dimension loop instead of writing them back to shared memory each iteration. Double buffering overlaps the load of the next tile with computation on the current tile.

## Files

| File | Description |
|------|-------------|
| `fig15_3.mojo` | Tiled matrix multiplication with all prior optimizations combined, the starting baseline for this chapter |
| `fig15_4.mojo` | Accumulator initialization; `clear()` function for zeroing the register tile |
| `fig15_5.mojo` | Tile load function; loads a tile from global memory into shared memory with coalesced access |
| `fig15_6.mojo` | Compute function; multiplies a shared memory tile against a register tile to accumulate results |
| `fig15_7.mojo` | Write function; stores the register tile back to global memory |
| `fig15_7_coalesced_exc.mojo` | Write with SIMD vector stores; uses vector-width writes for better memory coalescing |
| `fig15_9.mojo` | Register blocking; each thread computes a tM x tN submatrix of the output, keeping results in registers across the k-dimension loop |
| `fig15_14.mojo` | Double buffering (software pipelining); prefetches the next tile into a second shared memory buffer while computing with the current tile |
| `fig15_14_LayoutTensor.mojo` | Same double buffering kernel using `LayoutTensor`; shows how Mojo's tensor abstraction maps to the tiled memory layout |

## Mojo vs. CUDA

`fig15_14_LayoutTensor.mojo` has no direct CUDA equivalent. It uses `LayoutTensor` from Mojo's layout library to express the same memory access pattern more abstractly. Compare it with `fig15_14.mojo` to see both approaches side by side.

Read in order: `fig15_3.mojo` -> `fig15_9.mojo` -> `fig15_14.mojo` -> `fig15_14_LayoutTensor.mojo`.
