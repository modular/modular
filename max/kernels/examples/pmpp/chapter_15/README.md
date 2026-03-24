# Chapter 15: Performance optimizations

Starting from the tiled matrix multiply of Chapter 5, these examples add two
optimizations. Register blocking keeps partial results in registers across the
k-dimension loop instead of writing them back to shared memory each iteration.
Double buffering overlaps the load of the next tile with computation on the
current tile.

## Files

| File                         | Description                                                                                                                               |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `fig15_3.mojo`               | Tiled matrix multiplication with all prior optimizations combined, the starting baseline for this chapter                                 |
| `fig15_4.mojo`               | Accumulator initialization; `clear()` function for zeroing the register tile                                                              |
| `fig15_5.mojo`               | Tile load function; loads a tile from global memory into shared memory with coalesced access                                              |
| `fig15_6.mojo`               | Compute function; multiplies a shared memory tile against a register tile to accumulate results                                           |
| `fig15_7.mojo`               | Write function; stores the register tile back to global memory                                                                            |
| `fig15_7_coalesced_exc.mojo` | Write with SIMD vector stores; uses vector-width writes for better memory coalescing                                                      |
| `fig15_9.mojo`               | Register blocking; each thread computes a tM x tN submatrix of the output, keeping results in registers across the k-dimension loop       |
| `fig15_14.mojo`              | Double buffering (software pipelining); prefetches the next tile into a second shared memory buffer while computing with the current tile |
| `fig15_14_LayoutTensor.mojo` | Same double buffering kernel using `LayoutTensor`; shows how Mojo's tensor abstraction maps to the tiled memory layout                    |
