# Chapter 9: Parallel histogram

This chapter builds a parallel histogram, introducing privatization and aggregation as techniques for handling write contention across threads. The examples start with sequential CPU code and progressively improve the GPU implementation.

## Files

| File | Description |
|------|-------------|
| `fig9_2.mojo` | Sequential histogram, CPU baseline using a single-threaded loop |
| `fig9_4.mojo` | Sequential histogram variant, same computation with a different loop structure for comparison |
| `fig9_6.mojo` | Basic GPU histogram; each thread atomically increments the corresponding bin in global memory |
| `fig9_9.mojo` | Privatized histogram; each block maintains a private copy of the bins in global memory, merged to the output bins at the end |
| `fig9_10.mojo` | Privatized histogram with shared memory; each block maintains a private copy of the bins in shared memory, merged to global memory at the end |
| `fig9_12.mojo` | Privatized histogram with thread coarsening (coarse factor 4); each thread processes multiple input elements |
| `fig9_14.mojo` | Coarsened histogram with contiguous partitioning; each thread handles a contiguous range of input elements |
| `fig9_15.mojo` | Coarsened histogram with interleaved partitioning; threads process strided elements for better memory coalescing |

## Mojo vs. CUDA

Atomic operations use `Atomic` from `std.os`. Shared memory via `external_memory()` follows the same pattern as Chapters 5 and 8.

Read in order: `fig9_2.mojo` -> `fig9_6.mojo` -> `fig9_9.mojo` -> `fig9_12.mojo` -> `fig9_14.mojo` or `fig9_15.mojo`.
