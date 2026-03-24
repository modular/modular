# Chapter 9: Parallel histogram

When thousands of threads try to increment the same histogram bin, write
contention becomes the bottleneck. Privatization and aggregation solve it. The
examples start with sequential CPU code and progressively improve the GPU
implementation.

## Files

| File           | Description                                                                                                                         |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `fig9_2.mojo`  | Sequential histogram, CPU baseline using a single-threaded loop                                                                     |
| `fig9_4.mojo`  | Sequential histogram variant, functionally identical to `fig9_2` (matches a different figure in the book)                           |
| `fig9_6.mojo`  | Basic GPU histogram; each thread atomically increments the corresponding bin in global memory                                       |
| `fig9_9.mojo`  | Global memory privatization; each block maintains a private copy of the bins in global memory, merged to the output bins at the end |
| `fig9_10.mojo` | Shared memory privatization; each block maintains a private copy of the bins in shared memory, merged to global memory at the end   |
| `fig9_12.mojo` | Privatized histogram with thread coarsening (coarse factor 4); each thread processes multiple input elements                        |
| `fig9_14.mojo` | Coarsened histogram with contiguous partitioning; each thread handles a contiguous range of input elements                          |
| `fig9_15.mojo` | Coarsened histogram with interleaved partitioning; threads process strided elements for better memory coalescing                    |
