# Chapter 10: Reduction and minimizing divergence

Parallel reduction computes a single value (like a sum) from an array, but naive
implementations suffer from severe thread divergence. The examples show how to
restructure the work to keep threads active and minimize idle lanes, then
introduce warp shuffles as a lower-overhead alternative to shared memory.

## Files

| File            | Description                                                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------|
| `fig10_5.mojo`  | Simple sum reduction; naive tree reduction with high thread divergence                                                         |
| `fig10_8.mojo`  | Convergent sum reduction; restructured to access contiguous memory and keep active threads together                            |
| `fig10_10.mojo` | Shared memory sum reduction; uses shared memory for the reduction tree                                                         |
| `fig10_14.mojo` | Warp-level reduction with shuffle; uses `shuffle_down()` to reduce within a warp without shared memory                         |
| `fig10_16.mojo` | Warp reduction combined with shared memory; warp results stored in shared memory for final block-level reduction               |
| `fig10_18.mojo` | Multi-block reduction; each block produces a partial result, accumulated globally via atomic add                               |
| `fig10_20.mojo` | Reduction with thread coarsening (coarse factor 4); each thread processes multiple elements before entering the reduction tree |
