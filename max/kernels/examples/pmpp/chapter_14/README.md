# Chapter 14: Sorting

This chapter covers two parallel sorting algorithms. Odd-even sort is simple to parallelize but not work-efficient. Radix sort is practical at scale and forms the basis of GPU sorting libraries.

## Files

| File | Description |
|------|-------------|
| `fig14_2.mojo` | Parallel odd-even transposition sort; alternating odd and even passes, each thread compares and swaps adjacent elements |
| `fig14_7.mojo` | Radix sort iteration; one pass of least-significant-digit radix sort, using scan to compute output positions for each bit bucket |

## Mojo vs. CUDA

Both examples use standard `block_idx` and `thread_idx` for indexing. `random_ui64()` from `std.random` generates test data.
