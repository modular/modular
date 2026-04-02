# Chapter 16: Dynamic programming

Dynamic programming algorithms pose a particular challenge for GPUs: each result
depends on previously computed subproblems. The examples cover Fibonacci (simple
1D recurrence), Floyd-Warshall (all-pairs shortest path), and Smith-Waterman
(local sequence alignment), each with a different dependency structure that maps
to GPU execution differently.

## Files

| File                                     | Description                                                                                                                                           |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `fig16_1_fibonacci.mojo`                 | Fibonacci; top-down (memoization) and bottom-up (tabulation) CPU implementations for contrast                                                         |
| `fig16_3_floyd_warshall.mojo`            | Floyd-Warshall CPU baseline; all-pairs shortest path with the classic triple-nested loop                                                              |
| `fig16_4_floyd_warshall_gpu.mojo`        | Floyd-Warshall GPU kernel; each thread computes one cell per outer iteration, inner two loops parallelized                                            |
| `fig16_8_9_10_11_12_smith_waterman.mojo` | Smith-Waterman local sequence alignment; GPU kernel using diagonal wavefront parallelism with shared memory tiling; covers Figures 16.8 through 16.12 |

## Notes

Smith-Waterman's anti-diagonal dependency structure (each cell depends on its
left, top, and top-left neighbors) requires a wavefront traversal to expose
parallelism. The tiled GPU version processes one diagonal tile per kernel
launch.
