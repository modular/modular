# Chapter 11: Prefix sum (scan)

Computing the running total of an array in parallel (prefix sum, or "scan") is
a building block for stream compaction, sorting, and histogram algorithms. The
examples work through several scan strategies and show how to scale from
single-block to multi-block inputs.

## Files

| File            | Description                                                                                                                              |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `fig11_1.mojo`  | Sequential inclusive scan, CPU baseline                                                                                                  |
| `fig11_3.mojo`  | Kogge-Stone scan using shared memory; parallel scan within a single block                                                                |
| `fig11_5.mojo`  | Kogge-Stone scan with double-buffering; avoids the read-after-write hazard in `fig11_3` by alternating between two shared memory buffers |
| `fig11_8.mojo`  | Warp-level scan using `shuffle_up()`; scan within a single warp without shared memory                                                    |
| `fig11_9.mojo`  | Block-level scan built from warp scans; warp results stored in shared memory, then combined                                              |
| `fig11_10.mojo` | Block-level scan variant; adjusted warp-to-block aggregation                                                                             |
| `fig11_12.mojo` | Block scan with thread coarsening (coarse factor 4); each thread handles multiple elements before entering the scan                      |
| `fig11_13.mojo` | Coarsened block scan variant; uses a different accumulation strategy                                                                     |
| `fig11_17.mojo` | Hierarchical multi-block scan; scans blocks independently, then scans the block sums, then adds the block sum back to each block         |
| `fig11_18.mojo` | Hierarchical scan with thread coarsening; combines multi-block scan with the coarsening from `fig11_12`                                  |
| `fig11_15`      | *(no Mojo port)* Three-kernel reduce-scan-scan; refer to the book for the CUDA version                                                   |
