# Chapter 11: Prefix sum (scan)

This chapter covers parallel prefix sum — computing the running total of an array in parallel. Scan underlies many other algorithms (stream compaction, sorting, histogram). The chapter works through several scan algorithms and shows how to scale from single-block to multi-block inputs.

## Files

| File | Description |
|------|-------------|
| `fig11_1.mojo` | Sequential inclusive scan, CPU baseline |
| `fig11_3.mojo` | Kogge-Stone scan using shared memory; parallel scan within a single block |
| `fig11_5.mojo` | Kogge-Stone scan with double-buffering; avoids the read-after-write hazard in `fig11_3` by alternating between two shared memory buffers |
| `fig11_8.mojo` | Warp-level scan using `shuffle_up()`; scan within a single warp without shared memory |
| `fig11_9.mojo` | Block-level scan built from warp scans; warp results stored in shared memory, then combined |
| `fig11_10.mojo` | Block-level scan variant; adjusted warp-to-block aggregation |
| `fig11_12.mojo` | Block scan with thread coarsening (coarse factor 4); each thread handles multiple elements before entering the scan |
| `fig11_13.mojo` | Coarsened block scan variant; uses a different accumulation strategy |
| `fig11_17.mojo` | Hierarchical multi-block scan; scans blocks independently, then scans the block sums, then adds the block sum back to each block |
| `fig11_18.mojo` | Hierarchical scan with thread coarsening; combines multi-block scan with the coarsening from `fig11_12` |

## Mojo vs. CUDA

Warp lane operations use `lane_id()` and `warp_id()` from `std.gpu.primitives.id`, and `shuffle_up()` from `std.gpu.primitives.warp`. Multi-block synchronization uses `Atomic` from `std.os`. Mojo does not yet have a direct equivalent to CUDA's cooperative groups for grid-wide sync, so `fig11_17` uses an atomic-based approach instead.

**Note:** `fig11_15` (three-kernel reduce-scan-scan) has no Mojo port yet. The original CUDA implementation can be found in the PMPP book.

Read in order: `fig11_1.mojo` -> `fig11_3.mojo` -> `fig11_8.mojo` -> `fig11_9.mojo` -> `fig11_12.mojo` -> `fig11_17.mojo`.
