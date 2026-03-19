# Chapter 12: Stream compaction and parallel partition

This chapter covers stream compaction — selecting elements from an array that satisfy a predicate and packing them into a contiguous output. The examples use "keep even numbers" as the predicate. Stream compaction relies on scan (from Chapter 11) and is used in many GPU algorithms that deal with irregular data.

## Files

| File | Description |
|------|-------------|
| `fig12_2.mojo` | Basic compaction; sequential scan to compute output positions, then scatter |
| `fig12_3.mojo` | Warp-vote compaction; uses `vote()` to count matching elements per warp before computing positions |
| `fig12_4.mojo` | Compaction with atomic positions; threads compute their output index using atomic increments |
| `fig12_6.mojo` | Compaction with shared memory scan; uses a shared memory prefix sum to compute per-block output positions |
| `fig12_8.mojo` | Compaction with warp-level scan; uses `shuffle_up()` inside shared memory reduction for better efficiency |

## Mojo vs. CUDA

Warp vote operations use `vote()` from `std.gpu.primitives.warp`. Lane index is from `lane_id()` in `std.gpu.primitives.id`. Atomic position tracking uses `Atomic` from `std.os`.

Read in order: `fig12_2.mojo` -> `fig12_3.mojo` -> `fig12_6.mojo` -> `fig12_8.mojo`.
