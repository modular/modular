# Chapter 12: Stream compaction and parallel partition

Given an array, select only the elements that match a predicate and pack them
into contiguous output. That is stream compaction. The examples use "keep even
numbers" as the predicate and build on scan (Chapter 11). Stream compaction
appears in many GPU algorithms that deal with irregular data.

## Files

| File           | Description                                                                                               |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `fig12_2.mojo` | Basic compaction; uses atomic fetch-and-add to compute output positions, then scatter                     |
| `fig12_3.mojo` | Warp-vote compaction; uses `vote()` to count matching elements per warp before computing positions        |
| `fig12_4.mojo` | Compaction with atomic positions; threads compute their output index using atomic increments              |
| `fig12_6.mojo` | Compaction with shared memory scan; uses a shared memory prefix sum to compute per-block output positions |
| `fig12_8.mojo` | Compaction with warp-level scan; uses `shuffle_up()` inside shared memory reduction for better efficiency |
