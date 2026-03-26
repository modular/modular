# Chapter 13: Merge sort

Combining two sorted arrays into a single sorted output (parallel merge) is
harder than reduction or scan because each output element's position depends on
elements from both input arrays. The examples progress from a simple per-thread
merge to tiled approaches that use shared memory.

## Files

| File                     | Description                                                                                                                                         |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `fig13_9.mojo`           | Basic merge kernel; each thread independently merges a segment of the two input arrays using binary search to find the split point                  |
| `fig13_11_12_13.mojo`    | Tiled merge kernel; threads in a block cooperate to load input tiles into shared memory before merging; covers Figures 13.11, 13.12, and 13.13      |
| `fig13_16_18_19_20.mojo` | Circular buffer merge; uses a circular buffer in shared memory to avoid re-loading data across tiles; covers Figures 13.16, 13.18, 13.19, and 13.20 |
