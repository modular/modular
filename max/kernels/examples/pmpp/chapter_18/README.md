# Chapter 18: Graph traversal

This chapter covers parallel breadth-first search (BFS) on graphs stored in sparse formats. Graph algorithms are irregular: the work per vertex varies, memory access is non-contiguous, and level dependencies complicate parallel execution. The examples cover three fundamental BFS strategies, then show how privatization and frontier queues improve performance.

## Files

| File | Description |
|------|-------------|
| `graph_utils.mojo` | Shared utilities; `CSRGraph`, `CSCGraph`, and `COOGraph` data structures, plus graph generation and verification functions |
| `fig18_06.mojo` | Vertex-centric BFS, push-based (CSR); each thread checks one vertex and pushes updates to unvisited neighbors if it was visited last level |
| `fig18_08.mojo` | Vertex-centric BFS, pull-based (CSC); each unvisited vertex checks whether any incoming neighbor was visited last level |
| `fig18_10.mojo` | Edge-centric BFS (COO); each thread handles one edge, checking if the source was visited last level and marking the destination if so |
| `fig18_12.mojo` | Frontier-based BFS (CSR); maintains an explicit queue of active vertices, only frontier vertices are processed each level |
| `fig18_15.mojo` | Frontier BFS with privatization; each block maintains a private shared memory frontier, merged to global memory at the end |
| `fig18_17.mojo` | Frontier BFS with privatization, multi-launch; uses separate kernel launches per level instead of grid sync |

## Mojo vs. CUDA

The original `fig18_17` uses CUDA's cooperative groups for grid-wide synchronization. The Mojo version uses multi-launch instead, which is functionally equivalent for level-by-level BFS traversal.

`graph_utils.mojo` must be in the same directory as the figure files when compiling.
