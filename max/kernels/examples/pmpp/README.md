# PMPP Mojo examples

Mojo implementations of the code examples from [Programming Massively Parallel Processors, 5th Edition](https://a.co/d/0bZCoTKY) by David Kirk and Wen-mei W. Hwu.

If you're reading PMPP and want to follow along in Mojo, this is the companion. Each chapter directory contains Mojo files named to match the corresponding figures in the book — `fig5_10.mojo` maps to Figure 5.10 in the text. The original CUDA implementations are in the book itself.

The Mojo versions currently target NVIDIA GPUs. Apple Silicon GPU support is not yet available for these examples (see `BUILD.bazel` for compatibility details).

## Structure

```text
chapter_5/
├── BUILD.bazel
├── fig5_1.mojo
├── fig5_10.mojo
├── fig5_14.mojo
└── dynamic_smem.mojo
```

## Chapters covered

| Chapter | Topics |
|---------|--------|
| 2 | Heterogeneous data parallel computing, vector addition |
| 3 | Multidimensional grids, image processing |
| 4 | Compute architecture and scheduling |
| 5 | Memory architecture, shared memory, tiled matrix multiplication |
| 7 | Convolution |
| 8 | Stencil |
| 9 | Parallel histogram |
| 10 | Reduction and minimizing thread divergence |
| 11 | Prefix sum (scan) |
| 12 | Stream compaction and parallel partition |
| 13 | Merge sort |
| 14 | Sorting (odd-even, radix) |
| 15 | Performance optimizations: register blocking, software pipelining |
| 16 | Dynamic programming |
| 17 | Sparse matrix-vector multiplication |
| 18 | Graph traversal (BFS) |
| 19 | Convolutional layers |
| 20 | Softmax and attention |
| 21 | Electrostatic potential map (Direct Coulomb Summation) |

## Running the examples

Examples are built and tested using Bazel. Each chapter directory has a `BUILD.bazel` file that defines test targets for the Mojo files.

```bash
# Run a single example via Bazel
bazel test //max/kernels/examples/pmpp/chapter_5:fig5_10.test --test_tag_filters=gpu

# Run all examples in a chapter
bazel test //max/kernels/examples/pmpp/chapter_5/... --test_tag_filters=gpu
```

Not every file is a standalone runnable program. Some are code snippets or utility modules meant to be read alongside the book (for example, `fig2_10.mojo` shows only the kernel function; `vec_add.mojo` is the complete runnable version for that chapter). Each chapter README identifies the runnable files and the recommended reading order.

## Notes on the Mojo implementations

These are direct ports of the CUDA originals, same algorithms and structure. They're written to be readable alongside the book, not to demonstrate idiomatic Mojo GPU programming style.

A few things look different from the CUDA versions:

- Memory management uses Mojo's ownership model instead of `cudaMalloc()`/`cudaFree()`
- Kernel launches use the `DeviceContext` API
- Some examples use `LayoutTensor` where it maps cleanly to the memory layout patterns in the chapter
- A small number of examples note where the Mojo API differs from CUDA (for example, constant memory, grid sync)

For idiomatic GPU programming in Mojo and MAX, see:

- [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles): problem sets for learning GPU programming in Mojo; draws from PMPP concepts without reproducing the examples directly
- [MAX documentation](https://docs.modular.com/max/): higher-level GPU programming with MAX
