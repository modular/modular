# Programming Massively Parallel Processors Mojo examples

Mojo implementations of the code examples from
[Programming Massively Parallel Processors, 5th Edition](https://a.co/d/0bZCoTKY)
by David Kirk, Wen-mei W. Hwu, and Izzat El Hajj.

If you're reading PMPP and want to follow along in Mojo, this is the companion.
Each chapter directory contains files named to match the corresponding figures
in the book. `fig5_10.mojo` maps to Figure 5.10 in the text.

## GPU compatibility

These examples currently target NVIDIA GPUs. Apple Silicon GPU support is not
yet available for these examples.

## Structure

```text
chapter_05/
├── BUILD.bazel
├── fig5_1.mojo
├── fig5_10.mojo
├── fig5_14.mojo
└── dynamic_smem.mojo
```

Mojo files live directly in each chapter directory. The original CUDA
implementations are in the book itself.

## Chapters covered

| Chapter | Topics                                                                                                       |
|---------|--------------------------------------------------------------------------------------------------------------|
| 2       | Heterogeneous data parallel computing, vector addition                                                       |
| 3       | Multidimensional grids, image processing                                                                     |
| 4       | Compute architecture and scheduling                                                                          |
| 5       | Memory architecture, shared memory, tiled matrix multiplication                                              |
| 6       | *(absent; Chapter 6 covers performance considerations and is mostly prose with no code figures in the book)* |
| 7       | Convolution                                                                                                  |
| 8       | Stencil                                                                                                      |
| 9       | Parallel histogram                                                                                           |
| 10      | Reduction and minimizing thread divergence                                                                   |
| 11      | Prefix sum (scan)                                                                                            |
| 12      | Stream compaction and parallel partition                                                                     |
| 13      | Merge sort                                                                                                   |
| 14      | Sorting (odd-even, radix)                                                                                    |
| 15      | Performance optimizations: register blocking, software pipelining                                            |
| 16      | Dynamic programming                                                                                          |
| 17      | Sparse matrix-vector multiplication                                                                          |
| 18      | Graph traversal (BFS)                                                                                        |
| 19      | Convolutional layers                                                                                         |
| 20      | Softmax and attention                                                                                        |
| 21      | Electrostatic potential map (Direct Coulomb Summation)                                                       |

## Running the examples

Build and run examples using Bazel:

```bash
# Run a single example
bazel test //max/kernels/examples/pmpp/chapter_05:fig5_10.test

# Run all examples in a chapter
bazel test //max/kernels/examples/pmpp/chapter_05/...

# Run all PMPP examples
bazel test //max/kernels/examples/pmpp/...
```

Not every file is a standalone runnable program. Some are code snippets or
utility modules meant to be read alongside the book (for example, `fig2_10.mojo`
shows only the kernel function; `vec_add.mojo` is the complete runnable version
for that chapter).
