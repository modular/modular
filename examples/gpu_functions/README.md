# Compiling and running Mojo functions on a GPU

Mojo functions can be compiled and dispatched on a GPU, and these examples
show a few different ways of doing so. These examples are complementary with
[those that show how to program custom graph operations](../custom_ops/) to
run on the CPU or GPU in Mojo.

A [MAX-compatible GPU](https://docs.modular.com/max/faq/#gpu-requirements) is
necessary to run these examples.

The five examples of GPU functions defined in Mojo consist of:

- **vector_addition.mojo**: A common "hello world" example for GPU programming,
  this adds two vectors together in the same way as seen in Chapter 2 of
  ["Programming Massively Parallel Processors"](https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors).
- **grayscale.mojo**: The parallelized conversion of an RGB image to grayscale,
  as seen in Chapter 3 of "Programming Massively Parallel Processors".
- **naive_matrix_multiplication.mojo**: An implementation of naive matrix
  multiplication, again inspired by Chapter 3 of "Programming Massively
  Parallel Processors".
- **mandelbrot.mojo**: A parallel calculation of the number of iterations to
  escape in the Mandelbrot set. An example of the same computation performed as
  a custom graph operation can be found [here](../custom_ops/).
- **reduction.mojo**: A highly performant reduction kernel.
For a detailed explanation see [this blogpost](https://veitner.bearblog.dev/very-fast-vector-sum-without-cuda/).

These examples can be run either through a single [Pixi](https://pixi.sh)
command:

```sh
pixi run vector_addition
pixi run grayscale
pixi run naive_matrix_multiplication
pixi run mandelbrot
pixi run reduction
```

or directly with Mojo if it has been installed in the local environment:

```sh
mojo vector_addition.mojo
mojo grayscale.mojo
mojo naive_matrix_multiplication.mojo
mojo mandelbrot.mojo
mojo reduction.mojo
```

For larger computations, we recommend staging them as part of a
[MAX Graph](https://docs.modular.com/max/tutorials/get-started-with-max-graph-in-python).
The graph compiler within MAX performs intelligent operator fusion,
orchestrates efficient runtime execution, and more. The same Mojo code running
on a GPU in one of these examples can be easily translated to
[a custom operation](https://docs.modular.com/max/tutorials/build-custom-ops)
and placed inside a node in a MAX Graph.
