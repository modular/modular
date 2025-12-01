# Mojo GPU block and warp operations examples

This directory contains code examples for the
[GPU block and warp operations and
synchronization](https://docs.modular.com/mojo/manual/gpu/block-and-warp)
section of the [Mojo Manual](https://docs.modular.com/mojo/manual).

**Note:** These examples require a [supported
GPU](https://docs.modular.com/max/faq/#gpu-requirements) to compile and run the
kernels. If your system doesn't have a supported GPU, you can compile the
programs but the only output you'll see when you run them is the message:

```output
No GPU detected - this example requires a supported GPU
```

## Files

This directory contains the following examples:

- `tiled_matmul.mojo`: A tiled matrix multiplication example to demonstrate the
  proper use of
  [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/sync/barrier/) for
  thread block synchronization in GPU kernels.

- `pixi.toml`: a [Pixi](https://pixi.sh) project file containing the project
  dependencies and task definitions.

- `BUILD.bazel`: a Bazel BUILD file for building and running the examples with
  the [Bazel](https://bazel.build/) build system.

## Run the code

This example project uses the [Pixi](https://pixi.sh/latest/) package and
virtual environment manager. Once you have installed `pixi`, you can run the
examples like this:

```bash
pixi run mojo tiled_matmul.mojo
```
