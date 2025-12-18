# Mojo code examples

This directory contains a collection of sample programs written in the
[Mojo](https://docs.modular.com/mojo/manual/) programming language.

## Getting started

You can use `git` to clone the repository containing all of these sample
programs:

```bash
git clone https://github.com/modular/modular.git
```

For the most reliable experience building and running these examples, we
recommend using [Pixi](https://pixi.sh/latest/). It's both a package manager and
virtual environment manager—which alone makes development a lot easier—but it's
also fast, language agnostic, and provides lock files for package dependencies.
For more help with it, see our
[Pixi basics guide](https://docs.modular.com/pixi).

Each subdirectory of this directory is a self-contained project that
demonstrates features of the Mojo programming language and its standard library.
Each contains a `README.md` file and a `pixi.toml` file that specifies the
required dependencies. Simply follow the instructions in each `README.md` file
to use `pixi` to download and install the dependencies for the project and to
build and run the examples.

For more information on system requirements, installing Mojo and the Mojo
extension for VS Code, and getting started with Mojo programming, see the
[Install Mojo](https://docs.modular.com/mojo/manual/install/) section of the
[Mojo Manual](https://docs.modular.com/mojo/manual/).

## Example subdirectories

- `life/`: The [Get started with Mojo](https://docs.modular.com/mojo/manual/get-started)
  tutorial solution. A complete implementation of Conway's Game of Life cellular
  automaton, demonstrating Mojo basics including structs, modules, and Python
  interoperability.

- `gpu-intro/`: The
  [Get started with GPU programming](https://docs.modular.com/mojo/manual/gpu/intro-tutorial)
  tutorial solution. An introduction to GPU programming in Mojo with a
  simple vector addition kernel.

- `gpu-functions/`: GPU kernel implementations including vector addition,
  grayscale conversion, matrix multiplication, Mandelbrot set calculation, and
  reduction operations.

- `gpu-block-and-warp/`: Advanced GPU programming demonstrating
  block-level and warp-level synchronization and operations.

- `python-interop/`: Calling Mojo functions from Python
  code, enabling progressive migration of Python hotspots to Mojo.

- `layouts/`: Using Mojo's
  [`layout` package](https://docs.modular.com/mojo/manual/layout/layouts) for
  working with dense multidimensional arrays.

- `layout_tensor/`: Companion code for
  [Using LayoutTensor](https://docs.modular.com/mojo/manual/layout/tensors)

- `operators/`:
  [Implementing operators for a custom Mojo type](https://docs.modular.com/mojo/manual/operators#an-example-of-implementing-operators-for-a-custom-type).

- `testing/`: Writing and running unit tests using the [Mojo testing
  framework](https://docs.modular.com/mojo/tools/testing).

## License

The Mojo examples in this repository are licensed under the Apache License v2.0
with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)).

## Contributing

As a contributor, your efforts and expertise are invaluable in driving the
evolution of the Mojo programming language. The [Mojo contributor
guide](../../CONTRIBUTING.md) provides all the information necessary to make
meaningful contributions—from understanding the submission process to
adhering to best practices.
