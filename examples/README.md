# MAX examples

These examples demonstrate the power and flexibility of
[MAX](https://docs.modular.com/max/). They include:

## [Mojo code examples](mojo/)

A collection of sample programs written in the
[Mojo](https://docs.modular.com/mojo/manual/) programming language.

## [Custom GPU and CPU operations in Mojo](custom_ops/)

The [MAX Graph API](https://docs.modular.com/max/graph/) provides a powerful
framework for staging computational graphs to be run on GPUs, CPUs, and more.
Each operation in one of these graphs is defined in
[Mojo](https://docs.modular.com/mojo/), an easy-to-use language for writing
high-performance code.

The examples here illustrate how to construct custom graph operations in Mojo
that run on GPUs and CPUs, as well as how to build computational graphs that
contain and run them on different hardware architectures.

## [Compiling and running Mojo functions on a GPU](gpu_functions/)

In addition to placing custom Mojo functions within a computational graph, the
MAX Driver API can handle direct compilation of GPU functions written in Mojo
and can dispatch them onto the GPU. This is a programming model that may be
familiar to those who have worked with CUDA or similar GPGPU frameworks.

These examples show how to compile and run Mojo functions, from simple to
complex, on an available GPU. Note that
[a MAX-compatible GPU](https://docs.modular.com/max/faq/#gpu-requirements) will
be necessary to build and run these.

## [Using Mojo from Python](python_mojo_interop/)

To enable progressive introduction of Mojo into an existing Python codebase,
Mojo modules and functions can be referenced as if they were native Python
code. This interoperability between Python and Mojo can allow for slower Python
algorithms to be selectively replaced with faster Mojo alternatives.

These examples illustrate how that can work, including using Mojo functions
running on a compatible GPU.

## [PyTorch custom operations in Mojo](pytorch_custom_ops/)

PyTorch custom operations can be defined in Mojo to try out new algorithms on
GPUs. These examples show how to extend PyTorch layers using custom operations
written in Mojo.

## [PyTorch inference on MAX](inference/)

MAX has the power to accelerate existing PyTorch models directly, and
provides Python, Mojo, and C APIs for this. These examples showcase common
models and how to run them even faster via MAX.

## [Jupyter notebooks](notebooks/)

Jupyter notebooks that showcase PyTorch models being accelerated
through MAX.

## [Build custom neural network modules with MAX Python API](python_modules/)

The [MAX Python API](https://docs.modular.com/max/api/python/) provides a
PyTorch-like interface for building neural network components that compile to
highly optimized graphs. These examples demonstrate how to create reusable,
modular components using MAX's `nn.Module` class.

The examples include custom layers, blocks, and architectural patterns that
showcase the flexibility of MAX's Python API for deep learning development, from
simple MLP blocks to more complex neural network architectures.
