# MAX examples

These examples demonstrate the power and flexibility of
[MAX](https://docs.modular.com/max/). They include:

## [Mojo code examples](mojo/)

A wide variety of [Mojo](https://docs.modular.com/mojo/manual/) programs
to help you learn the language.

## [Basic MAX graph](max-graph/)

The [MAX Python API](https://docs.modular.com/max/api/python/) provides a
PyTorch-like interface for building neural network components that compile to
highly optimized graphs. This is a simple example of how to build a graph in
Python, before moving on to more complex custom ops using Mojo.

## [Custom MAX graph module](custom-graph-module/)

This example shows how to create a reusable, modular component for a MAX graph,
using the `nn.Module` class. It include custom layers, blocks, and
architectural patterns that showcase the flexibility of MAX's Python API for
deep learning development, from simple MLP blocks to more complex neural
network architectures.

## [Custom GPU and CPU graph ops in Mojo](custom_ops/)

Building upon the basic MAX graph concepts from the above examples, this is a
collection of examples that how to construct custom graph operations in Mojo
that run on both GPUs and CPUs that run on different hardware architectures. It
includes GPU kernels written in Mojo for algorithms such as top-k, matrix
multiplication, fused attention, and more.

## [GPU functions written in Mojo](mojo/gpu-functions/)

In addition to placing custom Mojo functions within a computational graph, Mojo
can handle direct compilation and dispatch of GPU functions. This is a
programming model that may be familiar to those who have worked with CUDA or
similar GPGPU frameworks.

These examples show how to compile and run Mojo functions, from simple to
complex, on an available GPU, without using a MAX graph.

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

## [Offline inference](offline-inference/)

A simple example showing how to directly send inference to an LLMs using the
MAX Python API, without starting a webserver (without an endpoint).
