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

## [Using Mojo from Python](mojo/python-interop/)

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

---

## Example code tests

Whenever possible, code examples should be tested in the following ways.

### Pre-submit build test

Use a `BUILD.bazel` file to create an executable binary for the code.

Any code used in a build function (such as `modular_by_binary()` or
`mojo_binary()`), is automatically built as part of the "build everything" CI
commands that run during PR pre-submit. For
[example](https://github.com/modular/modular/tree/main/examples/max-graph/BUILD.bazel):

```bzl
modular_py_binary(
    name = "addition",
    srcs = ["addition.py"],
    imports = ["."],
    deps = [
        "//SDK/lib/API/python/max",
        requirement("numpy"),
    ],
)
```

You can run it directly to confirm it works like this:

```sh
br //open-source/max/examples/max-graph:addition
```

For help writing this code, look at the other `BUILD.bazel` files in the
examples subdirectories. For more about the `bazel` command, see [Using
bazel](https://github.com/modular/modular/blob/main/bazel/docs/usage.md).

But this build target isn't actually called by CI and it doesn't need to be. By
merely building the code through these Bazel targets (via "build everything"
workflows), we confirm that the Python or Mojo code actually compiles. That is,
we can be sure that all Python APIs and Mojo APIs imported and called by
the files (still) exist.

This is good for catching major breaking changes like an API getting removed.

However, there still could be runtime issues.

### Pre-submit smoke test

It's possible that an API changed in a way that passes the above build but
doesn't break until the code is executed. To verify that the code actually runs
successfully, add a `modular_run_binary_test()` function to the same
`BUILD.bazel` file that calls the previously-defined binary. For
[example](https://github.com/modular/modular/blob/8dbd252/examples/mojo/gpu-intro/BUILD.bazel#L15-L19):

```bzl
mojo_binary(
    name = "vector_addition",
    srcs = [
        "vector_addition.mojo",
    ],
    target_compatible_with = ["//:has_gpu"],
    deps = [
        "@mojo//:layout",
        "@mojo//:stdlib",
    ],
)

modular_run_binary_test(
    name = "vector_addition_test",
    binary = "vector_addition",
    tags = ["gpu"],
)
```

You can run it directly to confirm it works like this (but notice this example
requires a GPU):

```sh
bt //open-source/max/examples/mojo/gpu-intro:vector_addition_test
```

With this, we can now be sure that the code runs under specific build
conditions, but it doesn't necessarily match the end-user's environment
and packages.

### Nightly smoke test

The Bazel test above uses pinned package dependencies and users won't actually
run the examples through Bazel. So before we release a new nightly build, we
want to mimic the user environment the best we can using
[Pixi](https://pixi.sh/latest/) and then execute the code again. (The Pixi
environment might have different package versions.)

To make every example easy to use, they should already have a `pixi.toml` file
that specifies the code's dependencies. To add a test, just add a task named
`test` to the code's local `pixi.toml` file that executes the code. For
[example](https://github.com/modular/modular/blob/8d0650d/examples/offline-inference/pixi.toml):

```toml
[tasks]
basic = "python basic.py"
test = "python basic.py"
```

All this does is execute the `basic.py` file. As long as that doesn't crash,
it passes. If it does crash, then it will halt the release until the code is
fixed (hopefully).

**NOTE:** If the test requires GPU, make sure that it's included in the
`testMaxGPUExamples` workflow because it must be skipped by
`testMaxCondaExamples`.

### Nightly functional test (optional)

The simple `pixi` test above is good enough for most cases, but it doesn't
assert that the result produced is what's expected. To go to that next level,
we might add an actual [pytest](https://docs.pytest.org/en/stable/). For
[example](https://github.com/modular/modular/tree/main/examples/max-graph/pixi.toml):

```toml
[tasks]
addition = "python3 addition.py"
test = "pytest test_addition.py"
```

This is ideal because it uses an actual test program to confirm that the result
from the example code is actually what we expect.
