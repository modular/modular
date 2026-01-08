This directory contains code examples for the
[GPU programming fundamentals](../../../../../docs/manual/gpu/fundamentals.mdx)
section of the Mojo Manual.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).

**Note:** These examples require a [supported
GPU](https://docs.modular.com/max/faq/#gpu-requirements) to compile and run the
kernels. If your system doesn't have a supported GPU, you can compile the
programs but the only output you'll see when you run them is the message:

```output
No GPU detected
```
