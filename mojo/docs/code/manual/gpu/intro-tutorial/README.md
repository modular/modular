This directory contains code examples for the intermediate steps of the
[Get started with GPU programming](../../../../../docs/manual/gpu/intro-tutorial.mdx)
section of the Mojo Manual. The fully completed reference solution is maintained
in [examples/mojo/gpu-intro](../../../../../../examples/mojo/gpu-intro) and
published to the public GitHub repo.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).

**Note:** `vector_addition_checked.mojo` is a version of the fully completed
reference solution that uses `enqueue_function_checked()`. After kernel
typechecking becomes the default behavior, this file will be no longer needed
and can be deleted.

**Note:** These examples require a [supported
GPU](https://docs.modular.com/max/faq/#gpu-requirements) to compile and run the
kernels. If your system doesn't have a supported GPU, you can compile the
programs but the only output you'll see when you run them is the message:

```output
No compatible GPU found
```
