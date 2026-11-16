# Code examples and tests for language interoperability

This directory contains code examples and tests for the
[Language interoperability](../../../../manual/interop/interoperability.mdx)
section of the Mojo Manual.

Contents:

- Each `.mojo` file is a standalone Mojo application, except files ending in
  `_c_host.mojo`.
- Each `*_c_host.mojo` file is a Mojo shared library that a C program calls.
  Its `RUN:` lines build the library, run `c_host` against it, and check the
  output with FileCheck.
- `c_host.c` is the C program. It loads the library with `dlopen()` and calls
  the exported function.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each standalone `.mojo` file (using the file
    name without extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
  - A `c_host` target for the C program, and a `lit_tests` target that runs
    the `*_c_host.mojo` files.
