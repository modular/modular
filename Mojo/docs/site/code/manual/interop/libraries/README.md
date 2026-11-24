# Code examples and tests for calling C ABI functions

This directory contains code examples and tests for the
[Calling C ABI functions from Mojo](../../../../manual/interop/libraries.mdx)
page of the Mojo Manual.

Contents:

- `tests.mojo` is a standalone Mojo application. It tests the page's claims
  and examples that don't need the custom C library.
- `custom.c` is the page's one-function C library, `custom_abs()`.
- Each `*_lit.mojo` file is a lit test for an example that builds against or
  loads `libcustom`. Its `RUN:` lines copy the library into a temporary
  folder, build the example with the page's commands, run it, and check the
  output with FileCheck. macOS and Linux commands differ, so the `RUN:` lines
  use `%if system-darwin`.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for `tests.mojo` and a `modular_run_binary_test`
    target that runs it (`tests_test`).
  - `libcustom.dylib` (macOS) and `libcustom.so` (Linux), built from
    `custom.c`.
  - A `lit_tests` target that runs the `*_lit.mojo` files.
