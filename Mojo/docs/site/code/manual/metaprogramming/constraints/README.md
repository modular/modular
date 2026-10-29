# Code examples and tests for constraints

This directory contains Code examples and tests for the
[Constraints](../../../../manual/metaprogramming/constraints.mdx) section of
the Mojo Manual.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- A `*_compile_error.mojo` file is a program the compiler is supposed to reject.
  Its `# RUN:` and `# CHECK:` lines assert both the rejection and the diagnostic
  text. See
  [Examples that are supposed to fail](../../../README.md#examples-that-are-supposed-to-fail).
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each non-error `.mojo` file (using the file name
    without extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
  - A `lit_tests` target covering every `*_compile_error.mojo` file, which
    generates one `<file>.mojo.test` target per source.
