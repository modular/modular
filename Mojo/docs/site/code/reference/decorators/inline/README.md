# Code examples and tests for the `@inline` decorator

This directory contains code examples and tests for the
[`@inline`](../../../../reference/decorators/inline.mdx)
decorator reference page.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the filename without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
