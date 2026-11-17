# Code examples and tests for passing data to C ABI functions

This directory contains tests for the
[Passing data to C ABI
functions](../../../../manual/interop/passing-data-to-c.mdx) page of the
Mojo Manual.

Contents:

- `tests.mojo` checks the page's claims. Its header maps each claim to a test
  and lists what isn't tested, with reasons.
- The `BUILD.bazel` file defines a `mojo_binary` target for each `.mojo` file
  and a `modular_run_binary_test` target for each binary (with a `_test`
  suffix).
