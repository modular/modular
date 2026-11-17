# Code examples and tests for passing data to exported Mojo functions

This directory contains tests for the
[Passing data to exported Mojo
functions](../../../../manual/interop/passing-data-to-mojo.mdx) page of the
Mojo Manual.

Contents:

- `exports_c_host.mojo` is a Mojo shared library that holds the page's
  exported functions. Its header maps each page claim to a check and lists
  what isn't tested. Its `RUN:` lines build the library, run `c_host` against
  it, and check the output with FileCheck.
- `c_host.c` is the C program. It loads the library with `dlopen()` and makes
  the calls that the page's C, C++, and Rust callers make.
- The `BUILD.bazel` file defines a `c_host` target for the C program and a
  `lit_tests` target that runs the `*_c_host.mojo` files.
