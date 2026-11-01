# Code examples and tests for destruction and ownership transfer

This directory contains code examples and tests for the
[Ownership transfer and destruction](../../../../manual/lifecycle/destruction-transfer.mdx)
section of the Mojo Manual.

This page sits alongside
[Automatic value destruction](../../../../manual/lifecycle/death.mdx). Claims
that page owns, such as ASAP ordering and field-level ASAP inside a
deinitializer, are tested in `../death/` and are deliberately not duplicated
here.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
