This directory contains code examples for the
[Tips for Python devs](../../../../docs/manual/python-to-mojo)
section of the Mojo Manual.

The `BUILD.bazel` file defines:

- A `mojo_binary` target for each `.mojo` standalone application, consisting of
  the file name without an extension (for example, `python_to_mojo` for `python_to_mojo.mojo`)
- A `modular_run_binary_test` target named `python_to_mojo_test` to run the
  `python_to_mojo.mojo` application as a test target (it should raise no errors)

Only the Mojo elements are tested.
