# Python -> Mojo Bindings Integration Tests

This directory contains tests for calling Mojo from Python.

## Tests

> [!WARNING]
>
> These tests are not representative of what the final state of the Python <=>
> Mojo interop experience will look like. Do not attempt to copy the structure
> used in these tests, as it will likely change and your code will break. This
> functionality is not ready for usage yet.

The tests can be run using Bazel:

```shell
bazel test //open-source/mojo/integration-test:lit
```

### Test #1: [./basic](./basic/)

This is a minimal smoke test, testing calling a simple Mojo function from
Python.

The initial contents are:

```text
basic
├── bindings.mojo
└── use_bindings.py
```

Build manually using:

```shell
mojo-pybind bindings.mojo
```

Which will result in a `bindings.so` being built and placed alongside the
existing files:

```text
basic
├── bindings.a
├── bindings.mojo
├── bindings.so
└── use_bindings.py
```

(The bindings.a file is an intermediate artifact that can be deleted.)

Running the Python `use_bindings.py` code will load and run compiled Mojo code
from `bindings.so`:

```shell
% python use_bindings.py
Result from Mojo 🔥: 2
```

### Test #2: [./feature-overview](./feature-overview/)

This test aims to include a basic test for each supported Mojo language
feature that is usable across the Python <=> Mojo interop boundary.
