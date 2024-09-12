# mojo-pybind

`mojo-pybind` is a temporary CLI for generating CPython a
[extension module](khttps://docs.python.org/3/extending/extending.html)
from a Mojo program, used for testing the development of the Python <=> Mojo
interop work.

This tool is ONLY for the development of the Python <=> Mojo interop work, and
is NOT intended to be used widely.

## Tests

This functionality is tested indirectly as part of
`//open-source/mojo/integration-test/python-extension-modules` test suite.

## Quick command reference

### Usage

#### Run `mojo-pybind` using Bazel

```shell
bazel run //KGEN/MojoBindings/mojo-pybind -- --help
```

This should print out the `mojo-pybind` help content.
