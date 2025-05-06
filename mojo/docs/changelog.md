# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

- Parts of the Kernel library continue to be progressively open sourced!
  Packages that are open sourced now include:
  - `kv_cache`
  - `quantization`
  - `nvml`
  - Benchmarks
  - `Mogg` directory which contains registration of kernels with the Graph
    Compiler

- Implicit trait conformance is deprecated. Each instance of implicit
  conformance results in a warning, but compilation still goes through. Soon it
  will be upgraded into an error. Any code currently relying on implicit
  conformance should either declare conformances explicitly or, if appropriate,
  replace empty, non-load-bearing traits with trait compositions.

### Language changes

### Standard library changes

- The `CollectionElement` trait has been removed.

- Added support for NVIDIA RTX 2060 GPUs, enabling Mojo programs to run
  on a wider range of consumer-grade hardware.

Changes to Python-Mojo interoperability:

- `Python.{unsafe_get_python_exception, throw_python_exception_if_error_state}`
  have been removed in favor of `CPython.{unsafe_get_error, get_error}`.

- Since virtually any operation on a `PythonObject` can raise, the
  `PythonObject` struct no longer implements the `Indexer` and `Intable` traits.
  Instead, it now conforms to `IntableRaising`, and users should convert
  explictly to builtin types and handle exceptions as needed. In particular, the
  `PythonObject.__int__` method now returns a Python `int` instead of a mojo
  `Int`, so users must explicitly convert to a mojo `Int` if they need one (and
  must handle the exception if the conversion fails, e.g. due to overflow).

- `PythonObject` no longer implements `Stringable`. Instead, the
  `PythonObject.__str__` method now returns a Python `str` object and can raise.
  The new `Python.str` function can also be used to convert an arbitrary
  `PythonObject` to a Python `str` object.

- `String` now implements `ConvertibleFromPython`.

- The `bitset` datastructure was added to the `collections` package. This is a
  fixed `bitset` that simplifies working with a set of bits and perform bit
  operations.

- A new `json` module was added the provides a way to deserialize JSON objects
  into Mojo.

- A new `regex` module was added. The regex module provides functionality
  for pattern matching and manipulation of strings using regular
  expressions. This is a simple implementation that supports basic regex
  operations.

### Tooling changes

- Added support for emitting LLVM Intermediate Representation (.ll) using `--emit=llvm`.
  - Example usage: `mojo build --emit=llvm YourModule.mojo`

- Removing support for command line option `--emit-llvm` infavor of `--emit=llvm`.

- Added support for emitting assembly code (.s) using `--emit-asm`.
  - Example usage: `mojo build --emit=asm YourModule.mojo`

- Added `associated alias` support for documentation generated via `mojo doc`.

### ❌ Removed

### 🛠️ Fixed
