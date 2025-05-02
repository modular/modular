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

### Language changes

### Standard library changes

- `Python.{unsafe_get_python_exception, throw_python_exception_if_error_state}`
  have been removed in favor of `CPython.{unsafe_get_error, get_error}`.

- The `CollectionElement` trait has been removed.

- Since virtually any operation on a `PythonObject` can raise, the
  `PythonObject` struct no longer implements the following traits: `Indexer`,
  `Intable`. Instead, it now conforms to `IntableRaising`, and users should
  convert explictly to builtin types and handle exceptions as needed. In
  particular, the `PythonObject.__int__` method now returns a Python `int`
  instead of a mojo `Int`, so users must explicitly convert to a mojo `Int` if
  they need one (and must handle the exception if the conversion fails, e.g. due
  to overflow).

### Tooling changes

### ❌ Removed

### 🛠️ Fixed
