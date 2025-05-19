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

- The Mojo compiler will now synthesize `__moveinit__` and `__copyinit__` and
  `copy()` methods for structs that conform to `Movable`, `Copyable`, and
  `ExplicitlyCopyable` (respectively) but that do not implement the methods
  explicitly.

- A new `@fieldwise_init` decorator can be attached to structs to synthesize a
  fieldwise initializer - an `__init__` method that takes the same arguments as
  the fields in the struct.  This gives access to this helpful capability
  without having to opt into the rest of the methods that `@value` synthesizes.
  This decorator allows an optional `@fieldwise_init("implicit")` form for
  single-element structs, which marks the initializer as `@implicit`.

- `try` and `raise` now work at comptime.

- "Initializer lists" are now supported for creating struct instances with an
  inferred type based on context, for example:

  ```mojo
  fn foo(x: SomeComplicatedType): ...

  # Example with normal initializer.
  foo(SomeComplicatedType(1, kwarg=42))
  # Example with initializer list.
  foo({1, kwarg=42})
  ```

- List literals have been redesigned to work better.  They produce homogenous
  sequences by invoking the `T(<elements>, __list_literal__: ())` constructor
  of a type `T` that is inferred by context, or otherwise defaulting to the
  standard library `List[Elt]` type.  The `ListLiteral` type has been removed
  from the standard library.

- Dictionary and set literals now work and default to creating instances of the
  `Dict` and `Set` types in the collections library.

### Standard library changes

- The `CollectionElement` trait has been removed.

- Added support for a wider range of consumer-grade hardware, including:
  - NVIDIA RTX 2060 GPUs
  - NVIDIA RTX 4090 GPUs

- The `bitset` datastructure was added to the `collections` package. This is a
  fixed `bitset` that simplifies working with a set of bits and perform bit
  operations.

- A new `json` module was added the provides a way to deserialize JSON objects
  into Mojo.

Changes to Python-Mojo interoperability:

- Python lists are now constructible with list literal syntax, e.g.:
  `var list: PythonObject = [1, "foo", 2.0]` will produce a Python list
  containing other Python objects.

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

- `PythonObject` no longer implements the `KeyElement` trait. Since Python
  objects may not be hashable, and even if they are, could theoretically raise
  in the `__hash__` method, `PythonObject` cannot conform to `Hashable`.
  This has no effect on accessing Python `dict` objects with `PythonObject`
  keys, since `__getitem__` and `__setitem__` should behave correctly and raise
  as needed. Two overloads of the `Python.dict` factory function have been added
  to allow constructing dictionaries from a list of key-value tuples and from
  keyword arguments.

- `String` and `Bool` now implement `ConvertibleFromPython`.

- A new `def_function` API is added to `PythonModuleBuilder` to allow declaring
  Python bindings for arbitrary functions that take and return `PythonObject`s.
  Similarly, a new `def_method` API is added to `PythonTypeBuilder` to allow
  declaring Python bindings for methods that take and return `PythonObject`s.

### Tooling changes

- Added support for emitting LLVM Intermediate Representation (.ll) using `--emit=llvm`.
  - Example usage: `mojo build --emit=llvm YourModule.mojo`

- Removing support for command line option `--emit-llvm` infavor of `--emit=llvm`.

- Added support for emitting assembly code (.s) using `--emit-asm`.
  - Example usage: `mojo build --emit=asm YourModule.mojo`

- Added `associated alias` support for documentation generated via `mojo doc`.

### ❌ Removed

### 🛠️ Fixed

- [#4352](https://github.com/modular/modular/issues/4352) - `math.sqrt`
  products incorrect results for large inputs.
- [#4518](https://github.com/modular/modular/issues/4518) - Try Except Causes
  False Positive "Uninitialized Value".
