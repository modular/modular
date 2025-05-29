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

- The Python-Mojo bindings are available as a preview release!  This is the
  ability to call into Mojo functions from existing Python codebases.  The use
  case is to speed up hot spots/slow Python code by rewriting certain portions
  of your code in Mojo to achieve performance.

- List, Set and Dict literals have been reimplemented to provide
  Python-equivalent features and syntax, including simple literals like
  `[1, 2, 3]` and `{k1: v1, k2: v2}` as well as fancy "comprehensions" like
  `[a*b for a in range(10) if isprime(a) for b in range(20)]`.

- Parts of the Kernel library continue to be progressively open sourced!
  Packages that are open sourced now include:
  - `kv_cache`
  - `quantization`
  - `nvml`
  - Benchmarks
  - `Mogg` directory which contains registration of kernels with the Graph
    Compiler

### Language changes

- The type [`Dict`](/mojo/stdlib/collections/dict/Dict/) is now part of the
  prelude, so there is no need to import them anymore.

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

- Implicit trait conformance is deprecated. Each instance of implicit
  conformance results in a warning, but compilation still goes through. Soon it
  will be upgraded into an error. Any code currently relying on implicit
  conformance should either declare conformances explicitly or, if appropriate,
  replace empty, non-load-bearing traits with trait compositions.

### Standard library changes

- The `CollectionElement` trait has been removed.

- Added support for a wider range of consumer-grade hardware, including:
  - NVIDIA RTX 2060 GPUs
  - NVIDIA RTX 4090 GPUs

- The `bitset` datastructure was added to the `collections` package. This is a
  fixed `bitset` that simplifies working with a set of bits and perform bit
  operations.

- Fixed GPU `sum` and `prefix_sum` implementations in `gpu.warp` and `gpu.block`
  modules. Previously, the implementations have been incorrect and would either
  return wrong results or hang the kernel (due to the deadlock). [PR
  4508](https://github.com/modular/modular/pull/4508) and [PR
  4553](https://github.com/modular/modular/pull/4553) by [Kirill
  Bobyrev](https://github.com/kirillbobyrev) mitigate the found issues and add
  tests to ensure correctness going forward.

Changes to Python-Mojo interoperability:

- Python objects are now constructible with list/set/dict literal syntax, e.g.:
  `var list: PythonObject = [1, "foo", 2.0]` will produce a Python list
  containing other Python objects and `var d: PythonObject = {}` will construct
  an empty dictionary.

- `Python.{unsafe_get_python_exception, throw_python_exception_if_error_state}`
  have been removed in favor of `CPython.{unsafe_get_error, get_error}`.

- Since virtually any operation on a `PythonObject` can raise, the
  `PythonObject` struct no longer implements the `Indexer` and `Intable` traits.
  Instead, it now conforms to `IntableRaising`, and users should convert
  explictly to builtin types and handle exceptions as needed. In particular, the
  `PythonObject.__int__` method now returns a Python `int` instead of a mojo
  `Int`, so users must explicitly convert to a mojo `Int` if they need one (and
  must handle the exception if the conversion fails, e.g. due to overflow).

- `PythonObject` no longer implements the following traits:
  - `Stringable`. Instead, the `PythonObject.__str__` method now returns a
    Python `str` object and can raise. The new `Python.str` function can also be
    used to convert an arbitrary `PythonObject` to a Python `str` object.
  - `KeyElement`. Since Python objects may not be hashable, and even if they
    are, could theoretically raise in the `__hash__` method, `PythonObject`
    cannot conform to `Hashable`. This has no effect on accessing Python `dict`
    objects with `PythonObject` keys, since `__getitem__` and `__setitem__`
    should behave correctly and raise as needed. Two overloads of the
    `Python.dict` factory function have been added to allow constructing
    dictionaries from a list of key-value tuples and from keyword arguments.
  - `EqualityComparable`. The `PythonObject.{__eq__, __ne__}` methods need to
    return other `PythonObject` values to support rich comparisons.
    Code that previously compared `PythonObject` values should be wrapped in
    `Bool(..)` to perform the fallible conversion explicitly:
    `if Bool(obj1 == obj2): ...`.
  - `Floatable`. An explicit, raising constructor is added to `SIMD` to allow
    constructing `Float64` values from `PythonObject` values that implement
    `__float__`.

- `String` and `Bool` now implement `ConvertibleFromPython`.

- A new `def_function` API is added to `PythonModuleBuilder` to allow declaring
  Python bindings for arbitrary functions that take and return `PythonObject`s.
  Similarly, a new `def_method` API is added to `PythonTypeBuilder` to allow
  declaring Python bindings for methods that take and return `PythonObject`s.

- The `ConvertibleFromPython` trait is now public. This trait is implemented
  by Mojo types that can be constructed by converting from a `PythonObject`.
  This is the reverse operation of the `PythonConvertible` trait.

- `PythonObject(alloc=<value>)` is a new constructor that can be used to
  directly store Mojo values in Python objects.

  This initializer will fail if the type of the provided Mojo value has not
  previously had a corresponding Python 'type' object globally registered using
  `PythonModuleBuilder.add_type[T]()`.

- `PythonObject` has new methods for downcasting to a pointer to a contained
  Mojo value, for use in Python/Mojo interop.

  ```mojo
  struct Person:
      var name: String

  fn greet(obj: PythonObject) raises:
    var person = obj.downcast_value_ptr[Person]()

    print("Hello ", person[].name, "from Mojo🔥!")
  ```

  - `PythonObject.downcast_value_ptr[T]()` checks if the object is a wrapped
    instance of the Mojo type `T`, and if so, returns an `UnsafePointer[T]`.
    Otherwise, an exception is raised.

  - `PythonObject.unchecked_downcast_value_ptr[T]()` unconditionally
    returns an `UnsafePointer[T]` with any runtime type checking. This is useful
    when using Python/Mojo interop to optimize an inner loop and minimizing
    overhead is desirable.

    Also added equivalent `UnsafePointer` initializer for downcasting from a
    `PythonObject`.

- The `Python.is_type(x, y)` static method has been removed. Use the
  expression `x is y` instead.

- `os.abort(messages)` no longer supports generic variadic number of `Writable`
  messages.  While this API was high-level and convenient, it generates a lot of
  IR for simple and common cases, such as when we have a single `StringLiteral`
  message.  We now no longer need to generate a bunch of bloated IR, and
  instead, callers must create the `String` on their side before calling
  `os.abort(message)`.

- The function `atof` has been entirely rewritten as it produced incorrect
  results for very low and very high exponents.
  It now works correctly for strings with less than
  19 digits left of the `e`. For example `1.1385616158185648648648648648616186186e-3`
  won't work, and will raise an error. Anything that does
  not produce an error is now garanteed to be correct.
  While the current implementation is not the fastest, it's based on the paper
  [Number Parsing at a Gigabyte per Second](https://arxiv.org/abs/2101.11408) by
  Daniel Lemire. So with a bit of effort to
  pinpoints the slow parts, we can easily have state of the
  art performance in the future.

### Tooling changes

- Added support for emitting LLVM Intermediate Representation (.ll) using `--emit=llvm`.
  - Example usage: `mojo build --emit=llvm YourModule.mojo`

- Removing support for command line option `--emit-llvm` infavor of `--emit=llvm`.

- Added support for emitting assembly code (.s) using `--emit-asm`.
  - Example usage: `mojo build --emit=asm YourModule.mojo`

- Added `associated alias` support for documentation generated via `mojo doc`.

### ❌ Removed

- `VariadPack.each` and `VariadPack.each_idx` methods have been removed.
  Use the `@parameter for` language construct to achieve this now.

### 🛠️ Fixed

- [#4352](https://github.com/modular/modular/issues/4352) - `math.sqrt`
  products incorrect results for large inputs.
- [#4518](https://github.com/modular/modular/issues/4518) - Try Except Causes
  False Positive "Uninitialized Value".
- [#4677](https://github.com/modular/modular/issues/4677),
- [#4684](https://github.com/modular/modular/issues/4684) - Failure inferring
  type of initializer list from field of struct.
- [#4688](https://github.com/modular/modular/issues/4668) - Incorrect result for
  unsigned `gt` and `le` comparisions.
- [#4694](https://github.com/modular/modular/issues/4694) - Compiler error
  handling `x or y` expressions with PythonObject.
