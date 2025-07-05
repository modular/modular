# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

- `@parameter for` now works on a broader range of collection types, enabling
  things like `@parameter for i in [1, 2, 3]: ...`.

- Parametric aliases are now supported: Aliases can be specified with an
  optional parameter list (just like functions). Parametric aliases are
  considered first class parameter values, too.

- The compiler now detects attempts to materialize references (and related types
  like slices/pointers) to comptime interpreter stack memory into runtime code.
  The compiler cannot currently track the lifetime of internal stack objects
  when materialized to runtime, which could cause memory leaks.  Consider this
  example:

  ```mojo
  fn test_comptime_materialize():
    # This is ok! Forms a comptime reference to a comptime "stack" value of String
    # type.
    alias bad = String("foo" + "bar").unsafe_ptr()
    # This is ok too, dereferences the pointer at comptime loading the byte.
    alias byte = bad[]
    # This materializes a Byte from comptime to runtime.
    var rt_byte = byte
    # Error: cannot materialize to runtime value, the type contains an origin
    # referring to a compile-time value
    var use_bad = bad
  ```

  Previously the compiler would materialize the memory representation of the
  `String` value but not know it needs to be destroyed.  It now detects the
  problem. If you run into this, rework the code to materialize the full object
  (e.g. the String) to runtime explicitly.

- `StringLiteral` now automatically materializes to a `String` when used at
  runtime:

  ```mojo
  alias param = "foo"        # type = StringLiteral
  var runtime_value = "bar"  # type = String
  var runtime_value2 = param # type = String
  ```

  This enables all the behavior users expect without having to convert
  or annotate types, for example:

  ```mojo
  var string = "hello"
  string += " world"

  var if_result = "foo" if True else "bar"
  ```

Initializing a `String` from a `StringLiteral` initially points to static
constant memory, and does not perform SSO or allocate until the first
mutation.

### Language changes

- The `@value` decorator has been formally deprecated with a warning, it will
  be removed in the next release of Mojo.  Please move to the `@fieldwise_init`
  and synthesized `Copyable` and `Movable` trait conformance.

- Implicit trait conformance is removed. All conformances must be explicitly
  declared.

- The `owned` argument convention is being renamed to `var`. This reflects that
  `var` is used consistently for a "named, scoped, owning of a value" already
  which is exactly what the `owned` convention does.  In this release, both
  `var` and `owned` are allowed in an argument list, but `owned` will be removed
  in a subsequent release, so please move your code over.

### Standard library changes

- The `Hashable` trait has been updated to use a new data flow strategy.
  - Users are now required to implement the method
    `fn __hash__[H: Hasher](self, mut hasher: H):`
    (see `Hashable` docstring for further details).

- Added support for a wider range of consumer-grade AMD hardware, including:
  - AMD Radeon RX 7xxx GPUs
  - AMD Radeon RX 9xxx GPUs
- Compile-time checks for AMD RDNA3+ GPUs are now provided by the functions:
  - `_is_amd_rdna3()`
  - `_is_amd_rdna4()`
  - `_is_amd_rdna()`
- Added WMMA matrix-multiplication instructions for RDNA3+ GPUs to help support
  running AI models on those GPUs.

- `memory.UnsafePointer` is now implicitly included in all mojo files. Moreover,
  `OpaquePointer` (the equivalent of a `void*` in C) is moved into the `memory`
  module, and is also implicitly included.

- Python interop changes:

  - The `PythonTypeBuilder` utility now allows:
    - registering bindings for Python static methods, i.e. methods that don't
      require an instance of the class.
    - registering initializers that take arguments. Types no longer need to be
      `Defaultable` to be exposed and created from Python.

- Added `Iterator` trait for modeling types that produce a sequence of values.

  A type can implement `Iterator` by providing `__next__()` and `__has_next__()`
  methods. This naming and behavior is based on
  the Python
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)
  typing annotation, diverging slightly due to constraints present in Mojo today.

  Any type that implements `Iterator` can be used within `for` and
  `@parameter for` looping syntax.

  `Iterator` does not currently have a variant for supporting iteration over
  borrowed `ref` values.

- `InlineArray` can now be constructed with a size of 0. This makes it easier to
  use `InlineArray` in situations where the number of elements is generic and
  could also be 0.

### Tooling changes

- Added progress reporting support to the Mojo language server. This will emit progress
  notifications in your editor when the server is currently parsing a document.

### ❌ Removed

- Various functions from the `sys.info` have been removed.  Use the appropriate method
  on `CompilationTarget` from `sys.info` instead.
  - `is_x86()`
  - `has_sse4()`

### 🛠️ Fixed

- [#4121](https://github.com/modular/modular/issues/4121) - better error message
  for `.value()` on empty `Optional`.

- [#4566](https://github.com/modular/modular/issues/4566) - Hang when assigning
  loop variable inside `@parameter for`.

- [#4820](https://github.com/modular/modular/issues/4820) - `math.exp2` picks
  the wrong implementation for `float64`.

- [#4836](https://github.com/modular/modular/issues/4836) - Else path in
  `@parameter for` broken.

- [#4499](https://github.com/modular/modular/issues/4499) - Traits with
  `ref self` cause issues when used as parameter.

- [#4911](https://github.com/modular/modular/issues/4911) - `InlineArray`
  now calls the move constructor for its elements when moved.

- [#3927](https://github.com/modular/modular/issues/3927) - `InlineArray`
  now can be constructed with a size of 0.
