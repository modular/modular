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

- Methods on structs may now declare their `self` argument with a `deinit`
  argument convention.  This argument convention is used for methods like
  `__del__` and `__moveinit__` to indicate that they tear down the corresponding
  value without needing its destructor to be run again. Beyond these two
  methods, this convention can be used to declare "named" destructors, which are
  methods that consume and destroy the value without themselves running the
  values destructor.  For example, the standard `VariadicPack` type has these
  methods:

  ```mojo
  struct VariadicPack[...]:
      # implicit destructor
      fn __del__(deinit self): ...
      # move constructor
      fn __moveinit__(out self, deinit existing: Self): ...
      # custom explicit destructor that destroys "self" by transferring all of
      # the stored elements.
      fn consume_elements[
        elt_handler: fn (idx: Int, var elt: element_type) capturing
    ](deinit self): ...
  ```

  This argument convention is a fairly narrow power-user feature that is
  important to clarify the destruction model and make linear types fit into the
  model better.  A linear types are just types where all of the destructors are
  explicit - it has no `__del__`.

- Uncaught exceptions or segmentation faults in Mojo programs can now
  generate stack traces. This is currently only for CPU-based code. To generate
  a fully symbolicated stack trace, set the `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
  environment variable, use `mojo build` with debug info enabled, e.g.
  `-debug-level=line-tables`, and then run the resulting binary.

### Language changes

- The `__del__` and `__moveinit__` methods should now take their `self` and
  `existing` arguments as `deinit` instead of either `owned`.

- The Mojo compiler now warns about use of the deprecated `owned` keyword,
  please move to `var` or `deinit` as the warning indicates.

- The `__disable_del` keyword and statement has been removed, use `deinit`
  methods instead.

### Standard library changes

- The `Copyable` trait now requires `ExplicitlyCopyable`, ensuring that all
  all types that can be implicitly copied may also be copied using an explicit
  `.copy()` method call.

  If a type conforms to `Copyable` and an `ExplicitlyCopyable` `.copy()`
  implementation is not provided by the type, a default implementation will be
  synthesized by the compiler.

  - The following standard library types and functions now require only
    `ExplicitlyCopyable`, enabling their use with types that are not implicitly
    copyable:
    `List`, `Span`, `InlineArray`, `Optional`, `Variant`, `Tuple`, `Dict`,
    `Set`, `Counter`, `LinkedList`, `Deque`, `reversed`.

    Additionally, the following traits now require `ExplicitlyCopyable` instead
    of implicit `Copyable`:
    `KeyElement`

- A new `Some` utility is introduced to reduce the syntactic load of declaring
  function arguments of a type that implements a given trait or trait
  composition. For example, instead of writing

  ```mojo
  fn foo[T: Intable, //](x: T) -> Int:
      return x.__int__()
  ```

  one can now write:

  ```mojo
  fn foo(x: Some[Intable]) -> Int:
      return x.__int__()
  ```

- The comparison operators (e.g. `__eq__` and `__le__`) of the `SIMD` type now
  return a single `Bool` instead of a boolean `SIMD` mask. Moreover, `SIMD` now
  has explicit elementwise comparisons that return boolean masks, e.g. `eq()`
  and `le()`.
  - This allows `SIMD` to conform to the `EqualityComparable` trait, enabling
    the use of `SIMD` vectors in sets, as keys to dictionaries, generic search
    algorithms, etc. Moreover, `Scalar` now conforms to the `Comparable` trait,
    i.e. `SIMD` conforms to `Comparable` when the size is 1.
  - As a consequence, `SIMD.__bool__` no longer needs to be restricted to
    scalars, and instead performs an `any` reduction on the elements of vectors.

- Non-scalar `SIMD` constructors no longer allow implicit splatting of `Bool`
  values. This could lead to subtle bugs that cannot be caught at compile time,
  for example:

  ```mojo
  fn foo[w: Int](v: SIMD[_, w]) -> SIMD[DType.bool, w]:
    return v == 42  # this silently reduced to a single bool, and then splat
  ```

  Similarly to `InlineArray`, an explicit constructor with the `fill`
  keyword-only argument can be used to express the same logic more safely:

  ```mojo
  ```mojo
  fn foo[w: Int](v: SIMD[_, w]) -> SIMD[DType.bool, w]:
    return SIMD[DType.bool, w](fill=(v == 42))  # highlights the splat logic

  fn bar(Scalar[_]) -> Scalar[DType.bool]:
    # still works, since implicit splatting to a scalar is never ambiguous
    return v == 42
  ```

- Added `os.path.realpath` to resolve symbolic links to an absolute path and

  remove relative path components (`.`, `..`, etc.). Behaves the same as the
  Python equivalent function.

- `Span` is now `Representable` if its elements implement trait
  `Representable`.

- `Optional` and `OptionalReg` can now be composed with `Bool` in
  expressions, both at comptime and runtime:

  ```mojo
  alias value = Optional[Int](42)

  @parameter
  if CompilationTarget.is_macos() and value:
      print("is macos and value is:", value.value())

- Added `sys.info.platform_map` for specifying types that can have different

  values depending on the platform:

  ```mojo
  from sys.info import platform_map

  alias EDEADLK = platform_map["EDEADLK", linux = 35, macos = 11]()
  ```

- Added support for AMD RX 6900 XT consumer-grade GPU.

- Added support for AMD RDNA3.5 consumer-grade GPUs in the `gfx1150`,
`gfx1151`, and `gfx1152` architectures. Representative configurations have been
added for AMD Radeon 860M, 880M, and 8060S GPUs.

- Updated `layout_tensor` copy related functions to support 2D and 3D
  threadblock dimensions.

- The `compile.reflection.get_type_name` utility now has limited capability to
  print parametric types, e.g. `SIMD[DType.float32, 4]` instead of just `SIMD`.
  If the parameter is not printable, an `<unprintable>` placeholder is printed
  instead. A new `qualified_builtins` flag also allows users to control the
  verbosity for the most common (but not all) builtin types.

- Add `repr` support for `List`, `Deque`, `Dict`, `LinkedList`, `Optional`, `Set`.
  [PR #5189](https://github.com/modular/modular/pull/5189) by rd4com.

- `InlineArray` now automatically detects whether its element types are
  trivially destructible to not invoke the destructors in its `__del__`
  function.  This improves performance for trivially destructible types
  (such as `Int` and friends).

### Tooling changes

- `mojo test` now ignores folders with a leading `.` in the name. This will
  exclude hidden folders on Unix systems ([#4686](https://github.com/modular/modular/issues/4686))

- Nightly `mojo` Python wheels are now available. To install everything needed
  for Mojo development in a Python virtual environment, you can use

  ```sh
  pip install mojo --index-url https://dl.modular.com/public/nightly/python/simple/
  ```

### Kernels changes

- A fast matmul for SM100 is available in Mojo. Please check it out in `matmul_sm100.mojo`.

- Moved `mojo/stdlib/stdlib/gpu/comm/` to `max/kernels/src/comm/`

### ❌ Removed

### 🛠️ Fixed

- Fixed <https://github.com/modular/modular/issues/5190>
- Fixed <https://github.com/modular/modular/issues/5139> - Crash on malformed initializer.
