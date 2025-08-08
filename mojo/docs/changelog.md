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

### Language changes

- The `__del__` and `__moveinit__` methods should now take their `self` and
  `existing` arguments as `deinit` instead of either `owned`.

- The Mojo compiler now warns about use of the deprecated `owned` keyword,
  please move to `var` or `deinit` as the warning indicates.

- The `__disable_del` keyword and statement has been removed, use `deinit`
  methods instead.

### Standard library changes

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

### Tooling changes

- `mojo test` now ignores folders with a leading `.` in the name. This will
  exclude hidden folders on Unix systems ([#4686](https://github.com/modular/modular/issues/4686))

### ❌ Removed

### 🛠️ Fixed
