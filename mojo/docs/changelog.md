# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

- Mojo now differentiates between `...` and `pass` in trait methods. The use of
  `...` continues to denote no default implementation - `pass` now specifies a
  default do-nothing implementation. For example:

  ```mojo
  trait T:
      # No default implementation
      fn foo(self): ...

      # Default implementation that does nothing
      fn bar(self) : pass
  ```

  The compiler will error on the use of `pass` to define a default
  implementation for a trait method with results:

  ```mojo
  trait T:
      foo.mojo:2:26: error: trait method has results but default implementation returns no value; did you mean '...'?
      fn foo(self) -> Int: pass
                           ^
      trait.mojo:2:8: note: in 'foo', declared here
      fn foo(self) -> Int: pass
         ^
  ```

- The `deinit` argument convention can now be applied to any argument of a
  struct method, but the argument type still must be of the enclosing struct
  type.

### Language changes

- The compiler will now warn on unqualified access to struct parameters, e.g.

  ```mojo
  @fieldwise_init
  struct MyStuff[my_param: Int]:
      fn give_me_stuff(self) -> Int:
          # Warning: unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead
          return my_param
  ```

### Library changes

- Tuples have been improved:
  - Tuples can now be concatenated with `Tuple.concat(other)`.
  - Tuple can now be reversed with `Tuple.reverse()`.

- New `ContiguousSlice` and `StridedSlice` types were added to
  the `builtin_slice` module to support specialization for slicing without strides.

- `List` slicing without a stride now returns a `Span`, instead of a `List` and
  no longer allocates memory.

- The `random` module now uses a pure Mojo implementation based on the Philox
  algorithm (via an internal wrapper), replacing the previous `CompilerRT` C++
  dependency. The Philox algorithm provides excellent statistical quality, works
  on both CPU and GPU, and makes random number generation fully transparent and
  source-available. Note that this changes the random number sequence for a given
  seed value, which may affect tests or code relying on reproducible sequences.

- Implicit conversion between `Int` and `UInt` have been removed.

- `UnsafePointer` can now be initialized from a raw memory address using the
  `unsafe_from_address` initializer.

- `alloc()` now has a `debug_assert` ensuring count is non-negative.

- The `EqualityComparable` trait has been deprecated in favor of `Equatable`,
  which has identical functionality.

### Tooling changes

- The Mojo compiler now "diffs" very long types in error messages to explain
  what is going on in a more easy to understand way.
- Specifying CUDA architectures with `--target-accelerator` now expects a sm
  version string rather than just a compute capability. For example
  `--target-accelerator=nvidia:80` should be changed to
  `--target-accelerator=nvidia:sm_80`. If an incorrect format is used for the
  version, the compiler will default to the lowest supported sm version.

### Experimental changes

Changes described in this section are experimental and may be changed, replaced,
or removed in future releases.

- Mojo now supports compile-time trait conformance check (via `conforms_to()`)
  and downcast (via `trait_downcast()`). This allows users to implement features
  like static dispatching based on trait conformance. For example:

  ```mojo
  fn maybe_print[T : AnyType](maybe_printable : T):
    @parameter
    if conforms_to(T, Writable):
      print(trait_downcast[Writable](maybe_printable))
    else:
      print("[UNPRINTABLE]")
  ```

- Added support for `DType` expressions in `where` clauses:

  ```mojo
  fn foo[dt: DType]() -> Int where dt is DType.int32:
      return 42
  ```

  Currently, the following expressions are supported:
  - equality and inequality
  - `is_signed()`, `is_unsigned()`, `is_numeric()`, `is_integral()`,
    `is_floating_point()`, `is_float8()`, `is_half_float()`

### ❌ Removed

### 🛠️ Fixed
