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

- `@register_passable("trivial")` is now deprecated,
   conform to `TrivialRegisterType` trait instead.
   The decorator will be removed after next release.

### Language changes

- Slice literals in subscripts has changed to be more similar to collection
  literals. They now pass an empty tuple as a required `__slice_literal__`
  keyword argument to disambiguate slices. If you have defined your own range
  types, please add a `__slice_literal__: () = ()` argument to their
  constructors.

- `trait` declarations no longer automatically inherit from
  `ImplicitlyDestructible`. `struct` declarations are not changed, and continue
  to inherit from `ImplicitlyDestructible`.

  Previously, the `@explicit_destroy` annotation was required to opt-out of
  `ImplicitlyDestructible` conformance. Now, if a trait's usage depends on
  implicit destructibility, it must opt-in explicitly:

  ```mojo
  # Before
  trait Foo:
      ...

  # After:
  trait Foo(ImplicitlyDestructible):
      ...
  ```

  Conversely, if a trait wanted to support non-implicitly-destructible types,
  it no longer needs to be annotated with `@explicit_destroy`:

  ```mojo
  # Before
  @explicit_destroy
  trait Foo:
      ...

  # After
  trait Foo:
      ...
  ```

  Making `struct` continue to inherit from `ImplicitlyDestructible` and not
  `trait` is intended to balance usability and familiarity in the common case,
  with the need to foster broad Mojo ecosystem support for explicitly destroyed
  types.

  It's not a problem if the majority of `struct` types are
  `ImplicitlyDestructible` in practice. However, if many ecosystem libraries are
  written with unnecessary `ImplicitlyDestructible` bounds, that would hamper
  the usability of any individual `struct` type that opts-in to being explicitly
  destroyed.

  Libraries with generic algorithms and types should be written to accomodate
  linear types. Making `ImplicitlyDestructible` opt-in for traits
  encourages a default stance of support, with specific types and functions
  only opting-in to the narrower `ImplicitlyDestructible` requirement if they
  truly need it.

  The majority of generic algorithms that take their inputs by reference should
  not be affected.

### Library changes

- The `builtin.math` module has been merged into `math`. The traits `Absable`,
  `DivModable`, `Powable`, `Roundable` and functions `abs()`, `divmod()`,
  `max()`, `min()`, `pow()`, `round()` are now part of the `math` module and
  continue to be available in the prelude. Code that explicitly imported from
  `builtin.math` should update to import from `math` instead.

- The `ffi` module is now a top-level module in the standard library, rather
  than being nested under `sys`. This improves discoverability of FFI
  functionality. Update your imports from `from sys.ffi import ...` to
  `from ffi import ...`.

- The `itertools` module now includes three new iterator combinators:
  - `cycle(iterable)`: Creates an iterator that cycles through elements
    indefinitely
  - `take_while[predicate](iterable)`: Yields elements while the predicate
    returns True
  - `drop_while[predicate](iterable)`: Drops elements while the predicate
    returns True, then yields the rest

- Math functions in `std.math` (`exp`, `exp2`, `log2`, `erf`, `tanh`, `sin`,
  `cos`, `tan`, `acos`, `asin`, `atan`, `atan2`, `acosh`, `asinh`, `atanh`,
  `cosh`, `sinh`, `expm1`, `log10`, `log1p`, `logb`, `cbrt`, `erfc`, `j0`,
  `j1`, `y0`, `y1`) now use `where dtype.is_floating_point()` clauses on their
  signatures instead of `__comptime_assert` checks in their bodies.
  This provides better compile-time error messages at the call site. Callers
  using these functions with generic `dtype` parameters may need to add
  evidence proving (either a `where` clause or `__comptime_assert`) that their
  type is floating point.

- Many kernels in `nn` have been migrated to use `TileTensor`. We will have
  more documentation on `TileTensor` and its uses over the coming weeks.

- `InlineArray` now requires explicitly using literals for construction. E.g.

  ```Mojo
  var a: InlineArray[UInt8, 4] = [1, 2, 3, 4]
  # instead of InlineArray[UInt8, 4](1, 2, 3, 4)
  ```

- The following types now conform to `Writable` and have custom implementations
  of `write_to` and `write_repr_to`.
  - `Tuple`
  - `Variant`

- The `testing` module now provides `assert_equal` and `assert_not_equal`
  overloads for `Tuple`, enabling direct tuple-to-tuple comparisons in tests
  instead of element-by-element assertions. Element types must conform to
  `Equatable & Writable`.

- The `__reversed__()` method on `String`, `StringSlice`, and `StringLiteral`
  has been deprecated in favor of the new `codepoints_reversed()` method. The
  new method name makes it explicit that iteration is over Unicode codepoints
  in reverse order, maintaining consistency with the existing `codepoints()`
  and `codepoint_slices()` methods. The deprecated `__reversed__()` methods
  will continue to work but will emit deprecation warnings.

- The `StringSlice` constructor from `String` now propagates mutability. If you
  have a mutable reference to a `String`, `StringSlice(str)` returns a mutable
  `StringSlice`. The `String.as_string_slice()` method is now deprecated in
  favor of the `StringSlice(str)` constructor, and `String.as_string_slice_mut()`
  has been removed.

- `String.ljust`, `String.rjust`, and `String.center` have been renamed to
  `String.ascii_ljust`, `String.ascii_rjust`, and `String.ascii_center`.
  Likewise for their mequivalents on `StringSlice` and `StaticString`

- `String.resize` will now panic if the new length would truncate a codepoint.
  Previously it would result in a string with invalid UTF-8.

- `String.resize` will now panic if `fill_byte` is >=128. Previously it would
  create invalid UTF-8.

- Subscripting into `String` and `StringSlice` will now panic if the index falls
  in the middle of a UTF-8 encoded code-point. Previously they would return invalid
  UTF-8. This panic is unconditional. Use `.as_bytes()[...]` if you really want
  the previous behavior.

- `StringSlice[byte=]` subscripting now returns a `StringSlice` instead of a `String`,
  This is consistent with range-based subscripting.

- Subscripting `String` and `StringSlice` by byte position will
  now return an entire Unicode codepoint. Previously it would
  return a single byte, and produce invalid UTF-8 if the index fell on
  the starting byte of a multi-byte codepoint.
- The following types now correctly implement `write_repr_to`
  - `List`, `Set`

- `assert_equal` and `assert_not_equal` now work with types implementing
  `Writable`.

- All traits and structs with `@register_passable("trivial")` decorator are now
  extending `TrivialRegisterType` trait. The decorator is removed from them.

- `String`, `StringSlice`, and `StringLiteral`'s `.format()` method now require
  their arguments to be `Writable`.

- Formatting compile-time format strings (`StringLiteral`s) no longer allocates
  memory! It uses `global_constant` to store what would be heap allocated
  parsed formatting data.

- The `Int.__truediv__` method is temporarily deprecated in favor of explicitly
  casting the operands to Float64 before dividing. This deprecation is to help
  prepare to migrate `Int.__truediv__` to return `Int`, which could be a quietly
  breaking change.

### Tooling changes

- The Mojo compiler now accepts conjoined `-D` options in addition to the
  non-conjoined form as before. Now, both `-Dfoo` and `-D foo` are accepted.

- `mojo build` now supports several `--print-*` options for discovering target
  configuration and supported architectures:
  - `--print-effective-target`: Shows the resolved target configuration after
    processing all command-line flags.
  - `--print-supported-targets`: Lists all available LLVM target architectures.
  - `--print-supported-cpus`: Lists valid CPU names for a given target triple
    (requires `--target-triple`).
  - `--print-supported-accelerators`: Lists all supported GPU and accelerator
    architectures (NVIDIA, AMD, Apple Metal).

### ❌ Removed

### 🛠️ Fixed

- [Issue #5845](https://github.com/modular/modular/issues/5845): Functions
  raising custom type with conversion fails when returning StringSlice

- [Issue #5875](https://github.com/modular/modular/issues/5875): Storing
  `SIMD[DType.bool, N]` with width > 1 to a pointer and reading back
  element-wise now returns correct values.

- `StringSlice.find`: Fixed integer overflow bug in SIMD string search that
  caused searches to fail when searching for strings longer than
  `simd_width_of[DType.bool]()` and haystacks larger than UInt16.MAX.
