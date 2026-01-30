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

### Library changes

- Many kernels in `nn` have been migrated to use `TileTensor`. We will have
  more documentation on `TileTensor` and its uses over the coming weeks.

- `InlineArray` now requires explicitly using literals for construction. E.g.

  ```Mojo
  var a: InlineArray[UInt8, 4] = [1, 2, 3, 4]
  # instead of InlineArray[UInt8, 4](1, 2, 3, 4)
  ```

- `Tuple` now conforms to `Writable`, implementing `write_to()` and
  `write_repr_to()` methods. `write_to()` formats elements with their string
  representation (e.g., `(1, hello, 3)`), while `write_repr_to()` includes the
  type name and uses each element's repr
  (e.g., `Tuple[Int, String](Int(1), 'hello')`). The `repr()` function also
  now supports tuples.

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
