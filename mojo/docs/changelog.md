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

- `String.ljust` and `String.rjust` have been renamed to
  `String.ascii_ljust` and `String.ascii_rjust`. Likewise for their
  equivalents on `StringSlice` and `StaticString`

- `String.resize` will now panic if the new length would truncate a codepoint.
  Previously it would result in a string with invalid UTF-8.

- `String.resize` will now panic if `fill_byte` is >=128. Previously it would
  create invalid UTF-8.

- All traits and structs with `@register_passable("trivial")` decorator are now
  extending `TrivialRegisterType` trait. The decorator is removed from them.

### Tooling changes

- The Mojo compiler now accepts conjoined `-D` options in addition to the
  non-conjoined form as before. Now, both `-Dfoo` and `-D foo` are accepted.

### ❌ Removed

### 🛠️ Fixed
