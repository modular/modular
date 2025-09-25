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

Literals now have a default type. For example, you can now bind
[1,2,3] to T in a call to a function defined as
 fn zip[T: Iterable](impl:T) because it will default to the
 standard library's List type.

### Language changes

### Standard library changes

- Added `unsafe_get`, `unsafe_swap_elements` and `unsafe_subspan` to `Span`.

- The deprecated `DType.index` is now removed in favor of the `DType.int`.

- `String` can now decode UTF-16 input using the `String(from_utf16=...)`
  constructor. PR [#5255](https://github.com/modular/modular/pull/5255) by
  [@martinvuyk](https://github.com/martinvuyk).

### Tooling changes

### ❌ Removed

### 🛠️ Fixed
