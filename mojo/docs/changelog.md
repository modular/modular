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

### Language changes

### Library changes

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

- The `EqualityComparable` trait has been deprecated in favor of `Equatable`,
  which has identical functionality.

### Tooling changes

- The Mojo compiler now "diffs" very long types in error messages to explain
  what is going on in a more easy to understand way.

### ❌ Removed

### 🛠️ Fixed
