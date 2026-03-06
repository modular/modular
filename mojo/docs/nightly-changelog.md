---
title: Mojo nightly
---

This version is still a work in progress.

## ✨ Highlights

## Documentation

## Language enhancements

## Language changes

- Support for "set-only" accessors has been removed. You need to define a
  `__getitem__` or `__getattr__` to use a type that defines the corresponding
  setter. This eliminates a class of bugs determining the effective element
  type.

## Library changes

- `SIMD[DType.bool, N]` now has two new methods:
  - `first_true()` -- returns the index of the first `True` lane, or `-1` if
    all lanes are `False`. Replaces the manual `pack_bits` +
    `count_trailing_zeros` pattern.
  - `count_true()` -- returns the number of `True` lanes. A bool-specific
    alias for `reduce_bit_count()`.

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
