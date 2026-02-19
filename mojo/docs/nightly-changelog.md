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

- `StringSlice.startswith()` and `StringSlice.endswith()` now use a direct
  `memcmp` against the prefix/suffix bytes instead of going through `find()`,
  avoiding the cost of a full search when the answer is determined by a single
  bounded comparison.

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
