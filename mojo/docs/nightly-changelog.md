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

- Added `ones[dtype](start, end)` to `std.bit.mask`. Returns a `Scalar[dtype]`
  with bits in the half-open range `[start, end)` set to 1 and all other bits
  set to 0. Asserts that `0 <= start < end <= bitwidth(dtype)`.

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
