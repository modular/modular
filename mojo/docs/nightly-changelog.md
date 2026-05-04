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

- Added `comb(n, k)` and `perm(n, k)` to `std.math`, matching Python's
  `math.comb()` and `math.perm()`. `comb(n, k)` computes the binomial
  coefficient C(n, k) without computing full factorials, returning 0 when
  `k > n`. `perm(n, k)` computes permutations P(n, k); omitting `k` (default
  `-1`) returns `n!`. `factorial()`, `comb()`, and `perm()` also accept
  `Scalar[dtype]` arguments for any integer dtype (e.g. `Int32`, `Int64`,
  `UInt32`).

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
