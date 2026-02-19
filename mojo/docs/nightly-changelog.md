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

- `StringSlice._justify()` (used by `ljust`, `rjust`, `center`) now uses
  `memset`/`memcpy` instead of repeated single-character concatenation.

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
