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

- `Span` now has `find()` and `rfind()` methods which work for any
  `Span[Scalar[D]]` e.g. `Span[Byte]`. PR
  [#3548](https://github.com/modularml/mojo/pull/3548)
  by [@martinvuyk](https://github.com/martinvuyk).

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
