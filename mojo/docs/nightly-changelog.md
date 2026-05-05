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

- `String` can now decode UTF-16 input using the `String(from_utf16=...)`
  constructor. PR [#5255](https://github.com/modular/modular/pull/5255) by
  [@martinvuyk](https://github.com/martinvuyk).

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
