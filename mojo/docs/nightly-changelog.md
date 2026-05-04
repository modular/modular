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

- Added `InlineArray.fill(value)` to set all elements to a given value.
- Added `InlineArray.fill(value)` to set all elements to a given value.

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
