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

- Added `List.remove(value)` method that removes the first occurrence of a
  value from the list, raising an error if the value is not found.

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
