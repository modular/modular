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

- `String` and `StringSlice` now have `is_ascii_alpha()`, `is_ascii_alnum()`,
  `capitalize()`, and `title()` methods:
  String("hello world").capitalize()        # "Hello world"
  StringSlice("hello world").title()        # "Hello World"
  String("abc").is_ascii_alpha()            # True
  StringSlice("abc123").is_ascii_alnum()    # True

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
