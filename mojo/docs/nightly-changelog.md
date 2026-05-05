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

- `String` can now decode UTF-32 input using the `String(from_utf32=...)`
  constructor. It can also decode shorter unicode codepoint encodings like
  ISO-8859-1 (aka. Latin-1) by using the `String(from_codepoints=...)`
  constructor. PR [#5258](https://github.com/modular/modular/pull/5258) by
  [@martinvuyk](https://github.com/martinvuyk).

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed
