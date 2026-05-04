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

## Tooling changes

## GPU programming

## ❌ Removed

## 🛠️ Fixed

- `FileDescriptor.write_bytes()`: Fixed silent data loss on partial writes by
  looping until all bytes are written, matching `FileHandle.write_bytes()`.
