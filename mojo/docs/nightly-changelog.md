---
title: Mojo nightly
---

This version is still a work in progress.

## ✨ Highlights

## Documentation

## Language enhancements

- Types can parameterize the `out` argument modifier when they want into being
  bindable to alternate address spaces, e.g.:

  ```mojo
  struct MemType(Movable):
    # Can be constructed into any address space.
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        ...

    # Only constructable into GLOBAL address space.
    def __init__(arg: Int, out[AddressSpace.GLOBAL] self):
        ...
  ```

## Language changes

- Support for "set-only" accessors has been removed. You need to define a
  `__getitem__` or `__getattr__` to use a type that defines the corresponding
  setter. This eliminates a class of bugs determining the effective element
  type.

## Library changes

- `String.as_bytes_mut()` has been renamed to `String.unsafe_as_bytes_mut()`, to
  reflect that writing invalid UTF-8 to the resulting `Span[Byte]` can lead to
  later issues like out of bounds access.

## Tooling changes

## GPU programming

## ❌ Removed

- The legacy `fn` keyword now produces an error instead of a warning. Please
  move to `def`.

## 🛠️ Fixed
