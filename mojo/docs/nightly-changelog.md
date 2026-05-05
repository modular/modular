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

- The previously-deprecated `constrained[cond, msg]()` function has been
  removed. Use `comptime assert cond, msg` instead.

- The previously-deprecated `Int`-returning overload of `normalize_index` has
  been removed. Use the `UInt`-returning overload (or write the index
  arithmetic inline, e.g. `x[len(x) - 1]`).

- The previously-deprecated default `UnsafePointer()` null constructor has
  been removed. To model a nullable pointer use
  `Optional[UnsafePointer[...]]`. For a non-null placeholder for delayed
  initialization, use `UnsafePointer.unsafe_dangling()`.

- The deprecated free-function reflection API in `std.reflection` has been
  removed. Use the unified `reflect[T]() -> Reflected[T]` API instead.

  Migration table:

  - `struct_field_count[T]()` → `reflect[T]().field_count()`
  - `struct_field_names[T]()` → `reflect[T]().field_names()`
  - `struct_field_types[T]()` → `reflect[T]().field_types()`
  - `struct_field_index_by_name[T, name]()` →
    `reflect[T]().field_index[name]()`
  - `struct_field_type_by_name[T, name]()` →
    `reflect[T]().field_type[name]()`
  - `struct_field_ref[idx](s)` → `reflect[T]().field_ref[idx](s)`
  - `is_struct_type[T]()` → `reflect[T]().is_struct()`
  - `offset_of[T, name=...]()` → `reflect[T]().field_offset[name=...]()`
  - `offset_of[T, index=...]()` → `reflect[T]().field_offset[index=...]()`
  - `ReflectedType[T]` → `Reflected[T]`

## 🛠️ Fixed
