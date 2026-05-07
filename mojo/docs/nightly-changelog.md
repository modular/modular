---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

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

- Added `TileTensor.copy_from()` and `TileTensor.split()` for copying between
  compatible tile views and splitting tiles into static or runtime-sized
  partitions.

- `String.as_bytes_mut()` has been renamed to `String.unsafe_as_bytes_mut()`, to
  reflect that writing invalid UTF-8 to the resulting `Span[Byte]` can lead to
  later issues like out of bounds access.

- `reflect[T]` is now a `comptime` alias for the `Reflected[T]` handle type
  rather than a function returning a zero-sized handle instance. All methods on
  `Reflected[T]` are `@staticmethod`s, and the type is no longer constructible.
  Drop the parens at call sites:

  ```mojo
  # Before
  comptime r = reflect[Point]()
  print(r.field_count())
  print(reflect[Point]().name())
  comptime y_handle = reflect[Point]().field_type["y"]()
  var v: y_handle.T = 3.14

  # After
  comptime r = reflect[Point]
  print(r.field_count())
  print(reflect[Point].name())
  comptime y_handle = reflect[Point].field_type["y"]
  var v: y_handle.T = 3.14
  ```

  `field_type[name]` is now a parametric `comptime` member alias that yields
  `Reflected[FieldT]` directly — no trailing `()`, and the result is fully
  composable (e.g. `reflect[T].field_type["x"].name()`). The previously
  deprecated free functions `get_type_name`, `get_base_type_name`, and the
  `struct_field_*` family (along with the `ReflectedType[T]` wrapper) have been
  removed; use the corresponding methods on `reflect[T]`:

  <!-- markdownlint-disable MD013 -->

  | Removed                                 | Replacement                              |
  |-----------------------------------------|------------------------------------------|
  | `get_type_name[T]()`                    | `reflect[T].name()`                      |
  | `get_base_type_name[T]()`               | `reflect[T].base_name()`                 |
  | `is_struct_type[T]()`                   | `reflect[T].is_struct()`                 |
  | `struct_field_count[T]()`               | `reflect[T].field_count()`               |
  | `struct_field_names[T]()`               | `reflect[T].field_names()`               |
  | `struct_field_types[T]()`               | `reflect[T].field_types()`               |
  | `struct_field_index_by_name[T, name]()` | `reflect[T].field_index[name]()`         |
  | `struct_field_type_by_name[T, name]()`  | `reflect[T].field_type[name]`            |
  | `struct_field_ref[idx, T](s)`           | `reflect[T].field_ref[idx](s)`           |
  | `offset_of[T, name=name]()`             | `reflect[T].field_offset[name=name]()`   |
  | `offset_of[T, index=index]()`           | `reflect[T].field_offset[index=index]()` |
  | `ReflectedType[T]`                      | `Reflected[T]`                           |

  <!-- markdownlint-enable MD013 -->

## Tooling changes

## GPU programming

- `DeviceContext.enqueue_function[func]` and
  `DeviceContext.compile_function[func]` now accept a single kernel argument
  instead of requiring it to be passed twice. The previous two-argument forms
  `enqueue_function[func, func]` and `compile_function[func, func]` are
  deprecated. The transitional `enqueue_function_experimental` and
  `compile_function_experimental` aliases are also deprecated; switch to
  `enqueue_function` / `compile_function`.

  ```mojo
  # Before
  ctx.enqueue_function[my_kernel, my_kernel](grid_dim=1, block_dim=1)
  ctx.enqueue_function_experimental[my_kernel](grid_dim=1, block_dim=1)

  # After
  ctx.enqueue_function[my_kernel](grid_dim=1, block_dim=1)
  ```

## Removed

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

## Fixed

- Reduced the virtual address space reserved by every `mojo` invocation by
  ~1 GiB. The JIT memory mapper's reservation granularity was 1 GiB, so each
  fresh reservation was rounded up to that size and mmapped
  `PROT_READ|PROT_WRITE`, inflating `VmPeak` and counting against Linux
  `RLIMIT_AS`. This caused non-deterministic OOM crashes in
  `libKGENCompilerRTShared.so` when two `mojo` processes ran concurrently on
  memory-constrained CI runners (e.g. GitHub Actions free-tier, 7 GiB). The
  granularity is now 64 MiB; large compiles still work because the mapper
  reserves additional slabs on demand.
  ([Issue #6433](https://github.com/modular/modular/issues/6433))
