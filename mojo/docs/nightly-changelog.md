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

- Mojo now supports building types that support implicit conversions for
  widening origins, allowing code like this to "just work" without rebind:

  ```mojo
  def origin_superset_conversion(
    a: String, b: String, c: Bool
  ) -> Pointer[String, origin_of(a, b)]:
    if c:  # These pointers implicitly convert.
        return Pointer(to=a)
    else:
        return Pointer(to=b)
  ```

## Language changes

- Support for "set-only" accessors has been removed. You need to define a
  `__getitem__` or `__getattr__` to use a type that defines the corresponding
  setter. This eliminates a class of bugs determining the effective element
  type.

## Library changes

- `Coord`, `coord()`, `Idx`, `ComptimeInt`, `RuntimeInt`, and related coordinate
  helpers now live in the standard library module
  [`std.utils.coord`](/docs/std/utils/coord/). The
  [`layout.coord`](/mojo/layout/coord/) module re-exports the same symbols for
  layout and kernel code; `layout` also hoists the common names at package
  scope for convenience.

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

- Added `ReflectedFn[func]`, a function-side reflection handle accessed via
  the `reflect_fn[func]` `comptime` alias. Exposes function introspection
  through static methods, paralleling the type-side `Reflected[T]` API:

  ```mojo
  from std.reflection import reflect_fn

  def my_func(x: Int) -> Int:
      return x + 1

  def main():
      print(reflect_fn[my_func].display_name())  # "my_func"
      print(reflect_fn[my_func].linkage_name())  # mangled symbol name

- Added `alloc`, `free`, and `Layout` in `memory.alloc` for layout-aware memory
  allocation. A `Layout[T]` bundles an element count and alignment into a
  single value that is passed to both `alloc` and `free`, keeping size and
  alignment requirements explicit and co-located at every call site.

  ```mojo
  from memory import alloc, free, Layout

  var layout = Layout[Int32](count=4)
  var ptr = alloc(layout)
  # ... initialize & use ptr ...
  free(ptr, layout)
  ```

- The default `seed` for `random.Random`, `random.NormalRandom`, and the
  internal `_PhiloxWrapper` has changed from `0` to `0x3D30F19CD101`
  (67280421310721) to match PyTorch's `at::Philox4_32_10` default. Calls
  that omitted the `seed` argument will now produce a different output
  stream; pass `seed=0` explicitly to keep the previous behavior.

- Added `nth()` as a default method on the `Iterator` trait. It advances the
  iterator by `n` elements (destroying them) and returns the next element, or
  `None` if the iterator runs out before reaching index `n`.

  ```mojo
  var l = [10, 20, 30, 40]
  print(iter(l).nth(0).value())   # 10
  print(iter(l).nth(3).value())   # 40
  var missing = iter(l).nth(10)   # None (Optional)
  ```

- PythonObject convertibility got simplified and cleaned up. When working with
  types that required custom conversions to `PythonObject`, we used to write
  code like this:

  ```mojo
  struct MyCustomType(ConvertibleToPython, ImplicitlyCopyable):
     def to_python_object(var self) raises -> PythonObject:
        return PythonObject( ... custom logic ...)

  def hi_python(a: Some[ImplicitlyCopyable & ConvertibleToPython]) raises:
      print(t"Hi, {a.to_python_object()}!")

  def example():
      hi_python(MyCustomType())
  ```

  This approach allows custom types to implement `ConvertibleToPython` to get a
  domain specific encoding as a Python object. Mojo has simplified this by
  making all `ConvertibleToPython` types implicitly convert to `PythonObject`,
  so this can/should be simplified to:

  ```mojo
  def hi_python(a: PythonObject) raises:
      print(t"Hi, {a}!")
  ```

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

- Attempting to import a source Mojo package from a broken symlink will no
  longer result in a compiler crash.
  ([Issue #6424](https://github.com/modular/modular/issues/6424))
