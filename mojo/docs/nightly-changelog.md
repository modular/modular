---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Keyword variadic arguments can now be forwarded to another function that takes
  keyword variadics, using Python style `**` syntax:

  ```mojo
  def takes_them(**kwargs: Int): ...
  def pass_them(**kwargs: Int):
    takes_them(**kwargs^)
  ```

- Struct fields are no longer allowed to hide `UnsafeAnyOrigin` within a
  struct, e.g. this is no longer accepted:

  ```mojo
  struct Example:
    # error: cannot use UnsafeAnyOrigin in a struct field.
    var ptr: UnsafePointer[Int, MutUnsafeAnyOrigin]
  ```

  This is because Mojo doesn't know that uses of `Example` contain an
  `UnsafeAnyOrigin` and therefore doesn't do lifetime extension for values in
  its context. The typical solution for this is to add an `Origin` parameter but
  you can also use `UntrackedOrigin` if you explicitly manage the lifetime of
  the underlying data:

  ```mojo
  struct Example[origin: Origin]:
    var ptr: UnsafePointer[Int, Self.origin]

  @export("new")
  def new() abi("C"): pass
  ```

## Library changes

- The `ImplicitlyCopyable`, `Intable`, and `Equatable` traits no longer
  inherit from `ImplicitlyDestructible`. Generic code that relied on
  receiving the destructor bound transitively through these traits (or
  through `Comparable`, which inherits from `Equatable`) must now spell it
  out explicitly, for example
  `T: ImplicitlyCopyable & ImplicitlyDestructible`. In practice, most
  generic code should prefer `T: Copyable` instead, per the guidance in
  `ImplicitlyCopyable`'s docstring.

- Changed `Idx` to a `comptime` alias for `ComptimeInt`. Use `Idx[value]`
  instead of `Idx[value]()` for compile-time coordinates.

- Added `is_trivially_movable`, `is_trivially_copyable`, and
  `is_trivially_destructible` to `std.memory`. These helper functions
  return whether a type's move constructor, copy constructor, or destructor is
  trivial (i.e., a bit-copy or a no-op).

- Added `std.gpu.host.CompletionFlag`, a non-owning handle to an MLRT
  `M::Driver::CompletionFlag` (an 8-byte slot in pinned host memory mapped
  into a device's address space). Pairs with the new
  `DeviceStream.wait_for_host_value(flag, value)` method, which stalls the
  stream until the flag's 64-bit slot equals the given value. Corresponds to
  CUDA's `cuStreamWaitValue64` and captures cleanly into a CUDA graph as a
  wait-value node, letting a CPU thread (or an AsyncRT worker dispatched by
  `enqueue_host_func`) gate a GPU stream on host-produced data without a
  second stream or a blocking host-function callback. Currently CUDA-only;
  other backends raise.

- `Coord`, `coord()`, `Idx`, `ComptimeInt`, `RuntimeInt`, and related coordinate
  helpers now live in the standard library module
  [`std.utils.coord`](/docs/std/utils/coord/). The
  [`layout.coord`](/docs/layout/coord/) module re-exports the same symbols for
  layout and kernel code; `layout` also hoists the common names at package
  scope for convenience.

- Python -> Mojo FFI calls registered through `PythonModuleBuilder` and
  `PythonTypeBuilder` have significantly reduced per-call overhead:

  - Non-kwargs callables registered with `def_function` / `def_method` /
    `def_staticmethod` now use CPython's `METH_FASTCALL` calling
    convention rather than `METH_VARARGS`. Kwargs-accepting functions
    still use `METH_VARARGS | METH_KEYWORDS`.

  - `PythonObject.__del__` skips the `PyGILState_Ensure` /
    `PyGILState_Release` round-trip when the current thread already
    holds the GIL (checked via `PyGILState_Check`). On the common
    Python -> Mojo FFI path (where CPython hands the callee an
    already-held GIL) the destructor pays just the check and a direct
    `Py_DecRef`. The public contract is unchanged - dropping a
    `PythonObject` from a thread that does not hold the GIL remains
    safe.

  - `Int(py=obj)` and `Scalar[IntDType](py=obj)` fast-path exact
    Python `int` via `PyLong_AsSsize_t`.

  - `String(py=obj)` fast-paths exact Python `str` via
    `PyUnicode_AsUTF8AndSize`, reading the cached UTF-8 buffer directly
    and skipping the `py.__str__()` round trip.

- Added `TileTensor.copy_from()` and `TileTensor.split()` for copying between
  compatible tile views and splitting tiles into static or runtime-sized
  partitions.

- `String.as_bytes_mut()` has been renamed to `String.unsafe_as_bytes_mut()`, to
  reflect that writing invalid UTF-8 to the resulting `Span[Byte]` can lead to
  later issues like out of bounds access.

- A new `BinaryHeap` collection has been added to the `std.collections` module.
  This is a list-backed binary max-heap.

- The core collection types no longer require their element type to be
  `Copyable` — `Movable & ImplicitlyDestructible` is now the minimum bound.
  This applies to `List[T]`, `Deque[T]`, `LinkedList[T]`, `InlineArray[T,
  size]`, both type parameters of `Dict[K, V, H]` (along with
  `SwissTable`/`SwissTableEntry`/`OwnedKwargsDict` and the loosened
  `KeyElement` trait), and `Set[T]`. Copy-requiring methods stay gated on
  `Copyable`. `Counter[V]` is unchanged. `Dict.setdefault` and `Set.add`
  now take their argument by `var T`; for move-only types call them as
  `d.setdefault(key^, default)` or `set.add(value^)`.

- `reflect[T]` is now a `comptime` alias for the `Reflected[T]` handle type
  rather than a function returning a zero-sized handle instance. All methods on
  `Reflected[T]` are `@staticmethod`s, and the type is no longer constructible.
  Drop the parens at call sites:
  # OR

  struct Example:
    var ptr: UnsafePointer[Int, MutUntrackedOrigin]
  ```

  As a temporary workaround, you can decorate fields with
  `@__allow_legacy_any_origin_fields` to ignore the compiler error, however this
  decorator is not stable and will eventually be removed.

## Language changes

- Relative imports must now use `from` (`from . import foo`); the `import .foo`
  form is no longer accepted.

## Library changes

- `Int` is now an alias for `Scalar[DType.int]` and integer literals materialize
  to this `Scalar` type. Because of this some conversions have become more
  strict.

  A new `SIMDSize` type has been added for the width of `SIMD` itself and must
  be used when inferring a parameter based on a SIMD argument like so:

  ```mojo
  def frob[w: SIMDSize](v: SIMD[DType.int, w]): ...
  ```

  Alternitively the width can be unbound if you simply want to be parametric
  over any `SIMD` type:

  ```mojo
  def frob(v: SIMD[DType.int, _])
  ```

  The new `Int` should still be used in all other situations.

- `ImplicitlyDestructible` has been renamed to `ImplicitlyDeletable`, for better
  name consistency with its required `__del__()` "delete" special method.

- The `Reflected.field_type[name]` reflection member has been renamed to
  `Reflected.field[name]`, because it returns a chainable `Reflected` handle
  for the named field rather than the field's bare type, so the old name was
  not accurate. Retrieve the field's type from the handle's `.T` member, as in
  `reflect[T].field["x"].T`. Update call sites such as
  `reflect[T].field_type["x"]` to `reflect[T].field["x"]`.

- Several collection types now *conditionally* conform to `ImplicitlyDeletable`,
  conforming only when their element type does. This lets a collection hold
  non-`ImplicitlyDeletable` elements at all (previously such a collection failed
  to compile); a collection of non-deletable elements is itself linear and must
  be drained explicitly with the new `destroy_with()` method, which calls a
  closure on each element:

  ```mojo
  collection^.destroy_with(my_destroy_closure)
  ```

  Generic code that takes one of these collections by value may now need
  `& ImplicitlyDeletable` added to its element bound so the collection can be
  dropped:

  ```mojo
  def foo[T: Movable & ImplicitlyDeletable, //](var arr: InlineArray[T, 3]):
      pass
  ```

  Affected types:

  - `InlineArray[ElementType, size]`.
  - `Deque[ElementType]`
    - Element-destroying operations (`append`, `appendleft`, `extend`,
      `extendleft`, `insert`, `clear`, `remove`, etc.) still require
      `ElementType` to be `ImplicitlyDeletable`.
    - Consuming iteration (`for x in deque^`, the `IterableOwned` conformance)
      is likewise conditional, requiring `ElementType` to be
      `ImplicitlyDeletable`; generic code bounded on `IterableOwned` now rejects
      a non-conforming element type at the bound rather than failing later
      inside `__iter__()`. For deletable element types (the common case) this is
      transparent.

- The `IterableOwned` conformance on `List` and `InlineArray` (consuming
  iteration via `for x in collection^`) is now conditional, requiring the
  element type to be `Movable & ImplicitlyDeletable`. Consuming iteration moves
  elements out of the collection rather than copying them, so it no longer
  requires `Copyable`. Generic code bounded on `IterableOwned` now rejects a
  collection of non-conforming elements at the bound, rather than failing later
  inside `__iter__()`.

- The implicit conversion constructors that cast an `UnsafePointer` to
  `MutUnsafeAnyOrigin` or `ImmutUnsafeAnyOrigin` are now deprecated and emit a
  deprecation warning when used. `UnsafeAnyOrigin` is an unsafe escape hatch
  that silently extends unrelated lifetimes and disables exclusivity checking,
  so it should never be applied implicitly. Prefer keeping a concrete origin;
  if you must discard it, make the cast explicit with the
  `as_unsafe_any_origin()` method.

- The traits `ImplicitlyDeletable`, `Movable`, `Copyable`, and
  `ImplicitlyCopyable` are now stable.

- Added `raise_python_exception()` to `std.python.bindings`, which translates a
  Mojo `Error` into a Python exception via `PyErr_SetString` and returns a null
  `PyObjectPtr`.

## Tooling changes

- Added a `--lld-path` CLI flag. This overrides the LLD path that Mojo uses.

## GPU programming

- Added an 8x8 `simdgroup_matrix` matrix multiply-accumulate primitive
  (`_mma_apple_8x8()`) with `apple_mma_load_8x8()` / `apple_mma_store_8x8()`
  fragment helpers for Apple Silicon GPUs in `std.gpu.compute.arch`. Unlike
  the 16x16 path (Apple M5 only), the 8x8 primitive is available on all Apple
  GPU generations (M1-M5). It accepts `Float16`, `BFloat16`, and `Float32`
  inputs with a `Float32` accumulator.

- Apple M5 `simdgroup_matrix` MMA now accepts FP8 (`float8_e4m3fn`,
  `float8_e5m2`) inputs with an F32 accumulator, alongside the existing
  F16/BF16/F32 and 8-bit integer types.

- Added `warp.match_any()`, which returns, for each warp lane, the mask of
  lanes whose value has the same bits. It uses NVIDIA's `match.any.sync`
  instruction, a `readfirstlane` ballot fold on AMD, and a shuffle-based
  emulation on Apple Silicon GPUs.

- Added `warp.match_all()`, which returns the warp's active-lane mask if every
  lane holds the same bits and 0 otherwise. It uses NVIDIA's `match.all.sync`
  instruction, a `readfirstlane` ballot fold on AMD, and a shuffle-based check
  on Apple Silicon GPUs.

## Removed

## Fixed
