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

## Language changes

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

- `InlineArray[ElementType, size]` now conditionally conforms to
  `ImplicitlyDeletable`, only when its `ElementType` does. This lets an
  `InlineArray` hold `@explicit_destroy` elements without leaking them when the
  array is dropped.

  As a result, generic code that takes an `InlineArray` by value with only a
  `Movable` element bound now fails to compile for every `ElementType`
  (previously the error was deferred until `ElementType` was a non-deletable
  type). Add `& ImplicitlyDeletable` to the element bound:

  ```mojo
  def foo[T: Movable & ImplicitlyDeletable, //](var arr: InlineArray[T, 3]):
      pass
  ```

  To consume an `InlineArray` of explicitly-destroyed elements, drain it with
  the new `destroy_with()` method, which calls a closure on each element:

  ```mojo
  arr^.destroy_with(my_destroy_closure)
  ```

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
