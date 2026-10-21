---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- `with` statements now no longer copy when passed a reference to a context
  manager like in `with foo: pass`.

- `__exit__(self, error: E)` now additionally supports `var self` and
  `deinit self`, similar to the non-error `__exit__`.

## Language changes

## Library stabilizations

## Library performance improvements

## Library changes

- `Tuple` element types no longer need to be `Movable`. A `Tuple` is now
  `Movable` only when all of its elements are.
  `Tuple` has a new `init_with` initializer, which constructs each element in
  place from a closure parameterized on the element index:

  ```mojo
  comptime T = Tuple[Pinned, Pinned]  # `Pinned` is not `Movable`.
  var t = T(init_with=lambda [i: Int]() -> T.Ts[i]: Pinned(i * 10))
  ```

- `Layout` now names its element count `length`, matching `Span` and `List`.
  Construct one with `Layout[T](length=n)` and read the count back with
  `len(layout)`: `Layout` and `ManagedAllocation` now conform to `Sized`. The
  `Layout(count=)` initializer and the `Layout.count()` method are deprecated.

  ```mojo
  var allocation = alloc(Layout[Int32](length=4))
  for i in range(len(allocation)):
      allocation.unsafe_ptr().unsafe_offset(i).unsafe_write(Int32(i))
  dealloc(allocation^)
  ```

## GPU programming

- Added the `apple-m6` and `apple-m6-metal4` accelerator targets, and
  `CompilationTarget.is_apple_m6()` to `std.sys`. Apple M6 GPUs are now
  detected automatically, so you no longer need to pass
  `--target-accelerator=apple-m5` on M6 hardware.

## Tooling changes

## Removed

## Fixed
