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

- You can transfer ownership of a payload from an rvalue enum/tuple subject with
  a `var` case binding, enabling `case .Some(var value):` and
  `case (var x, var y):` without copies. This allows `match` and `if case`
  statements work more conveniently with linear types, and reduces copies in
  general.

## Language changes

- `@explicit_destroy("<message>")` on a struct is deprecated and now emits a
  warning. Give the message on the struct's `Deinitable` conformance instead:

  ```mojo
  # Before
  @explicit_destroy("call `close()` instead")
  struct File(not Deinitable):
      ...

  @explicit_destroy("use `deinit_with()` for non-Deinitable elements")
  struct Box[T: Movable](Deinitable where conforms_to(T, Deinitable)):
      ...

  # After
  struct File(not Deinitable else "call `close()` instead"):
      ...

  struct Box[T: Movable](
      Deinitable where conforms_to(T, Deinitable) else (
          "use `deinit_with()` for non-Deinitable elements"
      ),
  ):
      ...
  ```

  The message on a conditional `Deinitable where <condition> else "<message>"`
  conformance is now reported when a value is abandoned for parameters that
  don't satisfy the condition. Previously, only an always-false conformance's
  message was reported.

- Precompiled packages now no longer leak names and provide the same surface as
  the same packages compiled from source. This may break your code (especially
  from `std`):

  ```mojo
  from std.os import rmdir

  def main():
    std.os.rmdir(...) # This no longer works - use rmdir, or import std.os.
  ```

## Library stabilizations

- `List.insert`
- `List.clear`

## Library performance improvements

- Reduced the size of `Optional[T]` to the size of `T` when `T` is one of the
  following types, so code that stores them in an `Optional` uses less memory:

  - `Allocation`
  - `Layout`
  - `List`
  - `ManagedAllocation`
  - `Span`
  - `String`
  - `StringSpan`
  - `ThinAllocation`

  Don't depend on this size. It relies on bit patterns that each type's current
  representation never uses, and that representation could change in the
  future.

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

- Raising an integral `SIMD` value (including `Int`) or a `SIMDLength` to a
  negative power now aborts at run time instead of silently returning `0`. Use
  a floating point base instead.

- Raising an `IntLiteral` to a negative `IntLiteral` power now computes the
  exact result as a float literal, the reciprocal of the positive power,
  instead of evaluating to `0`. The result is a float literal even when the
  base is `1` or `-1`, so `var x = (-1)**-3` is now a `Float64` holding
  `-1.0`. Zero to a negative power evaluates to `nan`.

  ```mojo
  var a = 2**-1             # 0.5, same as 1 / 2
  var b = Float64(2) ** -1  # 0.5
  var n = 2
  var c = n**-1             # aborts at run time
  ```

- Added `FileDescriptor` to `std.io`, an owning wrapper around
  `RawFileDescriptor`. It is a linear type: close it explicitly with
  `fd^.close()`, which raises if the close fails. It is movable but not
  copyable, and `Optional[FileDescriptor]` is the size of a single
  descriptor. `RawFileDescriptor` gains a raising `close()` method.

- Added `BorrowedFileDescriptor` to `std.io`, a copyable, non-owning view
  of a `FileDescriptor` that carries the owner's origin, so it can be
  passed around without closing the descriptor or outliving its owner.

- `ArcPointer` element types no longer need to be `Deinitable`. Destroy an
  `ArcPointer` of a non-`Deinitable` type with the new `deinit_with()` method.

## GPU programming

- Added the `apple-m6` and `apple-m6-metal4` accelerator targets, and
  `CompilationTarget.is_apple_m6()` to `std.sys`. Apple M6 GPUs are now
  detected automatically, so you no longer need to pass
  `--target-accelerator=apple-m5` on M6 hardware.

## Tooling changes

## Removed

- Legacy closures are removed. The `capturing` function effect and the
  `@__parameter` (and `@parameter`), `@__copy_capture`, and
  `@__move_capture` decorators are now errors. Declare a nested function's
  captures in a capture list instead, and pass the closure as a value:

  ```mojo
  def apply[F: def(Int) -> Int](f: F, x: Int) -> Int:
      return f(x)

  def main():
      var offset = 10
      def add(x: Int) {var offset} -> Int:  # was @__copy_capture(offset)
          return x + offset
      print(apply(add, 1))
  ```

  A function passed as a compile-time parameter must now be `thin`, for
  example `[f: def(Int) thin -> Int]`.

- `CompiledFunctionInfo` no longer has `num_captures`, `capture_sizes`, or
  `populate` members. Kernels can no longer capture implicitly, so these
  were always empty.

- The AMD-only intrinsics `readfirstlane()`, `sendmsg()`, and
  `implicitarg_ptr()` moved from `std.sys.intrinsics` to `max.gpu.intrinsics`,
  next to the other vendor-specific GPU intrinsics. Import them from there
  instead. `std.sys.intrinsics.ballot()` is removed; use
  `max.gpu.primitives.warp.vote()`, which works on every GPU vendor.

## Fixed

- `SIMD.slice()` no longer passes `llvm.vector.extract` an offset that is not
  a multiple of the output width, which LLVM rejects. A wide slice at such an
  offset, like `slice[16, offset=8]()` on a 32-element vector, now takes the
  element-wise path, and aligned slices narrower than the target's native
  vector width, such as the halves `split()` returns for small vectors, now
  use the intrinsic.

- Fixed rejection of negative integer and floating-point literals in `match`
  patterns.
