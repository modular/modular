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

- You can bind a non-`Movable` payload from an rvalue enum or tuple subject
  with a `var` case binding, enabling `case .Some(var value):` and
  `case (var x, var y):` without copies. This allows `match` and `if case`
  statements to work with non-movable and linear types, and reduces copies in
  general.

- Auto-parameterization now reaches `_` holes nested inside other parameterized
  types, so an unbound parameter at any depth of an argument's type becomes an
  an implicit parameter of the function:

  ```mojo
  def process(p: Optional[Pointer[Int, _]]): ...
  ```

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
- `Pointer.__init__(to=)`
- `Pointer.__eq__`
- `Pointer.__ne__`
- `Pointer.unsafe_bitcast`

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

- An optional mutable pointer now implicitly converts to an optional
  immutable pointer, as `Pointer` does.

- Added `normpath()` to `std.os.path`, matching Python's `os.path.normpath()`.

- Added a builtin `filter()`, which takes an iterable and a predicate closure
  and yields the elements the predicate accepts:
  `filter(items, lambda (x: Int) -> Bool: x > 0)`.

- `take_while()` and `drop_while()` in `std.itertools` now take the predicate
  as an argument instead of a compile-time parameter, so it can be a closure
  that captures values. Write `take_while(items, pred)` instead of
  `take_while[pred](items)`.

- `Span.binary_search_by()` no longer takes the comparator as a compile-time
  parameter. Pass it as an argument, `span.binary_search_by(cmp)`, which also
  lets it capture the value being searched for.

- `QuickBench.run()` in `std.benchmark` now takes a single closure with no
  arguments in place of its eleven overloads for functions of 0 to 10
  arguments. Capture the inputs instead of passing them separately:
  `qb.run(lambda () {x} -> Float32: exp(x), bench_id=BenchId("exp"))`.

- `ord()` and `Codepoint.ord()` on a runtime string (`StringSpan`, `String`)
  now raise an `Error` instead of aborting when the string is empty. They also
  raise when the string holds more than one codepoint; previously they
  silently returned the first codepoint unless assertions were enabled.
  Callers must now handle the error with `try` or a `raises` function. On a
  string literal, both never raise, and a literal that isn't exactly one
  codepoint is a compile-time error.

  ```mojo
  var text = String("🔥")
  var codepoint = ord(text)  # 128293; raises if `text` isn't one codepoint
  comptime letter = ord("a")  # 97; never raises
  ```

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
  `FileDescriptor(copy_borrowed=)` duplicates a descriptor with
  close-on-exec set, so child processes don't inherit the copy.

- `RawFileDescriptor.read_bytes()`, `RawFileDescriptor.fchdir()`,
  `RawFileDescriptor.close()`, `FileDescriptor.close()`, and
  `FileDescriptor(copy_borrowed=)` now raise `IOError`, so callers can
  match on the error's `kind()`. Code in a plain `raises` function still
  compiles unchanged.

- Added `BorrowedFileDescriptor` to `std.io`, a copyable, non-owning view
  of a `FileDescriptor` that carries the owner's origin, so it can be
  passed around without closing the descriptor or outliving its owner.

- `ArcPointer` element types no longer need to be `Deinitable`. Destroy an
  `ArcPointer` of a non-`Deinitable` type with the new `deinit_with()` method.

- Added an `OwnedPointer(init_with=)` initializer that constructs the value in
  place, so the element type doesn't need to be `Movable`, and an
  `OwnedPointer.deinit_with()` method that destroys an `OwnedPointer` whose
  element type isn't `Deinitable`. The initializers that copied a value into the
  allocation are now deprecated.

- `FileHandle` now owns its descriptor through a `FileDescriptor` and always
  refers to an open file. It is no longer `Defaultable`, and `close()`
  consumes the handle (`f^.close()`) instead of leaving it in an invalid
  state. It still raises if the close fails. Use `Optional[FileHandle]`,
  which is the same size as `FileHandle`, for a file that may not be open.
  `NamedTemporaryFile` follows the same rules: it always holds an open file,
  `close()` consumes it, and `Optional[NamedTemporaryFile]` is the same size
  as `NamedTemporaryFile`.

- The `OptionalPointer`, `OptionalMutPointer`, and `OptionalImmPointer`
  aliases are deprecated. Write `Optional[Pointer[T, origin]]`,
  `Optional[MutPointer[T, origin]]`, or `Optional[ImmPointer[T, origin]]`
  instead. An `_` origin nested inside the `Optional` makes the function
  generic over it, as it did with the aliases:

  ```mojo
  # Before
  def fill(buf: OptionalMutPointer[Byte, _]): ...

  # After
  def fill(buf: Optional[MutPointer[Byte, _]]): ...
  ```

- The `StringSlice`, `ImmStringSpan`, and `ImmStringSlice` aliases are
  deprecated. Use `StringSpan` instead. `StringSpan` is always immutable, so
  the `Imm` variants are redundant.

- `Span` now conforms to `Equatable` when its element type does.

- `Optional.value()`, `Optional.unsafe_value()`, and `Optional.__getitem__()`
  now return a reference with an interior origin of the `Optional`.
  Reassigning the `Optional`, or calling a `mut self` method on it
  invalidates outstanding references to its value. Reading through
  such a reference afterward used to compile and read a destroyed value; it is
  now a compile error:

  ```mojo
  var opt = Optional(5)
  ref i = opt.value()
  opt = None
  print(i)  # error: use of invalidated interior reference 'opt["value"]'
  ```

## GPU programming

- Added the `apple-m6` and `apple-m6-metal4` accelerator targets, and
  `CompilationTarget.is_apple_m6()` to `std.sys`. Apple M6 GPUs are now
  detected automatically, so you no longer need to pass
  `--target-accelerator=apple-m5` on M6 hardware.

- `Bool`, `Scalar` (including `Int`), `String`, `StringSpan`, and
  `StringLiteral` now conform to `ConvertibleToPython`, so they can be passed
  to generic code bounded on that trait. The dedicated `PythonObject`
  constructors for these types are removed; implicit conversion still works
  through the single `ConvertibleToPython` constructor.
  `ConvertibleToPython` gains an associated `ConversionToPythonErrorType`
  (default `Never`), so conversions that cannot fail, such as `Bool` and
  `Scalar`, stay non-raising, while `String` conversions keep raising
  `Error`. This is a breaking change. If a conversion is non-raising, the
  `raises` effect can be removed from the `ConvertibleToPython` constructor.
  If a conversion is raising, an error type must be assigned to
  `ConversionToPythonErrorType`.

## Tooling changes

- `mojo build` and `mojo run` can now report what elaboration produced, to
  show which compile-time code makes a compile slow. `--elaboration-stats`
  lists each `comptime for` loop with its iteration count and the operations
  its unrolled iterations produced. It also lists the calls that compile-time
  evaluation made to the interpreter, summed by function and by call site,
  with their cache hits and the CPU time spent interpreting. The
  report has one part for each pipeline that elaborated: the host and each
  accelerator target. This option is hidden; use `--help-hidden` to list it.

  Like the timing reports, the report prints as text to stderr by default.
  `--timing-json` prints it as JSON instead, under an `elaboration` key, and
  `--timing-file` writes it to a file instead of stderr. If the compilation
  hits the cache, elaboration is skipped and nothing is reported. Clear
  compilation cache or point `MODULAR_CACHE_DIR` at an empty directory
  to get a full report.

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

- The `Powable` and `CeilDivableRaising` traits are removed. A type that
  defines `__pow__()` still supports the `**` operator without conforming to a
  trait, and `pow()` now takes `SIMD` values only. Nothing conformed to
  `CeilDivableRaising`, so the raising `ceildiv()` overload is removed too.

- Removed APIs that were deprecated in Mojo 1.1 or earlier. Use the
  replacement in each case:

  - The `UnsafePointer` alias and the `std.memory.unsafe_pointer` module that
    defined it. Use `Pointer`.

  - The `Pointer` methods `load()`, `store()`, `strided_load()`,
    `strided_store()`, `gather()`, `scatter()`, `free()`, `bitcast()`,
    `address_space_cast()`, and `take_pointee()`. Use their `unsafe_`-prefixed
    spellings, such as `unsafe_load()`.

  - `Pointer.mut_cast()`. Spell the mutability in the pointer type
    (`ImmPointer` or `MutPointer`), or use `unsafe_mut_cast()`.

  - Subtracting an integer from a `Pointer` and the `+=` and `-=` operators.
    Use `unsafe_offset()`. Adding an integer and positional indexing such as
    `ptr[i]` remain deprecated.

  - `memmove()`. Use `unsafe_memmove()`.

  - The `alloc[T](count)` overload. Use `alloc(Layout[T](length=count))`,
    which returns an `Allocation`, or `unsafe_alloc[T](count)` for a raw
    pointer.

  - `is_trivially_movable()`, `is_trivially_copyable()`, and
    `is_trivially_deletable()`. Use `IsTriviallyMovable[T]`,
    `IsTriviallyCopyable[T]`, and `IsTriviallyDeinitable[T]` from
    `std.traits`.

  - The `CStringSlice` alias and `as_c_string_slice()` on `String`,
    `StringSlice`, and `StringLiteral`. Use `CStringSpan` and
    `as_c_string_span()`.

  - `unsafe_ptr()` on `ArcPointer`, `OwnedPointer`, `CStringSpan`, and
    `StringLiteral`. Use `ptr()`.

  - `String.capacity()` and `String.reserve()`. Use `capacity_bytes()` and
    `reserve_bytes()`.

  - `Tuple.element_types` and `VariadicPack.element_types`. Use `Ts`.

  - The `AnyType` and `Deinitable` aliases in `std.builtin.anytype`, and the
    `Movable`, `Copyable`, and `ImplicitlyCopyable` aliases in
    `std.builtin.value`. These traits live in `std.traits` and are still
    exported from the prelude.

## Fixed

- `SIMD.slice()` no longer passes `llvm.vector.extract` an offset that is not
  a multiple of the output width, which LLVM rejects. A wide slice at such an
  offset, like `slice[16, offset=8]()` on a 32-element vector, now takes the
  element-wise path, and aligned slices narrower than the target's native
  vector width, such as the halves `split()` returns for small vectors, now
  use the intrinsic.

- Fixed rejection of negative integer and floating-point literals in `match`
  patterns.

- A `comptime if conforms_to(T, Deinitable)` guard now correctly refines `T`
  and deinits it after its last use.
