---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Added experimental `__match` / `case` pattern matching for early testing.
  Patterns include literals, or-patterns (`|`), guards (`if`), `var` / `ref` /
  `as` bindings, tuples, structs, and `EnumLike` types such as `Optional`.
  Nested patterns can dig through several layers in one case — for example
  matching an optional point without a nested `match`:

  ```mojo
  def describe(p: Optional[Point]) -> String:
      __match p:
      case .Some(Point(x=0, y=0)):
          return "origin"
      case .Some(Point(x=var x, y=0)) | .Some(Point(x=0, y=var x)):
          return String("axis:", x)
      case .Some(Point(x=x, y=y)) if x == y:
          return "diagonal"
      case .None:
          return "missing"
      case _:
          return "unreachable"
  ```

  Exhaustiveness is not checked yet for enums or `Bool`, so you may still need
  a redundant `case _` even when the other cases appear complete. The spelling
  remains `__match` while the feature is experimental.

- The message on a `where` clause can now be written
  `where <condition> else "<message>"`, as the preferred alternative to the
  existing `where (<condition>, "<message>")`, which will be deprecated over
  time.

  ```mojo
  def foo[sc: Int]() where sc > 1 else "scaling factor must be greater than 1":
      ...

  struct Box[T: Deinitable](
      Marker where conforms_to(T, Marker) else "Box[T] is a Marker only when T is",
  ):
      ...
  ```

- A struct can now opt out of a trait by writing `not <Trait>` in its
  conformance list. It is the preferred alternative to the existing
  `<Trait> where False`, which will continue to work:

  ```mojo
  struct Handle(not Movable, Writable):
      ...
  ```

  An opt-out can record why, with the same `else "<message>"` spelling a
  `where` clause uses:

  ```mojo
  struct Handle(not Movable else "a Handle is pinned to the port it opened"):
      ...
  ```

## Language changes

## Library stabilizations

## Library changes

- Hashing a byte sequence is now spelled `hash_bytes()`, which takes an
  `ImmSpan[Byte]`. The pointer-and-length `hash()` overload is deprecated:

  ```mojo
  var digest = hash_bytes(text.as_bytes())  # was: hash(text.ptr(), text.byte_length())
  ```

  `hash()` keeps its `Hashable` meaning, so `hash(my_list)` still hashes a list
  element-by-element while `hash_bytes(my_list)` hashes its bytes.

- `Coord.product()` has a new parameterized overload, `product[T: DType]()`,
  which accumulates and returns the product at `T` rather than at
  `Coord.DTYPE`:

  ```mojo
  var c = Coord(Idx[4], Int(8), Int32(3))
  var n = c.product[DType.uint32]()  # UInt32(96)
  ```

  The unparameterized `product()` is unchanged and still returns
  `Scalar[Coord.DTYPE]`. A `T` too narrow to hold the result wraps rather
  than widening, so a caller that picks one owns the overflow.

- `StringSpan` now only provides immutable byte access. `MutStringSpan` and
  `MutStringSlice` have been removed, and the `mut` parameter on `StringSpan`
  has been removed.

- Introduced a new `CompilationTarget` public type, which is now used in
  most APIs that are parameterized on compilation target, such as `size_of()`,
  `align_of()`, `compile_info()`, and more.

  Convenience accessors are provided for querying the current host and
  accelerator targets:

  - `CompilationTarget.current()` — either the current host target, or
    accelerator target in an offload compilation.

  - `CompilationTarget.current_accelerator()` — the default accelerator target,
    either determined automatically or as specified by `--target-accelerator`.

- Added new `TargetAccelerator` type for representing accelerator metadata at
  compile time, combining `GPUInfo` and `CompilationTarget` values.

  Target constants like `B200` or `MI355X` have changed from being `GPUInfo`
  instances to `TargetAccelerator`.

  Use `.gpu_info` to access the prior device metadata, or `.target` for the
  compilation target.

- `Layout` now carries its alignment as a keyword-only parameter of
  the new `Alignment` type, instead of storing it as a runtime field. It
  defaults to the element type's natural alignment, so `Layout[T](count=n)` is
  unchanged. Build an `Alignment` with `Alignment.of[T]()` for a type's natural
  alignment or `Alignment.of_bytes[n]()` for an explicit byte count.

  `Allocation` and `ManagedAllocation` carry the same parameter.
  `ThinAllocation` deliberately does not: it records only the element type, so
  the alignment travels with the `Layout` you supply to `unsafe_with_layout`.

  ```mojo
  var layout = Layout[Int32, alignment = .of_bytes[64]()](count=8)
  var thin = alloc(layout).into_thin()
  dealloc(thin^.unsafe_with_layout(layout))
  ```

  Only compile-time alignments are supported for now. This makes the common case
  (natural type alignment) simple - so the `Layout` and `Allocation` type don't
  pay the cost of holding an extra `Int` field. Dynamic (runtime) alignment will
  eventually be supported after some more design considerations.

- `Span` has a new `unsafe_deinit_elements()` method, which destroys every
  element in place and leaves the memory uninitialized.

- `Span` has new methods for initializing a span of `MaybeUninit[T]` elements
  and viewing the result as a span of `T`:

  - `unsafe_init_with()` initializes each element with the result of calling a
    function with that element's index.
  - `unsafe_init_copy_from()` copies from another span.
    `unsafe_init_move_from()` moves out of another span.
  - `unsafe_assume_init()` reinterprets the span as initialized, for memory
    that was initialized some other way.

## Tooling changes

## Removed

- Removed `sum()` from the `CoordLike` trait and from its implementations
  (`Coord`, `ComptimeInt` and the `All` marker). Nothing called it: a
  coordinate's elements are extents and indices, so adding them together
  has no meaning the way `product()` does, where the result is the number
  of elements a shape describes. Use `product()` for that, or iterate the
  elements and add them yourself if you really want a sum.

## Fixed
