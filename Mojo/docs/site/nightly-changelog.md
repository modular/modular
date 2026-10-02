---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Added experimental `__match` / `case` pattern matching for early testing,
  including `comptime __match` for compile-time subjects. Patterns include
  literals, contextual initializer lists such as `{}` and `{1, offset=2}`,
  or-patterns (`|`), guards (`if`), `as` / bare name bindings, tuples, structs,
  and `EnumLike` types such as `Optional`. An initializer-list pattern
  constructs a value of the subject type and compares it to the subject.
  Dynamic match also supports `var` / `ref` bindings; comptime match binds
  names as parameter values and rejects `var` / `ref`. Nested patterns can dig
  through several layers in one case—for example, matching an optional point
  at runtime or specializing a kernel path on a compile-time tile size:

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

  # comptime match is guaranteed evaluated at compile time.  The subject must be
  # a parameter and the bound names are also parameters. var and ref specifiers
  # are not allowed in comptime match, because they are not meaningful.
  def matmul_tile[m: Int, n: Int, k: Int](...):
      comptime __match (m, n, k):
          case (16, 16, 16):
              # Hand-tuned 16³ path.
              ...
          case (m_tile, n_tile, k_tile) if m_tile * n_tile <= 256:
              # Small-tile path; bound sizes stay available as parameters.
              ...
          case _:
              # Generic fallback.
              ...
  ```

  Both forms warn on non-exhaustive `EnumLike` subjects (including `Bool` and
  tuples/structs of such types) and on unreachable or duplicate cases. Open
  subjects such as `Int` and `String` diagnose duplicate literal cases but are
  not required to be exhaustive.

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

- `Tuple` gained `first_of[T]()`, which returns a reference to the first
  element of type `T`. The lookup is resolved at compile time, and it is a
  compile-time error if the tuple has no element of that type:

  ```mojo
  var t = (1, String("two"), 3.0)
  t.first_of[String]() += "!"
  ```

- `SIMD.from_bytes()` and `SIMD.as_bytes()` now take an `endian` parameter of
  the new `Endian` type (`Endian.big` or `Endian.little`) instead of the
  `big_endian` `Bool`:

  ```mojo
  var port = UInt16.from_bytes[endian=.big](bytes)  # was: big_endian=True
  ```

  The default is still the target's byte order, which `Endian.native()`
  returns. `is_big_endian()` and `is_little_endian()` are deprecated; compare
  `Endian.native()` with `.big` or `.little` instead.

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

  Additionally, `CompilationTarget` provides predicate methods for checking the
  vendor identity of a given target. (The preexisting free functions with the
  same name are now implemented in terms of these new methods.) These include:

  - `.is_nvidia_gpu()` for checking whether a compilation target is an NVIDIA
    GPU.
  - `.is_amd_gpu()` and `.is_apple_gpu()` provide the equivalent checks for AMD
    and Apple GPUs respectively.

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

- The default `Writable` implementation now formats `EnumLike` types by their
  active case rather than their fields. A case prints as
  `TypeName.case(payload)`, or as `TypeName.case` when it has no payload
  (`NoneType`):

  ```mojo
  print(Shape.circle(3))       # Shape.circle(3)
  print(repr(Shape.circle(3))) # Shape.circle(Int(3))
  print(Shape.empty())         # Shape.empty
  ```

  Types that implement their own `write_to()` or `write_repr_to()`, such as
  `Optional` and `Bool`, are unaffected.

- `Tuple.consume_elements()`, `Tuple.deinit_with()`, and
  `VariadicPack.consume_elements()` now take the element handler as a runtime
  closure argument instead of a compile-time `capturing` parameter, matching
  `List.deinit_with()` and `VariadicList.consume_elements()`:

  ```mojo
  def handler[idx: Int](var elt: t.Ts[idx]) {mut collected}:
      collected[idx] = elt.data

  t^.consume_elements(handler)  # was: t^.consume_elements[handler]()
  ```

- `SIMD.reduce()` now takes a closure reduction function as a runtime
  argument instead of a compile-time `capturing` parameter. The `thin`
  function-pointer form is unchanged:

  ```mojo
  def add[width: SIMDLength](
      lhs: SIMD[DType.int32, width], rhs: SIMD[DType.int32, width]
  ) -> SIMD[DType.int32, width]:
      return lhs + rhs

  v.reduce(add)          # was: v.reduce[add]()
  v.reduce[2](add)       # was: v.reduce[add, 2]()
  ```

## Tooling changes

## Removed

- Removed `sum()` from the `CoordLike` trait and from its implementations
  (`Coord`, `ComptimeInt` and the `All` marker). Nothing called it: a
  coordinate's elements are extents and indices, so adding them together
  has no meaning the way `product()` does, where the result is the number
  of elements a shape describes. Use `product()` for that, or iterate the
  elements and add them yourself if you really want a sum.

## Fixed

- Splitting on an empty separator no longer puts the trailing empty slice out
  of bounds.
