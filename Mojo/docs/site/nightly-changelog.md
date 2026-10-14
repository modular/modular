---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Added `match` / `case` pattern matching, including `comptime match` for
  compile-time subjects. Patterns include
  literals, contextual initializer lists such as `{}` and `{1, offset=2}`,
  explicit value patterns such as `== expected`, or-patterns (`|`), guards
  (`if`), `as` / bare name bindings, tuples, structs, and `EnumLike` types such
  as `Optional`, plus fixed and starred list patterns. An initializer-list
  pattern constructs a value of the subject type and compares it to the
  subject. An explicit value pattern compares
  against an existing value or expression without binding its name. Dynamic
  match also supports `var` / `ref` bindings; comptime match binds names as
  parameter values and rejects `var` / `ref`. Nested patterns can dig through
  several layers in one case—for example, matching an optional point at
  runtime or specializing a kernel path on a compile-time tile size:

  ```mojo
  def describe(p: Optional[Point]) -> String:
      match p:
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
      comptime match (m, n, k):
          case (16, 16, 16):
              # Hand-tuned 16³ path.
              ...
          case (m_tile, n_tile, k_tile) if m_tile * n_tile <= 256:
              # Small-tile path; bound sizes stay available as parameters.
              ...
          case _:
              # Generic fallback.
              ...

  # comptime match can also be used as a "type switch" since traits are
  # comparable to specific types with `==`.
  def classify_type[T: AnyType]() -> String:
      comptime match T:
          case == Int:
              return "integer"
          case == String:
              return "string"
          case _:
              return "other"
  ```

  Both forms warn on non-exhaustive `EnumLike` subjects (including `Bool` and
  tuples/structs of such types) and on unreachable or duplicate cases. Open
  subjects such as `Int` and `String` diagnose duplicate literal cases but are
  not required to be exhaustive. The Mojo language server offers an "Add
  missing cases" code action on any `match` whose cases leave values
  uncovered, including one with no cases yet, appending a `case` arm with a
  `pass` body for each missing value.

  `EnumLike` types can now set `_enum_is_exhaustive` to `False` when their
  known cases do not cover every possible value. This is useful for extensible
  domains such as error codes, where an operating system or library can add
  values that the type does not know about:

  ```mojo
  struct ErrorCode(EnumLike):
      comptime FileNotFound = Self(2)
      comptime PermissionDenied = Self(13)
      comptime OutOfMemory = Self(12)
      comptime TimedOut = Self(110)
      comptime Unavailable = Self(503)

      comptime _enum_is_exhaustive = False

      # EnumLike metadata and methods omitted.
      ...

  def describe(error: ErrorCode) -> String:
      match error:
          case .FileNotFound:
              return "file not found"
          case .PermissionDenied:
              return "permission denied"
          case .OutOfMemory:
              return "out of memory"
          case .TimedOut:
              return "operation timed out"
          case .Unavailable:
              return "service unavailable"
          case _:
              return "unknown error"
  ```

  Open enums preserve separate compilation between libraries and their
  clients. A library can add a known case without breaking previously compiled
  client matches when they compile against a new version of the package that
  adds a case they didn't know about.

- Mojo supports list destructuring assignments and list patterns for
  collections that conform to `Sized` and that are subscriptable. A single
  `*name` captures the elements between any leading and trailing patterns:

  ```mojo
  var [first, *middle, last] = values

  match values:
      case []:
          print("empty")
      case [1, *middle, last]:
          print(len(middle), last)
      case [first, *rest]:
          print(first, len(rest))
  ```

  The subject must conform to `Sized` and support integer and
  `ContiguousSlice` subscripting if a `*` pattern is used. A destructuring
  assignment with the wrong length raises an error, while a list pattern with
  the wrong length does not match.

- `if case` and `while case`: Conditional patterns are also available in
  `if`, `elif`, and `while` statements using `case pattern = value`. The value
  is evaluated once, and names bound by a successful pattern are available in
  the statement body. An `if case` pattern failure continues to the next
  `elif` or `else` arm. A `while case` pattern failure exits the loop normally
  and runs its `else` block, if present:

  ```mojo
  def describe_optional(value: Optional[Int]) -> String:
      if case .Some(item) = value:
          return String("some:", item)
      else:
          return "none"

  def consume():
      while case .Some(item) = get_next():
          process(item)
  ```

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

- The `async` and `await` keywords are renamed to `__async` and `__await` to
  mark async support as unstable. Write `__async def` and `__await expr`. The
  unprefixed spellings still work but emit a warning that async is unstable,
  with a fix-it to the prefixed spelling. The prefixed spelling suppresses the
  warning but doesn't make async stable.

## Library stabilizations

## Library changes

- `Defaultable` now provides a default `__init__()` implementation that
  default-constructs each field using reflection. A struct can conform to
  `Defaultable` without defining an initializer when all its fields also
  conform to `Defaultable`.

- The enum-behaving structs across the standard library, MAX kernels, and
  MAX runtime now conform to `EnumLike`, so they support enum-style pattern
  matching (`match` with `case .NAME`) and reflective case metadata:
  `Scope`, `FastMathFlag`, and `TestResult` in the standard library;
  `AllReduceAlgorithm`, `RasterOrder`, `MatmulSchedule`, `ReductionMode`,
  `GEMMKind`, `ThreadScope`, `FP6Format`, `MXFormat`, `Backend`,
  `AMDIGLPStrategy`, `MHASchedule`, `ScatterOobIndexStrategy`,
  `SchedulingStrategy`, `Phase`, `PhaseAction`, and `BlockPhase` in the
  kernels; and `TraceCategory`, `TraceLevel`, `LogLevel`, `CacheEviction`,
  `Fill`, and `Consistency` in the runtime and driver APIs.
  `FlashAttentionAlgorithm` conforms non-exhaustively: its `-1`
  "unspecified" value selects no named case, matching its runtime
  resolution semantics.

  ```mojo
  match AllReduceAlgorithm.ONE_STAGE:
      case .ONE_STAGE:
          launch_1stage()
      case .TWO_STAGE | .LAMPORT:
          launch_2stage()
  ```

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

- `SIMD` gained `to_endian()` and `from_endian()`, which convert an integer
  value to and from a given byte order.

  ```mojo
  var wire = UInt16(8080).to_endian[.big]()  # stored as [0x1F, 0x90]
  var port = UInt16.from_endian[.big](wire)  # 8080
  ```

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

- `FileDescriptor` now stores its underlying descriptor as a `c_int`, matching
  the C `int` type that the underlying descriptor APIs take. Integer literals
  keep constructing one unchanged, but passing an `Int` variable now requires
  an explicit `c_int()` conversion, and `.value` is a `c_int`.

- Introduced a new `CompilationTarget` public type, which is now used in
  most APIs that are parameterized on compilation target, such as `size_of()`,
  `align_of()`, `compile_info()`, and more.

  Convenience accessors are provided for querying the current host and
  accelerator targets:

  - `CompilationTarget.current()` — either the current host target, or
    accelerator target in an offload compilation.

  - `CompilationTarget.default_accelerator()` — the default accelerator target,
    either determined automatically or as specified by `--target-accelerator`.

  Additionally, `CompilationTarget` now provides predicate methods for checking
  the vendor identity of a given target. (The preexisting free functions with
  the same name are now implemented in terms of these new methods.) These
  include:

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

- `Coord` has a new `replace_dynamic[at](value)` method, the in-place
  counterpart to `replace`. It keeps the coordinate's type, so the element at
  `at` must already be dynamic, and `value` takes that element's own dtype.

  ```mojo
  var c = Coord(ComptimeInt[3](), Int64(0))
  c.replace_dynamic[1](Int64(7))  # Coord(ComptimeInt[3](), Int64(7))
  ```

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

- The TMA descriptor types `TMATensorTile`, `TMATensorTileArray`,
  `TMATensorTileIm2col`, and `SplitLastDimTMATensorTile` in
  `max.kernels.layout.tma_async` now take `tile_shape` and `desc_shape` as
  flat `Coord` values instead of `IndexList[rank]`, and the separate `rank`
  parameter is gone (rank is now derived as `tile_shape.rank`). Shape helpers
  `_default_desc_shape`, `_padded_shape`, `_ragged_shape`, and
  `_im2col_desc_shape` are now out-param functions returning concrete `Coord`
  types. Migrate callers by passing `coord[...]` values directly and reading
  shape dimensions via `.element_types[i].static_value`:

  ```mojo
  # was: TMATensorTile[dtype, rank=3, tile_shape=Index(BM,1,depth),
  #                     desc_shape=Index(BM,1,depth)]
  # now:
  var tile = TMATensorTile[
      dtype, coord[BM, 1, depth], coord[BM, 1, depth]
  ]()
  comptime BM_i = type_of(tile).tile_shape.element_types[0].static_value
  ```

  `QTMATile` split into `QTMATilePrefill` (rank-3, prefill/non-fused) and
  `QTMATileFused` (rank-4, decoding or fused-GQA); the rank choice must now
  be spelled at each use site. The companion factories `q_tma_prefill` and
  `q_tma_fused` replace the deleted `q_tma`. The `QTMATile` alias itself was
  removed outright after all in-tree consumers migrated; spell the rank
  choice explicitly with `QTMATilePrefill` or `QTMATileFused`.

- `std.math.scalb()` is now implemented in Mojo and runs on GPU as well as
  CPU. The `float64` type is not supported on Apple GPU. On targets that
  flush subnormals, subnormal inputs and subnormal results are handled;
  the results are bit-exact against IEEE targets.

- `std.utils.numerics.nextafter()` no longer calls the system libm and now
  runs on GPU as well as CPU. Both share one Mojo implementation that steps
  the IEEE-754 bit pattern directly, so the result is exact and matches the
  system libm for non-NaN inputs, and — since every ordering, zero, and
  equality predicate reads the raw bits — it is unaffected by targets that
  flush subnormal arithmetic. `std.math.ulp()`, which is defined through
  `nextafter`, follows it onto the GPU.
- `std.math.atan()` and `std.math.atan2()` no longer call the system libm and
  now run on GPU as well as CPU. On CPU their results come from the same Mojo
  implementations, so they can differ from the system libm in the last ulp.
  On targets that flush subnormals, subnormal inputs are handled and normal
  results are correct; subnormal results flush to zero.
- `std.math.gamma` no longer calls the system libm and now runs on GPU as
  well as CPU. On CPU its results come from the same Mojo implementation, so
  they can differ from the system libm in the last ulp. Subnormal inputs and
  results are also handled on targets that flush subnormals, such as Apple
  GPUs in float32.
- `float64` `expm1()` in `std.math` no longer calls the system libm and now
  runs on GPU as well as CPU. On CPU its result comes from the same Mojo
  implementation, so it can differ from the system libm in the last ulp. On
  targets that flush subnormals, subnormal inputs are handled; an input
  whose result would be subnormal is returned bit-exactly, so no
  subnormal result is ever flushed.
- `std.math.logb` no longer calls the system libm and now runs on GPU as well
  as CPU. It reads the exponent from the bit pattern, so subnormal inputs
  return their true exponent even on GPUs that flush subnormals.
- `std.math.lgamma()` no longer calls the system libm and now runs on GPU as
  well as CPU. On CPU its result comes from the same Mojo implementation, so
  it can differ from the system libm in the last ulp. It stays within about 2
  ulp for negative inputs, including next to the negative real zeros of
  `lgamma` (such as -2.457 and -2.748), where the result is near 0.
- The Bessel functions `j0`, `j1`, `y0`, and `y1` in `std.math` no longer
  call the system libm and now run on GPU as well as CPU. On CPU their
  results come from the same Mojo implementations, so they can differ from
  the system libm in the last ulp. For `float32`, relative error grows near
  the zeros of each function, while absolute error stays small there. On
  targets that flush subnormals, subnormal inputs and subnormal results
  are handled; only `j1` can produce a subnormal result in `float32`, and
  it is kept.
- `std.math.log10()` no longer calls the system libm or GPU intrinsics: CPU
  and every GPU now run the same Mojo implementation, accurate to within 1
  ulp on measured targets, and `float64` `log10()` now runs on AMD GPUs. On
  CPU, results can differ from the system libm in the last ulp, and results
  can differ between targets in the last ulp where the compiler contracts a
  multiply-add into an fma. On targets that flush subnormals, subnormal
  inputs are handled and normal results are correct; subnormal results
  flush to zero.
- `float64` `erfc()` no longer calls the system libm and now runs on GPU as
  well as CPU, and `float64` `erf()` now uses a `float64` approximation and
  is accurate to about 1 ulp. Both share one Mojo implementation with
  `float32` `erfc()`, so on CPU their results can differ from the system
  libm in the last ulp.
- `std.math.tan` no longer calls the system libm and now runs on GPU as well
  as CPU. On CPU its results come from the same Mojo implementation, so they
  can differ from the system libm in the last ulp, and CPU and GPU results
  can differ from each other in the last ulp. On targets that flush
  subnormals, subnormal inputs are handled and normal results are correct;
  subnormal results flush to zero.
- `cbrt()` no longer calls the system libm and now runs on GPU for every
  floating-point type, including `float64`. On CPU its results come from the
  same Mojo implementation, so they can differ from the system libm in the
  last ulp; `float32` results are within 0.505 ulp of the true root.

- `std.math.remainder()` no longer calls the system libm and now runs on GPU
  as well as CPU. It is exact, so it matches the system libm bit for bit. On
  targets that flush subnormals, subnormal inputs and subnormal results
  are handled; the results match the system libm bit for bit.
- `std.math.hypot` no longer calls the system libm and now runs on GPU as well
  as CPU. On CPU its result comes from the same Mojo implementation, so it can
  differ from the system libm in the last ulp.

- `float64` `log()` now uses double-precision coefficients and is within
  1 ulp of the correctly rounded result; it was previously off by up to about
  2e7 ulp. It runs the same Mojo code on CPU and GPU.
- `log1p()` now evaluates in the input's own precision with a Mojo
  implementation based on musl, shared by CPU and GPU, and is accurate to
  within 1 ulp for `float32` and `float64`. `float16` and `bfloat16` inputs
  are computed in `float32`. `float64` `log1p()` is no longer supported on
  Apple GPUs, which have no `float64` arithmetic.
- `atanh()` in `std.math` no longer calls the system libm for `float64`, so
  its `float64` form now runs on GPUs that have `float64` as well as on CPU.
  All floating-point types share one Mojo implementation on CPU and GPU, so
  `float32` GPU results, previously up to 16 ulp off through a
  target-dependent `log`, are now within 1 ulp of the CPU. Results can
  differ from the system libm, and between targets, in the last ulp. On
  targets that flush subnormals, subnormal inputs are handled and normal
  results are correct; subnormal results flush to zero.
- `std.math` `acosh` and `asinh` no longer call the system libm and now run
  on GPU as well as CPU. On CPU their results come from the same Mojo
  implementations, so they can differ from the system libm in the last ulp.

- `Bool`, `Scalar` (including `Int`), `String`, `StringSpan`, and
  `StringLiteral` now conform to `ConvertibleToPython`, so they can be passed
  to generic code bounded on that trait. The dedicated `PythonObject`
  constructors for these types are removed; implicit conversion still works
  through the single `ConvertibleToPython` constructor.
  `ConvertibleToPython` gains an associated `ConversionToPythonErrorType`
  (default `Never`), so conversions that cannot fail, such as `Bool` and
  `Scalar`, stay non-raising, while `String` conversions keep raising
  `Error`.

## Tooling changes

- The `mojo` compiler now uses jemalloc as its allocator on Linux. Compiling
  large GPU kernel files is roughly 10 to 13 percent faster, because the
  compiler spends far less system time in glibc's arena management.

- The Mojo language server now reports a value's address space as a semantic
  token modifier (`addressSpace0` through `addressSpace15`, matching the
  `AddressSpace` index) so editors can color shared, local, and constant memory
  differently. The address space is found in the value's type or in a type
  nested in its parameters, so an `Array` of shared pointers is marked too.

## Removed

- Removed `has_nvidia_gpu_accelerator()`, `has_amd_gpu_accelerator()` and
  `has_apple_gpu_accelerator()` from `std.sys`, including the sub-architecture
  overloads of the NVIDIA variant. Ask the target itself instead, using the
  new `CompilationTarget` predicate methods:

  ```mojo
  # Before
  from std.sys import has_nvidia_gpu_accelerator

  comptime if has_nvidia_gpu_accelerator():
      ...

  # After
  from std.sys import default_accelerator

  comptime if default_accelerator().is_nvidia_gpu():
      ...
  ```

  Code that already holds a device context can check its target directly,
  for example `ctx.target.is_nvidia_gpu()`.

- Removed `sum()` from the `CoordLike` trait and from its implementations
  (`Coord`, `ComptimeInt` and the `All` marker). Nothing called it: a
  coordinate's elements are extents and indices, so adding them together
  has no meaning the way `product()` does, where the result is the number
  of elements a shape describes. Use `product()` for that, or iterate the
  elements and add them yourself if you really want a sum.

## Fixed

- `c_char` now reflects the target's actual `char` signedness: unsigned by
  default on Linux running on ARM (AAPCS64), signed everywhere else Mojo
  currently targets (x86-64 Linux, macOS on Apple silicon). Previously
  `c_char` was always signed, which could bind an FFI declaration to the
  wrong C type on Linux/ARM.

- `atof()` now rounds large mantissas correctly. Values whose significant
  digits exceed 53 bits, such as `atof("123456789012345678")`, could round
  to the wrong neighbouring double, and exact ties such as
  `atof("4503599627370497.5")` did not round to even.

- Splitting on an empty separator no longer puts the trailing empty slice out
  of bounds.

- `atof()` accepts numbers with more than 19 significant digits, which
  previously raised "The number is too long". Malformed inputs such as
  `"1..2"`, `"1e5.0"`, or `"1e--5"`, which the old parser silently misread
  by skipping the offending characters, now raise.

- Struct layouts that recurse only through a parametric indirection now
  compile. A field such as `Optional[List[Pointer[Node, origin]]]` or
  `Tuple[Pointer[Node, origin], Int]` inside `struct Node` was spuriously
  rejected with `struct has recursive reference to itself`, despite the
  recursion being behind a pointer.

- Passing `x^` to an `imm` or `ref` argument now warns that the move has
  no effect instead of it being silently ignored.
- On macOS, `scalb()` now returns NaN for a non-integral exponent, as it does
  on Linux.
- `float32` `expm1()` no longer overflows early for inputs just below 88.7,
  and it now returns `-0.0` for `-0.0` and `inf` for `inf`.
- `float32` `erfc()` is now accurate in its tail, for large positive inputs,
  and returns subnormal results instead of flushing them to zero. On targets
  that flush subnormals, subnormal inputs are handled and normal results are
  correct; subnormal results flush to zero.
- `cbrt()` of `inf` and `-inf` now returns `inf` and `-inf` instead of
  `Float32.MAX` and `-Float32.MAX`, and `float32` `cbrt()` is correct near
  `Float32.MAX` and for subnormal inputs. On targets that flush subnormals,
  subnormal inputs are handled and normal results are correct; subnormal
  results flush to zero, though `cbrt()` never produces one.
- `log()`, `log2()`, and `log1p()` of `inf` now return `inf` instead of `nan`.
- `float64` `log1p()` is now accurate for inputs whose `1 + x` falls in
  `[sqrt(2)/2, sqrt(2))`, where it was off by up to about 7e9 ulp.
