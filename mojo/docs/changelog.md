# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

- Mojo now supports `comptime if` and `comptime for` as the preferred syntax
  for compile-time conditional and loop constructs, replacing the legacy
  `@parameter if` and `@parameter for` decorator forms:

  ```mojo
  # Old syntax (still accepted, deprecated in a future release)
  @parameter
  if some_condition:
      ...

  @parameter
  for i in range(10):
      ...

  # New syntax
  comptime if some_condition:
      ...

  comptime for i in range(10):
      ...
  ```

  Both syntaxes are accepted in this release. The `@parameter` forms will be
  deprecated soon, and the parser will generate a warning and a fixit suggestion
  to migrate to the new syntax.

- `@register_passable("trivial")` is now deprecated,
   conform to `TrivialRegisterPassable` trait instead.
   The decorator will be removed after next release.

- `@register_passable` is now deprecated,
   conform to `RegisterPassable` trait instead.
   The decorator will be removed after next release.

- Mojo now supports more flexible default arguments and parameters, which can
  mismatch on declared type when their types are parametric.  This allows
  inferring parameters from these when they are used as a default value, for
  example:

  ```mojo
    fn take_string_slice[o: ImmOrigin](str: StringSlice[o] = ""): ...
    fn use_it():
      take_string_slice() # Ok, defaults to empty string, inferring "o".
      # Explicit calls also work of course.
      take_string_slice(StaticString("hello"))

    # Default value is checked for validity at the call site.
    fn defaultArgumentBadType2[T: AnyType](a: T = 1.0): pass
    fn callDefaultArgumentBadType2():
        # Ok!
        defaultArgumentBadType2[Float64]()
        # error: value passed to 'a' cannot be converted from 'FloatLiteral[1]' to 'Int'
        defaultArgumentBadType2[Int]()
  ```

- Mojo now supports the `@align(N)` decorator to specify minimum alignment for
  structs, similar to C++'s `alignas` and Rust's `#[repr(align(N))]`. The value
  `N` must be a positive power of 2 and specifies the minimum alignment in
  bytes. The actual alignment will be `max(N, natural_alignment)` - you cannot
  use `@align` to reduce alignment below the struct's natural alignment. For
  example, `@align(1)` on a struct containing an `Int` (8-byte aligned) will
  emit a warning and the struct will remain 8-byte aligned.

  ```mojo
  from sys import align_of

  @align(64)
  struct CacheAligned:
      var data: Int

  fn main():
      print(align_of[CacheAligned]())  # Prints 64
  ```

  Both stack and heap allocations respect `@align`.

  The alignment value can also be a struct parameter, enabling generic
  aligned types:

  ```mojo
  @align(Self.alignment)
  struct AlignedBuffer[alignment: Int]:
      var data: Int

  fn main():
      print(align_of[AlignedBuffer[64]]())   # Prints 64
      print(align_of[AlignedBuffer[128]]())  # Prints 128
  ```

- Mojo now allows move constructors to take their "take" argument as either
  `deinit` or `var`. `deinit` should be generally preferred, but `var` can be
  useful in unusual cases where you want C++-like behavior where the destructor
  still runs on the value after the move constructor is done with it.

  ```mojo
  fn __init__(out self, *, var take: Self):
    self.data = munge(take.data)
    # take.__del__() runs here.
  ```

### Language changes

- `**_` and `*_` are no longer supported in parameter binding lists. Use a more
  concise `...` to unbind any unspecified parameter explicitly.

- As part of "init unification", the `__moveinit__` and `__copyinit__` methods
  [are now
  renamed](https://github.com/modular/modular/blob/main/mojo/proposals/remove_move_and_copy_init.md)
  to `__init__(*, take=)` and `__init__(*, copy=)` respectively.  Mojo now
  accepts these names for initializers, but also supports the legacy
  `__moveinit__` and `__copyinit__` names as well (for now).  However, the
  argument name for these legacy methods must now be named `take` and `copy`
  respectively.  Please move to the more modern names.

- As part of move unification, the `__moveinit__is_trivial` and
  `__copyinit__is_trivial` members of `Movable` and `Copyable` have been renamed
  to `__move_ctor_is_trivial` and `__copy_ctor_is_trivial` respectively.

- Slice literals in subscripts has changed to be more similar to collection
  literals. They now pass an empty tuple as a required `__slice_literal__`
  keyword argument to disambiguate slices. If you have defined your own range
  types, please add a `__slice_literal__: () = ()` argument to their
  constructors.

- `trait` declarations no longer automatically inherit from
  `ImplicitlyDestructible`. `struct` declarations are not changed, and continue
  to inherit from `ImplicitlyDestructible`.

  Previously, the `@explicit_destroy` annotation was required to opt-out of
  `ImplicitlyDestructible` conformance. Now, if a trait's usage depends on
  implicit destructibility, it must opt-in explicitly:

  ```mojo
  # Before
  trait Foo:
      ...

  # After:
  trait Foo(ImplicitlyDestructible):
      ...
  ```

  Conversely, if a trait wanted to support non-implicitly-destructible types,
  it no longer needs to be annotated with `@explicit_destroy`:

  ```mojo
  # Before
  @explicit_destroy
  trait Foo:
      ...

  # After
  trait Foo:
      ...
  ```

  Making `struct` continue to inherit from `ImplicitlyDestructible` and not
  `trait` is intended to balance usability and familiarity in the common case,
  with the need to foster broad Mojo ecosystem support for explicitly destroyed
  types.

  It's not a problem if the majority of `struct` types are
  `ImplicitlyDestructible` in practice. However, if many ecosystem libraries are
  written with unnecessary `ImplicitlyDestructible` bounds, that would hamper
  the usability of any individual `struct` type that opts-in to being explicitly
  destroyed.

  Libraries with generic algorithms and types should be written to accommodate
  linear types. Making `ImplicitlyDestructible` opt-in for traits
  encourages a default stance of support, with specific types and functions
  only opting-in to the narrower `ImplicitlyDestructible` requirement if they
  truly need it.

  The majority of generic algorithms that take their inputs by reference should
  not be affected.

- Unstable `__comptime_assert` syntax is now finalized as `comptime assert`. A
  deprecation warning is emitted with a fixit for the old syntax.

- `comptime assert` no longer errors on always false conditions. The assertion
  will only trigger if its parent scope is concretized.

### Library changes

- `Set.pop()` now uses `Dict.popitem()` directly, avoiding a redundant rehash.
  Order changes from FIFO to LIFO, matching Python's unordered `set.pop()`.

- `Set.__gt__()` and `Set.__lt__()` now use an O(1) `len()` check plus a single
  `issubset()` traversal instead of two full traversals.

- Added `uninit_move_n()`, `uninit_copy_n()`, and `destroy_n()` functions to the
  `memory` module for efficient bulk memory operations. These functions handle
  moving, copying, and destroying multiple values in contiguous memory, with
  automatic optimization for trivial types using `memcpy`. They encapsulate the
  common pattern of checking `__move_ctor_is_trivial`, `__copy_ctor_is_trivial`,
  or `__del__is_trivial` and selecting the appropriate implementation. The `List`
  collection now uses these functions internally for improved code clarity and
  maintainability.

- `Dict` internals have been replaced with a Swiss Table implementation using
  SIMD group probing for lookups. This improves lookup, insertion, and deletion
  performance — especially when looking up keys not in the dict — while
  increasing the load factor from 2/3 to 7/8 for better memory efficiency.
  The `power_of_two_initial_capacity` keyword argument has been renamed to
  `capacity` and now accepts any positive integer (it is rounded up to the
  next power of two internally, minimum 16).

- Implicit conversions from `Int` to `SIMD` are now deprecated, and will be
  removed in a future version of Mojo. This includes deprecating converions from
  `Int` to specific `SIMD` scalar types like `Int8` or `Float32`.

  > Note: The experimental `mojo build --experimental-fixit` command may be useful
  in assiting with migrating your code to reflect this change.

  Code currently relying on implicit conversions, e.g.:

  ```mojo
  fn foo(arg: Int) -> Float32:
      return arg
  ```

  should be changed to use an explicit conversion:

  ```mojo
  fn foo(arg: Int) -> Float32:
      return Float32(arg)
  ```

  Occasionally, when an implicit conversion was happening from a
  scalar `Int` to a wider `SIMD` value, the compiler will suggest a
  change to make the conversion explicit that is somewhat verbose,
  performing both the type conversion and the "splat" in a single step:

  ```diff
  var bits: SIMD[DType.uint8, 16] = ...
  - var y = bits >> (j * 4)
  + var y = bits >> SIMD[DType.uint8, 16](j * 4)
  ```

  a simpler change is to make the type conversion explicit, and
  let the "splat" continue to happen implicitly (which will remain supported):

  ```diff
  var bits: SIMD[DType.uint8, 16] = ...
  - var y = bits >> (j * 4)
  + var y = bits >> UInt8(j * 4)
  ```

  A similar overly verbose suggestion can be emitted relating
  to `LayoutTensor.element_type`, where the compiler may suggest
  a change of the form:

  ```diff
  - tensor[i] = i * l
  + tensor[i] = LayoutTensor[DType.uint32, Layout(IntTuple(-1)), stack].element_type(i * l)
  ```

  where a simpler change is sufficient:

  ```diff
  - tensor[i] = i * l
  + tensor[i] = UInt32(i * l)

  # Note: If the tensors `dtype` is not statically known, this works as well:
  + tensor[i] = Scalar[tensor.dtype](i * l)
  ```

- The `builtin.math` module has been merged into `math`. The traits `Absable`,
  `DivModable`, `Powable`, `Roundable` and functions `abs()`, `divmod()`,
  `max()`, `min()`, `pow()`, `round()` are now part of the `math` module and
  continue to be available in the prelude. Code that explicitly imported from
  `builtin.math` should update to import from `math` instead.

- The `ffi` module is now a top-level module in the standard library, rather
  than being nested under `sys`. This improves discoverability of FFI
  functionality. Update your imports from `from sys.ffi import ...` to
  `from ffi import ...`.

- Added `UnsafeUnion[*Ts]`, a C-style untagged union type for FFI
  interoperability. Unlike `Variant`, `UnsafeUnion` does not track which type
  is stored (no discriminant), making it suitable for interfacing with C unions
  and low-level type punning. The memory layout exactly matches C unions (size
  is max of elements, alignment is max of elements).

  All element types must have trivial copy, move, and destroy operations,
  matching C union semantics where types don't have constructors or destructors.

  Construction is explicit (no implicit conversion) to emphasize the unsafe
  nature of this type. All accessor methods are prefixed with `unsafe_` to make
  it clear that these operations are unsafe.

  ```mojo
  from ffi import UnsafeUnion

  # Define a union that can hold Int32 or Float32
  comptime IntOrFloat = UnsafeUnion[Int32, Float32]

  var u = IntOrFloat(Int32(42))
  print(u.unsafe_get[Int32]())  # => 42
  print(u)  # => UnsafeUnion[Int32, Float32](size=4, align=4)

  # Type punning (reinterpreting bits)
  var u2 = IntOrFloat(Float32(1.0))
  print(u2.unsafe_get[Int32]())  # => 1065353216 (IEEE 754 bits)
  ```

- The `itertools` module now includes three new iterator combinators:
  - `cycle(iterable)`: Creates an iterator that cycles through elements
    indefinitely
  - `take_while[predicate](iterable)`: Yields elements while the predicate
    returns True
  - `drop_while[predicate](iterable)`: Drops elements while the predicate
    returns True, then yields the rest

- Math functions in `std.math` (`exp`, `exp2`, `log2`, `erf`, `tanh`, `sin`,
  `cos`, `tan`, `acos`, `asin`, `atan`, `atan2`, `acosh`, `asinh`, `atanh`,
  `cosh`, `sinh`, `expm1`, `log10`, `log1p`, `logb`, `cbrt`, `erfc`, `j0`,
  `j1`, `y0`, `y1`) now use `where dtype.is_floating_point()` clauses on their
  signatures instead of `__comptime_assert` checks in their bodies.
  This provides better compile-time error messages at the call site. Callers
  using these functions with generic `dtype` parameters may need to add
  evidence proving (either a `where` clause or `__comptime_assert`) that their
  type is floating point.

- Many kernels in `nn` have been migrated to use `TileTensor`. We will have
  more documentation on `TileTensor` and its uses over the coming weeks.

- `InlineArray` now requires explicitly using literals for construction. E.g.

  ```Mojo
  var a: InlineArray[UInt8, 4] = [1, 2, 3, 4]
  # instead of InlineArray[UInt8, 4](1, 2, 3, 4)
  ```

- The following types now conform to `Writable` and have custom implementations
  of `write_to` and `write_repr_to`.
  - `Tuple`
  - `Variant`
  - `Optional`

- The `testing` module now provides `assert_equal` and `assert_not_equal`
  overloads for `Tuple`, enabling direct tuple-to-tuple comparisons in tests
  instead of element-by-element assertions. Element types must conform to
  `Equatable & Writable`.

- The `__reversed__()` method on `String`, `StringSlice`, and `StringLiteral`
  has been deprecated in favor of the new `codepoints_reversed()` method. The
  new method name makes it explicit that iteration is over Unicode codepoints
  in reverse order, maintaining consistency with the existing `codepoints()`
  and `codepoint_slices()` methods. The deprecated `__reversed__()` methods
  will continue to work but will emit deprecation warnings.

- The `Origin` struct now takes the underlying MLIR origin as a parameter
  instead of storing it. This follows the design of `IntLiteral` and related
  types, and fixes some memory safety problems.

- The `StringSlice` constructor from `String` now propagates mutability. If you
  have a mutable reference to a `String`, `StringSlice(str)` returns a mutable
  `StringSlice`. The `String.as_string_slice()` method is now deprecated in
  favor of the `StringSlice(str)` constructor, and `String.as_string_slice_mut()`
  has been removed.

- `String.ljust`, `String.rjust`, and `String.center` have been renamed to
  `String.ascii_ljust`, `String.ascii_rjust`, and `String.ascii_center`.
  Likewise for their mequivalents on `StringSlice` and `StaticString`

- `String.resize` will now panic if the new length would truncate a codepoint.
  Previously it would result in a string with invalid UTF-8.

- `String.resize` will now panic if `fill_byte` is >=128. Previously it would
  create invalid UTF-8.

- Subscripting into `String` and `StringSlice` will now panic if the index falls
  in the middle of a UTF-8 encoded code-point. Previously they would return invalid
  UTF-8. This panic is unconditional. Use `.as_bytes()[...]` if you really want
  the previous behavior.

- `StringSlice[byte=]` subscripting now returns a `StringSlice` instead of a `String`,
  This is consistent with range-based subscripting.

- Subscripting `String` and `StringSlice` by byte position will
  now return an entire Unicode codepoint. Previously it would
  return a single byte, and produce invalid UTF-8 if the index fell on
  the starting byte of a multi-byte codepoint.
- The following types now correctly implement `write_repr_to`
  - `List`, `Set`

- `assert_equal` and `assert_not_equal` now work with types implementing
  `Writable`.

- All traits and structs with `@register_passable("trivial")` decorator are now
  extending `TrivialRegisterPassable` trait. The decorator is removed from them.

- `String`, `StringSlice`, and `StringLiteral`'s `.format()` method now require
  their arguments to be `Writable`.

- Formatting compile-time format strings (`StringLiteral`s) no longer allocates
  memory! It uses `global_constant` to store what would be heap allocated
  parsed formatting data.

- `Int.__truediv__` now performs truncating integer division, returning `Int`
  instead of the previously deprecated `Float64`. Use explicit `Float64` casts
  for floating-point division.

- Documentation for `SIMD.__round__` now clarifies the pre-existing behavior
  that ties are rounded to the nearest even, not away from zero.

- A new `utils/type_functions` module was added for holding type functions. Type
  functions are `comptime` declarations that produce a type from compile-time
  parameter inputs. Unlike regular `fn` functions which accept and return runtime
  values, type functions operate entirely at compile time -- they take `comptime`
  parameters and evaluate to a type, with no runtime component.
  - `ConditionalType` - A type function that conditionally selects between two
    types based on a compile-time boolean condition. It is the type-level
    equivalent of the ternary conditional expression `Then if If else Else`.

- `UnsafeMaybeUninit` has been renamed as such, and it's methods have had their
  names updated to reflect the `init` name. It also now exposes a `zeroed()` method
  to get zeroed out uninitialized memory. It also no longer calls `abort()` when
  being copied or moved, allowing for more practical uses.

- `Int.write_padded` now accounts for a negative sign when calculating the
  width, resulting in a consistent width regardless of sign:

  ```Mojo
  Int(1).write_padded(s, 4)  # writes "   1"
  Int(-1).write_padded(s, 4) # writes "  -1"
  ```

- `SIMD` now has a `write_padded` method for integral `DType`s, matching the
  behaviour of `Int.write_padded`. The padding is added elementwise. Unlike
  `SIMD` regular printing, there are no spaces added between elements by
  default:

  ```Mojo
  Int32(1).write_padded(s, 4)  # writes "   1"
  Int32(-1).write_padded(s, 4) # writes "  -1"

  # "[   1,  -1,   0,1234]"
  SIMD[DType.int32, 4](1,-1,0,1234).write_padded(s, 4)
  # "[255,255]"
  SIMD[DType.uint8, 2](255).write_padded(s, 1)
  ```

- `SIMD`'s safe division operators (`__floordiv__`, `__mod__`, `__divmod__`)
  now return safe results in an elementwise fashion. They previously returned a
  zero result if *any* element of the divisor was zero:

  ```Mojo
  comptime Int32x2 = SIMD[DType.int32, 2]

  var a = Int32x2(99, 99)
  var b = Int32x2(4, 0)
  # Now returns [24, 0]
  # Previously returned [0, 0]
  print(a.__floordiv__(b))
  ```

### Tooling changes

- The Mojo compiler now accepts conjoined `-D` options in addition to the
  non-conjoined form as before. Now, both `-Dfoo` and `-D foo` are accepted.

- `mojo build` now supports several `--print-*` options for discovering target
  configuration and supported architectures:
  - `--print-effective-target`: Shows the resolved target configuration after
    processing all command-line flags.
  - `--print-supported-targets`: Lists all available LLVM target architectures.
  - `--print-supported-cpus`: Lists valid CPU names for a given target triple
    (requires `--target-triple`).
  - `--print-supported-accelerators`: Lists all supported GPU and accelerator
    architectures (NVIDIA, AMD, Apple Metal).

- `mojo format` now only formats Mojo files (`.mojo`, `.🔥`) by default when
  run on a directory. Previously it would also format Python files, which
  conflicted with Python-specific formatters in pre-commit hooks. Users who
  want to format Python files can use `mblack` directly.

### ❌ Removed

- The `owned` keyword has been removed. Use `var` for parameters or `deinit`
  for `__moveinit__`/`__del__` arguments as appropriate.

- `Dict.EMPTY` and `Dict.REMOVED` comptime aliases have been removed. These
  were internal implementation details of the old hash table design.

- The `@nonmaterializable` decorator has been renamed to `@__nonmaterializable`.
  This decorator should not be used outside the standard library, and might be
  removed in a future release.

### 🛠️ Fixed

- [Issue #5845](https://github.com/modular/modular/issues/5845): Functions
  raising custom type with conversion fails when returning StringSlice

- [Issue #5722](https://github.com/modular/modular/issues/5722): `__del__`
  incorrectly runs when `__init__` raises before all fields are initialized.

- [Issue #5875](https://github.com/modular/modular/issues/5875): Storing
  `SIMD[DType.bool, N]` with width > 1 to a pointer and reading back
  element-wise now returns correct values.

- `StringSlice.find`: Fixed integer overflow bug in SIMD string search that
  caused searches to fail when searching for strings longer than
  `simd_width_of[DType.bool]()` and haystacks larger than UInt16.MAX.

- `LinkedList.reverse()`: Fixed missing `prev` pointer updates, which caused
  `__reversed__()` to produce wrong results after reversing.

- Mojo would previously consider types Movable if they had a `__moveinit__`
  method, even if they didn't conform to `Movable`.  Movability is now tied to
  the conformance which implies the method.
