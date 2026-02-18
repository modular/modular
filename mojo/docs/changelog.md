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

### Language changes

- The `__moveinit__` and `__copyinit__` methods are being renamed to `__init__`
  to standardize construction. As such, the argument name for `__moveinit__`
  must now be named `take` and the argument name for `__copyinit__` must now be
  named `copy`.

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

  Libraries with generic algorithms and types should be written to accomodate
  linear types. Making `ImplicitlyDestructible` opt-in for traits
  encourages a default stance of support, with specific types and functions
  only opting-in to the narrower `ImplicitlyDestructible` requirement if they
  truly need it.

  The majority of generic algorithms that take their inputs by reference should
  not be affected.

- Unstable `__comptime_assert` syntax is now finalized as `comptime assert`. A
  deprecation warning is emitted with a fixit for the old syntax.

### Library changes

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

### ❌ Removed

- The `owned` keyword has been removed. Use `var` for parameters or `deinit`
  for `__moveinit__`/`__del__` arguments as appropriate.

- `Dict.EMPTY` and `Dict.REMOVED` comptime aliases have been removed. These
  were internal implementation details of the old hash table design.

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
