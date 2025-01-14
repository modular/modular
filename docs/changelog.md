# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language changes

- Initializers are now treated as static methods that return an instance of
  `Self`.  This means the `out` argument of an initializer is now treated the
  same as a any other function result or `out` argument. This is generally
  invisible, except that patterns like `instance.__init__()` and
  `x.__copyinit__(y)` no longer work.  Simply replace them with `instance = T()`
  and `x = y` respectively.

- The legacy `borrowed`/`inout` keywords and `-> T as foo` syntax now generate
  a warning.  Please move to `read`/`mut`/`out` argument syntax instead.

- The `@value` decorator now additionally derives an implementation of the
  `ExplicitlyCopyable` trait. This will ease the transition to explicit
  copyablility requirements by default in the Mojo collection types.

- Indexing into a homogenous tuple now produces the consistent element type
  without needing a rebind:

  ```mojo
    var x = (1, 2, 3, 3, 4)
    var y : Int = x[idx]     # Just works!
  ```

### Standard library changes

- The `int` function to construct an `Int` has been deprecated, this was a
  temporary workaround when Mojo didn't have a way to distinguish between
  implicit and explicit constructors. You can do a search and replace from
  `int(` to `Int(` to update your programs, the `int` function will be removed
  in the next release.

- `UnsafePointer`'s `bitcast` method has now been split into `bitcast`
  for changing the type, `origin_cast` for changing mutability,
  `static_alignment_cast` for changing alignment,
  and `address_space_cast` for changing the address space.

- `UnsafePointer` is now parameterized on mutability. Previously,
  `UnsafePointer` could only represent mutable pointers.

  The new `mut` parameter can be used to restrict an `UnsafePointer` to a
  specific mutability: `UnsafePointer[T, mut=False]` represents a pointer to
  an immutable `T` value. This is analogous to a `const *` pointer in C++.

  - `UnsafePointer.address_of()` will now infer the origin and mutability
    of the resulting pointer from the argument. For example:

    ```mojo
    var local = 10
    # Constructs a mutable pointer, because `local` is a mutable memory location
    var ptr = UnsafePointer.address_of(local)
    ```

    To force the construction of an immutable pointer to an otherwise mutable
    memory location, use a cast:

    ```mojo
    var local = 10
    # Cast the mutable pointer to be immutable.
    var ptr = UnsafePointer.address_of(local).origin_cast[mut=False]()
    ```

  - The `unsafe_ptr()` method on several standard library collection types have
    been updated to use parametric mutability: they will return an `UnsafePointer`
    whose mutability is inherited from the mutability of the `ref self` of the
    receiver at the call site. For example, `ptr1` will be immutable, while
    `ptr2` will be mutable:

    ```mojo
    fn take_lists(read list1: List[Int], mut list2: List[Int]):
        # Immutable pointer, since receiver is immutable `read` reference
        var ptr1 = list1.unsafe_ptr()

        # Mutable pointer, since receiver is mutable `mut` reference
        var ptr2 = list2.unsafe_ptr()
    ```

- Added `Optional.copied()` for constructing an owned `Optional[T]` from an
  `Optional[Pointer[T]]` by copying the pointee value.

- Added `Dict.get_ptr()` which returns an `Optional[Pointer[V]]`. If the given
  key is present in the dictionary, the optional will hold a pointer to the
  value. Otherwise, an empty optional is returned.

- Added new `List.extend()` overloads taking `SIMD` and `Span`. These enable
  growing a `List[Scalar[..]]` by copying the elements of a `SIMD` vector or
  `Span[Scalar[..]]`, simplifying the writing of some optimized SIMD-aware
  functionality.

- Added `Char`, for representing and storing single Unicode characters.
  - `Char` implements `CollectionElement`, `EqualityComparable`, `Intable`, and
    `Stringable`.
  - Added `String` constructor from `Char`
  - `Char` can be converted to `UInt32` via `Char.to_u32()`.

- `chr(Int)` will now abort if given a codepoint value that is not a valid
  `Char`.

- Added `StringSlice.from_utf()` factor method, for validated construction of
  a `StringSlice` from a buffer containing UTF-8 encoded data. This method will
  raise if the buffer contents are not valid UTF-8.

- Added `StringSlice.chars()` which returns an iterator over `Char`s. This is a
  compliant UTF-8 decoder that returns each Unicode codepoint encoded in the
  string.

- Several standard library functions have been changed to take `StringSlice`
  instead of `String`. This generalizes them to be used for any appropriately
  encoded string in memory, without requiring that the string be heap allocated.

  - `atol()`
  - `atof()`
  - `ord()`
  - `ascii()`
  - `b64encode()`
    - Additionally, the `b64encode()` overload that previously took `List` has
      been changed to
      take a `Span`.
  - `b64decode()`
  - `b16encode()`
  - `b16decode()`

- Various functionality has moved from `String` and `StringRef` to the more
  general `StringSlice` type.

  - `StringSlice` now implements `Representable`, and that implementation is now
    used by `String.__repr__()` and `StringRef.__repr__()`.

- `StringSlice` now implements `EqualityComparable`.

  Up until now, `StringSlice` has implemented a more general `__eq__` and
  `__ne__` comparision with `StringSlice` types that had arbitrary other
  origins. However, to satisfy `EqualityComparable`, `StringSlice` now also
  has narrower comparison methods that support comparing only with
  `StringSlice`'s with the exact same origin.

- Added `StringSlice.char_length()` method, to pair with the existing
  `StringSlice.byte_length()` method.

  In a future version of Mojo, `StringSlice.__len__()` may be changed to return
  the length in bytes, matching the convention of string length methods in
  languages like C++ and Rust. Callers that know they need the length in
  Unicode codepoints should update to calling `StringSlice.char_length()`
  instead.

- Removed `@implicit` decorator from some standard library initializer methods
  that perform allocation. This reduces places where Mojo code could implicitly
  allocate where the user may not be aware.

  Remove `@implicit` from:

  - `String.__init__(out self, StringRef)`
  - `String.__init__(out self, StringSlice)`
  - `List.__init__(out self, owned *values: T)`
  - `List.__init__(out self, span: Span[T])`

- The `ExplicitlyCopyable` trait has changed to require a
  `fn copy(self) -> Self` method. Previously, an initializer with the signature
  `fn __init__(out self, *, other: Self)` had been required by
  `ExplicitlyCopyable`.

  This improves the "greppability" and at-a-glance readability when a programmer
  is looking for places in their code that may be performing copies

- `bit_ceil` has been renamed to `next_power_of_two`, and `bit_floor` to
  `prev_power_of_two`. This is to improve readability and clarity in their use.

- The `Indexer` and `IntLike` traits which were previously both used for
  indexing have been combined. This enables SIMD scalar integer types and UInt
  to be used for indexing into all of the collection types, as well as
  optimizing away normalization checks for UInt indexing.

- The `ImplicitlyIntable` trait has been added, allowing types to be implicitly
  converted to an `Int` by implementing the `__as_int__` method:

  ```mojo
  @value
  struct Foo(ImplicitlyIntable):
      var i: Int

      fn __as_int__(self) -> Int:
          return self.i
  ```

- You can now cast SIMD types using constructors:

  ```mojo
  var val = Int8(42)
  var cast = Int32(val)
  ```

  It also works when passing a scalar type to larger vector size:

  ```mojo
  var vector = SIMD[DType.int64, 4](cast) # [42, 42, 42, 42]
  ```

  For values other than scalars the size of the SIMD vector needs to be equal:

  ```mojo
  var float_vector = SIMD[DType.float64, 4](vector)
  ```

  `SIMD.cast` still exists to infer the size of new vector:

  ```mojo
  var inferred_size = float_vector.cast[DType.uint64]() # [42, 42, 42, 42]
  ```

- You can now use `max()` and `min()` with variadic number of arguments.

### Tooling changes

- mblack (aka `mojo format`) no longer formats non-mojo files. This prevents
  unexpected formatting of python files.

- Full struct signature information is now exposed in the documentation
  generator, and in the symbol outline and hover markdown via the Mojo Language
  Server.

### ❌ Removed

- `StringRef` is being deprecated. Use `StringSlice` instead.
  - Changed `sys.argv()` to return list of `StringSlice`.
  - Added `Path` explicit constructor from `StringSlice`.
  - removed `StringRef.startswith()` and `StringRef.endswith()`
  - removed `StringRef.strip()`
- The `Tuple.get[i, T]()` method has been removed. Please use `tup[i]` or
  `rebind[T](tup[i])` as needed instead.

### 🛠️ Fixed

- The Mojo Kernel for Jupyter Notebooks is working again on nightly releases.

- The command `mojo debug --vscode` now sets the current working directory
  properly.

- [Issue #3796](https://github.com/modularml/mojo/issues/3796) - Compiler crash
  handling for-else statement.

- [Issue #3540](https://github.com/modularml/mojo/issues/3540) - Using named
  output slot breaks trait conformance

- [Issue #3617](https://github.com/modularml/mojo/issues/3617) - Can't generate
  the constructors for a type wrapping `!lit.ref`

- The Mojo Language Server doesn't crash anymore on empty **init**.mojo files.
  [Issue #3826](https://github.com/modularml/mojo/issues/3826).

- [Issue #3935](https://github.com/modularml/mojo/issues/3935) - Confusing OOM
   error when using Tuple.get incorrectly.
