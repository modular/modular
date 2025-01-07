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

### Standard library changes

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

- Added `StringSlice.from_utf()` factor method, for validated construction of
  a `StringSlice` from a buffer containing UTF-8 encoded data. This method will
  raise if the buffer contents are not valid UTF-8.

- Several standard library functions have been changed to take `StringSlice`
  instead of `String`. This generalizes them to be used for any appropriately
  encoded string in memory, without requiring that the string be heap allocated.

  - `atol()`
  - `atof()`

- Removed `@implicit` decorator from some standard library initializer methods
  that perform allocation. This reduces places where Mojo code could implicitly
  allocate where the user may not be aware.

  Remove `@implicit` from:

  - `String.__init__(out self, StringRef)`
  - `String.__init__(out self, StringSlice)`

- The `ExplicitlyCopyable` trait has changed to require a
  `fn copy(self) -> Self` method. Previously, an initializer with the signature
  `fn __init__(out self, *, other: Self)` had been required by
  `ExplicitlyCopyable`.

  This improves the "greppability" and at-a-glance readability when a programmer
  is looking for places in their code that may be performing copies

- `bit_ceil` has been renamed to `next_power_of_two`, and `bit_floor` to
  `prev_power_of_two`. This is to improve readability and clarity in their use.

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
