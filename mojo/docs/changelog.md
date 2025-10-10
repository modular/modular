# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

- Literals now have a default type. For example, you can now bind `[1,2,3]` to
  `T` in a call to a function defined as `fn zip[T: Iterable](impl:T)` because
  it will default to the standard library's List type.

- Mojo now has a `__functions_in_module` experimental intrinsic that allows
  reflection over the functions declared in the module where it is called. For
  example:

  ```mojo
  fn foo(): pass

  def bar(x: Int): pass

  def main():
    alias funcs = __functions_in_module()
    # equivalent to:
    alias same_funcs = Tuple(foo, bar)
  ```

  The intrinsic is currently limited for use from within `main`.

### Language changes

### Standard library changes

- Added `unsafe_get`, `unsafe_swap_elements` and `unsafe_subspan` to `Span`.

- The deprecated `DType.index` is now removed in favor of the `DType.int`.

- `math.isqrt` has been renamed to `rsqrt` since it performs reciprocal square
  root functionality.

- Added `swap_pointees` function to `UnsafePointer` as an alternative to `swap`
  when the pointers may potentially alias each other.

- `memcpy` and `parallel_memcpy` without keyword arguments are deprecated.

- The `math` package now has a mojo native implementation of `acos`, `asin`,
  `cbrt`, and `erfc`.

- Added support for NVIDIA GeForce GTX 970.

- Added support for NVIDIA Jetson Thor.

- `Optional` now conforms to `Iterable` and `Iterator` acting as a collection of
  size 1 or 0.

- `String.splitlines()` now returns an iterator instead of a
  `List[StringSlice]`.

### Tooling changes

- Error messages now preserve symbolic calls to `always_inline("builtin")`
  functions rather than inlining them into the error message.

### ❌ Removed

### 🛠️ Fixed

- The `math.cos` and `math.sin` function can now be evaluated at compile time
  (fixes #5111).
