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

- The `@implicit` decorator now accepts an optional `deprecated` keyword
  argument. This can be used to phase out implicit conversions instead of just
  removing the decorator (which can result in another, unintended implicit
  conversion path). For example, the compiler now warns about the following:

  ```mojo
  struct MyStuff:
    @implicit(deprecated=True)
    fn __init__(out self, value: Int):
      pass

  fn deprecated_implicit_conversion():
    # warning: deprecated implicit conversion from 'IntLiteral[1]' to 'MyStuff'
    _: MyStuff = 1

    _ = MyStuff(1)  # this is okay, because the conversion is already explicit.
  ```

The `@deprecated` decorator can now take a target symbol with the `use` keyword
argument. This is mutually exclusive with the existing positional string
argument. A deprecation warning will be automatically generated.

```mojo
@deprecated(use=new)
fn old():
  pass

fn new():
  pass

fn main():
  old() # 'old' is deprecated, use 'new' instead
```

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

- `origin_cast` for `LayoutTensor`, `NDBuffer` and `UnsafePointer` has been
  deprecated and removed. `LayoutTensor` and `NDBuffer` now supports a safer
  `as_any_origin()` origin casting. `UnsafePointer` has the same
  safe alternative and in addition, it has an additional safe `as_immutable`
  casting function and explicitly unsafe `unsafe_mut_cast` and
  `unsafe_origin_cast` casting function.

- The `@implicit` decorator on `UInt.__init__(Int)` has been deprecated.
  Conversion from `Int` to `UInt` should now be done explicitly using
  `UInt(int_value)`.

- `assert_equal` now displays colored character-by-character diffs when string
  comparisons fail, making it easier to spot differences. Differing characters
  are highlighted in red for the left string and green for the right string.
  
- Added `sys.compile.SanitizeAddress` providing a way for mojo code to detect
  `--sanitize address` at compile time.

### Tooling changes

- Error messages now preserve symbolic calls to `always_inline("builtin")`
  functions rather than inlining them into the error message.

### ❌ Removed

### 🛠️ Fixed

- The `math.cos` and `math.sin` function can now be evaluated at compile time
  (fixes #5111).

- Fixed `LayoutTensor` element-wise arithmetic operations (`+`, `-`, `*`, `/`)
  between tensors with different memory layouts. Previously, operations like
  `a.transpose() - b` would produce incorrect results when the operands had
  different layouts, because the same layout index was incorrectly used for both
  operands. This now correctly computes separate indices for each tensor based
  on its layout.

- Fixed `arange()` function in `layout._fillers` to properly handle nested
  layout structures. Previously, the function would fail when filling
  tensors with nested layouts like
  `Layout(IntTuple(IntTuple(16, 8), IntTuple(32, 2)), ...)` because it
  attempted to extract shape values from nested tuples incorrectly.
