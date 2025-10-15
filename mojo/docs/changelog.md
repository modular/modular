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
  deprecated. `LayoutTensor` and `NDBuffer` now supports a safer
  `as_any_origin()` origin casting. `UnsafePointer` has the same
  safe alternative and in addition, it has an additional safe `as_immutable`
  casting function and explicitly unsafe `unsafe_mut_cast` and
  `unsafe_origin_cast` casting function.

- The `Roundable` trait has been refactored and now allows several different
  rounding modes and the ability to round to a multiple of a given value. Having
  the multiple to round to as a parameter allows for some optimizations. An
  example implementation:

  ```mojo
  @fieldwise_init
  struct Complex(Roundable):
      alias _RoundMultipleType = Float64
      alias _RoundMultipleDefault = Float64(1)
      alias _RoundModeDefault = RoundMode.HalfToEven

      var re: Float64
      var im: Float64

      fn __round__[
          mode: RoundMode, to_multiple_of: Self._RoundMultipleType
      ](self) -> Self:
          return Self(
              round[mode, to_multiple_of=to_multiple_of](self.re),
              round[mode, to_multiple_of=to_multiple_of](self.im),
          )

      fn __round__(self, ndigits: Int) -> Self:
          return Self(round(self.re, ndigits), round(self.im, ndigits))
  ```

  And how one could use it:

  ```mojo
  round[RoundMode.Up, to_multiple_of=3](4) # 6
  round[RoundMode.Down, to_multiple_of=3](4) # 3
  round[RoundMode.ToEven, to_multiple_of=3](4) # 6
  round[RoundMode.ToZero, to_multiple_of=3](4) # 3
  round[RoundMode.HalfUp, to_multiple_of=2](5) # 6
  round[RoundMode.HalfDown, to_multiple_of=2](5) # 4
  round[RoundMode.HalfToZero, to_multiple_of=2](5) # 4
  ```

### Tooling changes

- Error messages now preserve symbolic calls to `always_inline("builtin")`
  functions rather than inlining them into the error message.

### ❌ Removed

### 🛠️ Fixed

- The `math.cos` and `math.sin` function can now be evaluated at compile time
  (fixes #5111).
