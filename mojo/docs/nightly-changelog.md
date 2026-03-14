# Nightly: v0.26.3

This version is still a work in progress.

## ✨ Highlights

## Language enhancements

- `is_run_in_comptime_interpreter()` has been renamed to
  `__is_run_in_comptime_interpreter` keyword which provides
  a mechanism for supporting different code execution
  path in comptime interpreter from runtime generated code.
  This value cannot be used as a constant in any comptime expressions because
  it is always evaluated as `True` for comptime expressions.

## Language changes

- Mojo supports indexing with subscripts and names using the standard Python
  `__getitem__` and `__getattr__` methods.  Previously it used a heuristic to
  determine whether a `__get*__` method was operated with dynamic or parameter
  indexes.  Mojo now has a simple and explicit policy: if a type implements a
  `__getitem_param__` or `__getattr_param__` method and the indices are valid
  parameter expressions, the compiler will pick it (but will not support a
  `__setitem__` pair).  If not or if the indices are only valid runtime values,
  Mojo will try `getitem`/`setitem` as usual. This makes the behavior more
  predictable and explicit, but requires types to switch to the "param" method
  names if they desire parameter-style subscripting.  This only affects a small
  number of special types like `Tuple` and `VariadicPack`.

## Library changes

- Standard library types now use conditional conformances, replacing previous
  `_constrained_conforms_to` checks:
  - `List`: `Hashable`
  - `Optional`: `Hashable`
  - `Tuple`: `Equatable`, `Hashable`

- `perf_counter_ns()` now returns correct nanoseconds on GPU instead of raw
  cycle counts. Previously on NVIDIA GPUs it used `clock64` (a cycle counter
  dependent on GPU core clock frequency); it now uses `globaltimer` which
  provides actual nanosecond resolution. On AMD GPUs, it now uses
  `s_memrealtime` (a constant-speed real-time clock) instead of `s_memtime`
  (a cycle counter).

- The `DimList` type has moved to representing its dimensions as parameters to
  the type instead of values inside the type, directly reflecting that the
  dimensions are known at compile time.  Please change `DimList(x, y)`
  into `DimList[x, y]()`.

- `Random` and `NormalRandom` in `std.random.philox` now conform to `Writable`,
  making it possible to print or inspect their internal state (seed key and
  counter). Example output: `Random[10](key=[42, 0], counter=[0, 0, 0, 0])`.

- T-strings now support the raw prefix (`rt"..."`) which preserves backslashes
  as literal characters while still supporting interpolation.

```mojo
  var name = "Mojo"
  print(t"C:\{name}\Documents") # prints "C:\Mojo\Documents"
```

- Subscripting `String` and `StringSlice` now requires a named parameter for range
  indexing, for example `s[1:3]` is now `s[byte=1:3]`.

## Tooling changes

## ❌ Removed

## 🛠️ Fixed
