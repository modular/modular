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

## Library changes

- Standard library types now use conditional conformances, replacing previous
  `_constrained_conforms_to` checks:
  - `List`: `Hashable`
  - `Tuple`: `Equatable`, `Hashable`

- The `DimList` type has moved to representing its dimensions as parameters to
  the type instead of values inside the type, directly reflecting that the
  dimensions are known at compile time.  Please change `DimList(x, y)`
  into `DimList[x, y]()`.

- T-strings now support the raw prefix (`rt"..."`) which preserves backslashes
  as literal characters while still supporting interpolation.

```mojo
  var name = "Mojo"
  print(t"C:\{name}\Documents") # prints "C:\Mojo\Documents"
```

## Tooling changes

## ❌ Removed

## 🛠️ Fixed
