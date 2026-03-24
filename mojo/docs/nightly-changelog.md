# Nightly: v0.26.3

This version is still a work in progress.

## ✨ Highlights

## Documentation

## Language enhancements

- String literals now support `\uXXXX` and `\UXXXXXXXX` unicode escape
  sequences, matching Python. The resulting code point is stored as UTF-8.
  Invalid code points and surrogates are rejected at parse time.

## Language changes

- Mojo now warns on uses of the legacy `fn` keyword. Please move to `def` as
  this will upgrade to an error in the future.

## Library changes

- Standard library types now use conditional conformances, replacing previous
  `_constrained_conforms_to` checks:
  - `Span`: `Writable`

- Added `IterableOwned` trait to the iteration module. Types conforming to
  `IterableOwned` implement `__iter__(var self)`, which consumes the collection
  and returns an iterator that owns the underlying elements.
  - `List` now conforms to `IterableOwned`.

- `CStringSlice` can no longer represent a null pointer. To represent
  nullability use `Optional[CStringSlice]` which is guaranteed to have the same
  size and layout as `const char*`, where `NULL` is the empty `Optional`.

- `external_call`'s `return_type`'s requirements has been relaxed from
  `TrivialRegisterPassable` to `RegisterPassable`.

- `alloc[T](count, alignment)` will now `abort` if the underlying allocation
  failed.

- `Span` now has `find()` and `rfind()` methods which work for any
  `Span[Scalar[D]]` e.g. `Span[Byte]`. PR [#3548](https://github.com/modularml/mojo/pull/3548)
  by [@martinvuyk](https://github.com/martinvuyk).

## Tooling changes

- The Mojo debugger now displays scalar types (e.g. `UInt8`, `Float32`) as
  plain values instead of `([0] = value)`, and elides internal `_mlir_value`
  wrapper fields from struct display.

## ❌ Removed

## 🛠️ Fixed

- Fixed `atof` producing incorrect results for floats near the
  normal/subnormal boundary (e.g., `Float64("4.4501363245856945e-308")`
  returned half the correct value).
  ([#6196](https://github.com/modular/modular/issues/6196))
