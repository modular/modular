# Nightly: v0.26.3

This version is still a work in progress.

## ✨ Highlights

## Documentation

## Language enhancements

- String literals now support `\uXXXX` and `\UXXXXXXXX` unicode escape
  sequences, matching Python. The resulting code point is stored as UTF-8.
  Invalid code points and surrogates are rejected at parse time.

- Variadic packs can be forwarded through runtime calls with `*pack` when the
  callee takes a compatible variadic pack parameter.

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

- `CStringSlice` can no longer represent a null pointer. To represent nullability
  use `Optional[CStringSlice]` which is guaranteed to have the same size and layout
  as `const char*`, where `NULL` is the empty `Optional`.

- `external_call`'s `return_type`'s requirements has been relaxed from
  `TrivialRegisterPassable` to `RegisterPassable`.

- `alloc[T](count, alignment)` will now `abort` if the underlying allocation
  failed.

- Added `Variadic.contains_value` comptime alias to check whether a variadic
  sequence contains a specific value at compile time.
  
- `ArcPointer` now conditionally conforms to `Hashable` and `Equatable` when
  its inner type `T` does. Both `__eq__` and `__hash__` delegate to the managed
  value, matching C++ `shared_ptr` and Rust `Arc` semantics. This makes
  `ArcPointer` usable as a `Dict` key or `Set` element with value-based
  equality. Pointer identity is still available via the `is` operator.

- `Path` now conforms to `Comparable`, enabling lexicographic ordering and use
  with `sort()`.

- `Consistency` now has a default constructor that selects `RELEASE` ordering on
  Apple GPU and `SEQUENTIAL` on all other targets. All `Atomic` methods and
  `fence` use this platform-aware default instead of hard-coding `SEQUENTIAL`.

- `SIMD[DType.bool, N]` now has two new methods:
  - `first_true()` -- returns the index of the first `True` lane, or `-1` if
    all lanes are `False`. Replaces the manual `pack_bits` +
    `count_trailing_zeros` pattern.
  - `count_true()` -- returns the number of `True` lanes. A bool-specific
    alias for `reduce_bit_count()`.

## Tooling changes

- The Mojo debugger now displays scalar types (e.g. `UInt8`, `Float32`) as
  plain values instead of `([0] = value)`, and elides internal `_mlir_value`
  wrapper fields from struct display.

- `mojo format` no longer supports the deprecated `fn` keyword, nor the
  removed `owned` argument convention.

## GPU programming

- Added support for AMD MI250X accelerators.

## ❌ Removed

- The deprecated `@doc_private` decorator has been removed. Use `@doc_hidden`
  instead.

## 🛠️ Fixed

- Fixed `mojo format` crashing after upgrading Mojo versions due to a stale
  grammar cache. ([Issue #6144](https://github.com/modular/modular/issues/6144))

- Fixed `atof` producing incorrect results for floats near the
  normal/subnormal boundary (e.g., `Float64("4.4501363245856945e-308")`
  returned half the correct value).
  ([#6196](https://github.com/modular/modular/issues/6196))
