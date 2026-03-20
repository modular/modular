# Nightly: v0.26.3

This version is still a work in progress.

## ✨ Highlights

## Documentation

## Language enhancements

## Language changes

## Library changes

- Standard library types now use conditional conformances, replacing previous
  `_constrained_conforms_to` checks:
  - `Span`: `Writable`

- Added `IterableOwned` trait to the iteration module. Types conforming to
  `IterableOwned` implement `__iter__(var self)`, which consumes the collection
  and returns an iterator that owns the underlying elements.
  - `List` now conforms to `IterableOwned`.

- `external_call`'s `return_type`'s requirements has been relaxed from
  `TrivialRegisterPassable` to `RegisterPassable`.

## Tooling changes

## ❌ Removed

## 🛠️ Fixed
