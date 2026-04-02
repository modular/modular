# Nightly: v0.26.3

This version is still a work in progress.

## ✨ Highlights

## Documentation

## Language enhancements

- String literals now support `\uXXXX` and `\UXXXXXXXX` unicode escape
  sequences, matching Python. The resulting code point is stored as UTF-8.
  Invalid code points and surrogates are rejected at parse time.

- Added support for conditional `RegisterPassable` conformance.

- Variadic packs can be forwarded through runtime calls with `*pack` when the
  callee takes a compatible variadic pack parameter.

  ```mojo
  def callee[*Ts: Writable](*args: *Ts):
      comptime for i in range(args.__len__()):
          print(args[i])

  def forwarder[*Ts: Writable](*args: *Ts):
      callee(*args)

  forwarder(1, "hello", 3.14)  # prints each value on a separate line
  ```

## Language changes

- Mojo now warns on uses of the legacy `fn` keyword. Please move to `def` as
  this will upgrade to an error in the future.

- Import statements of the form `from pkg import ...` no longer make `pkg`
  available to the module.

## Library changes

- `abort(message)` now includes the call site location in its output. The
  location is automatically captured and printed alongside the message. You can
  also pass an explicit `SourceLocation` to override it:

  ```mojo
  abort("something went wrong")
  # prints: ABORT: path/to/file.mojo:42:5: something went wrong

  var loc = current_location()
  abort("something went wrong", location=loc)
  ```

- `SourceLocation` fields (`line`, `col`, `file_name`) are now private.
  Use the new accessor methods `line()`, `column()`, and `file_name()` instead.

- Fixed default alignment in `TileTensor.load()` and `TileTensor.store()` to
  use the caller-specified `width` parameter instead of `Self.element_size`.

- Added `CompilationTarget.is_apple_m5()` to `std.sys` for detecting Apple M5
  targets at compile time. `is_apple_silicon()` now includes M5 in its check.

- Standard library types now use conditional conformances, replacing previous
  `_constrained_conforms_to` checks:
  - `Span`: `Writable`, `Hashable`
  - `Tuple`, `Optional`, `Variant`, and `UnsafeMaybeUninit`: `RegisterPassable`
  - `Variant`: `Copyable`, `ImplicitlyCopyable`

- GPU primitive id accessors (e.g. `thread_idx`) have migrated from `UInt` to
  `Int`.

  This is part of a broader migration to standardize on the `Int` type for all
  sizes and offsets in Mojo.

  To provide a gradual migration path, explicitly typed aliases are
  available temporarily.

  | Base         | `UInt` Accessor   | `Int` Accessor    |
  |--------------|-------------------|-------------------|
  | `thread_idx` | `thread_idx_uint` | `thread_idx_int`  |
  | `thread_dim` | `thread_dim_uint` | `thread_dim_int`  |
  | `block_dim`  | `block_dim_uint`  | `block_dim_int`   |
  | `grid_dim`   | `grid_dim_uint`   | `grid_dim_int`    |
  | `global_idx` | `global_idx_uint` | `global_idx_int`  |
  | `lane_id`    | `lane_id_uint`    | `lane_id_int`     |
  | `warp_id`    | `warp_id_uint`    | `warp_id_int`     |

  Code can preserve its prior behavior by using a renaming import of the
  `thread_idx_uint` alias:

  ```diff
  - from std.gpu import thread_idx
  + from std.gpu import thread_idx_uint as thread_idx
  ```

  Note that `thread_idx_uint` and the other `_*uint` aliases will eventually
  be deprecated and removed as well.

  After a temporary deprecation acting as a "speed bump" in the 2026-03-29
  nightly release, `thread_idx` etc. have changed from `UInt` to `Int`.

  Code built with a version where `thread_idx` is still `UInt`, can proactively
  migrate to the eventual `Int` behavior using the `thread_idx_int` alias:

  ```diff
  - from std.gpu import thread_idx
  + from std.gpu import thread_idx_int as thread_idx

  # ... update file to reflect change from `UInt` to `Int` ...
  ```

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

- Added `Variadic.contains_value` comptime alias to check whether a variadic
  sequence contains a specific value at compile time.

- `ArcPointer` now conditionally conforms to `Hashable` and `Equatable` when
  its inner type `T` does. Both `__eq__` and `__hash__` delegate to the managed
  value, matching C++ `shared_ptr` and Rust `Arc` semantics. This makes
  `ArcPointer` usable as a `Dict` key or `Set` element with value-based
  equality. Pointer identity is still available via the `is` operator.

- `Path` now conforms to `Comparable`, enabling lexicographic ordering and use
  with `sort()`.

- `range()` overloads that took differently-typed arguments or arguments that
  were `Intable`/`IntableRaising` but not `Indexer` have been removed. Callers
  should ensure they're passing consistent integral argument types when calling
  `range()`.

- `Consistency` now has a default constructor that selects `RELEASE` ordering on
  Apple GPU and `SEQUENTIAL` on all other targets. All `Atomic` methods and
  `fence` use this platform-aware default instead of hard-coding `SEQUENTIAL`.

- `NDBuffer` has been fully removed. Please migrate to `TileTensor`.

- Added a generic `__contains__` method to `Span` for any element type
  conforming to `Equatable`, not just `Scalar` types.

- Fixed `blocked_product` in `tile_layout` to zip block and tiler dimensions
  per mode, matching the legacy `blocked_product` behavior.

- Added `Span`-based overloads for `enqueue_copy`, `enqueue_copy_from`, and
  `enqueue_copy_to` on `DeviceContext`, `DeviceBuffer`, and `HostBuffer`,
  providing a safer alternative to raw `UnsafePointer` for host-device memory
  transfers.

## Tooling changes

- The Mojo debugger now displays scalar types (e.g. `UInt8`, `Float32`) as
  plain values instead of `([0] = value)`, and elides internal `_mlir_value`
  wrapper fields from struct display.

- `mojo format` no longer supports the deprecated `fn` keyword, nor the
  removed `owned` argument convention.

- Comptime function calls now print more nicely in error messages and generated
  documentation, not including `VariadicList`/`VariadicPack` and including
  keyword argument labels when required.

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

- [Issue #5872](https://github.com/modular/modular/issues/5872): Fixed a
  compiler crash ("'get_type_name' requires a concrete type") when using
  default `Writable`, `Equatable`, or `Hashable` implementations on structs
  with MLIR-type fields (e.g. `__mlir_type.index`). The compiler now correctly
  reports that the field does not implement the required trait.
