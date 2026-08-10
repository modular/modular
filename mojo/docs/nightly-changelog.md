---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

- Code that performs many implicit conversions, most visibly large collection
  literals, compiles faster: the compiler no longer runs parameter inference on
  constructors that cannot be used for an implicit conversion in the first
  place. Files that are mostly data, such as the standard library's Unicode
  lookup tables, compile about 1.3x faster.

## Documentation

## Language enhancements

## Language changes

- The module & package system:

  - Directories may now have "namespace" semantics; a single directory name may
    resolve across distinct locations on disk which share that name.

    ```mojo
    # .
    # ├── one
    # │   └── foo
    # │       └── bar.mojo
    # └── two
    #     └── foo
    #         └── baz.mojo
    #
    # Compiles with -Ione -Itwo
    import foo.bar
    import foo.baz
    ```

  - Importing functions with the same name from different modules, combining
    them into one overload set, is now an error, following a period of
    deprecation.

  - Intra-package accesses without explicit `import`s are now an error,
    following a period of deprecation.

## Library stabilizations

## Library changes

- `StringDict` now conforms to `Writable` when its value type is `Writable`,
  matching the existing behavior of `Dict`. This lets you `print()` a
  `StringDict` or convert it to a `String`.

- The `chars` argument of `strip()`, `lstrip()` and `rstrip()` on `StringSpan`,
  `String` and `StringLiteral` is now an `ImmStringSpan`, so a mutable string
  is accepted as `chars`, including the string being stripped (`s.strip(s)`).

- `StringDict.__getitem__()` now accepts a `StringSpan`, so you can index a
  `StringDict` with a borrowed string view without first allocating a
  `String` just to perform the lookup.

- Renamed the variadic type-list parameter on `Tuple` and `VariadicPack` to
  `Ts`, standardizing the naming convention used across the standard library.
  The old name, `element_types`, remains as a deprecated alias.

- Added experimental `DType.float6_e2m3fn` and `DType.float6_e3m2fn`, the two
  6-bit encodings from the
  [Open Compute microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
  Both are finite-only, so neither has an inf nor a NaN encoding.

  These are experimental storage formats for packed weights rather than
  general-purpose numeric types, and standard library support is deliberately
  partial. As with the existing `DType.float4_e2m1fn`, they are excluded from
  `is_numeric()`, arithmetic is not implemented, and converting to or from
  another floating-point type is unsupported on every target, so values cannot
  be printed either.

- `Array` now conforms to `Defaultable` when its type `T` is also `Defaultable`.

- Deprecated `is_trivially_movable()`, `is_trivially_copyable()`, and
  `is_trivially_deletable()` in `std.memory` in favor of
  `IsTriviallyMovable[T]`, `IsTriviallyCopyable[T]`, and
  `IsTriviallyDeinitable[T]` in `std.traits`. The replacements are `comptime`
  predicates rather than functions, so drop the call parens at use sites, for
  example `IsTriviallyCopyable[T]` instead of `is_trivially_copyable[T]()`.

- `Atomic` is now parameterized on a value type `T` instead of a `DType`.
  Update call sites from `Atomic[DType.float32]` to `Atomic[Float32]`. The
  atomic operations (`load()`, `store()`, `fetch_add()`, `compare_exchange()`,
  and so on) still only support `Scalar` types.

- The following APIs have been migrated to unified closures: `sort`,
  `debug_assert`, `Span.apply`.

## GPU programming

## Tooling changes

## Removed

This release completes the removal of APIs deprecated during the v1.0 cycle.
Each entry names its replacement.

- Removed the temporary `InlineArray` alias for `Array`, including its
  re-exports from `std.collections` and the prelude. Use `Array` directly.

- Removed the `std.gpu.profiler` module and its `ProfileBlock` context manager.
  It timed host wall-clock, not GPU work, and reported the elapsed time with the
  operands reversed. Time a block of host code with
  [`perf_counter_ns()`](/docs/std/time/time/perf_counter_ns/) directly, and use
  a GPU profiler such as Nsight Systems or `rocprof` for device timings.

## Fixed

- Parametric `raises` now accepts any primary expression as the thrown type in
  a function signature, matching the syntax positions where types otherwise
  appear. This most notably fixes `raises Self.SomeAssocType` on trait and
  struct methods, which would previously fail with an error. The parenthesized
  workaround (`raises (Self.DriveErrorType)`) is no longer required.

- An integer `range()` with a step of zero is now always empty. It previously
  used to be an infinite loop - iterating forever at runtime, and hanging the
  compiler at comptime.

- Fixed `ceildiv()` returning `0` for unsigned operands near the type's
  maximum value. The unsigned code path computed `numerator + denominator -
  1`, which overflows and wraps for large operands; it now derives the
  ceiling from the floor division and remainder instead.

- `Counter.most_common(n)` now returns all elements when `n` exceeds the
  number of unique elements, matching Python, instead of aborting.
