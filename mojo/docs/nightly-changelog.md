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

## Library stabilizations

## Library changes

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

- An integer `range()` with a step of zero is now always empty. It previously
  used to be an infinite loop - iterating forever at runtime, and hanging the
  compiler at comptime.
