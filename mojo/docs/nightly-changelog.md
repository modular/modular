---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

## Language changes

## Library stabilizations

## Library changes

## GPU programming

## Tooling changes

## Removed

- Removed the `std.gpu.profiler` module and its `ProfileBlock` context manager.
  It timed host wall-clock, not GPU work, and reported the elapsed time with the
  operands reversed. Time a block of host code with
  [`perf_counter_ns()`](/docs/std/time/time/perf_counter_ns/) directly, and use
  a GPU profiler such as Nsight Systems or `rocprof` for device timings.

## Fixed
