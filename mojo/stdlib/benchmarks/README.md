# Mojo standard library benchmarks

This document covers the benchmarks provided for the Mojo
standard library.

## Layout

There is 1-1 correspondence between the directory structure of
the benchmarks and their source counterpart. For example,
consider `collections/bench_dict.mojo` and its source counterpart:
`collections/dict.mojo`. This organization makes it easy to stay
organized.

Benchmark files should be prefixed with `bench_` in the filename.
This is helpful for consistency, but also is recognized by tooling
internally.

## How to run the benchmarks

If you want to just compile and run all the benchmarks as-is,
we need to execute the following command:

```bash
./bazelw test mojo/stdlib/benchmarks/... --local_test_jobs=1 --test_output=all
```

This script builds the open source `stdlib.mojopkg` and then executes
all the benchmarks sequentially.

If you wish to test changes you are making on the current branch, remove the
`-t` flag on top of the `mojo/stdlib/benchmarks/BUILD.bazel` BAZEL file.

To run a specific benchmark, you need to change the following line BAZEL file:

```bash
for src in glob(["**/*.mojo"])
```

To something like this:

```bash
for src in glob(["**/bench_list.mojo"])
```

Remember to revert the `-t` flag and the `glob` changes again before pushing
any code.

## How to write effective benchmarks

All the benchmarks use the `benchmark` module. `Bench` objects are built
on top of the `benchmark` module. You can also use `BenchConfig` to configure
`Bench`. For the most part, you can copy-paste from existing
benchmarks to get started.

## Benchmarks in CI

Currently, there is no short-term plans for adding these benchmarks with regression
detection and such in the public Mojo CI. We're working hard to improve the processes
for this internally first before we commit to doing this in the external repo.

## Other reading

Check out our [blog post](https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking)
for more info on writing benchmarks.
