<!-- markdownlint-disable MD013 -->

# GraphRT MEF Executor Benchmarks

This directory contains benchmarks for the MEF (Modular Executor Format) executor
in GraphRT. These benchmarks measure key performance characteristics of the
executor including primitive dispatch, control flow, invokable calls, and
dataflow scheduling.

## Benchmarks

The benchmarks are organized into the following families:

- **Primitive Dispatch** - Measures scheduling and dispatch overhead for async primitives
  - Includes a baseline that measures pure scheduling overhead without waiting for completion
- **Control Flow** - Measures overhead of if/else branching, while loops, and repeat loops
- **Invokable Calls** - Measures function call depth and chaining overhead
- **Dataflow Scheduling** - Measures parallel and sequential dataflow execution
- **GPU Simultaneous Ready** - Measures serialization overhead when multiple primitives become ready at the same time (simulates multi-device scenarios)

## Running Benchmarks

**Run all benchmarks:**

```bash
./bazelw run --config=production //MLRT/benchmarks/GraphRT:MEFExecutorBenchmark
```

**Other commonly used options:**

```bash
./bazelw run --config=production //MLRT/benchmarks/GraphRT:MEFExecutorBenchmark -- \
    --benchmark_filter=BM_PrimitiveDispatch
    --benchmark_repetitions=10 --benchmark_display_aggregates_only=true \
    --benchmark_min_time=100000x \
    --benchmark_min_time=1s \
    --benchmark_format=json --benchmark_out=results.json
```

## Adding New Benchmarks

1. Add new MLIR testcases to the appropriate `.mlir` file
2. Add a new benchmark function in `MEFExecutorBenchmark.cpp`
3. Register the benchmark with `BENCHMARK(...)`
