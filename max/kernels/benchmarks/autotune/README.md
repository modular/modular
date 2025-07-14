# `kbench`: Benchmarking toolkit for Mojo kernels

`kbench` builds and executes through a grid of all the parameter combinations
 for a Mojo benchmark. This toolkit is used for benchmarking, tuning
 (finding optimal kernel parameters), and plotting Mojo kernels.

## Table of Contents

- [Why is `kbench` written in Python?](#why-is-kbench-written-in-python)
- [Setup for tuning](#setup-for-tuning)
- [Usage](#usage)
  - [Example](#example)
- [Design](#design)
  - [`kbench` YAML format](#kbench-yaml-format)
  - [Expanding spec's to get instances](#expanding-specs-to-get-instances)
  - [`kbench` loop: Enumerating over instances](#kbench-loop-enumerating-over-instances)
  - [`kbench` loop: Enumerating over instances with shapes](#kbench-loop-enumerating-over-instances-with-shapes)
- [Output pickle `.pkl` files](#output-pickle-pkl-files)
- [Compile-time Parameters vs. Runtime Variables](#compile-time-parameters-vs-runtime-variables)
- [`kbench` Object Cache](#kbench-object-cache)
  - [Enable object cache](#enable-object-cache)
  - [Clear object cache](#clear-object-cache)

## Why is `kbench` written in Python?

In other words, why the driver cannot be in the same process as the entity that
is autotuned?

- Not all parameters used for autotuning are valid. Invalid parameters may
cause the process to crash which brings down the driver, so we need to have
separation there.
- Abundance of utilities and libraries (Pandas, Plotly, Rich progress bar/console)

We could invent and build a fancy system, however, that’s not solving the
autotuning problem, so just leaned into building something simple.

## Setup for tuning

Compile using `--config=production`:

```bash
br //:install --config=production
setup-gpu-benchmarking
```

### Setup kbench

Before running you many need to export the `$KERNEL_BENCHMARKS_ROOT`
environment variable. This is the root directory of the kbench repository.

```bash
export KERNEL_BENCHMARKS_ROOT=$MODULAR_PATH/open-source/max/max/kernels/benchmarks
```

There are two ways to run kbench:

1. Using bazel:
   Running with bazel should automatically build and install mojo

    ```bash
    br //open-source/max/max/kernels/benchmarks/autotune:kbench -- --help

    ```

    When running from bazel, note your .yaml files reference a mojo file,
    you'll need to be in a directory where that file is accessible.

    ```bash
    br //open-source/max/max/kernels/benchmarks/autotune:kbench --  test.yaml --dryrun
    ```

2. Using uv:
    If you use uv, you'll need to install `modular` first, so that `mojo` will be
    available.

    ```bash
    uv run kbench --help
    ```

## Usage

0. Set the benchmark function properly
    The benchmark should run exactly one case, import `Bench` following the
    example in [`sample.mojo`](sample.mojo):

    ```mojo
    from sys import env_get_string, env_get_int
    from internal_utils import env_get_dtype, env_get_shape, int_list_to_tuple
    from benchmark import (
        BenchConfig,
        Bench,
        Bencher,
        BenchId,
        BenchMetric,
        ThroughputMeasure,
        keep,
    )
    ```

1. Define your input params in Mojo using the following sys env functions:

    ```mojo
    fn main():
        alias dtype = env_get_dtype["dtype", DType.float16]()
        alias shape_int_list = env_get_shape["shape", "1024x1024x1024"]()
        alias shape = int_list_to_tuple[shape_int_list]()
        alias stages = env_get_int["stages", 0]()
    ```

2. Ensure the input params are captured properly.

3. Define a config yaml file following the example of [`test.yaml`](test.yaml):

    ```yaml
    name: multistage_gemm
    file: sample.mojo
    params:

    - dtype: DType.float16
      shape: [1024x512x256, 32x32x32]
      stages: [4,8]

    - dtype: DType.float32
      shape: 64x64x64
      stages: 2
    ```

4. Run `kbench.py` script as follows:
    For simply running all the configs in the YAML file:

    ```bash
    $MODULAR_PYTHON kbench.py YAML_FILE --output OUTPUT_PATH
    ```

    For finding the best measured elapsed time add `--tune`:

    ```bash
    $MODULAR_PYTHON kbench.py YAML_FILE --output OUTPUT_PATH --tune
    ```

5. Avoid recompilation by enabling `kbench` object cache:
    Simply add `--cached` or `-c` to the of command.

    Note:
    - This doesn't check the changes in the source.
    - Caching is persistent for follow-up calls unless `kbench --clear-cache`
        or `kbench -cc` is invoked.

6. To include parameters from CLI without adding them to YAML files use `--param`:

    ```bash
    kbench --param NAME:VALUE # single value
    kbench --param NAME:[VALUE0, VALUE1] # Pythonic list of values
    ```

7. To filter out certain values without removing them from YAML files use
`--filter` in CLI:

    ```bash
    kbench --filter NAME=VALUE # only show the instances where parameter NAME=VALUE.
    ```

### Example

Just running [`sample.mojo`](sample.mojo) with parameters in [`test.yaml`](test.yaml):

```text
build-run ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 5/5
--------------------------------------------------------------------------------
Tuning [multistage_gemm] from [sample.mojo]
--------------------------------------------------------------------------------
Number of valid specs: 5
 mesh_idx                                           name  met (ms)  iters
        0 gemm/dtype=float16/m=1024/n=512/k=512/stages=4  0.000046      2
        1 gemm/dtype=float16/m=1024/n=512/k=512/stages=8  0.000037      2
        2     gemm/dtype=float16/m=32/n=32/k=32/stages=4  0.000033      2
        3     gemm/dtype=float16/m=32/n=32/k=32/stages=8  0.000034      2
        4     gemm/dtype=float32/m=64/n=64/k=64/stages=2  0.000034      2
--------------------------------------------------------------------------------
Elapsed tuning time: 22.5 (s)
wrote results to [output.csv]
```

Now, tuning [`sample.mojo`](sample.mojo) with parameters in [`test.yaml`](test.yaml):

```text
build-run ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 5/5
--------------------------------------------------------------------------------
Tuning [multistage_gemm] from [sample.mojo]
--------------------------------------------------------------------------------
Number of valid specs: 5
 mesh_idx                                           name  met (ms)  iters
        4     gemm/dtype=float32/m=64/n=64/k=64/stages=2  0.000031      2
        1 gemm/dtype=float16/m=1024/n=512/k=512/stages=8  0.000034      2
        2     gemm/dtype=float16/m=32/n=32/k=32/stages=4  0.000034      2
        3     gemm/dtype=float16/m=32/n=32/k=32/stages=8  0.000034      2
        0 gemm/dtype=float16/m=1024/n=512/k=512/stages=4  0.000034      2
top_spec_idx: 4
--------------------------------------------------------------------------------
Best Measured Time:
--------------------------------------------------------------------------------
{'dtype': 'DType.float32', 'shape': '64x64x64', 'stages': 2}
--------------------------------------------------------------------------------
Elapsed tuning time: 13.9 (s)
wrote results to [output.csv]
```

## Design

![`kbench` toolkit](data/kbench_toolkit.png)

### `kbench` YAML format

`kbench` compatible YAMLs should have the following structure:

```YAML
name: placeholder
file: path-to-mojo source
params:
    - spec #spec refers to a group of parameters, each with their own values.
        param_name: value-set|single-value #Each parameter can have 1+ values.
```

### Expanding spec's to get instances

```python
instance_list = product(<params, values>) for all specs in yaml
```

For example, consider the following YAML:

```yaml
name: multistage_gemm
file: sample.mojo
params:

- dtype: DType.float16
    shape: [1024x512x256, 32x32x32]
    stages: [4,8]

- dtype: DType.float32
    shape: 64x64x64
    stages: 2
```

The first `spec` will be expanded into 4 separate instances, the last one
remains as it is:

```YAML
- dtype: DType.float16
  shape: 1024x512x256
  stages: 4

- dtype: DType.float16
  shape: 1024x512x256
  stages: 8

- dtype: DType.float16
  shape: 32x32x32
  stages: 4

- dtype: DType.float16
  shape: 32x32x32
  stages: 8

- dtype: DType.float32
  shape: 64x64x64
  stages: 2
```

### `kbench` loop: Enumerating over instances

```python
for inst in instance_list:
    compile_and_run_kernel(inst)
```

For example:

```bash
kbench tuning_params.yaml
```

### `kbench` loop: Enumerating over instances with shapes

In certain use cases, we need to have two levels of parameters that should be
expanded separately. For example, when running a kernel with input shapes in
set `S` over a set of tuning parameters `T`, would like to avoid mixing shape
parameters and tuning parameters all at once, i.e., `expansion(SxT)`.
Instead, we are interested in `expansion(S) x expansion(T)`, writing the
results of each tuning step to `#S` separate output file

The following loop nest shows how `kbench` enumerates over shapes and instances:

```python
for shape in shapes:
    for bench_inst in benchmarking_instances:
        compile_and_run_kernel(shape + bench_inst)
    dump_results_for(shape)
```

For example:

```bash
kbench tuning_params.yaml --shapes input_shapes.yaml
```

## Output pickle `.pkl` files

For simply running all the configs in the YAML file:

```bash
kbench YAML_FILE -o/--output OUTPUT_PATH
```

This will automatically store the intermediate output in `OUTPUT_PATH.pkl`.
Please refer to [README_kprofile.md](README_kprofile.md) for details on how to
analyze the data in `.pkl` files.

## Compile-time Parameters vs. Runtime Variables

Building with various compile-time parameters does NOT hit Mojo-cache and
increases compilation time. Therefore, it is essential to minimize the number
of parameters, or simply replace them with runtime variables to reduce the time
spent in (re)compilation.

Following example shows how to define a runtime variable in Mojo using
`arg_parse` utility function. Note that the name in YAML is now prefixed with
`$`:

```mojo
from internal_utils import arg_parse
fn main():
var runtime_x = arg_parse("x", 0)
```

```bash
> mojo sample.mojo
> ./sample --x=123
```

```yaml
name: demo_sample
file: sample.mojo
params:
- dtype: DType.float16
  shape: [1024x512x256, 32x32x32]
  stages: [4,8]
  $x: [0,1,2,3]
```

## `kbench` Object Cache

Recompiling with same set of parameters and hitting the mojo-cache requires
parsing The baseline cost of parsing is not quite negligible. Still, an
unnecessary price. What if we keep track of the compiled binaries between
various launches of kbench? To that end, we have implemented `kbench`
object-cache available via `kbench --cached` or `kbench -c`.

NOTE: This doesn't check the changes in the source.

### Enable object cache

```bash
kbench --cached test.yaml
kbench -c test.yaml
```

### Clear object cache

```bash
kbench --clear-cache
kbench -cc
```
