# Find the optimal kernel parameters using `kbench.py`

This script runs a grid-search of all the parameters for a mojo benchmark and
picks the top config (with the lowest measured elapsed time).

## Why the tuning driver is written in Python?

The issue is that we cannot have the autotuning driver to be in the same
process as the thing to be autotuned.  A simple reason is that not all
params used for autotuning are valid and may cause the process to crash. A
crash will bring down the driver, so we need to have separation there. We
could invent and build a fancy system, but that’s not solving the
autotuning problem, so let’s just lean into building something simple.

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

1. Define your input params in mojo using the following sys env functions:

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

## Example

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
