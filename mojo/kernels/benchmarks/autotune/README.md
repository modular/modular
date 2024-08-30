# Find the optimal kernel parameters using `tune.py`

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
    from benchmark import (
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
    from sys import env_get_string, env_get_int
    ...

    fn main():
        alias dtype_str = env_get_string["DTYPE", "DType.float16"]()
        alias M = env_get_int["M", 0]()
        alias N = env_get_int["N", 0]()
    ```

2. Ensure the input params are captured properly.

3. Define a config yaml file following the example of [`test.yaml`](test.yaml):

    ```mojo
    name: multistage_gemm
    file: sample.mojo
    params:
        DTYPE: DType.float16
        M: [1024,512]
        N: [1024,512,256,64]
        STAGES: [4,8,12]
    ```

4. Run `tune.py` script as follows:
    For simply running all the configs in the YAML file:

    ```bash
    $MODULAR_PYTHON tune.py --yaml YAML_FILE --output OUTPUT_PATH
    ```

    For finding the best measured elapsed time add `--tune`:

    ```bash
    $MODULAR_PYTHON tune.py --yaml YAML_FILE --output OUTPUT_PATH --tune
    ```

## Example

Applying autotuning on `tune_multistage_gemm.mojo` with yaml in
[`demo/gemm.yaml`](demo/gemm.yaml):

```text
--------------------------------------------------------------------------------
Tuning [multistage_gemm] from [../tune_multistage_gemm.mojo]
--------------------------------------------------------------------------------
Number of valid specs: 29
 mesh_idx                                               name  met (ms)  iters  Arithmetic (GFLOPS/s)
       39 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  0.984400      2           17452.122292
       37 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  0.988528      2           17379.243870
       38 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  0.990560      2           17343.592699
       36 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.001488      2           17154.352085
       22 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.008256      2           17039.193602
       35 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.008432      2           17036.219779
       52 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.020560      2           16833.775183
       18 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.021231      2           16822.706306
       54 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.032655      2           16636.600979
       33 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.034704      2           16603.655909
       20 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.035279      2           16594.426127
       32 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.041600      2           16493.730015
       50 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.044896      2           16441.702508
       12 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.047984      2           16393.255225
       34 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.048367      2           16387.258460
       11 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.048560      2           16384.257817
       13 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.050191      2           16358.796642
       10 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.054864      2           16286.335664
       26 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.066896      2           16102.665287
       28 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.068880      2           16072.776349
       24 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.086640      2           15810.083546
        9 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.098816      2           15634.891723
        8 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.123552      2           15290.675629
        0 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.221232      2           14067.653963
       14 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.751696      2            9807.563175
       30 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.777440      2            9665.512863
        4 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  1.866032      2            9206.631603
        2 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  2.017520      2            8515.340212
        6 multistage_gemm/dtype=bfloat16/M=8192/N=8192/K=128  3.376528      2            5088.028052
top_spec_idx: 39
--------------------------------------------------------------------------------
Best Measured Time:
--------------------------------------------------------------------------------
params:
  M: 8192
  N: 8192
  K: 128
  BM: 128
  BN: 128
  WM: 64
  WN: 64
  NUM_STAGES: 4
  TRANSPOSE_B: False
--------------------------------------------------------------------------------
Elasped tuning time: 259.3 (s)
```
