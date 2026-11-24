# Mojo docstring coverage

Generated at 2026-10-28 20:47 UTC.

An API is complete when it has a docstring and describes every
argument, parameter, return value, and raised error that
`mojo doc --diagnose-missing-doc-strings` requires. Partial APIs
have a docstring with some of those missing; missing APIs have no
docstring.

## Summary

- 4978/13828 (35.9%) public APIs complete
- 1811 partial
- 7039 with no docstring
- 3/15 libraries complete

## Libraries

| Library              | Validation | APIs | Complete | Partial | Missing | %     |
|----------------------|------------|------|----------|---------|---------|-------|
| `algorithm`          | off        | 111  | 104      | 4       | 3       | 93.6% |
| `builtin_kernels`    | off        | 1028 | 508      | 166     | 354     | 49.4% |
| `comm`               | off        | 181  | 91       | 30      | 60      | 50.2% |
| `extensibility`      | off        | 195  | 105      | 17      | 73      | 53.8% |
| `kv_cache`           | off        | 275  | 61       | 95      | 119     | 22.1% |
| `layout`             | on         | 791  | 791      | 0       | 0       | 100%  |
| `linalg`             | off        | 4230 | 1120     | 530     | 2580    | 26.4% |
| `max`                | on         | 650  | 638      | 1       | 11      | 98.1% |
| `nn`                 | off        | 4859 | 946      | 669     | 3244    | 19.4% |
| `pipeline`           | off        | 403  | 195      | 99      | 109     | 48.3% |
| `profiling_range`    | on         | 7    | 7        | 0       | 0       | 100%  |
| `quantization`       | on         | 51   | 51       | 0       | 0       | 100%  |
| `shmem`              | off        | 332  | 93       | 72      | 167     | 28%   |
| `state_space`        | off        | 129  | 73       | 36      | 20      | 56.5% |
| `structured_kernels` | off        | 586  | 195      | 92      | 299     | 33.2% |

## `algorithm`

Bazel package: `//max/kernels/src/algorithm`.

5/8 modules complete.

| Module                    | APIs | Complete | Partial | Missing | %     |
|---------------------------|------|----------|---------|---------|-------|
| `algorithm.reduce_op`     | 16   | 12       | 4       | 0       | 75%   |
| `algorithm.rowwise`       | 25   | 23       | 0       | 2       | 92%   |
| `algorithm.rowwise_types` | 35   | 34       | 0       | 1       | 97.1% |

## `builtin_kernels`

Bazel package: `//max/kernels/src/graph_compiler/builtin_kernels`.

3/20 modules complete.

| Module                                  | APIs | Complete | Partial | Missing | %     |
|-----------------------------------------|------|----------|---------|---------|-------|
| `builtin_kernels.rowwise_normalization` | 9    | 1        | 0       | 8       | 11.1% |
| `builtin_kernels.rowwise_reductions`    | 5    | 1        | 2       | 2       | 20%   |
| `builtin_kernels.nan_check`             | 3    | 1        | 2       | 0       | 33.3% |
| `builtin_kernels.gather_scatter`        | 108  | 37       | 27      | 44      | 34.2% |
| `builtin_kernels.conv`                  | 51   | 20       | 9       | 22      | 39.2% |
| `builtin_kernels.quantization`          | 92   | 37       | 20      | 35      | 40.2% |
| `builtin_kernels.linalg`                | 85   | 39       | 9       | 37      | 45.8% |
| `builtin_kernels.kernels`               | 155  | 72       | 19      | 64      | 46.4% |
| `builtin_kernels.attention`             | 114  | 53       | 12      | 49      | 46.4% |
| `builtin_kernels.elementwise`           | 91   | 44       | 2       | 45      | 48.3% |
| `builtin_kernels.distributed`           | 24   | 12       | 8       | 4       | 50%   |
| `builtin_kernels.ep`                    | 58   | 29       | 28      | 1       | 50%   |
| `builtin_kernels.kv_cache`              | 38   | 19       | 1       | 18      | 50%   |
| `builtin_kernels.rowwise_softmax`       | 2    | 1        | 0       | 1       | 50%   |
| `builtin_kernels.short_conv_ring`       | 5    | 3        | 0       | 2       | 60%   |
| `builtin_kernels.monoids`               | 109  | 71       | 20      | 18      | 65.1% |
| `builtin_kernels.reductions`            | 60   | 49       | 7       | 4       | 81.6% |

## `comm`

Bazel package: `//max/kernels/src/comm`.

3/18 modules complete.

| Module                            | APIs | Complete | Partial | Missing | %     |
|-----------------------------------|------|----------|---------|---------|-------|
| `comm.allgather_rmsnorm`          | 5    | 1        | 2       | 2       | 20%   |
| `comm.reducescatter`              | 20   | 5        | 8       | 7       | 25%   |
| `comm.allreduce_residual_rmsnorm` | 3    | 1        | 2       | 0       | 33.3% |
| `comm.reducescatter_rmsnorm`      | 3    | 1        | 1       | 1       | 33.3% |
| `comm.rms_norm_fp8`               | 3    | 1        | 2       | 0       | 33.3% |
| `comm.scatter`                    | 3    | 1        | 2       | 0       | 33.3% |
| `comm.vendor.ccl`                 | 33   | 13       | 5       | 15      | 39.3% |
| `comm.device_query`               | 21   | 9        | 1       | 11      | 42.8% |
| `comm.allreduce`                  | 23   | 10       | 0       | 13      | 43.4% |
| `comm.allgather`                  | 5    | 3        | 0       | 2       | 60%   |
| `comm.allreduce_lamport_rmsnorm`  | 3    | 2        | 1       | 0       | 66.6% |
| `comm.broadcast`                  | 6    | 4        | 2       | 0       | 66.6% |
| `comm.relay`                      | 13   | 9        | 0       | 4       | 69.2% |
| `comm.sync`                       | 19   | 14       | 4       | 1       | 73.6% |
| `comm.lamport`                    | 18   | 14       | 0       | 4       | 77.7% |

## `extensibility`

Bazel package: `//max/kernels/src/graph_compiler/extensibility`.

4/9 modules complete.

| Module                               | APIs | Complete | Partial | Missing | %     |
|--------------------------------------|------|----------|---------|---------|-------|
| `extensibility.managed_tensor_slice` | 112  | 34       | 16      | 62      | 30.3% |
| `extensibility.operation_traits`     | 9    | 5        | 0       | 4       | 55.5% |
| `extensibility.stable_tensor`        | 15   | 9        | 0       | 6       | 60%   |
| `extensibility.tensor_tuple`         | 10   | 9        | 0       | 1       | 90%   |
| `extensibility.stable_addr_engine`   | 21   | 20       | 1       | 0       | 95.2% |

## `kv_cache`

Bazel package: `//max/kernels/src/kv_cache`.

1/3 modules complete.

| Module                                 | APIs | Complete | Partial | Missing | %     |
|----------------------------------------|------|----------|---------|---------|-------|
| `kv_cache.types`                       | 233  | 21       | 93      | 119     | 9%    |
| `kv_cache.paged_sparse_kv_index_remap` | 41   | 39       | 2       | 0       | 95.1% |

## `layout`

Bazel package: `//max/kernels/src/layout`.

18/18 modules complete.

## `linalg`

Bazel package: `//max/kernels/src/linalg`.

41/167 modules complete.

| Module                                                                                            | APIs | Complete | Partial | Missing | %     |
|---------------------------------------------------------------------------------------------------|------|----------|---------|---------|-------|
| `linalg.matmul.gpu.sm100_structured.structured_kernels.epilogue_components`                       | 131  | 3        | 30      | 98      | 2.2%  |
| `linalg.matmul.gpu.sm100.block_scaled_matmul`                                                     | 40   | 2        | 6       | 32      | 5%    |
| `linalg.matmul.gpu.sm100_structured.block_scaled.block_scaled_smem`                               | 58   | 3        | 24      | 31      | 5.1%  |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul_kernel`                    | 85   | 6        | 1       | 78      | 7%    |
| `linalg.matmul.gpu.sm100_structured.default.dispatch`                                             | 14   | 1        | 10      | 3       | 7.1%  |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_smem`                             | 41   | 3        | 16      | 22      | 7.3%  |
| `linalg.matmul.gpu.tile_scheduler_splitk`                                                         | 41   | 3        | 0       | 38      | 7.3%  |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_1d1d_smem`                  | 54   | 4        | 11      | 39      | 7.4%  |
| `linalg.matmul.vendor.blas`                                                                       | 25   | 2        | 3       | 20      | 8%    |
| `linalg.matmul.gpu.sm90.dispatch`                                                                 | 12   | 1        | 3       | 8       | 8.3%  |
| `linalg.matmul.gpu.sm100_structured.block_scaled.block_scaled_matmul_kernel`                      | 95   | 8        | 0       | 87      | 8.4%  |
| `linalg.matmul.gpu.sm100.block_scaled_matmul_small_bn`                                            | 58   | 5        | 1       | 52      | 8.6%  |
| `linalg.grouped_matmul`                                                                           | 11   | 1        | 10      | 0       | 9%    |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8_1d2d.blockwise_fp8_1d2d_matmul_kernel`          | 70   | 7        | 0       | 63      | 10%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_smem`               | 20   | 2        | 7       | 11      | 10%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.dispatch`                           | 10   | 1        | 3       | 6       | 10%   |
| `linalg.matmul.gpu.apple.matmul2d_fp8`                                                            | 19   | 2        | 3       | 14      | 10.5% |
| `linalg.matmul.gpu.sm100_structured.default.tuning_configs`                                       | 37   | 4        | 0       | 33      | 10.8% |
| `linalg.matmul.gpu.sm100_structured.fused_swiglu.tuning_configs`                                  | 18   | 2        | 0       | 16      | 11.1% |
| `linalg.matmul.gpu.tile_scheduler`                                                                | 45   | 5        | 0       | 40      | 11.1% |
| `linalg.matmul.gpu.sm100.tile_scheduler`                                                          | 26   | 3        | 0       | 23      | 11.5% |
| `linalg.matmul.gpu.amd.mxfp4_moe_matmul_amd`                                                      | 25   | 3        | 2       | 20      | 12%   |
| `linalg.matmul.gpu.__init__`                                                                      | 16   | 2        | 10      | 4       | 12.5% |
| `linalg.matmul.gpu.sm90.tuning_configs`                                                           | 24   | 3        | 0       | 21      | 12.5% |
| `linalg.matmul.gpu.sm100_structured.default.matmul_kernels`                                       | 187  | 24       | 4       | 159     | 12.8% |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_1d1d_matmul_kernel`         | 131  | 17       | 1       | 113     | 12.9% |
| `linalg.matmul.gpu.sm100_structured.fused_swiglu.matmul_swiglu_kernels`                           | 38   | 5        | 0       | 33      | 13.1% |
| `linalg.fp8_quantization`                                                                         | 15   | 2        | 12      | 1       | 13.3% |
| `linalg.grouped_matmul_tile_scheduler`                                                            | 30   | 4        | 0       | 26      | 13.3% |
| `linalg.matmul.gpu.amd_rdna.matmul`                                                               | 15   | 2        | 2       | 11      | 13.3% |
| `linalg.matmul.gpu.amd.block_scaled_matmul_amd`                                                   | 81   | 11       | 6       | 64      | 13.5% |
| `linalg.matmul.gpu.amd.amd_ping_pong_matmul`                                                      | 49   | 7        | 1       | 41      | 14.2% |
| `linalg.matmul.gpu.apple.fp4_matmul`                                                              | 21   | 3        | 1       | 17      | 14.2% |
| `linalg.matmul.gpu.apple.grouped_matmul`                                                          | 7    | 1        | 5       | 1       | 14.2% |
| `linalg.matmul.gpu.sm100.blockwise_fp8`                                                           | 7    | 1        | 3       | 3       | 14.2% |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8_1d2d.blockwise_fp8_1d2d_smem`                   | 14   | 2        | 8       | 4       | 14.2% |
| `linalg.matmul.gpu.sm90.matmul_output`                                                            | 21   | 3        | 0       | 18      | 14.2% |
| `linalg.utils_gpu`                                                                                | 42   | 6        | 2       | 34      | 14.2% |
| `linalg.matmul.gpu.amd.block_scaled_preshuffle_layouts`                                           | 27   | 4        | 5       | 18      | 14.8% |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul_kernel`      | 128  | 19       | 0       | 109     | 14.8% |
| `linalg.matmul.gpu.profiler`                                                                      | 20   | 3        | 0       | 17      | 15%   |
| `linalg.matmul.gpu.sm100_structured.fused_swiglu.config`                                          | 31   | 5        | 5       | 21      | 16.1% |
| `linalg.matmul.gpu.amd.block_scaled_grouped_matmul_amd`                                           | 18   | 3        | 2       | 13      | 16.6% |
| `linalg.matmul.gpu.sm90.matmul`                                                                   | 6    | 1        | 4       | 1       | 16.6% |
| `linalg.matmul.gpu.amd.amd_matmul`                                                                | 33   | 6        | 1       | 26      | 18.1% |
| `linalg.matmul.gpu.amd.block_scaled_matmul_amd_preb`                                              | 77   | 14       | 0       | 63      | 18.1% |
| `linalg.matmul.gpu.amd.amd_target`                                                                | 16   | 3        | 6       | 7       | 18.7% |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_output_writer`                    | 21   | 4        | 0       | 17      | 19%   |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.tile_scheduler`                            | 76   | 15       | 6       | 55      | 19.7% |
| `linalg.matmul.gpu.apple.fp8_gemv`                                                                | 10   | 2        | 7       | 1       | 20%   |
| `linalg.matmul.gpu.apple.matmul_8x8`                                                              | 5    | 1        | 2       | 2       | 20%   |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_accumulator`                      | 25   | 5        | 1       | 19      | 20%   |
| `linalg.matmul.vendor.matmul`                                                                     | 5    | 1        | 4       | 0       | 20%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_1d1d_tile_scheduler`        | 83   | 17       | 14      | 52      | 20.4% |
| `linalg.bmm`                                                                                      | 14   | 3        | 10      | 1       | 21.4% |
| `linalg.matmul.gpu.apple.matmul2d_fp4`                                                            | 42   | 9        | 8       | 25      | 21.4% |
| `linalg.gemv`                                                                                     | 45   | 10       | 12      | 23      | 22.2% |
| `linalg.matmul.gpu.sm100.block_scaled_dispatch`                                                   | 9    | 2        | 3       | 4       | 22.2% |
| `linalg.matmul.gpu.sm100_structured.default.matmul`                                               | 9    | 2        | 7       | 0       | 22.2% |
| `linalg.packing`                                                                                  | 27   | 6        | 5       | 16      | 22.2% |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.warp_context`                              | 53   | 12       | 5       | 36      | 22.6% |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.output_writer`                             | 99   | 23       | 8       | 68      | 23.2% |
| `linalg.matmul.gpu.apple.int8_matmul`                                                             | 30   | 7        | 2       | 21      | 23.3% |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.tile_pipeline`                             | 242  | 57       | 42      | 143     | 23.5% |
| `linalg.matmul.__init__`                                                                          | 4    | 1        | 3       | 0       | 25%   |
| `linalg.matmul.gpu.amd.matmul_mma`                                                                | 40   | 10       | 1       | 29      | 25%   |
| `linalg.mxfp6_dequant`                                                                            | 4    | 1        | 1       | 2       | 25%   |
| `linalg.arch.apple.mma`                                                                           | 34   | 9        | 2       | 23      | 26.4% |
| `linalg.matmul.gpu.sm100.matmul`                                                                  | 18   | 5        | 1       | 12      | 27.7% |
| `linalg.matmul.gpu.sm90.matmul_kernels`                                                           | 57   | 16       | 10      | 31      | 28%   |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.config`                                    | 94   | 27       | 1       | 66      | 28.7% |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.tile_scheduler_splitk`                     | 89   | 26       | 7       | 56      | 29.2% |
| `linalg.matmul.gpu.amd.amd_ping_pong_schedule`                                                    | 30   | 9        | 5       | 16      | 30%   |
| `linalg.matmul.cpu.apple_accelerate`                                                              | 13   | 4        | 9       | 0       | 30.7% |
| `linalg.matmul.gpu.apple.matmul_kernel`                                                           | 64   | 20       | 5       | 39      | 31.2% |
| `linalg.matmul.gpu.sm90.config`                                                                   | 25   | 8        | 4       | 13      | 32%   |
| `linalg.block_scaled_quantization`                                                                | 31   | 10       | 15      | 6       | 32.2% |
| `linalg.lora`                                                                                     | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.matmul.gpu.amd.smallm_streaming_matmul`                                                   | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.matmul.gpu.apple.gemv`                                                                    | 6    | 2        | 4       | 0       | 33.3% |
| `linalg.matmul.gpu.sm100.testbed_block_scaled_fp4`                                                | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8_1d2d.blockwise_fp8_1d2d_matmul`                 | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.matmul.gpu.sm100_structured.fused_swiglu.dispatch`                                        | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.matmul.gpu.sm80.dispatch`                                                                 | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.matmul.gpu.sm90.testbed_swapAB`                                                           | 3    | 1        | 2       | 0       | 33.3% |
| `linalg.arch.sm100.mma`                                                                           | 26   | 9        | 3       | 14      | 34.6% |
| `linalg.matmul.gpu.amd.block_scaled_preshuffle_loaders`                                           | 14   | 5        | 5       | 4       | 35.7% |
| `linalg.matmul.gpu.amd.amd_matmul_schedule`                                                       | 27   | 10       | 5       | 12      | 37%   |
| `linalg.utils`                                                                                    | 79   | 30       | 19      | 30      | 37.9% |
| `linalg.arch.amd.block_scaled_mma`                                                                | 13   | 5        | 1       | 7       | 38.4% |
| `linalg.fp6_quantization`                                                                         | 5    | 2        | 1       | 2       | 40%   |
| `linalg.fp6_utils`                                                                                | 25   | 10       | 5       | 10      | 40%   |
| `linalg.matmul.gpu.sm90.tile_writer`                                                              | 52   | 22       | 0       | 30      | 42.3% |
| `linalg.grouped_matmul_sm100_blockwise_fp8`                                                       | 16   | 7        | 4       | 5       | 43.7% |
| `linalg.grouped_matmul_sm100`                                                                     | 25   | 11       | 2       | 12      | 44%   |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.tmem`                                      | 109  | 48       | 21      | 40      | 44%   |
| `linalg.fp4_utils`                                                                                | 36   | 17       | 3       | 16      | 47.2% |
| `linalg.grouped_matmul_block_scaled_dispatch`                                                     | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd`                                                  | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul`                           | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_1d1d_matmul`                | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_matmul_block_scaled_swiglu` | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_matmul_swiglu_mxfp8`        | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_matmul_swiglu_nvfp4`        | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.tile_loader`                               | 22   | 11       | 0       | 11      | 50%   |
| `linalg.matmul.gpu.sm90.testbed`                                                                  | 2    | 1        | 1       | 0       | 50%   |
| `linalg.matrix_band_part`                                                                         | 2    | 1        | 1       | 0       | 50%   |
| `linalg.mxfp4_dequant`                                                                            | 2    | 1        | 1       | 0       | 50%   |
| `linalg.mxfp4_matmul_sm90`                                                                        | 2    | 1        | 1       | 0       | 50%   |
| `linalg.mx_format`                                                                                | 15   | 8        | 0       | 7       | 53.3% |
| `linalg.matmul.gpu.sm90.tile_loader`                                                              | 26   | 15       | 0       | 11      | 57.6% |
| `linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_tile_scheduler`                  | 78   | 47       | 7       | 24      | 60.2% |
| `linalg.matmul.gpu.amd.blockwise_scaled_fp8_grouped_matmul_amd`                                   | 3    | 2        | 1       | 0       | 66.6% |
| `linalg.matmul.gpu.apple.fp4_gemv`                                                                | 3    | 2        | 1       | 0       | 66.6% |
| `linalg.matmul.gpu.sm100_structured.default.dispatch_fused_bias_residual`                         | 3    | 2        | 0       | 1       | 66.6% |
| `linalg.matmul.gpu.sm90.grouped_matmul`                                                           | 3    | 2        | 1       | 0       | 66.6% |
| `linalg.matmul.gpu.sm100_structured.structured_kernels.row_scales`                                | 24   | 17       | 0       | 7       | 70.8% |
| `linalg.matmul.gpu.apple.fp4_dequant`                                                             | 4    | 3        | 1       | 0       | 75%   |
| `linalg.transpose`                                                                                | 9    | 7        | 2       | 0       | 77.7% |
| `linalg.matmul.gpu.amd.pipeline_body`                                                             | 14   | 11       | 2       | 1       | 78.5% |
| `linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd`                                           | 5    | 4        | 1       | 0       | 80%   |
| `linalg.matmul.cpu.impl`                                                                          | 18   | 15       | 3       | 0       | 83.3% |
| `linalg.matmul.gpu.sm100.warp_specialized_blockwise_fp8`                                          | 6    | 5        | 1       | 0       | 83.3% |
| `linalg.arch.cpu.vnni_intrinsics`                                                                 | 13   | 11       | 2       | 0       | 84.6% |
| `linalg.arch.cpu.apple_amx_intrinsics`                                                            | 32   | 30       | 2       | 0       | 93.7% |
| `linalg.matmul.gpu.amd.amd_4wave_matmul`                                                          | 73   | 72       | 0       | 1       | 98.6% |

## `max`

Bazel package: `//max/mojo/max`.

52/53 modules complete.

| Module              | APIs | Complete | Partial | Missing | %     |
|---------------------|------|----------|---------|---------|-------|
| `max.gpu.host.info` | 109  | 97       | 1       | 11      | 88.9% |

## `nn`

Bazel package: `//max/kernels/src/nn`.

49/196 modules complete.

| Module                                                                        | APIs | Complete | Partial | Missing | %     |
|-------------------------------------------------------------------------------|------|----------|---------|---------|-------|
| `nn.attention.gpu.nvidia.sm100.mla_decode_dispatch`                           | 23   | 0        | 10      | 13      | 0%    |
| `nn.attention.gpu.nvidia.sm100.mla_decode_sparse_kv_fp8`                      | 32   | 0        | 1       | 31      | 0%    |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_generic`                           | 12   | 0        | 0       | 12      | 0%    |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_utils`                             | 139  | 0        | 11      | 128     | 0%    |
| `nn.normalization`                                                            | 27   | 0        | 12      | 15      | 0%    |
| `nn.rope`                                                                     | 5    | 0        | 0       | 5       | 0%    |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils`                      | 143  | 1        | 0       | 142     | 0.6%  |
| `nn.attention.gpu.nvidia.sm100.attention`                                     | 117  | 1        | 52      | 64      | 0.8%  |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8`                    | 114  | 1        | 1       | 112     | 0.8%  |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_kv_fp8`                     | 79   | 1        | 0       | 78      | 1.2%  |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_sparse`                            | 65   | 1        | 0       | 64      | 1.5%  |
| `nn.attention.gpu.nvidia.sm100.mha_depth512.config`                           | 44   | 1        | 1       | 42      | 2.2%  |
| `nn.attention.gpu.nvidia.sm100.mla_decode_sparse_qkv_fp8`                     | 39   | 1        | 4       | 34      | 2.5%  |
| `nn.attention.gpu.nvidia.sm100.mla_decode_sparse_kv_bf16`                     | 28   | 1        | 1       | 26      | 3.5%  |
| `nn.attention.gpu.nvidia.sm100.kernel`                                        | 46   | 2        | 0       | 44      | 4.3%  |
| `nn.attention.gpu.nvidia.sm100.smem`                                          | 63   | 3        | 15      | 45      | 4.7%  |
| `nn.conv.gpu.nvidia.sm100.conv_smem`                                          | 40   | 2        | 6       | 32      | 5%    |
| `nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8_layout_g`                   | 31   | 2        | 0       | 29      | 6.4%  |
| `nn.conv.gpu.nvidia.sm100.conv_config`                                        | 44   | 3        | 12      | 29      | 6.8%  |
| `nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8_per_token_scale_rope_aware` | 29   | 2        | 0       | 27      | 6.8%  |
| `nn.attention.gpu.nvidia.sm100.mla_decode_sparse`                             | 43   | 3        | 0       | 40      | 6.9%  |
| `nn.kv_cache_ragged`                                                          | 26   | 2        | 24      | 0       | 7.6%  |
| `nn.attention.gpu.amd_rdna.attention`                                         | 72   | 6        | 0       | 66      | 8.3%  |
| `nn.attention.gpu.amd_rdna.softmax`                                           | 36   | 3        | 0       | 33      | 8.3%  |
| `nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8`                            | 22   | 2        | 0       | 20      | 9%    |
| `nn.attention.gpu.nvidia.sm100.mha_depth512.smem`                             | 32   | 3        | 8       | 21      | 9.3%  |
| `nn.attention.gpu.nvidia.sm100.mha_depth512.kernel`                           | 21   | 2        | 0       | 19      | 9.5%  |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_per_token_scale`                   | 21   | 2        | 2       | 17      | 9.5%  |
| `nn.attention.gpu.nvidia.sm100.mla_decode_combine`                            | 51   | 5        | 3       | 43      | 9.8%  |
| `nn.attention.gpu.nvidia.sm100.nvfp4_kv_dequant`                              | 10   | 1        | 7       | 2       | 10%   |
| `nn.attention.gpu.amd_structured.kv_buffer`                                   | 108  | 11       | 5       | 92      | 10.1% |
| `nn.attention.gpu.nvidia.sm100.mla_decode_kv_fp8`                             | 19   | 2        | 0       | 17      | 10.5% |
| `nn.conv.gpu.nvidia.sm100.conv2d_fprop_kernel`                                | 93   | 10       | 0       | 83      | 10.7% |
| `nn.attention.gpu.nvidia.sm100.mha_depth512.barriers`                         | 36   | 4        | 14      | 18      | 11.1% |
| `nn.attention.gpu.nvidia.sm100.mla_decode_kv_bf16`                            | 17   | 2        | 0       | 15      | 11.7% |
| `nn.kv_cache`                                                                 | 17   | 2        | 11      | 4       | 11.7% |
| `nn.attention.mha_mask`                                                       | 324  | 39       | 17      | 268     | 12%   |
| `nn.attention.gpu.amd_rdna.config`                                            | 16   | 2        | 0       | 14      | 12.5% |
| `nn.attention.gpu.nvidia.common`                                              | 64   | 8        | 12      | 44      | 12.5% |
| `nn.attention.gpu.nvidia.mha_tile_scheduler`                                  | 96   | 12       | 5       | 79      | 12.5% |
| `nn.attention.gpu.amd_rdna.buffers`                                           | 108  | 14       | 2       | 92      | 12.9% |
| `nn.attention.gpu.amd_structured.buffers`                                     | 45   | 6        | 3       | 36      | 13.3% |
| `nn.attention.gpu.nvidia.sm100.dispatch`                                      | 15   | 2        | 2       | 11      | 13.3% |
| `nn.attention.gpu.nvidia.sm100.mla_decode_utils`                              | 423  | 57       | 11      | 355     | 13.4% |
| `nn.toppminp_gpu`                                                             | 22   | 3        | 11      | 8       | 13.6% |
| `nn.attention.mha_utils`                                                      | 128  | 18       | 3       | 107     | 14%   |
| `nn.attention.gpu.sparse_index_fp8_sm100`                                     | 7    | 1        | 4       | 2       | 14.2% |
| `nn.slice`                                                                    | 7    | 1        | 6       | 0       | 14.2% |
| `nn.attention.gpu.amd_structured.attention`                                   | 88   | 14       | 1       | 73      | 15.9% |
| `nn.attention.gpu.amd_structured.softmax`                                     | 43   | 7        | 0       | 36      | 16.2% |
| `nn.attention.gpu.nvidia.sm100.mla_prefill_blockscale`                        | 12   | 2        | 2       | 8       | 16.6% |
| `nn.attention.gpu.nvidia.sm100.softmax_warp`                                  | 18   | 3        | 12      | 3       | 16.6% |
| `nn.attention.gpu.amd_structured.mha_prefill_v2`                              | 23   | 4        | 3       | 16      | 17.3% |
| `nn.attention.gpu.amd_structured.ps_metadata`                                 | 17   | 3        | 4       | 10      | 17.6% |
| `nn.attention.gpu.nvidia.sm100.attention_utils`                               | 252  | 45       | 57      | 150     | 17.8% |
| `nn.moe`                                                                      | 22   | 4        | 18      | 0       | 18.1% |
| `nn.topk`                                                                     | 22   | 4        | 12      | 6       | 18.1% |
| `nn.pool`                                                                     | 16   | 3        | 8       | 5       | 18.7% |
| `nn.attention.mha_operand`                                                    | 202  | 38       | 53      | 111     | 18.8% |
| `nn.conv.conv_utils`                                                          | 94   | 18       | 28      | 48      | 19.1% |
| `nn.conv.conv`                                                                | 101  | 20       | 10      | 71      | 19.8% |
| `nn.sampling.coop_row`                                                        | 10   | 2        | 5       | 3       | 20%   |
| `nn.attention.gpu.apple.fa_prefill`                                           | 8    | 2        | 1       | 5       | 25%   |
| `nn.attention.gpu.nvidia.sm100.mla_nvfp4_kv_to_fp8`                           | 4    | 1        | 3       | 0       | 25%   |
| `nn.attention.gpu.nvidia.sm100.mla_prefill`                                   | 4    | 1        | 3       | 0       | 25%   |
| `nn.concat`                                                                   | 8    | 2        | 6       | 0       | 25%   |
| `nn.conv.gpu.amd.rdna.conv2d_kernel`                                          | 8    | 2        | 0       | 6       | 25%   |
| `nn.pad_gpu`                                                                  | 8    | 2        | 5       | 1       | 25%   |
| `nn.tile`                                                                     | 4    | 1        | 3       | 0       | 25%   |
| `nn.attention.gpu.mla`                                                        | 23   | 6        | 7       | 10      | 26%   |
| `nn.attention.gpu.mla_graph`                                                  | 15   | 4        | 11      | 0       | 26.6% |
| `nn.attention.gpu.sparse_index_fp8_sm100_prefill`                             | 43   | 12       | 6       | 25      | 27.9% |
| `nn.attention.gpu.amd_structured.mla_prefill_v2`                              | 14   | 4        | 1       | 9       | 28.5% |
| `nn.roi_align`                                                                | 7    | 2        | 1       | 4       | 28.5% |
| `nn.attention.gpu.amd_structured.config`                                      | 17   | 5        | 0       | 12      | 29.4% |
| `nn.gather_scatter`                                                           | 27   | 8        | 15      | 4       | 29.6% |
| `nn.resize`                                                                   | 25   | 8        | 3       | 14      | 32%   |
| `nn.argsort`                                                                  | 3    | 1        | 2       | 0       | 33.3% |
| `nn.attention.gpu.amd_rdna.utils`                                             | 3    | 1        | 2       | 0       | 33.3% |
| `nn.attention.gpu.amd_structured.mask_op`                                     | 6    | 2        | 0       | 4       | 33.3% |
| `nn.attention.gpu.nvidia.sm100.fa4_splitk_combine`                            | 3    | 1        | 1       | 1       | 33.3% |
| `nn.attention.gpu.nvidia.sm100.mha_depth512.dispatch`                         | 3    | 1        | 1       | 1       | 33.3% |
| `nn.conv.conv_transpose`                                                      | 27   | 9        | 6       | 12      | 33.3% |
| `nn.conv.gpu.im2col_matmul_2d`                                                | 3    | 1        | 2       | 0       | 33.3% |
| `nn.fold`                                                                     | 3    | 1        | 2       | 0       | 33.3% |
| `nn.index_tensor`                                                             | 3    | 1        | 2       | 0       | 33.3% |
| `nn.indexer_candidates`                                                       | 3    | 1        | 2       | 0       | 33.3% |
| `nn.repeat_interleave`                                                        | 3    | 1        | 2       | 0       | 33.3% |
| `nn.roi_align_gpu`                                                            | 3    | 1        | 2       | 0       | 33.3% |
| `nn.rope_split_store`                                                         | 3    | 1        | 2       | 0       | 33.3% |
| `nn.sampling.__init__`                                                        | 3    | 1        | 2       | 0       | 33.3% |
| `nn.short_conv_ring`                                                          | 3    | 1        | 2       | 0       | 33.3% |
| `nn.attention.gpu.apple.naive_fa_decode`                                      | 8    | 3        | 1       | 4       | 37.5% |
| `nn.conv.gpu.nvidia.sm100.conv_tile_loader`                                   | 8    | 3        | 0       | 5       | 37.5% |
| `nn.image`                                                                    | 34   | 13       | 0       | 21      | 38.2% |
| `nn.sampling.sampling`                                                        | 5    | 2        | 3       | 0       | 40%   |
| `nn.attention.gpu.mha`                                                        | 31   | 13       | 12      | 6       | 41.9% |
| `nn.arg_nonzero`                                                              | 4    | 2        | 2       | 0       | 50%   |
| `nn.argmaxmin`                                                                | 2    | 1        | 1       | 0       | 50%   |
| `nn.argmaxmin_gpu`                                                            | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.compress_windows`                                               | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.gpu.amd_structured.utils`                                       | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.gpu.mha_cross`                                                  | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.gpu.mla_decode_dispatch_scalars`                                | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.gpu.nvidia.sm100.correction_warp`                               | 2    | 1        | 0       | 1       | 50%   |
| `nn.attention.gpu.nvidia.sm100.mma_warp`                                      | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.gpu.nvidia.sm90.mha`                                            | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.indexer_score`                                                  | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.latent_sparse_attention`                                        | 2    | 1        | 1       | 0       | 50%   |
| `nn.conv.gpu.amd.dispatch`                                                    | 2    | 1        | 1       | 0       | 50%   |
| `nn.conv.gpu.amd.dispatch_3d`                                                 | 2    | 1        | 1       | 0       | 50%   |
| `nn.conv.gpu.amd.rdna.dispatch`                                               | 2    | 1        | 1       | 0       | 50%   |
| `nn.conv.gpu.im2col_matmul_3d`                                                | 2    | 1        | 1       | 0       | 50%   |
| `nn.conv.gpu.matmul_1x1x1_conv3d`                                             | 2    | 1        | 1       | 0       | 50%   |
| `nn.conv.gpu.nvidia.sm100.qslice_conv3d`                                      | 2    | 1        | 1       | 0       | 50%   |
| `nn.kv_cache_gather`                                                          | 2    | 1        | 1       | 0       | 50%   |
| `nn.learnable_2d_interp_pos_emb`                                              | 2    | 1        | 1       | 0       | 50%   |
| `nn.rand_normal`                                                              | 2    | 1        | 1       | 0       | 50%   |
| `nn.randn`                                                                    | 2    | 1        | 1       | 0       | 50%   |
| `nn.shard_and_stack`                                                          | 2    | 1        | 1       | 0       | 50%   |
| `nn.attention.gpu.amd_structured.mla_components`                              | 23   | 12       | 1       | 10      | 52.1% |
| `nn.attention.gpu.mha_decode_partition_heuristic`                             | 11   | 6        | 3       | 2       | 54.5% |
| `nn.index_fp8`                                                                | 9    | 5        | 1       | 3       | 55.5% |
| `nn.attention.gpu.amd_structured.mma`                                         | 21   | 12       | 0       | 9       | 57.1% |
| `nn.sampling.topk_fi`                                                         | 21   | 12       | 1       | 8       | 57.1% |
| `nn.attention.gpu.amd_structured.mha_mma_op`                                  | 92   | 54       | 5       | 33      | 58.6% |
| `nn.irfft`                                                                    | 3    | 2        | 1       | 0       | 66.6% |
| `nn.rand_uniform`                                                             | 3    | 2        | 1       | 0       | 66.6% |
| `nn.spatial_merge`                                                            | 3    | 2        | 1       | 0       | 66.6% |
| `nn.topk_per_row`                                                             | 3    | 2        | 1       | 0       | 66.6% |
| `nn.toppminp`                                                                 | 6    | 4        | 2       | 0       | 66.6% |
| `nn.tpool_patch_merger`                                                       | 3    | 2        | 1       | 0       | 66.6% |
| `nn.attention.gpu.amd_structured.mha_softmax`                                 | 21   | 15       | 2       | 4       | 71.4% |
| `nn.bicubic`                                                                  | 7    | 5        | 2       | 0       | 71.4% |
| `nn.attention.gpu.mla_index_fp8`                                              | 44   | 32       | 12      | 0       | 72.7% |
| `nn.attention.gpu.amd_structured.iglp`                                        | 12   | 9        | 0       | 3       | 75%   |
| `nn.attention.gpu.mla_index_kpool`                                            | 8    | 6        | 1       | 1       | 75%   |
| `nn.attention.gpu.nvidia.sm90.attention`                                      | 4    | 3        | 1       | 0       | 75%   |
| `nn.nms`                                                                      | 13   | 10       | 1       | 2       | 76.9% |
| `nn.softmax`                                                                  | 9    | 7        | 2       | 0       | 77.7% |
| `nn.attention.gpu.amd_structured.mha_mask`                                    | 5    | 4        | 0       | 1       | 80%   |
| `nn.pad`                                                                      | 5    | 4        | 1       | 0       | 80%   |
| `nn.sampling.topk_fi_cluster`                                                 | 5    | 4        | 1       | 0       | 80%   |
| `nn.topk_bitonic`                                                             | 20   | 16       | 2       | 2       | 80%   |
| `nn.attention.cpu.mha`                                                        | 6    | 5        | 1       | 0       | 83.3% |
| `nn.conv.gpu.nvidia.sm100.epilogue_load_pipeline`                             | 26   | 22       | 1       | 3       | 84.6% |
| `nn.fused_qk_rope`                                                            | 7    | 6        | 1       | 0       | 85.7% |

## `pipeline`

Bazel package: `//max/kernels/src/pipeline`.

4/13 modules complete.

| Module                      | APIs | Complete | Partial | Missing | %     |
|-----------------------------|------|----------|---------|---------|-------|
| `pipeline.program_builder`  | 14   | 3        | 11      | 0       | 21.4% |
| `pipeline.compiler`         | 23   | 5        | 7       | 11      | 21.7% |
| `pipeline.pipeline_dsl`     | 26   | 6        | 15      | 5       | 23%   |
| `pipeline.phase_derivation` | 34   | 11       | 18      | 5       | 32.3% |
| `pipeline.program`          | 40   | 14       | 9       | 17      | 35%   |
| `pipeline.config`           | 88   | 36       | 19      | 33      | 40.9% |
| `pipeline.dependency_graph` | 13   | 6        | 4       | 3       | 46.1% |
| `pipeline.types`            | 103  | 55       | 13      | 35      | 53.3% |
| `pipeline.schedulers`       | 8    | 5        | 3       | 0       | 62.5% |

## `profiling_range`

Bazel package: `//max/kernels/src/profiling_range`.

2/2 modules complete.

## `quantization`

Bazel package: `//max/kernels/src/quantization`.

5/5 modules complete.

## `shmem`

Bazel package: `//max/kernels/src/shmem`.

1/6 modules complete.

| Module                | APIs | Complete | Partial | Missing | %     |
|-----------------------|------|----------|---------|---------|-------|
| `shmem.ep`            | 13   | 2        | 11      | 0       | 15.3% |
| `shmem.ep_comm`       | 242  | 56       | 42      | 144     | 23.1% |
| `shmem.shmem_api`     | 49   | 19       | 12      | 18      | 38.7% |
| `shmem.shmem_buffer`  | 11   | 5        | 1       | 5       | 45.4% |
| `shmem.shmem_context` | 16   | 10       | 6       | 0       | 62.5% |

## `state_space`

Bazel package: `//max/kernels/src/state_space`.

2/14 modules complete.

| Module                                    | APIs | Complete | Partial | Missing | %     |
|-------------------------------------------|------|----------|---------|---------|-------|
| `state_space.gated_group_rmsnorm`         | 6    | 1        | 1       | 4       | 16.6% |
| `state_space.varlen_causal_conv1d`        | 14   | 4        | 5       | 5       | 28.5% |
| `state_space.gated_delta_conv1d`          | 3    | 1        | 1       | 1       | 33.3% |
| `state_space.causal_conv1d`               | 21   | 10       | 11      | 0       | 47.6% |
| `state_space.rms_norm_fused_residual_ops` | 4    | 2        | 1       | 1       | 50%   |
| `state_space.selective_scan`              | 15   | 8        | 5       | 2       | 53.3% |
| `state_space.mamba2_ssd_chunked`          | 30   | 19       | 10      | 1       | 63.3% |
| `state_space.selective_scan_ops`          | 10   | 7        | 0       | 3       | 70%   |
| `state_space.causal_conv1d_ops`           | 7    | 5        | 0       | 2       | 71.4% |
| `state_space.mamba2_replay_fold`          | 4    | 3        | 1       | 0       | 75%   |
| `state_space.rms_norm_fused_residual`     | 4    | 3        | 1       | 0       | 75%   |
| `state_space.mamba2_ssd_scan`             | 5    | 4        | 0       | 1       | 80%   |

## `structured_kernels`

Bazel package: `//max/kernels/src/structured_kernels`.

1/11 modules complete.

| Module                                | APIs | Complete | Partial | Missing | %     |
|---------------------------------------|------|----------|---------|---------|-------|
| `structured_kernels.pipeline_storage` | 151  | 17       | 39      | 95      | 11.2% |
| `structured_kernels.amd_tile_io_conv` | 39   | 6        | 0       | 33      | 15.3% |
| `structured_kernels.kernel_common`    | 66   | 11       | 11      | 44      | 16.6% |
| `structured_kernels.barriers`         | 29   | 6        | 8       | 15      | 20.6% |
| `structured_kernels.smem_types`       | 15   | 4        | 5       | 6       | 26.6% |
| `structured_kernels.pipeline_backend` | 47   | 15       | 1       | 31      | 31.9% |
| `structured_kernels.tile_types`       | 78   | 32       | 11      | 35      | 41%   |
| `structured_kernels.amd_tile_io`      | 80   | 48       | 7       | 25      | 60%   |
| `structured_kernels.pipeline`         | 58   | 36       | 7       | 15      | 62%   |
| `structured_kernels.trace_buf`        | 22   | 19       | 3       | 0       | 86.3% |
