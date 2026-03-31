# Nightly: v26.3

This version is still a work in progress.

## Highlights {#26-3-highlights}

## Documentation {#26-3-docs}

## MAX models {#26-3-models}

- The `residual_threshold` parameter for FLUX first-block cache (FBCache) is
  now a per-request runtime parameter on `ImageProviderOptions`, allowing it
  to be tuned without recompiling the model graph.
- Added TaylorSeer denoising cache support to the FLUX.2 Klein pipeline,
  enabling significant speedups for image-to-image generation by skipping
  redundant transformer passes during the denoising loop.
- Added the Mamba state space model architecture.

## MAX framework {#26-3-max}

### Inference server {#26-3-max-serve}

- Added periodic "still building/compiling" log messages during model
  compilation so that long operations produce visible signs of progress.
- Consolidated KV connector CLI flags (`--host-kvcache-swap-space-gb`,
  `--disk-offload-dir`, `--disk-offload-max-gb`, `--disk-offload-direct-io`,
  `--lmcache-config-file`) into the `--kv-connector-config` JSON dict.
- Removed the `--allow-safetensors-weights-fp32-bf16-bidirectional-cast` CLI
  flag. Float32 <-> bfloat16 safetensors weight casting is now unconditionally
  enabled.

### `max` CLI {#26-3-max-cli}

### Python API {#26-3-max-python}

- Fixed slow `axis=None` reductions (`mean`, `sum`, `prod`, `max`, `min`) in
  `max.experimental.functional`. The previous implementation flattened the
  tensor before reducing, serializing the work onto a single GPU block.
  Reductions now iterate axis-by-axis to preserve parallelism.
- Renamed `Float8Config` to `QuantConfig` (and related types/functions)
  to reflect that the config now covers FP8, NVFP4, and MXFP4 quantization.
- Renamed related public Python quantization APIs from `Float8*` names to
  `Quant*` names, including `parse_float8_config()` to
  `parse_quant_config()`, and the public `quant` modules in `max.nn` and
  `max.pipelines.lib`.
- `max.diagnostics.gpu.BackgroundRecorder`'s sampling interval can now be
  configured.
- Added experimental `max.experimental.distributed` module with `DTensor`,
  `DeviceMesh`, and placement types (`Replicated`, `Sharded`, `Partial`) for
  expressing how tensors are distributed across multiple devices. Op dispatch
  is not yet supported.
- Improved experimental eager interpreter performance by enabling multi-threaded
  CPU execution and removing unnecessary GPU device synchronization after each
  op dispatch.
- Added `gather` and `gather_nd` op handlers to the experimental eager
  interpreter with full CPU and GPU support.
- Added `argmax` and `argmin` op handlers to the experimental eager interpreter
  with full CPU and GPU support, returning int64 indices along a specified axis.
- Added `split` op handler to the experimental eager interpreter with full CPU
  and GPU support, splitting a tensor into multiple outputs along a specified
  axis.
- Added `scatter` op handler to the experimental eager interpreter (CPU),
  scattering updates into a copy of the input tensor along a specified axis.
- Added `conv2d` and `conv2d_transpose` op handlers to the experimental eager
  interpreter with CPU and GPU support.
- Added `max_pool2d` op handlers (floor and ceil mode) to the experimental
  eager interpreter with CPU and GPU support.
- Added `tile` op handler to the experimental eager interpreter (CPU),
  repeating the input tensor along each dimension.
- Added `band_part` op handler to the experimental eager interpreter with
  CPU and GPU support, masking tensor matrices based on a diagonal band.
- Added `avg_pool2d` op handlers (floor and ceil mode) to the experimental
  eager interpreter with CPU and GPU support.
- Added `top_k` op handler to the experimental eager interpreter with CPU
  and GPU support, returning the top-k values and their original indices
  along a specified axis.
- `Module.compile()` now accepts a `custom_extensions` parameter for loading
  custom Mojo kernel libraries at graph construction time, fixing validation
  failures for kernels with struct-level parameters.
- Fixed `torch.compile(fullgraph=True)` failing with an "Unsupported context
  manager" error when accessing `CustomOpLibrary` ops inside the compiled
  function. Ops are now eagerly compiled during library initialization.

## Breaking changes {#26-3-breaking}

- Removed individual KV connector CLI flags (`--host-kvcache-swap-space-gb`,
  `--disk-offload-dir`, `--disk-offload-max-gb`, `--disk-offload-direct-io`,
  `--lmcache-config-file`). Use `--kv-connector-config` with a JSON dict
  instead.

- `max/python/max/benchmark/benchmark_throughput.py` has been deprecated and
  will be removed in a future MAX release.

- Removed `Dim` and `DimList` types from `buffer.dimlist`. Custom kernel code
  using these types should migrate to `IntTuple` and `TileLayout` from the
  `layout` package.

### Mojo API {#26-3-max-mojo}

### Custom ops {#26-3-custom-ops}

## MAX kernels {#26-3-max-kernels}

<!-- Please place Layout/LayoutTensor changes under "Library changes" in the
     **Mojo changelog**, since the layout package is packaged with and
     documented alongside Mojo. -->

- Added GPU kernel examples from the *Programming Massively Parallel Processors*
  (PMPP) textbook covering reductions, scans, histograms, sorting, sparse
  matrix operations, graph algorithms, convolutions, FlashAttention, and more.
- Improved NVFP4 grouped matmul kernel performance, now outperforming FlashInfer
  across all tested decoding and prefill shapes for Kimi K2.5 on B200.
- Optimized GPU `layer_norm` kernels with SIMD reductions, gamma/beta
  prefetch, and a `simd_width*2` warp tiling dispatch path.
- Optimized GPU `pad_constant` kernel with SIMD vectorization (`simd_width=4`)
  and added a kbench benchmark suite (`bench_pad`).
- Improved GPU `topk` and `argsort` kernel performance by nearly 2x.
- Optimized GPU `concat` with a flat-indexing kernel that avoids
  multi-dimensional index decomposition, using 128-bit vectorized loads with
  automatic fallback for unaligned shapes.

## Mojo language {#26-3-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](/mojo/changelog)
