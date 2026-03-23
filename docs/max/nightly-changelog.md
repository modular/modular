# Nightly: v26.3

This version is still a work in progress.

## Highlights {#26-3-highlights}

## Documentation {#26-3-docs}

## MAX models {#26-3-models}

## MAX framework {#26-3-max}

### Inference server {#26-3-max-serve}

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

## Breaking changes {#26-3-breaking}

- `max/python/max/benchmark/benchmark_throughput.py` has been deprecated and
  will be removed in a future MAX release.

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
