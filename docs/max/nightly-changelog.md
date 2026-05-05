---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

## MAX framework

### Inference server

### `max` CLI

- Added `--devices=gpu:all` to use every visible GPU (including MAX Serve).

### Python API

- `CPUMetricsCollector` in `max.diagnostics.cpu` is now used as a context
  manager instead of `start`/`stop` and now exposes `get_stats()` instead of
  `dump_stats()`, matching the interface of `GPUDiagContext`.

## Breaking changes

### Mojo API

### Custom ops

## MAX kernels

<!-- Please place Layout/LayoutTensor changes under "Library changes" in the
     **Mojo changelog**, since the layout package is packaged with and
     documented alongside Mojo. -->

## 🛠️ Fixed

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](https://www.mojolang.org/releases)
