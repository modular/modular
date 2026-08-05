---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

## MAX framework

### Inference server

### Server metrics

### `max` CLI

### Python API

### C API

## MAX kernels

## Breaking changes

- The legacy alias-buffer LoRA path has been removed. ModuleV3 LoRA (adapters
  passed as graph inputs) is now the only supported LoRA implementation.
  Serving a non-ModuleV3 architecture with `--lora-paths` now raises a clear
  error at startup instead of building a manager that never applies the
  adapters; serve the model's ModuleV3 variant (for example,
  `--prefer-module-v3`) to use LoRA adapters.

## Fixes

- Fixed `ops.group_norm()` raising `NotImplementedError` in eager mode on
  CPU. `group_norm` previously had a GPU-only kernel; it now has a CPU
  compute path too, so eager `group_norm` runs on CPU the same way
  `layer_norm`/`rms_norm` already do.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
