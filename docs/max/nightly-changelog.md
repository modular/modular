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

## Breaking changes {#26-3-breaking}

### Mojo API {#26-3-max-mojo}

### Custom ops {#26-3-custom-ops}

## MAX kernels {#26-3-max-kernels}

<!-- Please place Layout/LayoutTensor changes under "Library changes" in the
     **Mojo changelog**, since the layout package is packaged with and
     documented alongside Mojo. -->

## Mojo language {#26-3-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](/mojo/changelog)
