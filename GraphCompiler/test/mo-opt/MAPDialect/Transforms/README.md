# MAP dialect transform tests

Tests that validate the `map` dialect passes and transformations, such as
`mo-to-map` (lowering MO ops into `map.iter` / `imap` kernels) and
`map-fuse-kernels` (fusing producers into consumers).

`map-fuse-kernels` runs four fusers -- `EpilogueFuser`, `ElementwiseFuser`,
`ViewFuser`, `PrologueFuser` -- in sequence. `map-fuse-epilogue`,
`map-fuse-elementwise`, `map-fuse-views`, and `map-fuse-prologue` each run
exactly one of those fusers in isolation and exist for testing only (not
part of `buildMapPipeline`); use one of them when a test only needs to
exercise a single fuser's behavior, and fall back to `map-fuse-kernels` when
a test depends on more than one fuser running together (e.g. an elementwise
chain that must be merged before it can fold into a prologue, or into a
view).

Add `.mlir` files here; they are picked up automatically by the `**/*.mlir`
glob in `GraphCompiler/test/mo-opt/BUILD.bazel`. The usual pattern is:

```text
// RUN: mo-opt %s --map-fuse-kernels | FileCheck %s
```

Populate this directory as the passes gain functionality.
