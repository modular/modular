# Legacy tile traits removal checklist

The legacy tile traits (`_LegacyTileStoreAccess`, `_LegacyTileComputeOutput`,
and their capabilities) are a stopgap for tile-based epilogue fusion
(GEX-4340). When the tile epilogue interface of GEX-4467 lands, delete
everything below. Update this file in every pull request that adds or removes
a legacy tile site.

## Find the sites

Every site carries `TODO(GEX-4340)`:

```sh
git grep -n "TODO(GEX-4340)"
```

`bazel/lint/lint_legacy_tile_allowlist.py` lists every file that names a
legacy tile trait.

## Kernels

These kernels use the traits. `kLegacyTileRegistrations` in
`LegacyTileTraits.cpp` lists their registrations.

- Trainium, registered for `type("gpu") & api("neuron")`: `MatmulTileTrainium`
  (`mo.matmul`), `ReduceRMSNormTileTrainium`
  (`mo.composite.reduce.rms_norm`), and
  `ReduceRMSNormFusedResidualAddTrainium`
  (`mo.composite.rms_norm_fused_residual_add`).
- GPU, registered for sm_100a: `MatmulTileSM100` (`mo.matmul`), and the
  `tile_rms_norm` test kernels `RMSNormTileSM100` and `TileRMSNormStub`.

## Files to delete

- `max/kernels/src/graph_compiler/extensibility/legacy_tile_traits.mojo`.
- The graph compiler files `Analysis/LegacyTileTraits.h`,
  `Analysis/LegacyTileTraits.cpp`, and `Support/LegacyTileMojoTraits.h`, and
  the Trainium overlay copy of `LegacyTileMojoTraits.h`.
- The tests and test kernels named `*legacy-tile*` or `legacy_tile_*`.
- `bazel/lint/lint_legacy_tile_allowlist.py`.
- This file.

## Code to revert

Revert each item in `GraphCompiler/` and, where the overlay has a copy, in
`plugins/internal-trainium/GraphCompiler/`.

- The two access forms and four vocabulary rows in `KernelDeclAdaptor.h`,
  `KernelDeclAdaptor.cpp`, and `MOGGEnums.td`.
- The `$contract` of `mogg.tile.create.epilogue` in `MOGGOps.td` and
  `MOGGOps.cpp`.
- These functions, each with its one call: `applyLegacyTileTraitRules`,
  `verifyLegacyTileRegistration`, the legacy tile branch of
  `classifyEpilogueLambdaKind`, and `emitLegacyTileTraitEpilogue`.
- `_LegacyTileStoreConsumer` and `_LegacyTileComputeOp` in
  `builtin_kernels/_tile_fusion.mojo`.
- The re-exports in `extensibility/__init__.mojo` and its overlay copy.
- The lint rule and `multirun` entry in `bazel/lint/BUILD.bazel`.

## Removal order

1. Move the kernels to the GEX-4467 interface.
2. Delete the tests and test kernels.
3. Delete the graph compiler code: the emitter, the lowering branch, the
   import checks, and the vocabulary.
4. Delete the traits, the lint, and this file.
