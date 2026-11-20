// TODO(GEX-4340): Delete with the legacy tile traits (GEX-4467).
//
// The interim legacy tile traits at kernel import: the contracts the sm_100a
// tile kernels bake (legacy_tile_store = 8, legacy_tile_transform = 16), and
// one rejection per rule from the legacy_tile_invalid_stubs package, all
// reported in one run.

// `mogg-import-packages` prints the transitively loaded builtins too, so keep
// only the declarations under test.
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" 2>&1 | grep -E 'registration = "tile_rms_norm"|MatmulTileSM100' | FileCheck %s --check-prefix=IMPORT
// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%legacy_tile_invalid_pkg" 2>&1 | FileCheck %s --check-prefix=INVALID


// IMPORT-DAG: MatmulTileSM100{{.*}}[{access = 16 : i32, capabilities = 0 : i32, fused = true, role = "output"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}
// IMPORT-DAG: registration = "tile_rms_norm"{{.*}}[{access = 8 : i32, capabilities = 0 : i32, fused = true, role = "output"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}

// INVALID-DAG: error: _LegacyTileStoreAccess and _LegacyTileComputeOutput are only valid on a Fused tensor argument
// INVALID-DAG: error: a legacy tile access form requires an Output argument
// INVALID-DAG: error: declare _LegacyTileStoreAccess on the kernel signature and refine to _LegacyTileStorable in the body
// INVALID-DAG: error: declare _LegacyTileComputeOutput on the kernel signature and refine to _LegacyTileComputable in the body
// INVALID-DAG: error: a _LegacyTileStoreAccess argument cannot require TileTensorable
// INVALID-DAG: error: a _LegacyTileComputeOutput argument must also require TileTensorable
// INVALID-DAG: error: a Fused tensor argument must declare exactly one access form
// INVALID-DAG: error: kernel 'legacy_tile_not_listed' binds a legacy tile trait
// INVALID-DAG: error: kernel 'tile_rms_norm' (cond=type("gpu")) binds a legacy tile trait

module {}
