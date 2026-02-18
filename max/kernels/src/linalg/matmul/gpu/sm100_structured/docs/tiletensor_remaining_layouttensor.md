# Remaining LayoutTensor Dependencies in SM100 Structured Kernels

## Status (2026-02-11)

All 8 SM100 kernel families use new Layout types exclusively.
`block_scaled_matmul` is fully migrated to TileTensor (zero LayoutTensor
in the kernel body). The `TMATile` wrapper struct preserves new Layout
type parameters while delegating to `TMATensorTile` internally.
`_to_legacy_layout` is fully encapsulated inside `tile_types.mojo`.

LayoutTensor remains in 3 categories (down from 5):

1. **Host-side function signatures** (~10 refs, down from ~100) --
   All 8 SM100 kernel families migrated to TileTensor.
   Remaining: `conv2d` (im2col TMA), naive kernel dispatch path.
2. **TMATensorTile internals** (~80 refs in `tma_async.mojo`) --
   uses old IntTuple-based Layout. Requires big-bang refactor
   (193 change sites, 25+ consumer files) or old Layout trait
   conformance.
3. **TMA store boundary** (~15 refs in `epilogue_components.mojo`)
   -- `rebind` to SMemTile for `async_store`. Blocked on
   TMATensorTile accepting TileTensor natively.

---

## Completed Work

### Infrastructure

- ~~Dead code removal~~: Deleted `ScalesTileLoader` (zero callers)
  and `c_tiles_lt()` / `CTileArrayLT` from 3 pipeline storage
  structs (never called directly). -130 lines.
- ~~TileTensor `async_store[rank]`~~: Added TileTensor overload
  for `TMATensorTile.async_store[rank](StaticTuple)` (rank 2/3).
- ~~`size()` on new Layout~~: Added to both `Layout` struct and
  `TensorLayout` trait. Prerequisite for TMATensorTile migration.
- ~~`TMATile` wrapper~~: New struct parameterized on `TensorLayout`,
  wraps `TMATensorTile` via `_to_legacy_layout`. All 8 kernel
  families + conv2d use it.
- ~~`TMATile` in loaders~~: `TileLoader` and `ScalesLoader` use
  `TMATile.InnerType` instead of direct `_to_legacy_layout`.
- ~~Host TMA creation~~: `create_tma_tile` factory takes new
  Layout types directly. No `LegacyLayout` exposed to callers.
  Replaces `create_tensor_tile` with explicit legacy layouts.
- ~~`ViewType` on TileTensor~~: New type alias `ViewType[new_layout]`
  names the return type of `reshape()`. Enables properly-typed
  helper functions for batched/5D reshapes (Pattern 12).
- ~~TileTensor `create_tma_tile` overload~~: Calls
  `create_tensor_tile` directly, bypassing LayoutTensor. Works
  with any TileTensor including reshaped views (Pattern 13).

### Kernel migration (previous PRs)

- ~~Remove TMA legacy Layout params~~ Done for all 8 kernels.
- ~~Migrate fallback kernel C output~~ Done with `stride_layout`.
- ~~Migrate split-K reduction tensor~~ Done.
- ~~Migrate `RegTile`/`BlockwiseFP8Accumulator`~~ Done.
- ~~Implement `upcast`/`zipped_divide` for new Layout~~ Done.
- ~~Migrate SMEM epilogue (transpose + non-transpose)~~ Done.
- ~~Migrate `_store_with_bounds_check`~~ Done.
- ~~Enable parameter inference (`//`)~~ Done across ~20 functions.

### Host signature migration

- ~~`block_scaled_matmul`~~: Pure TileTensor with ViewType reshapes
  and module-level `_create_tma_and_launch` for TMA scoping.
- ~~`default/matmul` (3 functions)~~: Pure TileTensor with
  `create_tma_tile` TileTensor overload.
- ~~`default/dispatch`~~: `lt_to_tt` at NDBuffer boundary.
- ~~`blockwise_fp8_matmul`~~: Pure TileTensor.
- ~~`blockwise_fp8_1d2d_matmul`~~: Kernel takes `b_scales_layout`
  and `c_device_layout` as TensorLayout params from caller's
  TileTensor type. Types match by construction (Pattern 15).
- ~~`grouped_1d1d_matmul`~~: Kernel takes `c_device_layout` as
  TensorLayout param from caller's TileTensor type (Pattern 15).
- ~~`grouped_block_scaled_matmul`~~: Module-level `_GroupPtrTile`
  and `_ProblemSizesTile` aliases provide concrete TileTensor types
  for pointer arrays and problem sizes. Host constructs via kernel
  type aliases for enqueue_function type identity.

---

## Remaining Migration (blocked on external changes)

### TMATensorTile big-bang refactor

`TMATensorTile` in `tma_async.mojo` uses old `Layout` as its type
parameter. Old `Layout` does not implement the `TensorLayout` trait,
so the struct can't accept both old and new Layout. Migration requires
changing the struct + all ~25 consumer files simultaneously.

**Change sites**: 193 mechanical replacements in `tma_async.mojo`:

- `Self.layout.shape[i].value()` → `Self.layout.shape[i]().value()`
  (124 sites)
- `Self.desc_layout.size()` → `Self.desc_layout.static_size`
  (22 sites)
- `product(Self.layout.shape[i])` →
  `Self.layout.shape[i]().product()` (8 sites)
- `Layout.row_major(...)` → `LegacyLayout.row_major(...)` (34 sites)

**Options**:

1. Big-bang: change struct + all 25 consumers in one PR
2. Make old `Layout` conform to `TensorLayout` trait (enables
   gradual migration)
3. Keep `TMATile` wrapper indefinitely (current state -- works,
   just has the bridge internally)

### Host function signatures

All 8 host launch files accept `LayoutTensor` from the graph
compiler. Changing requires updating the dispatch layer
(`fp8_quantization.mojo`, `_matmul_dispatch_sm100`, etc.) to
produce TileTensor instead of LayoutTensor.

---

## Key Lessons Learned

- `_int_to_dim()` is NOT comptime-evaluable. Build Layout types
  directly with `ComptimeInt` and `RuntimeInt`.
- `TensorLayout` trait erases concrete type info. Use
  `tile[stride_layout=...]` for static strides, or pass concrete
  layout types before `//` for inference.
- TMA layouts are fully static. Compute inside kernel struct
  using `static_row_major`.
- Never use `rebind` for type mismatches. Use `TMATile` to derive
  `TMATensorTile` types from a single source of truth.
- `reshape(row_major[...])` is NOT `coalesce` -- preserves
  contiguous strides, not parent strides. Use explicit strides.
- New `upcast` keeps element-level strides (no `simd_size *`
  multiply needed).
- Old `Layout` does NOT implement `TensorLayout`. Can't use trait
  bounds to accept both. Use `TMATile` wrapper pattern instead.
- **Use `L._shape_types[i]` not `ComptimeInt[L.static_shape[i]]`**
  when computing derived layout types. The former preserves
  static/dynamic nature; the latter forces all dims to be static,
  which breaks for dynamic batch or M dims.
- **Use `tensor.layout.shape[i]()`** (not `Idx[L.static_shape[i]]()`)
  when constructing reshape layouts at runtime. The former carries
  actual runtime values; the latter uses compile-time constants that
  may not match the actual tensor dimensions.
- **TMA descriptors are scoped references** that can't be copied
  across `@parameter if` boundaries. Use module-level functions
  (Pattern 14) to keep TMA creation and kernel launch in one scope.
- **`lt_to_tt` only supports 2D**. For higher-rank tensors, use
  `TileTensor(ndbuffer)` directly. Beware: complex DimList
  expressions (e.g., `ceildiv(...)`) trigger a compiler bug;
  workaround is `DimList.create_unknown[rank]()`.
- **Closures can't have parameters in Mojo**. Local functions with
  `TileTensor[...]` args fail. Use module-level functions instead.
- **`RowMajorLayout` with conditional types doesn't work** at the
  type alias level. `ComptimeInt[X] if cond else ComptimeInt[Y]`
  produces `AnyStruct[...]` not a concrete type. Use separate
  type aliases for each branch instead.
- **Take layout as a kernel param, don't compute it internally**.
  If a TileTensor flows from the host to `enqueue_function`, the
  kernel struct should take its layout as a `TensorLayout` parameter
  (derived from `type_of(tt).LayoutType`). Computing the layout
  independently inside the kernel (e.g., from `static_N`) creates
  a different symbolic type that may not match -- this caused the
  DeepSeek-R1-NVFP4 pipeline failure (#77347/#77359).

---

## Current Architecture

```text
Host side
    ├── Kernel instantiated FIRST (computes TMA layouts from config)
    ├── create_tma_tile[Kernel.XTmaTile.tile_layout, ...](ctx, tensor) ✅
    ├── lt_to_tt / lt_to_tt_1d → TileTensor at enqueue boundary ✅
    └── enqueue_function passes TMATensorTile + TileTensor to kernel

Kernel struct
    ├── TMA layouts from config: static_row_major, tma_desc_layout_* ✅
    ├── TMATile[dtype, tile_layout, desc_layout] (new Layout types) ✅
    ├── ATmaOp = Self.ATmaTile.InnerType (for DevicePassable) ✅
    ├── Zero legacy Layout struct params ✅
    └── TensorLayout struct params only for dynamic C layouts

Kernel run() params
    ├── TMA ops: Self.ATmaOp (TMATensorTile via TMATile.InnerType) ✅
    ├── 1D data: TileTensor with GMEMLayout1D ✅
    ├── C device: TileTensor with TensorLayout ✅
    └── Scalars and other non-tensor params unchanged

TileLoader / ScalesLoader
    ├── Parameterized on new TensorLayout types ✅
    ├── Derive TMATensorTile via TMATile.InnerType ✅
    └── No direct _to_legacy_layout calls ✅

_to_legacy_layout (encapsulated in tile_types.mojo)
    ├── Used by TmaOpType / TmaOpTypeIm2col comptime aliases
    ├── Used by create_tma_tile factory (internal conversion)
    └── No external consumers ✅

Host signature migration progress (8 of 8 done)
    ├── block_scaled_matmul: TileTensor ✅
    ├── default/matmul (3 functions): TileTensor ✅
    ├── default/dispatch: lt_to_tt at boundary ✅
    ├── blockwise_fp8_matmul: TileTensor ✅
    ├── blockwise_fp8_1d2d_matmul: TileTensor (Pattern 15) ✅
    ├── grouped_1d1d_matmul: kernel CDeviceLayout (Pattern 15) ✅
    ├── grouped_block_scaled_matmul: TileTensor (module-level aliases) ✅
    └── conv2d: LayoutTensor (im2col TMA, separate concern) ⚠️

Remaining LayoutTensor (blocked on external changes)
    ├── TMATensorTile internals (old Layout in tma_async.mojo) ⚠️
    ├── TMA async_store boundary (rebind to SMemTile) ⚠️
    └── TileLoaderTMA (kept for conv2d) ⚠️
```
