# Remaining LayoutTensor Dependencies in SM100 Structured Kernels

## Status (2026-02-11)

All 8 SM100 kernel families use new Layout types exclusively.
The `TMATile` wrapper struct preserves new Layout type parameters
while delegating to `TMATensorTile` internally. `_to_legacy_layout`
is fully encapsulated inside `tile_types.mojo` -- no external
consumers remain.

LayoutTensor remains in 3 categories (down from 5):

1. **Host-side function signatures** (~100 refs) -- accept
   LayoutTensor from graph compiler callers. Blocked on upstream.
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

### Kernel migration (previous PRs)

- ~~Remove TMA legacy Layout params~~ Done for all 8 kernels.
- ~~Migrate fallback kernel C output~~ Done with `stride_layout`.
- ~~Migrate split-K reduction tensor~~ Done.
- ~~Migrate `RegTile`/`BlockwiseFP8Accumulator`~~ Done.
- ~~Implement `upcast`/`zipped_divide` for new Layout~~ Done.
- ~~Migrate SMEM epilogue (transpose + non-transpose)~~ Done.
- ~~Migrate `_store_with_bounds_check`~~ Done.
- ~~Enable parameter inference (`//`)~~ Done across ~20 functions.

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

Remaining LayoutTensor (blocked on external changes)
    ├── TMATensorTile internals (old Layout in tma_async.mojo) ⚠️
    ├── TMA async_store boundary (rebind to SMemTile) ⚠️
    ├── Host function signatures (LayoutTensor from callers) ⚠️
    └── TileLoaderTMA (kept for conv2d) ⚠️
```
