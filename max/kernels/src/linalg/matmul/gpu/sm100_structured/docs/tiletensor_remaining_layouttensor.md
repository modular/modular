# Remaining LayoutTensor Dependencies in SM100 Structured Kernels

## Status (2026-02-10)

All 8 SM100 kernel families have been migrated to new Layout types:

- `blockwise_fp8_1d2d` -- 0 legacy Layout struct params
- `blockwise_fp8` -- 0 legacy Layout struct params
- `block_scaled` -- 0 legacy Layout struct params
- `default` (BlackwellMatmulSM100Kernel) -- 0 legacy Layout struct params
- `grouped_block_scaled_1d1d` -- 0 legacy Layout struct params
  (1 TensorLayout for c_device)
- `grouped_block_scaled` -- 0 legacy Layout struct params
- `conv2d_fprop` -- 0 legacy Layout struct params
- `fallback` -- 0 legacy Layout struct params (1 TensorLayout for c_layout)

TMA layouts are computed from config via `static_row_major`/`RowMajorLayout`/
`tma_desc_layout_3d/4d/5d`, with TMATensorTile types derived via
`_to_legacy_layout`. `TileLoader`/`ScalesLoader` handle the legacy bridge.

The fallback kernel uses `tile_with_offset[stride_layout=...]` to produce
tiles with static strides from a dynamic-stride parent, enabling the full
`tile → vectorize → distribute_with_offset` chain.

LayoutTensor remains in:

1. Internal epilogue components (SMEM epilogue, layout algebra)
2. ~~`BlockwiseFP8Accumulator` / `RegTile`~~ -- DONE (TileTensor stack_allocation)
3. Host-side function signatures (from graph compiler)
4. `TileLoaderTMA` / `TileLoaderTMAIm2col` (legacy loader structs)

---

## Missing TileTensor Functionality

The following LayoutTensor features have no TileTensor equivalent and block
complete migration.

### 1. ~~`.reshape[layout]()` -- view with different layout~~ DONE

**Status**: Implemented. TileTensor now has a `.reshape(layout_val)` method
that accepts an explicit layout value and returns a new TileTensor viewing the
same memory with the target layout. All transpose paths in
`_write_transpose_tt` and `_store_transpose_tt` use this natively.

### 2. ~~`coalesce()` -- merge compatible dimensions~~ DONE (workaround)

**Status**: Handled via explicit `reshape` calls with manually computed
coalesced layouts. Since all dimensions are comptime-known, the coalesced
layout is expressed as an explicit `InternalLayout` with `Coord` shape/stride:

```mojo
comptime coalesced = InternalLayout(
    Coord(Idx[stageN](), Idx[tile_width]()),
    Coord(Idx[2 * tile_width](), Idx[1]()),
)
var flat_tile = tiled.reshape(coalesced)
```

TileTensor's `.coalesce()` still only works for row-major layouts, but
`reshape` provides a general escape hatch.

### 3. `.tile[]` with positional integer arguments

**Used in**: All transpose and non-transpose paths

**TileTensor status**: TileTensor has `.tile[*sizes](Coord(...))` which works,
but requires wrapping indices in `Coord(Idx(i), Idx(j), ...)`. The
LayoutTensor version uses `.tile[dims](i, j, ...)` with bare integer args.

**Status**: Already handled in the TileTensor overloads. Not a blocker.

### 4. `zipped_divide()` and `upcast()` -- layout algebra

**Used in**: `shared_memory_epilogue_transpose`, `shared_memory_epilogue`,
`_store_with_bounds_check`

**What they do**:

- `zipped_divide(layout, thread_layout)`: Divides a layout into thread-local
  fragments based on a thread distribution layout
- `upcast(layout, simd_size)`: Adjusts layout for SIMD vector access by
  grouping elements

These are complex layout algebra operations from `layout.layout` that compute
how threads map to memory elements in the SMEM epilogue. They produce new
layouts used for coordinate-to-index mapping.

**Why TileTensor can't do this**: These operations work on public `Layout`
(IntTuple-based) and produce new public Layouts. TileTensor uses internal
`Layout` (Coord-based). There's no equivalent in the internal layout system.

**Recommendation**: These operations would need to be reimplemented for
internal Layout, or the results pre-computed at compile time and expressed as
explicit Coord-based layouts. This is significant work and should be done
when TileTensor's layout algebra is more developed.

### 5. `.vectorize[]` and `.distribute[]` -- thread mapping

**Used in**: `shared_memory_epilogue`

**What they do**:

- `.vectorize[1, simd_size]()`: Groups elements into SIMD vectors
- `.distribute[thread_layout, swizzle=swizzle](lane_id)`: Maps thread ID to
  a fragment of the tensor, applying swizzle

These are LayoutTensor methods that compute per-thread fragments for the
SMEM-based epilogue path (used when `register_based_epilogue=False`).

**Why TileTensor can't do this**: TileTensor has no equivalent methods. These
would require implementing the same thread-to-element mapping logic.

**Recommendation**: Low priority. The SMEM-based epilogue is only used when
`elementwise_compute_lambda_fn` is set and `register_based_epilogue=False`,
which is a minority code path.

### 6. `blocked_product()` -- layout product

**Used in**: `shared_memory_epilogue`

**What it does**: Computes the blocked product of two layouts, producing a
combined layout for multi-level tiling.

**Same status as item 4** -- complex layout algebra with no TileTensor
equivalent.

---

## Remaining LayoutTensor Code Locations

### `epilogue_components.mojo`

| Function | Lines | LayoutTensor Usage | Status |
|----------|-------|-------------------|--------|
| `store_fragment_to_smem` (LT version) | 140-214 | `.layout.stride/shape` | Kept for SMemEpilogue callers |
| `TMAStoreExecutor.execute` (LT version) | 467-501 | Calls `_store_transpose/_non_transpose` | Kept for SMemEpilogue callers |
| `TMAStoreExecutor._store_transpose` | 505-589 | `.reshape[]`, `.tile[]` | Kept for SMemEpilogue callers |
| `TMAStoreExecutor._store_transpose_tt` | 685-803 | TileTensor reshape+tile, LT at TMA boundary | ✅ TT natively |
| `TMAStoreExecutor._store_non_transpose` | 591-631 | `.tile[]` | TT overload delegates here |
| `TMEMToSMemWriter.write_fragments` (LT) | 1149-1170 | Calls `_write_transpose/_non_transpose` | Kept for SMemEpilogue callers |
| `TMEMToSMemWriter._write_transpose` | 1172-1254 | `.reshape[]`, `.tile[]`, `coalesce()` | Kept for SMemEpilogue callers |
| `TMEMToSMemWriter._write_transpose_tt` | 1311-1394 | TileTensor reshape+tile natively | ✅ Fully TileTensor |
| `TMEMToSMemWriter._write_non_transpose` (LT) | 1256-1286 | `.tile[]` | Kept for SMemEpilogue callers |
| `SMemEpilogueWriter` (entire struct) | 1442-1733 | `SMemTileArray`, `.tile[]`, `.reshape[]` | Unblocked (reshape available) |
| `shared_memory_epilogue_transpose` | 1736-1919 | `zipped_divide`, `upcast`, `rt_crd2idx` | Item 4 |
| `shared_memory_epilogue` | 1922-2085 | `.vectorize[]`, `.distribute[]`, `blocked_product` | Items 4, 5, 6 |

### `output_writer.mojo`

| Function | Lines | LayoutTensor Usage | Status |
|----------|-------|-------------------|--------|
| `_store_with_bounds_check` | 740-854 | `zipped_divide`, `upcast`, `.tile[]` | Item 4 |
| `write_with_residual` | 856-900 | `CTileArrayTT` for source tiles | ✅ Converted to TileTensor |
| `_copy_to_gmem_with_residual` | 903-1113 | `CTileArrayTT` for source tiles | ✅ Converted to TileTensor |

---

## Priority Order for Future Migration

1. ~~**Add `tt_reshape` to TileTensor**~~ ✅ Done.
2. ~~**Convert residual source C tiles**~~ ✅ Done.
3. ~~**Move lt_to_tt to CPU side**~~ ✅ Done for all kernels.
4. ~~**Remove TMA legacy Layout params**~~ ✅ Done for all 8 kernels.
5. ~~**Apply kernel signature pattern to all kernels**~~ ✅ Done.
6. ~~**Migrate fallback kernel C output**~~ ✅ Done with `stride_layout`.
7. ~~**Migrate split-K reduction tensor**~~ ✅ Done.
8. **Convert `SMemEpilogueWriter` internals** to TileTensor.
9. **Implement layout algebra for TileTensor** (`zipped_divide`, `upcast`,
   `blocked_product`). Unlocks `shared_memory_epilogue*` and bounds-check path.
10. ~~**Migrate `RegTile`/`BlockwiseFP8Accumulator`**~~ DONE.
11. **Remove host-side LayoutTensor function signatures** (cross-cutting).

## Key Lessons Learned

- `_int_to_dim()` is NOT comptime-evaluable (runtime branching). Don't use
  it in comptime type aliases. Build Layout types directly with `ComptimeInt`
  and `RuntimeInt`.
- `TensorLayout` trait as a struct param **erases concrete type info** -- the
  compiler cannot prove `all_dims_known` through the trait even when the
  underlying strides are `ComptimeInt`. Use `tile[stride_layout=...]` to
  provide explicit static strides when the parent has dynamic stride types.
- TMA layouts are fully static and redundant with config. Compute them inside
  the kernel struct using `static_row_major` instead of passing as params.
- Never use `rebind` to solve type mismatches. Use `TmaOpType` to derive
  `TMATensorTile` types from a single source of truth.
- `tile()` inherits the parent's stride TYPES (not values). For dynamic-stride
  parents, use `tile[stride_layout=MyLayout]` to get static-stride tiles.

---

## Current Architecture (all kernels migrated)

```text
Host side
    ├── Kernel instantiated FIRST (computes TMA layouts from config)
    ├── create_tensor_tile with Kernel.XTmaOp.layout (types match) ✅
    ├── lt_to_tt / lt_to_tt_1d → TileTensor at enqueue boundary ✅
    └── enqueue_function passes TMATensorTile + TileTensor to kernel

Kernel struct
    ├── TMA layouts computed from config: static_row_major, RowMajorLayout,
    │   tma_desc_layout_3d/4d/5d ✅
    ├── TmaOpType / TmaOpTypeIm2col derive TMATensorTile types ✅
    ├── Zero legacy Layout struct params ✅
    └── TensorLayout struct params only for dynamic C layouts

Kernel run() params
    ├── TMA ops: Self.ATmaOp, Self.BTmaOp, Self.CTmaOp, etc. ✅
    ├── 1D data: TileTensor with GMEMLayout1D ✅
    ├── C device: TileTensor with TensorLayout ✅
    └── Scalars and other non-tensor params unchanged

Kernel output path (fallback kernel)
    ├── tile_with_offset[stride_layout=CGmemStrideLayout] ✅
    ├── vectorize[1, 2]() on static-stride tile ✅
    ├── distribute_with_offset[row_major[8, 4]()] ✅
    └── layout(coord[m, n]()) for element offset ✅

Remaining LayoutTensor (encapsulated)
    ├── SMEM epilogue: zipped_divide, upcast, blocked_product ⚠️
    ├── BlockwiseFP8Accumulator / RegTile ✅ (TileTensor)
    ├── TileLoaderTMA / TileLoaderTMAIm2col (legacy loaders) ⚠️
    └── Host function signatures (LayoutTensor from callers) ⚠️

✅ = new Layout + TileTensor
⚠️ = legacy Layout / LayoutTensor (encapsulated)
```
