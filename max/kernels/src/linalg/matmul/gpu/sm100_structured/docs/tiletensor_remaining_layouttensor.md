# Remaining LayoutTensor Dependencies in SM100 Structured Kernels

## Status

As of the TileTensor migration (PR #77201), all SM100 structured kernel
**entry points** and **primary data paths** use TileTensor. LayoutTensor
remains only in:

1. Internal epilogue component implementations (SMEM epilogue path)
2. Boundary conversions hidden inside TileTensor overloads (non-transpose
   store, TMA async_store in transpose store)
3. The `enqueue_function` kernel parameter signatures (compiler limitation)

Note: Transpose paths now use TileTensor natively for reshape/tile operations,
with LayoutTensor conversion only at the TMA `async_store` boundary.

These remaining usages are fully encapsulated -- no kernel code outside
`epilogue_components.mojo` and `output_writer.mojo` touches LayoutTensor.

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

1. ~~**Add `tt_reshape` to TileTensor**~~ ✅ Done. TileTensor now has
   `.reshape(layout_val)`. All transpose paths converted.

2. **Convert `SMemEpilogueWriter` internals** -- now that `reshape` is
   available, convert the struct to use TileTensor arrays and native tile ops.

3. ~~**Convert residual source C tiles**~~ ✅ Done. `SourceTileStorage`,
   `write_with_residual`, conv2d kernel all use TileTensor arrays now.

4. **Implement layout algebra for TileTensor** -- `zipped_divide`, `upcast`,
   `blocked_product` equivalents. This unlocks `shared_memory_epilogue*` and
   `_store_with_bounds_check`.

5. **Fix `_IntTupleToCoordLike` compiler crash** -- enables changing
   `enqueue_function` kernel signatures to accept TileTensor directly.

---

## Current Architecture

```text
Kernel entry (LayoutTensor args)
    │
    ├── lt_to_tt() conversion at entry
    │
    ▼
Kernel internals (ALL TileTensor)
    │
    ├── TMA loads: TileTensor directly ✅
    ├── MMA: TileTensor directly ✅
    ├── Accumulator: TileTensor ✅
    │
    ▼
Tile Writer (TileTensor interface)
    │
    ├── Non-transpose path: TileTensor natively ✅
    │   ├── write_fragments TT overload → _write_non_transpose_tt ✅
    │   └── execute TT overload → _store_non_transpose (LT internally) ⚠️
    │
    ├── Transpose path: TileTensor natively ✅
    │   ├── write_fragments TT overload → _write_transpose_tt ✅
    │   └── execute TT overload → _store_transpose_tt (LT at TMA boundary) ✅
    │
    ├── Residual path (write_with_residual): TileTensor natively ✅
    │   ├── src_tiles: SMemTileArray2DRowMajor (TileTensor) ✅
    │   └── add_residual: uses TileTensor .ptr for SMEM access ✅
    │
    └── SMEM epilogue path: TileTensor → LayoutTensor at SMemEpilogueWriter ⚠️
        ├── SMemEpilogueWriter.__init__ TT overload converts internally ⚠️
        ├── shared_memory_epilogue_transpose (LT) ⚠️
        └── shared_memory_epilogue (LT) ⚠️

✅ = fully TileTensor
⚠️ = LayoutTensor internally (encapsulated, zero-cost conversion)
```
