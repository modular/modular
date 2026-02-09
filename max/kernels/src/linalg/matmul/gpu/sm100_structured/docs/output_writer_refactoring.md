# Output Writer Refactoring Plan

## Overview

This document analyzes the `output_writer.mojo` and `tile_writer.mojo` files
to identify code duplication and consolidation opportunities before migrating
to TileTensor.

**Goal**: Simplify the output writer architecture, reduce parameter explosion,
and prepare for TileTensor migration.

---

## File Inventory

### tile_writer.mojo (Component Structs)

| Struct | Lines | Parameters | Purpose |
|--------|-------|------------|---------|
| `AccumTile` | 85-99 | 2 | Upper + lower TMEM fragments |
| `AccumBarrier` | 106-123 | 1 | Pipeline barrier helper for CTA arrival |
| `EpilogueConfig` | 214-235 | 5 | Computed epilogue parameters |
| `TMAStoreCoords` | 242-334 | 9 | TMA store coordinates + warp election |
| `TMAStoreExecutor` | 341-536 | 14 | Execute TMA store SMEM→GMEM |
| `FragmentCoords` | 544-566 | 2 | Coordinate tracking for fragments |
| `EpilogueApplier` | 574-715 | 6 | Apply element-wise ops on fragments |
| `TMEMToSMemWriter` | 723-892 | 12 | Write TMEM accumulators to SMEM |
| `SMemEpilogueWriter` | 908-1164 | 18 | SMEM-based epilogue with compute lambda |

**Standalone Functions:**

| Function | Lines | Purpose |
|----------|-------|---------|
| `tma_wait_pipelined` | 59-78 | Pipelined TMA wait helper |
| `store_fragment_to_smem` | 134-207 | Store fragment via st.matrix |
| `shared_memory_epilogue_transpose` | 1174-1359 | SMEM epilogue for transpose |
| `shared_memory_epilogue` | 1362-1527 | SMEM epilogue for non-transpose |

### output_writer.mojo (Main Writer)

| Struct | Lines | Parameters | Purpose |
|--------|-------|------------|---------|
| `TileWriter` | 66-913 | **27** | Main output writer for SM100 epilogue |

**TileWriter Parameters (27 total):**

```mojo
struct TileWriter[
    # Inferred from constructor (4)
    tma_origin: ImmutOrigin,
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    //,
    # Explicit config parameters (8)
    a_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    num_accum_pipeline_stages: Int,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    # Kernel-level parameters (7)
    c_smem_layout: Layout,
    num_output_stages: Int,
    stage_stride_cols: Int,
    num_output_warps: Int,
    elementwise_compute_lambda_fn: Optional[...],
    register_based_epilogue: Bool = True,
    batched: Bool = False,
]
```

**TileWriter Methods:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `write` | 176-185 | Write accumulated results (2D coords) |
| `write_batched` | 188-207 | Write with 3D batched coords |
| `write_splitk` | 210-240 | Write with split-K reduction |
| `_copy_to_gmem` | 243-414 | TMEM→Regs→SMEM→GMEM pipeline (2D) |
| `_copy_to_gmem_batched` | 417-621 | Same pipeline with 3D coords + alpha |
| `write_absolute_with_bounds_check` | 624-801 | Write with per-row bounds checking |
| `_store_with_bounds_check` | 804-913 | Element-by-element bounded store |

---

## Code Duplication Analysis

### 1. `_copy_to_gmem` vs `_copy_to_gmem_batched`

**Overlap**: ~85% identical code (~180 lines each)

**Differences:**

- 2D `Tuple[UInt32, UInt32]` vs 3D `Tuple[UInt32, UInt32, UInt32]` coordinates
- Batched version has `alpha: Float32` scaling
- `TMAStoreCoords` constructor differs (2D vs 3D)

**Solution**: Unify into single method with:

- Generic coordinate handling via `batched` parameter (already exists!)
- Optional alpha parameter with default 1.0

### 2. Redundant Config Computation

**In TileWriter (lines 124-162):**

```mojo
comptime epilogue_dtype = ...
comptime N_dim = ...
comptime stageN = ...
comptime stage_contiguous_size = ...
comptime data_paths = 16
comptime bits = 256
comptime rep = ...
comptime fragment_size = ...
comptime rep_frag_size = ...
comptime is_lower_frag_required = ...
comptime cg2_num_stages = ...
comptime cg1_num_stages = ...
comptime num_stages = ...
```

**In EpilogueConfig (tile_writer.mojo lines 214-235):**

```mojo
comptime is_lower_frag_required = ...
comptime cg2_num_stages = ...
comptime cg1_num_stages = ...
comptime num_stages = ...
comptime data_paths = 16
comptime bits = 256
comptime fragment_size = ...
```

**Solution**: Use `EpilogueConfig` in TileWriter instead of duplicating.

### 3. Transpose/Non-Transpose Path Duplication

Both `TMEMToSMemWriter` and `SMemEpilogueWriter` have:

- `_write_transpose` method
- `_write_non_transpose` method

With significant overlap in:

- Swizzle block layout computation
- Warp tile selection logic
- Fragment store patterns

**Solution**: Extract shared logic into helper functions or a shared base.

---

## Consolidation Plan

### Phase 1: Use EpilogueConfig in TileWriter (Low effort)

**Before:**

```mojo
struct TileWriter[...]:
    comptime is_lower_frag_required = not (Self.cta_group == 1 and Self.BM == 64)
    comptime cg2_num_stages = ...
    comptime cg1_num_stages = ...
    comptime num_stages = ...
    comptime data_paths = 16
    comptime bits = 256
    comptime fragment_size = ...
```

**After:**

```mojo
struct TileWriter[...]:
    comptime Config = EpilogueConfig[
        Self.MMA_M, Self.MMA_N, Self.stageN, Self.cta_group, Self.transpose_c
    ]
    # Use Self.Config.is_lower_frag_required, Self.Config.num_stages, etc.
```

**Impact**: Remove ~40 lines of redundant computation.

### Phase 2: Merge `_copy_to_gmem` Methods (Low-Medium effort)

**Before:** Two separate methods with 85% identical code.

**After:** Single unified method:

```mojo
@always_inline
fn _copy_to_gmem_impl[
    coord_dims: Int,  # 2 or 3
](
    self,
    c_tiles: Self.CTileArray,
    output_stage: Self.Stage,
    c_coord: CoordType,  # Tuple[UInt32, UInt32] or Tuple[UInt32, UInt32, UInt32]
    c_shape: Tuple[UInt32, UInt32],
    alpha: Float32 = Float32(1.0),
):
    ...
```

**Impact**: Remove ~150 lines of duplication.

### Phase 3: Create Shared EpilogueParams Config (Medium effort)

Create a consolidated config struct that groups related parameters:

```mojo
struct EpilogueParams[
    c_type: DType,
    accum_type: DType,
    c_smem_layout: Layout,
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    cta_group: Int,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    num_output_warps: Int,
]:
    # All computed values derived from these
    comptime Config = EpilogueConfig[...]
    comptime stageN = ...
    comptime stage_contiguous_size = ...
    # etc.
```

Then simplify TileWriter parameters:

```mojo
struct TileWriter[
    tma_origin: ImmutOrigin,
    c_layout: Layout,
    c_desc_layout: Layout,
    //,
    params: EpilogueParams,  # Replaces 12+ parameters
    num_output_stages: Int,
    stage_stride_cols: Int,
    elementwise_compute_lambda_fn: Optional[...] = None,
    register_based_epilogue: Bool = True,
    batched: Bool = False,
]:
```

**Impact**: Reduce TileWriter from 27 to ~10 parameters.

### Phase 4: TileTensor Migration (After consolidation)

Once consolidated, migrate to TileTensor:

1. Replace `from linalg.structuring import SMemTileArray, SMemTile` with:

   ```mojo
   from .tile_types import SMemTileArray2D, SMemTile
   ```

2. Update method signatures:

   ```mojo
   # Before
   fn write(self, c_tiles: SMemTileArray[c_type, c_smem_layout, ...], ...)

   # After
   fn write(self, c_tiles: SMemTileArray2D[c_type, dim0, dim1, ...], ...)
   ```

3. At TMA boundaries, use explicit LayoutTensor construction:

   ```mojo
   comptime CTileLT = LayoutTensor[Self.c_type, Self.c_smem_layout, ...]
   c_tma_op.async_store(CTileLT(c_smem_tile.ptr), coords)
   ```

---

## Implementation Order

| Step | Task | Est. Lines Changed | Risk | Status |
|------|------|-------------------|------|--------|
| 1 | Use EpilogueConfig in TileWriter | ~50 | Low | Pending |
| 2 | Merge _copy_to_gmem methods | ~200 | Low | **DONE** |
| 3 | Create EpilogueParams config | ~150 | Medium | Pending |
| 4 | TileTensor migration | ~100 | Medium | Pending |

**Completed**: Phase 2 reduced output_writer.mojo from 913 to 760 lines
(-153 lines).

---

## Phase 2 Completion Notes (2026-02-02)

Merged `_copy_to_gmem` and `_copy_to_gmem_batched` into single `_copy_to_gmem_impl`:

**Changes:**

1. `write()` now delegates to `_copy_to_gmem_impl` with `batch=0, alpha=1.0`
2. `write_batched()` delegates directly to `_copy_to_gmem_impl`
3. `write_splitk()` updated to use `_copy_to_gmem_impl`
4. Removed duplicate `_copy_to_gmem` method (~173 lines)

**API preserved**: `write()`, `write_batched()`, `write_splitk()` signatures unchanged.

**Tests verified**:

- test_matmul_sm100_blockwise_fp8_smoke.mojo ✓
- test_grouped_matmul_sm100_nvfp4.mojo ✓
- test_matmul_sm100_2sm_bf16.mojo ✓

---

## Testing Strategy

After each phase, run:

```bash
# Smoke tests
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_blockwise_fp8_smoke.mojo
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_2sm_bf16.mojo

# Block-scaled tests
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_block_scaled_nvfp4_1sm.mojo

# Grouped tests
mojo max/kernels/test/gpu/linalg/test_grouped_matmul_sm100_nvfp4.mojo
```

---

## Current Imports (for TileTensor migration reference)

**tile_writer.mojo:**

```mojo
from linalg.structuring import SMemTileArray, SMemTile
```

**output_writer.mojo:**

```mojo
from linalg.structuring import SMemTileArray, SMemTile
```

Both import from the legacy `linalg.structuring` module which uses LayoutTensor.
Migration will switch to our `tile_types.mojo` which uses TileTensor.
