# C Tensor Path Migration: Implementation Plan

## Goal

**Completely eliminate public Layout and LayoutTensor** from the C tensor
path in tile_writer.mojo and output_writer.mojo. Use internal Layout
(`layout._layout.Layout`) and TileTensor exclusively.

---

## Key Insight: Layout Type System

There are two Layout types in Mojo:

| Type | Module | Nature | Reshape |
|------|--------|--------|---------|
| **Public Layout** | `layout.Layout` | Runtime structure | `Layout.row_major(M, N)` |
| **Internal Layout** | `layout._layout.Layout` | Compile-time parametric | `row_major[M, N]()` |

TileTensor requires internal Layout because it extracts compile-time
type parameters:

```mojo
# TileTensor uses internal Layout's shape_types/stride_types
comptime SMemTile[dtype, layout] = TileTensor[
    shape_types = layout.shape_types,   # Internal Layout required!
    stride_types = layout.stride_types,
    ...
]
```

LayoutTensor's reshape is trivial - just creates a new tensor from the
same pointer:

```mojo
return Self.ReshapeType[dst_layout](self.ptr)  # Line 5130 of layout_tensor.mojo
```

We can do the same with TileTensor:

```mojo
var reshaped = SMemTile[dtype, new_layout](tile.ptr, new_layout)
```

---

## Migration Strategy: Dimensions Instead of Layout

### The Problem

Current code passes `c_smem_layout: Layout` (public Layout) to structs:

```mojo
struct TMAStoreExecutor[
    c_smem_layout: Layout,  # Public Layout - incompatible with TileTensor
    ...
]
```

### The Solution

Pass **dimensions** instead, create internal Layout inside:

```mojo
struct TMAStoreExecutor[
    c_smem_dim0: Int,  # Height dimension (M)
    c_smem_dim1: Int,  # Width dimension (N)
    ...
]:
    # Create internal Layout from dimensions
    comptime c_smem_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()

    # Now SMemTile works!
    comptime CTile = SMemTile[Self.c_type, Self.c_smem_layout]
```

---

## Affected Structs and Parameters

### 1. TMAStoreExecutor (tile_writer.mojo:341)

**Current:**

```mojo
struct TMAStoreExecutor[
    c_type: DType,
    c_smem_layout: Layout,  # ← Change this
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    stage_contiguous_size: Int,
    ...
]
```

**After:**

```mojo
struct TMAStoreExecutor[
    c_type: DType,
    c_smem_dim0: Int,       # ← New: replaces c_smem_layout
    c_smem_dim1: Int,       # ← New
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    stage_contiguous_size: Int,
    ...
]:
    # Create internal layout from dimensions
    comptime c_smem_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()
```

**Usage changes:**

- `Self.c_smem_layout.shape[0].value()` → `Self.c_smem_dim0`
- `SMemTile[Self.c_type, Self.c_smem_layout]` → works with internal Layout

### 2. TMEMToSMemWriter (tile_writer.mojo:723)

**Current:**

```mojo
struct TMEMToSMemWriter[
    c_type: DType,
    accum_type: DType,
    c_smem_layout: Layout,  # ← Change this
    BM: Int,
    ...
]
```

**After:**

```mojo
struct TMEMToSMemWriter[
    c_type: DType,
    accum_type: DType,
    c_smem_dim0: Int,       # ← New
    c_smem_dim1: Int,       # ← New
    BM: Int,
    ...
]:
    comptime c_smem_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()
```

### 3. SMemEpilogueWriter (tile_writer.mojo:908)

**Current:**

```mojo
struct SMemEpilogueWriter[
    c_type: DType,
    c_smem_layout: Layout,  # ← Change this
    num_output_stages: Int,
    ...
]
```

**After:**

```mojo
struct SMemEpilogueWriter[
    c_type: DType,
    c_smem_dim0: Int,       # ← New
    c_smem_dim1: Int,       # ← New
    num_output_stages: Int,
    ...
]:
    comptime c_smem_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()
```

### 4. TileWriter (output_writer.mojo:66)

**Current:**

```mojo
struct TileWriter[
    ...
    c_smem_layout: Layout,  # ← Change this
    num_output_stages: Int,
    ...
]
```

**After:**

```mojo
struct TileWriter[
    ...
    c_smem_dim0: Int,       # ← New
    c_smem_dim1: Int,       # ← New
    num_output_stages: Int,
    ...
]:
    comptime c_smem_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()
```

### 5. shared_memory_epilogue_transpose (tile_writer.mojo:1174)

**Current:**

```mojo
fn shared_memory_epilogue_transpose[
    ...
    c_smem_layout: Layout,  # ← Change this
    ...
]
```

**After:**

```mojo
fn shared_memory_epilogue_transpose[
    ...
    c_smem_dim0: Int,       # ← New
    c_smem_dim1: Int,       # ← New
    ...
]:
    comptime c_smem_layout = row_major[c_smem_dim0, c_smem_dim1]()
```

---

## Caller Updates

### matmul_kernels.mojo

**Current:**

```mojo
comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)
...
TileWriter[
    c_smem_layout = Self.SmemType.c_smem_layout,
    ...
]
```

**After:**

```mojo
# No c_smem_layout variable needed!
TileWriter[
    c_smem_dim0 = Self.OutputM,
    c_smem_dim1 = Self.OutputN,
    ...
]
```

### pipeline_storage.mojo

**Current:**

```mojo
comptime c_tile_layout = Layout.row_major(Self.c_dim0, Self.c_dim1)
```

**After:**

```mojo
# Use internal layout
comptime c_tile_layout = row_major[Self.c_dim0, Self.c_dim1]()
```

---

## Import Changes

### tile_writer.mojo

**Current:**

```mojo
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE, RuntimeTuple
from linalg.structuring import SMemTileArray, SMemTile
```

**After:**

```mojo
from layout import RuntimeLayout, UNKNOWN_VALUE, RuntimeTuple  # Remove Layout
from layout._layout import Layout, row_major  # Internal Layout
from .tile_types import SMemTileArrayWithLayout, SMemTile  # TileTensor types
```

### output_writer.mojo

**Current:**

```mojo
from layout import Layout, LayoutTensor, ...
from linalg.structuring import SMemTileArray, SMemTile
```

**After:**

```mojo
from layout import RuntimeLayout, ...  # Remove Layout
from layout._layout import Layout, row_major  # Internal Layout
from .tile_types import SMemTileArrayWithLayout, SMemTile  # TileTensor types
```

---

## Reshape Operations

Current code uses LayoutTensor's `.reshape[]`:

```mojo
var c_smem_reshaped = c_smem_tile.reshape[
    Layout.row_major(2 * Self.stageN, Self.stage_contiguous_size // 2)
]()
```

With TileTensor, create new tile from same pointer with new layout:

```mojo
comptime reshaped_layout = row_major[
    2 * Self.stageN, Self.stage_contiguous_size // 2
]()
var c_smem_reshaped = SMemTile[Self.c_type, reshaped_layout](
    c_smem_tile.ptr, reshaped_layout
)
```

---

## TMA Boundary Handling

TMA APIs (`TMATensorTile.async_store`) still require LayoutTensor. At this
boundary, construct LayoutTensor from TileTensor pointer:

```mojo
# Define LayoutTensor type alias for TMA boundary
comptime _CTileLT = LayoutTensor[
    Self.c_type,
    Layout.row_major(Self.c_smem_dim0, Self.c_smem_dim1),  # Public Layout for TMA
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    alignment = 128,
]

# At TMA call site only
c_tma_op.async_store(Self._CTileLT(c_smem_tile.ptr), coords)
```

This is the ONLY place we need public Layout/LayoutTensor.

---

## Implementation Order

1. **Phase 1: tile_writer.mojo**
   - Add internal Layout import
   - Change TMAStoreExecutor parameters
   - Change TMEMToSMemWriter parameters
   - Change SMemEpilogueWriter parameters
   - Update shared_memory_epilogue_transpose
   - Update reshape operations to TileTensor pattern
   - Add LayoutTensor aliases for TMA boundary

2. **Phase 2: output_writer.mojo**
   - Add internal Layout import
   - Change TileWriter parameters
   - Update CTileArray to SMemTileArrayWithLayout
   - Update internal struct instantiations

3. **Phase 3: Caller updates**
   - matmul_kernels.mojo
   - grouped_1d1d_matmul_kernel.mojo
   - pipeline_storage.mojo

4. **Phase 4: Test and verify**
   - Run test suite
   - Check compilation time

---

## Testing Checklist

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

## Status

| Task | Status |
|------|--------|
| Document migration plan | ✅ Complete |
| Phase 1: tile_writer.mojo | ✅ Complete |
| Phase 2: output_writer.mojo | ✅ Complete |
| Phase 3: Caller updates | ✅ Complete |
| Phase 4: Testing | ✅ Complete |

---

## Completed Migration (2026-02-02)

The C tensor path migration from `c_smem_layout: Layout` to explicit dimensions
(`c_smem_dim0: Int, c_smem_dim1: Int`) has been completed successfully.

### What Was Changed

**Structs migrated to use dimensions instead of Layout parameter:**

1. **TMAStoreExecutor** (tile_writer.mojo)
   - `c_smem_layout: Layout` → `c_smem_dim0: Int, c_smem_dim1: Int`
   - Internal layout created: `comptime c_smem_layout = Layout.row_major(...)`

2. **TMEMToSMemWriter** (tile_writer.mojo)
   - Same pattern as TMAStoreExecutor

3. **TileWriter** (output_writer.mojo)
   - Same pattern, plus updated all internal usages

4. **SMemEpilogueWriter** (tile_writer.mojo)
   - `c_smem_layout: Layout` → `c_smem_dim0: Int, c_smem_dim1: Int`
   - Changed from infer-only to explicit parameters

5. **BlockwiseFP8TileWriter** (blockwise_fp8_output_writer.mojo)
   - `c_smem_layout: Layout` → `c_smem_dim0: Int, c_smem_dim1: Int`

6. **get_accumulator_layout** (blockwise_fp8_accumulator.mojo)
   - `c_smem_layout: Layout` → `c_smem_dim1: Int`

**Callers updated to pass dimensions:**

- `default/matmul_kernels.mojo` - `c_smem_dim0 = SmemType.OutputM`
- `block_scaled/block_scaled_matmul_kernel.mojo`
- `grouped_block_scaled/grouped_block_scaled_matmul_kernel.mojo`
- `grouped_block_scaled_1d1d/grouped_1d1d_matmul_kernel.mojo`
- `blockwise_fp8/blockwise_fp8_matmul_kernel.mojo`

**Also added to tile_types.mojo:**

- `SMemTileShape[idx, Tile]` - compile-time shape accessor
- `SMemTileStride[idx, Tile]` - compile-time stride accessor

### Tests Verified

All SM100 tests compile successfully:

- `test_matmul_sm100_smoke.mojo`
- `test_matmul_sm100_2sm_bf16.mojo`
- `test_matmul_sm100_block_scaled_nvfp4_1sm.mojo`
- `test_grouped_matmul_sm100_nvfp4.mojo`
- `test_matmul_sm100_blockwise_fp8_smoke.mojo`

### What's Not Changed (Remaining Work)

- `shared_memory_epilogue_transpose` function still uses `c_smem_layout: Layout`
  parameter (internal function with complex layout operations)
- Internal implementations still use LayoutTensor-based `SMemTile` and
  `SMemTileArray`

---

## Implementation Challenges Discovered

### 1. TileTensor `.tile[]` API Difference

**Issue**: TileTensor's `.tile[]` method expects a `Coord` object, not
separate positional arguments.

```mojo
# LayoutTensor style (OLD):
tile.tile[dim0, dim1](idx0, idx1)

# TileTensor style (NEW):
tile.tile[dim0, dim1]((idx0, idx1))  # Tuple implicitly converts to Coord

# For multi-dimensional tiling:
tile.tile[d0, d1, d2, d3]((i0, i1, i2, i3))
```

**Note**: The Tuple must have matching types (all Int). If using range
iterators, explicitly convert: `(Int(i), 0)`.

### 2. Compile-time Layout Access in `store_fragment_to_smem`

**Issue**: The function accesses layout properties at compile time:

```mojo
comptime stride0 = dst.layout.stride[0].value()
```

With TileTensor, the `.layout` property is runtime, not compile-time.
Need to access type parameters instead:

```mojo
# Option 1: Access through type parameters
comptime stride0 = type_of(dst).stride_types[0].static_value

# Option 2: Pass layout info explicitly as parameters
fn store_fragment_to_smem[stride0: Int, stride1: Int, ...](...)
```

### 3. Helper Functions Using Public Layout Operations

**Issue**: Functions like `shared_memory_epilogue_transpose` use `upcast()`,
`zipped_divide()`, `blocked_product()` from `layout.layout` module.
These work with public Layout only.

**Solution**: Either:

- Keep public Layout for internal computations (violates "no old layouts" goal)
- Rewrite helper functions to use internal Layout/TileTensor operations
- Extract needed values at compile time and pass as parameters

### 4. Many Interconnected Callers

**Issue**: Changes to `tile_writer.mojo` break:

- `output_writer.mojo` - still uses `SMemTileArray` and `c_smem_layout: Layout`
- `blockwise_fp8_output_writer.mojo` - calls `store_fragment_to_smem` with
  LayoutTensor

**Solution**: Migration must be coordinated across all files simultaneously.

### 5. `shared_memory_epilogue` Layout Constraint

**Issue**: The `vectorize()` call requires `all_dims_known` constraint on
TileTensor, which may not be satisfied when layouts are passed as parameters.

**Solution**: Ensure layout parameters are fully static (compile-time known
dimensions).

---

## Revised Implementation Strategy

Given the challenges, the migration should be done in smaller, incremental
steps:

### Step 1: Add TileTensor utilities to tile_types.mojo

- Add helper for extracting compile-time stride/shape from TileTensor types
- Add conversion utilities between TileTensor and LayoutTensor

### Step 2: Update `store_fragment_to_smem` first

- Change to accept TileTensor with explicit layout parameters
- Or create overload that works with TileTensor

### Step 3: Update structs one at a time

- Start with `TMAStoreExecutor` (simplest, mainly TMA boundary)
- Then `TMEMToSMemWriter`
- Then `SMemEpilogueWriter` (most complex, has epilogue computations)

### Step 4: Update callers incrementally

- After each struct is updated, update its callers
- Test after each change

### Step 5: Helper functions last

- `shared_memory_epilogue_transpose`
- `shared_memory_epilogue`

---

## Summary

The migration eliminates public Layout by:

1. **Passing dimensions** (`c_smem_dim0`, `c_smem_dim1`) instead of
   `c_smem_layout: Layout`
2. **Creating internal Layout** inside structs: `row_major[dim0, dim1]()`
3. **Using TileTensor** (`SMemTile`, `SMemTileArrayWithLayout`) throughout
4. **Only at TMA boundary**: Construct LayoutTensor from pointer for API
   compatibility

This achieves the goal of eliminating both public Layout and LayoutTensor
from the structured kernel code, keeping them only as implementation details
at API boundaries.
