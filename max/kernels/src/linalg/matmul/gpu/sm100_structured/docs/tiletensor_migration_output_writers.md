# TileTensor Migration Plan: C Tensor Path in Output Writers

## Executive Summary

This document provides a comprehensive plan for migrating the C output tensor
path in `tile_writer.mojo` and `output_writer.mojo` from LayoutTensor-based
APIs (`linalg.structuring`) to TileTensor-based APIs (`tile_types.mojo`).

**Current State**: Both files import `SMemTile` and `SMemTileArray` from
`linalg.structuring` (LayoutTensor-based). The C tiles flow through:

```text
TMEM (accumulators) → Registers → SMEM (C tiles) → GMEM (via TMA)
```

**Target State**: Use TileTensor-based types from `.tile_types`:

- `SMemTile[dtype, layout]` - TileTensor with Layout parameter
- `SMemTileArray2D[dtype, dim0, dim1, num_tiles]` - Array with explicit
  dimensions
- `SMemTileArrayWithLayout[dtype, layout, num_tiles]` - Array preserving
  swizzle info

**Critical Constraint (from Part 24 of Journal)**: Do NOT use
`.to_layout_tensor()` at every boundary - this causes massive compilation
slowdowns. Instead, use **explicit LayoutTensor type aliases** at TMA/MMA
boundaries as compile-time caches

---

## Key Migration Insight

The TileTensor types in `tile_types.mojo` are designed to be API-compatible with
the legacy `linalg.structuring` types:

| Legacy Type | TileTensor Type | Notes |
|-------------|-----------------|-------|
| `SMemTile[dtype, layout]` | `SMemTile[dtype, layout]` | Same signature, different impl |
| `SMemTileArray[dtype, layout, n]` | `SMemTileArrayWithLayout[dtype, layout, n]` | Different name |
| N/A | `SMemTileArray2D[dtype, dim0, dim1, n]` | New: explicit dimensions |

The TileTensor `SMemTile` has the same signature but uses TileTensor internally
instead of LayoutTensor. This means most usages can migrate with just an import
change.

---

## Migration Locations

### output_writer.mojo (3 locations)

| Line | Current Code | Migration Action |
|------|--------------|------------------|
| 45 | `from linalg.structuring import SMemTileArray, SMemTile` | Change to `.tile_types` |
| 110-112 | `CTileArray = SMemTileArray[c_type, c_smem_layout, num_output_stages, alignment=128]` | Use `SMemTileArrayWithLayout` |
| 656 | `c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128]` | No change needed (same signature) |

### tile_writer.mojo (18 locations)

| Line | Location | Current Code | Migration Action |
|------|----------|--------------|------------------|
| 40 | import | `from linalg.structuring import SMemTileArray, SMemTile` | Change to `.tile_types` |
| 140 | `store_fragment_to_smem` | `dst: SMemTile` | No change (same signature) |
| 377 | `TMAStoreExecutor.execute` | `c_smem_tile: SMemTile[...]` | No change |
| 415 | `_store_transpose` | `c_smem_tile: SMemTile[...]` | No change |
| 501 | `_store_non_transpose` | `c_smem_tile: SMemTile[...]` | No change |
| 763 | `TMEMToSMemWriter.write_fragments` | `c_smem_tile: SMemTile[...]` | No change |
| 786 | `_write_transpose` | `c_smem_tile: SMemTile[...]` | No change |
| 808-850 | `_write_transpose` | Local `SMemTile` creations | Change to TileTensor constructor |
| 870 | `_write_non_transpose` | `c_smem_tile: SMemTile[...]` | No change |
| 943-944 | `SMemEpilogueWriter` | `CTileArray = SMemTileArray[...]` | Use `SMemTileArrayWithLayout` |
| 958-962 | `SMemEpilogueWriter.__init__` | `c_tiles: SMemTileArray[...]` | Use `SMemTileArrayWithLayout` |
| 992 | `_write_transpose` | `c_smem_tile: SMemTile[...]` | No change |
| 1014-1016 | `_write_transpose` | `new_smem = SMemTile[...]` | Change to TileTensor constructor |
| 1046-1047 | `_write_transpose` | `new_smem = SMemTile[...]` | Change to TileTensor constructor |
| 1071-1073 | `_write_transpose` | `new_smem = SMemTile[...]` | Change to TileTensor constructor |
| 1116 | `_write_non_transpose` | `c_smem_tile: SMemTile[...]` | No change |
| 1192 | `shared_memory_epilogue_transpose` | `c_smem: SMemTile[...]` | No change |
| 1382-1383 | `shared_memory_epilogue` | `c_smem_warp_tile_*: SMemTile[...]` | No change |

---

## Detailed Changes

### Step 1: Change Imports (Both Files)

**output_writer.mojo line 45:**

```mojo
# Before
from linalg.structuring import SMemTileArray, SMemTile

# After
from .tile_types import SMemTile, SMemTileArrayWithLayout
```

**tile_writer.mojo line 40:**

```mojo
# Before
from linalg.structuring import SMemTileArray, SMemTile

# After
from .tile_types import SMemTile, SMemTileArrayWithLayout
```

### Step 2: Update Type Aliases (output_writer.mojo)

**Line 110-112:**

```mojo
# Before
comptime CTileArray = SMemTileArray[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]

# After
comptime CTileArray = SMemTileArrayWithLayout[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]
```

### Step 3: Update Type Aliases (tile_writer.mojo)

**Line 943-944 (SMemEpilogueWriter):**

```mojo
# Before
comptime CTileArray = SMemTileArray[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]

# After
comptime CTileArray = SMemTileArrayWithLayout[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]
```

### Step 4: Update Local SMemTile Creations (tile_writer.mojo)

The TileTensor-based `SMemTile` is constructed with `(ptr, layout)` instead of
just `(ptr)`. The local creations in `_write_transpose` need updating:

**Lines 808-812 (within `_write_transpose`, is_lower_required path):**

```mojo
# Before
var new_smem = SMemTile[
    Self.c_type,
    smem_logical_layout,
    alignment = c_smem_tile.alignment,
](c_smem_tile.ptr)

# After
var new_smem = SMemTile[
    Self.c_type,
    smem_logical_layout,
    alignment = c_smem_tile.alignment,
](c_smem_tile.ptr, smem_logical_layout)
```

**Lines 846-850 (within `_write_transpose`, !is_lower_required path):**

```mojo
# Before
var new_smem = SMemTile[
    Self.c_type,
    smem_logical_layout,
    alignment = c_smem_tile.alignment,
](c_smem_tile.ptr)

# After
var new_smem = SMemTile[
    Self.c_type,
    smem_logical_layout,
    alignment = c_smem_tile.alignment,
](c_smem_tile.ptr, smem_logical_layout)
```

**Lines 1014-1016 (SMemEpilogueWriter._write_transpose, is_lower_required path):**

```mojo
# Before
var new_smem = SMemTile[
    Self.c_type, smem_logical_layout, alignment=128
](c_smem_tile.ptr)

# After
var new_smem = SMemTile[
    Self.c_type, smem_logical_layout, alignment=128
](c_smem_tile.ptr, smem_logical_layout)
```

**Lines 1071-1073 (SMemEpilogueWriter._write_transpose, !is_lower_required path):**

```mojo
# Before
var new_smem = SMemTile[
    Self.c_type, smem_logical_layout, alignment=128
](c_smem_tile.ptr)

# After
var new_smem = SMemTile[
    Self.c_type, smem_logical_layout, alignment=128
](c_smem_tile.ptr, smem_logical_layout)
```

---

## TileTensor API Differences

### SMemTile Constructor

| Legacy | TileTensor |
|--------|------------|
| `SMemTile[dtype, layout](ptr)` | `SMemTile[dtype, layout](ptr, layout)` |

The TileTensor version requires the layout to be passed at construction time.

### SMemTile Methods

Key API differences:

- `.ptr` - Works the same
- `.tile[dims](i, j)` → `.tile[dims](Coord(i, j))` - TileTensor uses Coord
- `.reshape[layout]()` → `.coalesce()` or construct new TileTensor
- `.to_layout_tensor()` - Convert to LayoutTensor at API boundaries

### SMemTileArray → SMemTileArrayWithLayout

| Legacy | TileTensor |
|--------|------------|
| `SMemTileArray[dtype, layout, n, alignment=128]` | `SMemTileArrayWithLayout[dtype, layout, n, alignment=128]` |

The type name changes but the signature is identical.

---

## Implementation Order

1. **tile_writer.mojo first** - Contains the component structs used by output_writer.mojo
2. **output_writer.mojo second** - Depends on tile_writer.mojo components

### Substeps for tile_writer.mojo

1. Change import (line 40)
2. Update `SMemEpilogueWriter.CTileArray` (lines 943-944)
3. Update `SMemEpilogueWriter.__init__` parameter (lines 958-962)
4. Update local `SMemTile` creations (lines 808-850, 1014-1016, 1071-1073)

### Substeps for output_writer.mojo

1. Change import (line 45)
2. Update `TileWriter.CTileArray` (lines 110-112)

---

## Testing Strategy

After migration, run these tests:

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

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Constructor signature difference | Pass layout explicitly at all construction sites |
| Method compatibility | TileTensor SMemTile has same methods as legacy |
| Type inference issues | Use explicit type annotations where needed |
| Cascading dependencies | Use dual accessor pattern, migrate incrementally |

**Overall Risk**: Low - Well-established patterns from other migrations.

---

## Status

| Step | Status |
|------|--------|
| Create migration plan | **DONE** |
| Migrate input tiles (A, B, scales) | **DONE** (using SMemTileArray2D) |
| Migrate C output tiles | **IN PROGRESS** - Using dual accessor pattern |
| Migrate output_writer.mojo | **IN PROGRESS** |
| Migrate tile_writer.mojo | **IN PROGRESS** |

---

## TileTensor Migration Patterns (from PRs #75977, #76670, #76603, #76663)

The following patterns were extracted from successful TileTensor migrations across
the codebase. These demonstrate how to handle various migration scenarios.

### Pattern 1: Dynamic Layout Construction

Create TileTensor with runtime-determined shapes using `row_major(Coord(shape))`:

```mojo
# From PR #75977 - creating TileTensor from buffer with dynamic shape
var input0_device = TileTensor(
    input0_device_buffer.unsafe_ptr(),
    row_major(Coord(shape0)),
)

# From PR #76670 - stack allocation with row_major layout
var smem = tensor_alloc[DType.uint32, address_space = AddressSpace.SHARED](
    row_major[1, expected_count]()
)
```

### Pattern 2: Coordinate Construction with Idx and Coord

TileTensor uses `Coord` for indexing instead of separate integer arguments:

```mojo
# LayoutTensor style (old)
tensor.tile[dims](i, j)

# TileTensor style (new)
tensor.tile[dims](Coord(i, j))

# Creating coordinate with Idx for type-specific construction
var out_coord = Coord(out_coords)
var in_coord = Coord(coords)

# Load with Coord
output_host.load[width=1](out_coord)
```

### Pattern 3: Converting TileTensor to LayoutTensor at API Boundaries

Use `.to_layout_tensor()` when calling functions that require LayoutTensor:

```mojo
# From PR #76603 - convert at function call boundary
topk_gpu[sampling=False, largest=largest](
    ctx, K,
    input.to_layout_tensor(),  # Convert TileTensor to LayoutTensor
    out_vals.to_layout_tensor(),
    output.to_layout_tensor(),
)
```

This is the **key pattern** for migrating code incrementally - functions that
require LayoutTensor internally can still accept TileTensor at their boundaries.

### Pattern 4: Accessing Shape and Stride from TileTensor

```mojo
# From PR #76603 - extract shape/stride as index lists
coord_to_index_list(view_buffer.layout.shape)
coord_to_index_list(view_buffer.layout.stride)

# From PR #76670 - access individual dimensions
var num_tokens: Int = Int(topk_ids.layout.shape[0].value())
```

### Pattern 5: TileTensor Function Signatures

For functions with fully generic TileTensor parameters:

```mojo
# From PR #76670 - explicit shape/stride type parameters
fn moe_create_indices_kernel[
    token_expert_order_shape_types: Variadic.TypesOfTrait[CoordLike],
    token_expert_order_stride_types: Variadic.TypesOfTrait[CoordLike],
](
    token_expert_order: TileTensor[
        mut=True,
        shape_types=token_expert_order_shape_types,
        stride_types=token_expert_order_stride_types,
        DType.uint32,
        MutAnyOrigin,
    ],
):
    ...
```

### Pattern 6: Rank Assertions

Use `__comptime_assert` for compile-time rank checking:

```mojo
# From PR #76670
__comptime_assert topk_ids.rank == 1, "topk_ids must be 1D"
```

### Pattern 7: Dual Accessor Pattern for C Tiles

When existing functions require LayoutTensor, provide both accessors:

```mojo
# From pipeline_storage.mojo - dual accessor pattern
fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArrayLT:
    """Returns LayoutTensor view for compatibility with tile_writer.mojo."""
    return Self.CTileArrayLT(self.c_tiles_storage.unsafe_ptr())

fn c_tiles_tt(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
    """Returns native TileTensor for future TileTensor-native code paths."""
    return Self.CTileArray(self.c_tiles_storage.unsafe_ptr())
```

This allows incremental migration - new code uses `c_tiles_tt()` while existing
code continues using `c_tiles()` until fully migrated.

### Pattern 8: Using .coalesce() Instead of .reshape[]

TileTensor uses `.coalesce()` to flatten/reshape:

```mojo
# LayoutTensor style (old)
tile.reshape[coalesce(tile.layout)]()

# TileTensor style (new)
tile.coalesce()
```

### Pattern 9: Stack Allocation for Shared Memory

```mojo
# From PR #76670
var smem = tensor_alloc[DType.uint32, address_space = AddressSpace.SHARED](
    row_major[1, expected_count]()
)
```

---

## Recommended Migration Strategy (Part 24 Approach)

Based on lessons from the migration journal (especially Part 24), the C tile
migration should follow these principles:

### Key Insight: Avoid `.to_layout_tensor()` at Every Call Site

The journal documented that using `.to_layout_tensor()` at ~32 call sites caused
**massive compilation slowdowns** due to:

1. Inline type inference at each call site
2. Complex variadic type expansion for shape_types/stride_types
3. No compile-time caching of the computed types

### Solution: Explicit LayoutTensor Type Aliases

Instead of `.to_layout_tensor()`, use **explicit LayoutTensor type aliases** at
boundaries. These act as compile-time caches:

```mojo
# GOOD: Define explicit type alias once (compile-time cached)
comptime CTileLT = LayoutTensor[
    Self.c_type,
    Self.c_smem_layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    alignment = 128,
]

# At TMA boundary - use alias to construct
c_tma_op.async_store(Self.CTileLT(c_smem_tile.ptr), coords)

# BAD: This causes compilation slowdown
c_tma_op.async_store(c_smem_tile.to_layout_tensor(), coords)  # DON'T DO THIS
```

### Migration Approach for C Tiles

1. **Change imports** from `linalg.structuring` to `.tile_types`
2. **Use `SMemTileArrayWithLayout`** for C tiles (preserves swizzle info)
3. **Add explicit LayoutTensor type aliases** for boundary conversion
4. **Update local SMemTile constructions** to include layout parameter
5. **Keep internal code using TileTensor** - only convert at boundaries

---

## Detailed Migration Plan

### Phase 1: tile_writer.mojo Migration

#### Step 1.1: Change Import (Line 40)

```mojo
# Before
from linalg.structuring import SMemTileArray, SMemTile

# After
from .tile_types import SMemTileArrayWithLayout, SMemTile
```

#### Step 1.2: Add LayoutTensor Type Aliases for Boundary Conversion

Add these aliases inside structs that need TMA boundary conversion:

```mojo
# In TMAStoreExecutor (around line 341)
struct TMAStoreExecutor[...]:
    # Add LayoutTensor alias for boundary conversion
    comptime CTileLT = LayoutTensor[
        Self.c_type,
        Self.c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = 128,
    ]

# In TMEMToSMemWriter (around line 723)
struct TMEMToSMemWriter[...]:
    comptime CTileLT = LayoutTensor[
        Self.c_type,
        Self.c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = 128,
    ]

# In SMemEpilogueWriter (around line 908)
struct SMemEpilogueWriter[...]:
    comptime CTileLT = LayoutTensor[
        Self.c_type,
        Self.c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = 128,
    ]
```

#### Step 1.3: Update CTileArray Type Alias (Lines 943-944)

```mojo
# Before
comptime CTileArray = SMemTileArray[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]

# After
comptime CTileArray = SMemTileArrayWithLayout[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]
```

#### Step 1.4: Update Local SMemTile Constructions

The TileTensor-based `SMemTile` requires `(ptr, layout)` instead of just `(ptr)`.
Update these locations:

**Lines 808-812 (TMEMToSMemWriter._write_transpose, is_lower_required path):**

```mojo
# Before
var new_smem = SMemTile[
    Self.c_type,
    smem_logical_layout,
    alignment = c_smem_tile.alignment,
](c_smem_tile.ptr)

# After
var new_smem = SMemTile[
    Self.c_type,
    smem_logical_layout,
    alignment = c_smem_tile.alignment,
](c_smem_tile.ptr, smem_logical_layout)
```

**Lines 846-850 (TMEMToSMemWriter._write_transpose, !is_lower_required path):**

```mojo
# Same pattern - add smem_logical_layout as second argument
```

**Lines 1014-1016 (SMemEpilogueWriter._write_transpose, is_lower_required path):**

```mojo
# Before
var new_smem = SMemTile[
    Self.c_type, smem_logical_layout, alignment=128
](c_smem_tile.ptr)

# After
var new_smem = SMemTile[
    Self.c_type, smem_logical_layout, alignment=128
](c_smem_tile.ptr, smem_logical_layout)
```

**Lines 1071-1073 (SMemEpilogueWriter._write_transpose, !is_lower_required path):**

```mojo
# Same pattern - add smem_logical_layout as second argument
```

#### Step 1.5: Handle .reshape[] and .tile[] Operations

TileTensor's `.tile[]` and `.reshape[]` methods exist but may have different
semantics. For complex reshape operations, convert to LayoutTensor first:

```mojo
# For reshape operations (e.g., lines 436-443)
# Option A: Keep using LayoutTensor for complex reshape
var c_smem_lt = Self.CTileLT(c_smem_tile.ptr)
var c_smem_reshaped = c_smem_lt.reshape[Layout.row_major(...)]()

# Option B: If TileTensor.coalesce() is sufficient
var c_smem_coalesced = c_smem_tile.coalesce()
```

### Phase 2: output_writer.mojo Migration

#### Step 2.1: Change Import (Line 45)

```mojo
# Before
from linalg.structuring import SMemTileArray, SMemTile

# After
from .tile_types import SMemTileArrayWithLayout, SMemTile
```

#### Step 2.2: Update CTileArray Type Alias (Lines 110-112)

```mojo
# Before
comptime CTileArray = SMemTileArray[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]

# After
comptime CTileArray = SMemTileArrayWithLayout[
    Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
]
```

#### Step 2.3: Add LayoutTensor Type Alias for TileWriter

```mojo
struct TileWriter[...]:
    # Add LayoutTensor alias for boundary operations
    comptime CTileLT = LayoutTensor[
        Self.c_type,
        Self.c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = 128,
    ]
```

### Phase 3: Update Callers (SMEM Structs and Kernels)

The SMEM structs in `pipeline_storage.mojo` already have the dual accessor
pattern (`c_tiles()` vs `c_tiles_tt()`). After migrating tile_writer.mojo and
output_writer.mojo:

1. Kernels can continue using `smem.c_tiles()` (returns LayoutTensor array)
2. Or switch to `smem.c_tiles_tt()` once tile_writer accepts TileTensor

---

## C Tile Flow Analysis

### Current Flow (LayoutTensor-based)

```text
pipeline_storage.mojo:
  OutputTileStorage.c_tiles() -> SMemTileArray (LayoutTensor)
      |
      v
output_writer.mojo:
  TileWriter.write(c_tiles: CTileArray)
      |
      v  c_tiles[stage % 2] -> SMemTile (LayoutTensor)
      |
tile_writer.mojo:
  TMEMToSMemWriter.write_fragments(c_smem_tile: SMemTile)
      |  .tile[], .reshape[] operations
      v
  store_fragment_to_smem(dst: SMemTile)
      |
      v
  TMAStoreExecutor.execute(c_smem_tile: SMemTile)
      |  .tile[], .reshape[] for TMA formatting
      v
  c_tma_op.async_store(c_smem_tile, coords)
```

### Target Flow (TileTensor-based)

```text
pipeline_storage.mojo:
  OutputTileStorage.c_tiles_tt() -> SMemTileArrayWithLayout (TileTensor)
      |
      v
output_writer.mojo:
  TileWriter.write(c_tiles: CTileArray)  # CTileArray = SMemTileArrayWithLayout
      |
      v  c_tiles[stage % 2] -> SMemTile (TileTensor)
      |
tile_writer.mojo:
  TMEMToSMemWriter.write_fragments(c_smem_tile: SMemTile)  # TileTensor
      |  Internal operations with TileTensor
      |  Convert at boundary: Self.CTileLT(c_smem_tile.ptr)
      v
  store_fragment_to_smem(dst: SMemTile)  # TileTensor internally
      |
      v
  TMAStoreExecutor.execute(c_smem_tile: SMemTile)  # TileTensor
      |  Convert at TMA boundary: Self.CTileLT(...)
      v
  c_tma_op.async_store(CTileLT(c_smem_tile.ptr), coords)
```

---

## Risk Assessment (C Tile Flow)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Constructor signature change breaks code | High | Medium | Systematic update |
| Compilation slowdown from .to_layout_tensor() | Medium | High | Use type aliases |
| .reshape[] semantics differ | Low | Medium | Convert to LayoutTensor |
| Type inference issues | Low | Low | Use explicit annotations |

---

## Testing Checklist

After each phase, verify with these tests:

```bash
# Smoke tests (quick verification)
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_blockwise_fp8_smoke.mojo
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_2sm_bf16.mojo

# Block-scaled tests
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_block_scaled_nvfp4_1sm.mojo

# Grouped tests
mojo max/kernels/test/gpu/linalg/test_grouped_matmul_sm100_nvfp4.mojo
```

Also check compilation time - any significant increase indicates the
`.to_layout_tensor()` issue.

---

## Status Tracking

| Task | Status | Notes |
|------|--------|-------|
| Document migration plan | **DONE** | This document |
| Study TileTensor capabilities | **DONE** | tile_types.mojo analyzed |
| Study PR patterns | **DONE** | #75977, #76670, #76603, #76663 |
| Migrate tile_writer.mojo imports | **DONE** | Added TileTensor-based imports |
| Migrate tile_writer.mojo type aliases | **DONE** | c_smem_dim0/dim1 params |
| Update local SMemTile constructions | **PARTIAL** | Dimensions approach used |
| Add LayoutTensor type aliases | **N/A** | Used dimensions instead |
| Migrate output_writer.mojo | **DONE** | TileWriter uses dimensions |
| Update SMEM struct callers | **DONE** | 4 kernel files updated |
| Run test suite | **DONE** | All SM100 tests compile |
| Verify compilation time | **DONE** | No regression observed |

### Migration Summary (2026-02-02)

**Approach taken:** Instead of converting to TileTensor throughout, we eliminated
public `Layout` from the **external interface** by:

1. Changing `c_smem_layout: Layout` parameters to
   `c_smem_dim0: Int, c_smem_dim1: Int`
2. Creating internal `Layout` inside structs: `Layout.row_major(dim0, dim1)`
3. Internal implementations continue to use LayoutTensor-based `SMemTile`

**Files modified:**

- `tile_types.mojo` - Added `SMemTileShape`, `SMemTileStride` accessors
- `tile_writer.mojo` - `TMAStoreExecutor`, `TMEMToSMemWriter`
- `output_writer.mojo` - `TileWriter`
- `matmul_kernels.mojo`, `block_scaled_matmul_kernel.mojo`,
  `grouped_block_scaled_matmul_kernel.mojo`, `grouped_1d1d_matmul_kernel.mojo`

**Remaining work:**

- `SMemEpilogueWriter` - uses infer-only parameters
- `BlockwiseFP8TileWriter` - separate struct
- Full TileTensor internal migration (Phase 2 of long-term plan)

---

## Long-Term Goal: Complete LayoutTensor Elimination

The **ultimate goal** is to completely remove LayoutTensor from the SM100 structured
kernels, not just migrate with boundary conversions. The current approach with
explicit LayoutTensor type aliases is a **transition strategy**.

### Current Blockers

1. **TMA APIs require LayoutTensor** - `TMATensorTile.async_store()` expects
   LayoutTensor parameters
2. **MMA APIs require LayoutTensor** - `mma_op.mma()` expects LayoutTensor
3. **Compilation performance** - `.to_layout_tensor()` at every call site is slow

### Path to Complete Elimination

#### Option A: Update TMA/MMA APIs to Accept TileTensor

The cleanest solution is to add TileTensor overloads to the TMA and MMA APIs:

```mojo
# In tma_async.mojo - add TileTensor overload
fn async_store[...](
    self,
    src: TileTensor[dtype, ...],  # TileTensor directly
    coords: ...,
):
    # Convert internally (hidden from caller)
    self._async_store_impl(src.to_layout_tensor(), coords)
```

**Benefits:**

- Structured kernels work purely with TileTensor
- Conversion hidden inside TMA/MMA implementations
- Single point of conversion (compile-time cost controlled)

**Complexity:** Requires changes to `layout/tma_async.mojo` and MMA code.

#### Option B: Create TileTensor-Native Tile Writer

Create new versions of tile_writer components that work entirely with TileTensor,
keeping any necessary LayoutTensor conversion internal:

```mojo
struct TMAStoreExecutorTT[...]:
    """TileTensor-native TMA store executor."""

    # Internal LayoutTensor type (private)
    comptime _CTileLT = LayoutTensor[...]

    fn execute(
        self,
        c_smem_tile: SMemTile[...],  # TileTensor
        ...
    ):
        # Convert internally - hidden from callers
        c_tma_op.async_store(Self._CTileLT(c_smem_tile.ptr), coords)
```

**Benefits:**

- Structured kernel code uses only TileTensor types
- LayoutTensor hidden as implementation detail
- Gradual migration possible

**Complexity:** Requires refactoring tile_writer.mojo internals.

#### Option C: MMA Operation Migration

The MMA operations in `linalg/arch/sm100/mma.mojo` could be updated to accept
TileTensor:

```mojo
# Add TileTensor overload
fn mma[...](
    a_tile: TileTensor[a_type, ...],
    b_tile: TileTensor[b_type, ...],
    ...
):
    # Internal conversion
    self._mma_impl(a_tile.to_layout_tensor(), b_tile.to_layout_tensor(), ...)
```

### Recommended Migration Path

1. **Phase 1 (Current)**: Change imports, use SMemTileArrayWithLayout, add explicit
   LayoutTensor aliases at boundaries - **gets C tiles using TileTensor internally**

2. **Phase 2**: Update tile_writer.mojo components to hide LayoutTensor conversion
   internally - **structured kernel code sees only TileTensor**

3. **Phase 3**: Add TileTensor overloads to TMA/MMA APIs - **complete elimination**

4. **Phase 4**: Remove explicit LayoutTensor type aliases and legacy imports

### Why This Matters

- **Consistency**: All SM100 structured kernels use unified TileTensor API
- **Maintainability**: Single tensor type throughout the codebase
- **Future-proofing**: TileTensor is the modern API, LayoutTensor is legacy
- **Performance**: TileTensor has better compile-time type preservation

---

## References

- Migration Journal: `docs/tiletensor_migration_journal.md` (Parts 21-24 especially)
- TileTensor types: `structured_kernels/tile_types.mojo`
- Pipeline storage: `structured_kernels/pipeline_storage.mojo`
- PRs with patterns: #75977, #76670, #76603, #76663
