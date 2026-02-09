# TileTensor Patterns for SM100 Structured Kernels

## Overview

This document summarizes the patterns used for TileTensor migration in SM100
structured kernels.

**Current Status (2026-02-02)**: Partial migration. TileTensor is used for
storage types, but explicit LayoutTensor type aliases are required at TMA/MMA
boundaries to avoid compilation slowdowns. See the migration journal Part 24
for details.

## Key Insight: Internal vs Public Layout

**Mojo has TWO Layout types - use the INTERNAL one for type-aware operations.**

| Module | Type | Type Params | Usage |
|--------|------|-------------|-------|
| `layout._layout` | Internal Layout | `shape_types`, `stride_types` | Compile-time type info, `.shape_types` accessible |
| `layout.layout` | Public Layout | None | Runtime `IntTuple` for shape/stride |

The **internal Layout** (from `layout._layout`) preserves compile-time type
parameters that can be accessed through struct parameters.

```mojo
# WORKS - using internal Layout from _layout.mojo
from layout._layout import Layout, row_major

comptime SMemTile[
    dtype: DType,
    tile_layout: Layout,  # INTERNAL Layout - type params preserved!
] = TileTensor[
    shape_types = tile_layout.shape_types,  # Can access!
    stride_types = tile_layout.stride_types,
    ...
]

struct MyStruct[layout: Layout]:  # INTERNAL Layout
    comptime Tile = SMemTile[Self.dtype, Self.layout]  # Works!
```

```mojo
# DOES NOT WORK - public Layout from layout.mojo
from layout import Layout  # Public Layout - no type params!

comptime SMemTile[
    dtype: DType,
    tile_layout: Layout,  # Public Layout - type params erased
] = TileTensor[
    shape_types = tile_layout.shape_types,  # ERROR: no attribute 'shape_types'
    ...
]
```

## Pattern 1: Internal Swizzled Layouts

**Create internal Layout aliases matching swizzled structures.**

The swizzled layouts from `tile_layout_k_major` have nested structures like
`((8, 8), (64, 1))`. We can create matching internal Layouts. These are now
unified into a single parametric function:

```mojo
# In tile_types.mojo
from layout._layout import Layout
from layout._coord import Coord, Idx

comptime _CM_NUM_ROWS = 8

# Parametric internal swizzled layout matching tile_layout_k_major
comptime internal_k_major[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_bytes: Int,  # 32, 64, or 128
] = Layout(
    Coord(
        Coord(Idx[BM // _CM_NUM_ROWS](), Idx[_CM_NUM_ROWS]()),
        Coord(
            Idx[swizzle_bytes // size_of[dtype]()](),
            Idx[BK * size_of[dtype]() // swizzle_bytes](),
        ),
    ),
    Coord(
        Coord(
            Idx[swizzle_bytes // size_of[dtype]()](),
            Idx[(BM // _CM_NUM_ROWS) * (swizzle_bytes // size_of[dtype]())](),
        ),
        Coord(Idx[1](), Idx[0]()),
    ),
)

# Convenience aliases for common swizzle sizes
comptime internal_k_major_128B[dtype, BM, BK] = internal_k_major[dtype, BM, BK, 128]
comptime internal_k_major_64B[dtype, BM, BK] = internal_k_major[dtype, BM, BK, 64]
comptime internal_k_major_32B[dtype, BM, BK] = internal_k_major[dtype, BM, BK, 32]
```

## Pattern 2: Swizzled Tile Types

**Use `SMemTile` with internal swizzled layouts.**

```mojo
# In SMEM struct:
comptime ATile = SMemTile[
    Self.a_type,
    internal_k_major_128B[Self.a_type, Self.BM, Self.BK]
]

# For backward compatibility with explicit dimensions:
comptime ATile2D = SMemTile2D[Self.a_type, Self.BM, Self.BK]  # Uses row_major internally
```

**IMPORTANT**: At TMA/MMA boundaries, use explicit LayoutTensor construction
instead of `.to_layout_tensor()`. The latter causes massive compilation
slowdowns (see migration journal Part 24).

```mojo
# In kernel - define explicit LayoutTensor type aliases
comptime ATileLT = LayoutTensor[
    Self.a_type,
    Self.SmemType.a_smem_layout,  # Public layout from SMEM struct
    address_space = AddressSpace.SHARED,
    alignment = 128,
]

# At TMA/MMA boundary - explicit construction from pointer
mma_op.mma(
    Self.ATileLT(a_tile.ptr),  # Explicit construction, not .to_layout_tensor()
    Self.BTileLT(b_tile.ptr),
    ...
)
```

## Pattern 3: Storage with Dimensions, Types with Layouts

**Separate storage allocation from type information.**

Storage types use explicit dimensions (for memory allocation):

```mojo
comptime ATileArray = SMemTileArray2D[
    Self.a_type, Self.BM, Self.BK, Self.num_pipeline_stages, 128
]
```

Type aliases use internal swizzled layouts (for correct swizzling):

```mojo
comptime ATile = SMemTile[
    Self.a_type, internal_k_major_128B[Self.a_type, Self.BM, Self.BK]
]
```

**At boundaries, use explicit LayoutTensor construction** (not `.to_layout_tensor()`):

```mojo
# Define LayoutTensor type alias with public layout from SMEM struct
comptime ATileLT = LayoutTensor[Self.a_type, Self.SmemType.a_smem_layout, ...]

# Construct LayoutTensor directly from TileTensor pointer
var tile = array[idx]  # TileTensor
a_loader.load(Self.ATileLT(tile.ptr), barrier, coords)
```

## Pattern 4: Explicit LayoutTensor Construction (RECOMMENDED)

**This is the PRIMARY pattern for TMA/MMA boundaries.**

Explicit LayoutTensor type aliases act as compile-time caches. The compiler
computes the type once, then reuses it at every call site. Using
`.to_layout_tensor()` forces the compiler to re-infer the type at each call
site, causing massive compilation slowdowns.

```mojo
# In kernel struct - define LayoutTensor type aliases ONCE
comptime ATileLT = LayoutTensor[
    Self.a_type,
    Self.SmemType.a_smem_layout,  # Public Layout from SMEM struct
    address_space = AddressSpace.SHARED,
    alignment = 128,
]
comptime BTileLT = LayoutTensor[
    Self.b_type,
    Self.SmemType.b_smem_layout,
    address_space = AddressSpace.SHARED,
    alignment = 128,
]

# At TMA boundary - explicit construction
tma_op.async_multicast_load_3d[...](
    Self.ATileLT(tile.ptr),  # Explicit, not .to_layout_tensor()
    barrier, coords, mask,
)

# At MMA boundary - explicit construction
mma_op.mma(
    Self.ATileLT(a_tile.ptr),
    Self.BTileLT(b_tile.ptr),
    ...
)
```

**Why NOT `.to_layout_tensor()`:**

- Forces type inference at every call site
- With ~32 boundary calls across kernels, this multiplies compile time
- See migration journal Part 24 for full analysis

## Architecture Summary

```text
tile_types.mojo
├── Parametric swizzled layout: internal_k_major[dtype, BM, BK, swizzle_bytes]
├── Convenience aliases: internal_k_major_128B, internal_k_major_64B, internal_k_major_32B
├── SMemTile[dtype, layout] - TileTensor with Layout parameter
├── SMemTile2D[dtype, dim0, dim1] - Backward-compatible alias (uses row_major)
├── SMemTileArray2D[dtype, dim0, dim1, ...] - Storage with explicit dimensions
└── SMemTileArrayWithLayout[dtype, layout, ...] - Storage with swizzled layout

SMEM Struct (*_smem.mojo)
├── Public layouts: a_smem_layout, b_smem_layout (from tile_layout_k_major)
├── Storage: BlockScaledTileStorage with SMemTileArray2D (allocation)
├── Swizzled types: ATile = SMemTile[internal_k_major_128B[...]]
└── Returns TileTensor tiles from accessors

Kernel (*_matmul_kernel.mojo)
├── Define LayoutTensor type aliases: ATileLT, BTileLT (from SMEM layouts)
├── Get TileTensor tiles from SMEM/payload
└── At TMA/MMA boundaries: Self.ATileLT(tile.ptr) - explicit construction
```

**Key insight**: Explicit LayoutTensor type aliases act as compile-time caches.
Converting at the API level (future work) is better than converting at every
call site.

## Files Involved

| File | Role |
|------|------|
| `tile_types.mojo` | Internal swizzled layouts (`internal_k_major`), `SMemTile`, `SMemTileArray2D` |
| `pipeline_storage.mojo` | Storage types with explicit dimensions |
| `tile_pipeline.mojo` | Payload types with TileTensor arrays |
| `*_smem.mojo` | SMEM structs with public layouts + TileTensor accessors |
| `*_matmul_kernel.mojo` | Kernel code with LayoutTensor type aliases at boundaries |

## Future Work

To complete TileTensor migration without compilation slowdown:

1. **Convert TileWriter to accept TileTensor** - Handle conversion internally
2. **Convert MMA ops to accept TileTensor** - Handle conversion internally
3. **Keep explicit LayoutTensor type aliases** - They act as compile-time caches

The key insight: convert at the API level, not at every call site.
