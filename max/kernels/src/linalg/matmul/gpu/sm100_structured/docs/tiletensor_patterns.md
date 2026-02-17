# TileTensor Patterns for SM100 Structured Kernels

## Overview

This document summarizes the patterns used for TileTensor migration in SM100
structured kernels.

**Current Status (2026-02-10)**: All 8 SM100 kernel families have been migrated
to new Layout types with zero legacy Layout struct params. TMA types are derived
from config via `TmaOpType`/`TmaOpTypeIm2col`. The fallback kernel's C output
path uses `tile_with_offset[stride_layout=...]` for the full
tile/vectorize/distribute chain.

## Naming Convention

- **New Layout** (`layout._layout.Layout`): Coord-based, compile-time type
  parameters. This is the target API.
- **Legacy Layout** (`layout.layout.Layout`): IntTuple-based, runtime. Import
  as `LegacyLayout` to make the distinction explicit.
- TileTensor types get clean names. LayoutTensor remnants get `LT` suffix.

## Key Insight: New vs Legacy Layout

**Mojo has TWO Layout types -- use the NEW one for TileTensor operations.**

| Module | Import As | Type Params | Usage |
|--------|-----------|-------------|-------|
| `layout._layout` | `Layout` | `shape_types`, `stride_types` | New API: compile-time type info, TileTensor |
| `layout.layout` | `LegacyLayout` | None (IntTuple) | Legacy: LayoutTensor, `enqueue_function` boundary |

The **new Layout** (from `layout._layout`) preserves compile-time type
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

## Pattern 5: Kernel TileTensor Parameters (NEW)

**Move `lt_to_tt` from GPU kernel to CPU-side `enqueue_function`.**

The kernel `run()` accepts TileTensor directly. The conversion from
LayoutTensor happens once at the enqueue boundary on the host.

```mojo
# Host side (blockwise_fp8_1d2d_matmul.mojo)
ctx.enqueue_function[kernel, kernel](
    KernelType.ATmaTile(a_tma_op),    # TMATile wrapper
    lt_to_tt(b_scales_2d),             # LayoutTensor → TileTensor
    lt_to_tt_1d(a_offsets),            # 1D conversion
    ...
)

# Kernel side (run() accepts TileTensor natively)
fn run(
    a_tma_op: Self.ATmaTile,
    b_scales: Self.BScalesTile,
    a_offsets: Self.WorkIterator.OffsetsTile,
    ...
):
    # No lt_to_tt needed -- already TileTensor
```

## Pattern 6: Static Row-Major Layouts (NEW)

**Build new Layout types from integer dimensions with provable rank.**

```mojo
# In tile_types.mojo
comptime static_row_major[dim0: Int, dim1: Int] = Layout[
    Variadic.types[T=CoordLike, ComptimeInt[dim0], ComptimeInt[dim1]],
    Variadic.types[T=CoordLike, ComptimeInt[dim1], ComptimeInt[1]],
]
```

Use for TMA layouts computed from config. Rank is provably 2 at compile time.

```mojo
# Kernel struct computes TMA layouts from config (no legacy Layout params)
comptime ATileLayout = static_row_major[Self.BM, Self.BK]
comptime ADescLayout = static_row_major[Self.BM, swizzle_elems]
comptime ATmaTile = TMATile[Self.a_type, Self.ATileLayout, Self.ADescLayout]
```

## Pattern 7: GMEMLayout1D for 1D Flat Arrays (NEW)

**1D layout with dynamic shape and stride 1, rank provably 1.**

```mojo
# In tile_types.mojo
comptime GMEMLayout1D = Layout[
    Variadic.types[T=CoordLike, RuntimeInt[DType.int64]],
    Variadic.types[T=CoordLike, ComptimeInt[1]],
]
```

Used for kernel parameters that are 1D lookup tables (offsets, expert IDs,
expert scales). The rank=1 is compile-time known, so `tensor[idx]` works.

**Key lesson**: `_int_to_dim()` is NOT comptime-evaluable (runtime branching).
Don't use it in comptime type aliases. Build Layout types directly with
`ComptimeInt` and `RuntimeInt`.

## Pattern 8: TMA Types as Single Source of Truth

**Kernel computes new Layout types from config. `TmaOpType` derives
`TMATensorTile` types. Host passes these layouts to `create_tensor_tile`
so types match by construction. No `rebind` anywhere.**

```mojo
# tile_types.mojo -- single derivation point
comptime TmaOpType[dtype, tile_layout: TensorLayout, desc_layout: TensorLayout]
    = TMATensorTile[dtype, _to_legacy_layout_2d[tile_layout](), ...]

# Kernel struct -- computes layouts from config
comptime ATileLayout = static_row_major[BM // CLUSTER_N, BK]
comptime ADescLayout = static_row_major[BM // CLUSTER_N, swizzle_elems]
comptime ATmaOp = TmaOpType[a_type, ATileLayout, ADescLayout]

fn run(a_tma_op: Self.ATmaOp, ...):
    var a_loader = TileLoader[...](Pointer(to=a_tma_op), mask)

# Host side -- pass kernel's layouts to create_tensor_tile
var a_tma_op = create_tensor_tile[
    tile_shape,
    __tile_layout = Kernel.ATmaOp.layout,
    __desc_layout = Kernel.ATmaOp.desc_layout,
](ctx, tensor)
# Types match by construction -- no rebind needed
ctx.enqueue_function[kernel](a_tma_op, ...)
```

**Key principles**:

- ONE source of truth: new Layout types in the kernel struct
- `TmaOpType` derives legacy layouts (the only place
  `_to_legacy_layout_2d` is called)
- Host passes `Kernel.XTmaOp.layout` to `create_tensor_tile`
  explicitly, ensuring type identity
- Never use `rebind` to solve type mismatches -- fix the types

## Pattern 9: TensorLayout as Struct Parameter (NEW)

**Pass new Layout type directly as a struct parameter. No bridge needed inside
the kernel.**

```mojo
# Kernel struct (new TensorLayout param, not LegacyLayout)
struct Kernel[
    b_scales_layout: TensorLayout,  # new Layout type
    ...
]:
    # TileTensor type directly -- no GMEMTile bridge
    comptime BScalesTile = TileTensor[
        Self.b_scales_type, Self.b_scales_layout, MutAnyOrigin
    ]

    # Compile-time constants via TensorLayout trait
    comptime b_scales_n = Self.b_scales_layout.static_shape[0]
    comptime b_scales_k = Self.b_scales_layout.static_shape[1]
```

At the host side, use `GMEMTile` to extract the new Layout type from a
legacy Layout:

```mojo
# Host side -- bridge from legacy to new at kernel instantiation
comptime BScalesTileType = GMEMTile[b_scales_type, b_scales_2d.layout]
comptime KernelType = Kernel[
    BScalesTileType.LayoutType,  # new Layout type
    ...
]
```

**Key insight**: `GMEMTile` is only needed on the host side to compute the
new Layout type from legacy Layout. Inside the kernel, `TensorLayout` works
directly -- `static_shape[i]`, `static_stride[i]` are accessible, and
`TileTensor[dtype, layout, origin]` accepts `TensorLayout` natively.
`enqueue_function` type matching works correctly.

## Pattern 10: tile/tile_with_offset with stride_layout (NEW)

**Override dynamic strides with explicit static strides for tiles.**

When a TileTensor has dynamic strides (e.g., from a `TensorLayout` trait
param), `tile()` inherits those dynamic stride *types*. This causes
`vectorize` and `distribute_with_offset` to fail their `all_dims_known`
constraint, even when the stride values are known at compile time.

The `stride_layout` named parameter provides explicit static strides:

```mojo
# Problem: c has dynamic strides (TensorLayout trait erases concrete types)
# tile() inherits RuntimeInt strides → vectorize fails
var ctile = c.tile[BM, BN](coord(...))
ctile.vectorize[1, 2]()  # ERROR: all_dims_known is false

# Solution: stride_layout provides a Layout whose shape IS the desired strides
comptime CGmemStrideLayout = Layout[
    Variadic.types[T=CoordLike, ComptimeInt[N], ComptimeInt[1]],
    Variadic.types[T=CoordLike, ComptimeInt[1], ComptimeInt[1]],
]

var ctile, coords, _ = c.tile_with_offset[
    BM, BN, stride_layout = CGmemStrideLayout
](coord(...))
# Now ctile has ComptimeInt strides → vectorize/distribute work!
ctile.vectorize[1, 2]().distribute_with_offset[row_major[8, 4]()](lane_id())
```

**Note**: The stride_layout's **shape types** are used as the tile's strides
(not the stride_layout's stride types). This is a Layout type that encodes
the desired strides in its shape.

**Root cause**: `TensorLayout` is a trait that erases concrete `Layout[...]`
type information. LayoutTensor avoided this because its `Layout` (IntTuple-based)
is always a concrete comptime value with strides embedded in the type.

## Pattern 11: Strided sub-tiles with _StridedLayout (NEW)

**Use explicit strides when sub-tiling preserves the parent's row stride.**

When creating a sub-tile that is narrower than its parent row, the
stride must be the parent's row stride, NOT a row-major stride of the
narrower shape. `static_row_major[height, width]` gives stride
`[width, 1]` which is wrong for strided views.

```mojo
# WRONG: row_major strides for the narrow sub-tile
# stride = [stage_width, 1] -- loses parent's row spacing
comptime SubLayout = static_row_major[16, stage_width]

# CORRECT: preserve parent's row stride
# stride = [MMA_N, 1] -- elements are MMA_N apart in memory
comptime SubLayout = _StridedLayout[16, stage_width, MMA_N]
```

This pattern appears in split-K reduction where workspace tiles
(BM x MMA_N) are decomposed into staged sub-tiles of width
`stage_width < MMA_N`. Each sub-tile's rows are still `MMA_N`
elements apart in memory.

```mojo
# tile_types.mojo
comptime _StridedLayout[dim0: Int, dim1: Int, stride0: Int] = Layout[
    Variadic.types[T=CoordLike, ComptimeInt[dim0], ComptimeInt[dim1]],
    Variadic.types[T=CoordLike, ComptimeInt[stride0], ComptimeInt[1]],
]

# Usage in _to_next_subtile:
comptime SubLayout = _StridedLayout[
    parent.static_shape[0],   # height preserved
    widths[curr_stage],       # width narrowed
    parent.static_stride[0],  # stride preserved from parent
]
```

**Rule of thumb**: If a sub-tile is a "window" into a larger row,
use `_StridedLayout`. If it's a standalone contiguous block, use
`static_row_major`.

## Architecture Summary (Updated)

```text
tile_types.mojo
├── Layouts: internal_k_major, static_row_major, GMEMLayout1D
├── TMA types: TmaOpType, TmaOpTypeIm2col (derive TMATensorTile from new Layout)
├── Desc layouts: tma_desc_layout_3d, tma_desc_layout_4d, tma_desc_layout_5d
├── Tiles: SMemTile, SMemTile2D, GMEMTile
├── Arrays: SMemTileArray2D, SMemTileArray2DRowMajor, SMemTileArrayWithLayout
├── Conversion: lt_to_tt, lt_to_tt_1d (host-side, at enqueue boundary)
└── Helpers: _to_legacy_layout, _int_to_dim

_tile_tensor.mojo
├── tile / tile_with_offset (standard + stride_layout overloads)
├── distribute / distribute_with_offset
├── vectorize
└── reshape, coalesce, load, store, dim, __getitem__, __setitem__

tile_loader.mojo
├── TileLoader (new Layout types, for A/B loads)
├── ScalesLoader (new Layout types, for scale factor loads)
├── TileLoaderTMA (legacy, still used by some code paths)
└── TileLoaderTMAIm2col (legacy, for conv2d)

Host (*_matmul.mojo, conv2d.mojo)
├── Kernel type instantiated FIRST (computes TMA types from config)
├── create_tensor_tile with Kernel.XTmaOp.layout (types match)
├── lt_to_tt / lt_to_tt_1d → TileTensor at enqueue boundary
└── enqueue_function passes TMATensorTile + TileTensor directly

Kernel (*_matmul_kernel.mojo, conv2d_fprop_kernel.mojo)
├── Zero legacy Layout struct params (all derived from config)
├── TMA types: TmaOpType / TmaOpTypeIm2col (single source of truth)
├── run() params: Self.ATmaOp, TileTensor
├── Output: tile_with_offset[stride_layout=...] + vectorize + distribute
└── LayoutTensor only in SMEM epilogue path (encapsulated)
```

## Gotchas

### `.tile[]` requires Coord, not positional arguments

```mojo
# LayoutTensor (old):
tile.tile[d0, d1](i, j)

# TileTensor (new) -- pass a tuple that converts to Coord:
tile.tile[d0, d1]((i, j))

# Multi-dimensional:
tile.tile[d0, d1, d2, d3]((i0, i1, i2, i3))
```

If using range iterators, convert explicitly: `(Int(i), 0)`.

### Compile-time layout access: use type params, not `.layout`

LayoutTensor's `.layout` is a comptime property. TileTensor's `.layout` is
a runtime value. For compile-time shape/stride access, use type parameters:

```mojo
# LayoutTensor (old):
comptime stride0 = dst.layout.stride[0].value()

# TileTensor (new):
comptime stride0 = type_of(dst).static_stride[0]
# Or pass as explicit parameters to the function
```

### Peer CTA tile construction requires layout parameter

LayoutTensor can be constructed from just a pointer. TileTensor requires
both pointer and layout:

```mojo
# LayoutTensor (old):
var peer_tile = type_of(tile)(tile.ptr + offset)

# TileTensor (new):
var peer_tile = type_of(tile)(tile.ptr + offset, tile.layout)
```

### Zero-cost pointer reinterpretation between LT and TT arrays

Both LayoutTensor-based `SMemTileArray` and TileTensor-based
`SMemTileArray2DRowMajor` use `InlineArray[Scalar[dtype], N]` for storage
and `UnsafePointer[Scalar[dtype], address_space=SHARED]` for access. They
share the exact same memory layout. You can take a pointer from one and
construct the other -- zero copy, zero cost:

```mojo
# Bridge from TileTensor array to LayoutTensor array
var lt_array = LTSMemTileArray(tt_array.ptr)
```

### Dimensions-as-parameters instead of legacy Layout

Instead of passing a legacy Layout as a struct parameter, pass integer
dimensions and compute the new Layout inside:

```mojo
# Old (legacy Layout as parameter):
struct MyStruct[c_smem_layout: LegacyLayout]:
    ...

# New (dimensions, compute layout inside):
struct MyStruct[c_smem_dim0: Int, c_smem_dim1: Int]:
    comptime c_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()
    comptime CTile = SMemTile[Self.c_type, Self.c_layout]
```

This pattern is already used by `TMAStoreExecutor`, `TMEMToSMemWriter`, and
`SMemEpilogueWriter` in `epilogue_components.mojo`.

## Pattern 12: ViewType for Reshape Return Types (NEW)

**Name the return type of `reshape()` so helper functions can
declare proper return types.**

LayoutTensor defines named type aliases like `TileType[*tile_sizes]`
so both internal methods and external code can name the return type of
complex operations. TileTensor now has `ViewType[new_layout]` which
preserves dtype, origin, address_space, etc. while replacing LayoutType.

```mojo
# On TileTensor struct (in _tile_tensor.mojo):
comptime ViewType[new_layout: TensorLayout] = TileTensor[
    dtype = Self.dtype,
    LayoutType = new_layout,
    origin = Self.origin,
    address_space = Self.address_space,
    linear_idx_type = Self.linear_idx_type,
    element_shape_types = Self.element_shape_types,
]
```

**Key use case**: Helper functions that reshape TileTensors can now
properly type their return values:

```mojo
# Compute the batched layout type from the input
comptime _Batched3DLayout[L: TensorLayout] = RowMajorLayout[
    ComptimeInt[1], L._shape_types[0], L._shape_types[1]
]

# Helper with a properly typed return via ViewType
fn _to_batched_3d(
    tensor: TileTensor[...],
) -> tensor.ViewType[_Batched3DLayout[type_of(tensor).LayoutType]]:
    return tensor.reshape(row_major(Coord(
        Idx[1](),
        tensor.layout.shape[0](),  # preserves runtime values
        tensor.layout.shape[1](),
    )))
```

**Why `L._shape_types[i]` instead of `ComptimeInt[L.static_shape[i]]`**:
Dimensions may be dynamic (RuntimeInt). Using `_shape_types[i]` preserves
the static/dynamic nature of each dimension. Using `ComptimeInt[...]`
forces all dims to be static, which fails for dynamic batch or M dims.

## Pattern 13: TileTensor Overload of create_tma_tile (NEW)

**Call `create_tensor_tile` directly with TileTensor, bypassing
LayoutTensor entirely.**

`create_tensor_tile` in `tma_async.mojo` already has a TileTensor
overload that reads `tensor.ptr`, `tensor.layout.shape[i]().value()`,
and `tensor.layout.stride[i]().value()`. These work on any TileTensor,
including reshaped views.

```mojo
# In tile_types.mojo -- TileTensor overload alongside the LayoutTensor one
def create_tma_tile[
    rank: Int, //,
    tma_tile_layout: TensorLayout,
    tma_desc_layout: TensorLayout,
    tile_shape: IndexList[rank], *,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: TileTensor[...]) -> TmaOpType[
    tensor.dtype, tma_tile_layout, tma_desc_layout
]:
    return create_tensor_tile[
        tile_shape, swizzle_mode=swizzle_mode,
        __tile_layout = _to_legacy_layout[tma_tile_layout](),
        __desc_layout = _to_legacy_layout[tma_desc_layout](),
    ](ctx, tensor)
```

After this, `create_tma_tile(ctx, any_tiletensor)` just works.
No `to_layout_tensor()` bridge needed.

## Pattern 14: Module-Level Launch Function for TMA Scoping (NEW)

**TMA descriptors are scoped references that can't cross `@parameter if`
boundaries. Put TMA creation + kernel launch in one scope.**

When batched vs non-batched code paths produce different TileTensor types
(e.g., 2D reshaped to 3D vs 3D pass-through), we can't pre-declare TMA
ops outside the branch because:

- TMA descriptors can't be copied (scoped by address space)
- Pre-declaring with explicit types fails (symbolic type identity)
- Local closures can't have parameters in Mojo

Solution: extract a module-level function that takes already-reshaped
TileTensors and does TMA creation + kernel launch in one scope:

```mojo
# Module-level function -- no scoping issues
fn _create_tma_and_launch[
    config: BlockScaledMatmulConfig[...], *,
    ...  # other comptime params
](
    a_3d: TileTensor[...],  # already 3D
    b_3d: TileTensor[...],
    c_3d: TileTensor[...],
    sfa_5d: TileTensor[...],  # already 5D
    sfb_5d: TileTensor[...],
    ctx: DeviceContext,
    alpha: Float32,
) raises:
    # TMA creation and kernel launch in same scope
    var a_tma_op = create_tma_tile[...](ctx, a_3d)
    ...
    ctx.enqueue_function[kernel, kernel](a_tma_op, ...)

# Caller dispatches with @parameter if
@parameter
if is_batched_matmul:
    _create_tma_and_launch[config=config, ...](
        a_tensor, b_tensor, c_tensor,  # already 3D
        _to_scales_5d_batched(a_scales_tensor),
        _to_scales_5d_batched(b_scales_tensor),
        ctx, alpha,
    )
else:
    _create_tma_and_launch[config=config, ...](
        _to_batched_3d(a_tensor),  # 2D -> 3D
        _to_batched_3d(b_tensor),
        _to_batched_3d(c_tensor),
        _to_scales_5d_non_batched(a_scales_tensor),
        _to_scales_5d_non_batched(b_scales_tensor),
        ctx, alpha,
    )
```

## Pattern 15: Layout Params from Caller's TileTensor (NEW)

**Take layout as a parameter to the kernel from the caller's TileTensor.
Types match by construction -- no fragile manual construction needed.**

Problem: `enqueue_function` requires exact type identity between
arguments and kernel `run()` parameter types. If the kernel computes
layouts independently (e.g., `RowMajorLayout[RuntimeInt, ComptimeInt[N]]`),
the symbolic type expression may differ from the caller's TileTensor
layout type even when they encode the same shape. This causes runtime
type validation failures that are hard to reproduce in unit tests.

Solution: The kernel struct takes layout as a `TensorLayout` parameter,
derived from the caller's TileTensor via `type_of(tt).LayoutType`.
The kernel's tile types are derived from the **same** layout expression
as the TileTensor being passed, so types match by construction:

```mojo
# Kernel struct -- takes layout from caller
struct Grouped1D1DMatmulKernel[
    ...,
    c_device_layout: TensorLayout,  # from caller's TileTensor
    ...
]:
    comptime CDeviceTile = TileTensor[
        Self.c_type, Self.c_device_layout, MutAnyOrigin
    ]

    fn run(c_device: Self.CDeviceTile, ...): ...
```

Host passes `type_of(tiletensor).LayoutType` -- types match by
construction, no `unsafe_from_address` or manual TileTensor construction:

```mojo
# Host -- types match by construction
var c_tt = lt_to_tt(c_device)
comptime KernelType = Grouped1D1DMatmulKernel[
    ...,
    c_device_layout=type_of(c_tt).LayoutType,
    ...
]
ctx.enqueue_function[KernelType.run, KernelType.run](
    ..., c_tt, ...  # exact type match -- derived from same layout
)
```

**Why NOT compute layouts internally**: An earlier approach (the kernel
computing `CDeviceLayout = RowMajorLayout[RuntimeInt, ComptimeInt[N]]`
from config) was tried and reverted. It required the host to construct
TileTensors via `KernelType.CDeviceTile(ptr, layout)` with
`unsafe_from_address` -- fragile and caused the DeepSeek-R1-NVFP4
pipeline failure (#77347/#77359) when the independently-computed layout
type differed symbolically from the caller's layout type.

**Rule of thumb**: If a TileTensor flows from the host to
`enqueue_function`, take its layout as a kernel struct parameter.
If a TileTensor is created entirely within the kernel (e.g., SMEM
tiles), compute the layout internally.

## Future Work

1. ~~**Remove all legacy Layout struct params from all kernels**~~ ✅ Done.
2. ~~**Migrate fallback kernel C output to TileTensor**~~ ✅ Done.
3. ~~**Migrate `RegTile`/`BlockwiseFP8Accumulator`**~~ ✅ Done.
4. ~~**Migrate split-K reduction tensor**~~ ✅ Done.
5. ~~**Migrate block_scaled host signatures to TileTensor**~~ ✅ Done.
6. ~~**Migrate default matmul + blockwise_fp8 host signatures**~~ ✅ Done.
7. ~~**Migrate blockwise_fp8_1d2d + grouped_1d1d host signatures**~~ ✅ Done.
8. ~~**Migrate grouped_block_scaled host signatures**~~ ✅ Done (8/8 kernels).
9. **Migrate `TMATensorTile` itself** to new Layout
   (eliminates `_to_legacy_layout` bridge)
10. **Layout algebra for TileTensor** (`zipped_divide`, `upcast`,
    `blocked_product`)
11. **Migrate conv2d** -- im2col TMA is a separate concern.
12. **Extend `lt_to_tt` to arbitrary ranks** (currently only 2D)
13. **Compiler bug**: `TileTensor(NDBuffer)` crashes when NDBuffer
    DimList contains complex expressions like `ceildiv(...)`.
    Workaround: create NDBuffer with `DimList.create_unknown[rank]()`.

## Gotchas (continued)

### Never use `rebind` to solve type mismatches

If two expressions compute the same layout through different paths,
the compiler treats them as different types. `rebind` hides this
mismatch and can crash at runtime. Instead:

- Use ONE type constructor for each layout, referenced everywhere
- Pass `Kernel.XTmaOp.layout` to `create_tensor_tile` so the
  returned `TMATensorTile` has the exact same type expression
- `TileLoader.__init__` accepts `AnyType` pointer and rebinds
  internally (this is the only acceptable rebind -- it's inside
  the loader's own type boundary)

### Don't wrap TMATensorTile in a wrapper struct

A `TMATile` wrapper that delegates TMA methods was tried and failed:
it crashes when loaders infer layout from the wrapper's type params.
Use `TmaOpType` to derive `TMATensorTile` directly instead.

### `enqueue_function` requires fully resolved function signatures

`enqueue_function[func]` needs `func` to be a non-parametric function.
You cannot add function-level type parameters to `run()` (like
`fn run[a_layout: Layout, ...](...)`). The layouts must be resolved at
the struct level or as comptime aliases derived from config.

### Sub-tiles must preserve parent strides, not recompute them

When creating a narrower view of a tile (e.g., splitting a row into
stages), the sub-tile's stride must match the parent's row stride.
Using `static_row_major[height, narrow_width]` gives stride
`[narrow_width, 1]` which is WRONG -- elements in adjacent rows
are actually `parent_row_width` apart, not `narrow_width` apart.

Use `_StridedLayout[height, narrow_width, parent_row_stride]` or
construct an explicit Layout with the correct strides. This bug
caused numerical errors in the split-K reduction path where staged
sub-tiles had incorrect memory access patterns.
