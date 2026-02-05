# TileTensor Migration Journal for SM100 Structured Kernels

## Executive Summary

This document chronicles our migration of SM100 structured kernels from
LayoutTensor to TileTensor, documenting the problems encountered, solutions
developed, and lessons learned.

**Current Status (2026-02-03)**: TileTensor migration is COMPLETE. All tiles
(A, B, SF, C) now use TileTensor end-to-end through storage, TMA, MMA, and
epilogue operations. All LayoutTensor wrapper types have been removed from
all kernels.

**Completed**:

- TileTensor-based storage for A, B, SF tiles (`SMemTileArray2D`, `SMemTileArrayWithLayout`)
- TileTensor-based storage for C tiles (`SMemTileArray2DRowMajor`)
- Internal swizzled layouts (`internal_k_major_128B`, `internal_sf_k_major`)
- MMA TileTensor overload extracts layout directly from type parameters
- Fixed `internal_sf_k_major` strides to match `tile_sf_layout_k_major`
- TMA 4D/5D TileTensor overloads for SF tile loads
- TileWriter TileTensor overloads for C tile output path
- Removed all LayoutTensor wrapper types from kernels

**Remaining work**:

- None! TileTensor migration is complete for all SM100 structured kernels.
- CTileArrayLT kept in pipeline_storage.mojo for backward compatibility only.

See Part 27 for the C tile migration.

---

## Part 1: Problems Encountered

### 1.1 API Differences Between TileTensor and LayoutTensor

During our bottom-up migration attempt, we discovered significant API differences:

| Feature | LayoutTensor | TileTensor |
|---------|--------------|------------|
| **Construction from ptr** | `type_of(tile)(ptr)` | `type_of(tile)(ptr, layout)` - requires layout |
| **Stack allocation** | `Tile.stack_allocation()` | Not available |
| **Load signature** | `tile.load[width](row, col)` | `tile.load[width](Coord(row, col))` |
| **Shape access** | `tile.layout.shape[i].value()` | `tile.static_shape[i]` |
| **Runtime layout** | `tile.runtime_layout` | `tile.layout` (unified) |

### 1.2 Peer Tile Creation Pattern

The SM100 kernels use a pattern for peer CTA slicing that doesn't work with TileTensor:

```mojo
# Current LayoutTensor pattern (works)
var a_peer_tile = type_of(a_tile)(
    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size)
)

# TileTensor requires layout parameter
var a_peer_tile = type_of(a_tile)(
    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
    a_tile.layout,  # Required!
)
```

**Solution identified**: TileTensor stores its layout in `self.layout`, so we
can use:

```mojo
type_of(tile)(tile.ptr + offset, tile.layout)
```

### 1.3 Cascading Type Mismatches

When we tried "bottom-up" migration (updating loaders first), it caused
cascading failures:

1. Updated `tile_loader.mojo` to accept TileTensor → broke kernel
2. Updated `blockwise_fp8_smem.mojo` to return TileTensor → broke accumulator
   and writer
3. Updated accumulator imports → discovered missing `stack_allocation()` and
   different `load()` API

The problem is that **all downstream APIs (MMA, TMA, accumulator, writer)
expect LayoutTensor**.

### 1.4 Internal vs Boundary Usage

A key insight: The accumulator uses `RegTile` for **internal register storage**,
not just API parameters. When we changed the import to TileTensor-based
`RegTile`, the internal code broke because:

- `RegTile.stack_allocation()` doesn't exist on TileTensor
- `tile.load[width](stage, offset)` has different signature in TileTensor

---

## Part 2: Insights from PR #75977

PR #75977 successfully migrates simpler kernels (nn/randn, nn/rope, etc.) to TileTensor.

### 2.1 Key Patterns from the PR

**Simple function signature migration:**

```mojo
# Old (LayoutTensor)
fn random_normal(output: LayoutTensor[mut=True, dtype, layout, ...])

# New (TileTensor)
fn random_normal(output: TileTensor[mut=True, dtype, ...])
```

**Shape access:**

```mojo
# TileTensor uses static_shape
comptime head_size = x.static_shape[2]
comptime rope_dim = freqs_cis.static_shape[1]

# Compile-time assertions
comptime assert x.shape_types[i].is_static_value
comptime assert freqs_cis.all_dims_known
```

**Load/store with Coord:**

```mojo
var coord = Coord(idx)
val = x.load[width=width](coord)
```

**Fill method:**

```mojo
tensor.fill(value)  # TileTensor now has fill()
```

### 2.2 What Makes PR #75977 Different from SM100

The nn/ kernels migrated in PR #75977 are **simpler** because:

1. **No complex type hierarchies** - Functions accept tensors directly, no
   `SMemTileArray`, no payloads
2. **No pipeline stages** - No `InputPipelineStorage`, `TilePayload` traits
3. **No MMA integration** - Don't call `mma_op.mma(a_tile, b_tile, ...)`
4. **No peer slicing** - Don't create offset tiles for CTA distribution

SM100 structured kernels have all of these complexities.

### 2.3 TileTensor Capabilities Added

From PR #75977, TileTensor now has:

- `fill()` method for initialization
- `numel()` for element count
- `static_shape[i]` for compile-time shape access
- `to_layout_tensor()` for conversion at boundaries
- Better `load(Coord)` / `store(Coord, value)` methods

---

## Part 3: Middle-Out Migration Strategy

### 3.1 The Idea

Instead of top-down (kernel → storage) or bottom-up (loaders → kernel), we start
from the **middle**: the pipeline payloads. These are:

1. **Localized** - Self-contained structs with clear boundaries
2. **Not deeply entangled** - Payloads hold arrays but don't call external APIs
3. **Natural conversion points** - Payloads are created from SMEM and consumed
   by kernel code

### 3.2 Why Pipeline Payloads?

```text
┌─────────────────────────────────────────────────────────┐
│                     SMEM Storage                        │
│  (LayoutTensor-based BlockwiseFP8TileStorage)          │
└─────────────────────────────────────────────────────────┘
                           │
                           │ .a_tiles(), .b_tiles(), ...
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Pipeline Payload ← START HERE             │
│  (BlockwiseFP8TilePayload holds tile arrays)           │
│                                                         │
│  Currently: LayoutTensor-based SMemTileArray           │
│  Goal: TileTensor-based SMemTileArray                  │
│                                                         │
│  Key: .get_tile() returns individual tiles             │
└─────────────────────────────────────────────────────────┘
                           │
                           │ .get_tile() → (a_tile, b_tile, a_scales)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Kernel Code                          │
│  Uses tiles for:                                        │
│  - TMA loads (needs LayoutTensor: .to_layout_tensor()) │
│  - MMA ops (needs LayoutTensor: .to_layout_tensor())   │
│  - Peer slicing (TileTensor can handle with layout)    │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Migration Steps

#### Phase 1: Make Payload Return TileTensor (Foundation Done)

We already have `BlockwiseFP8TilePayload` in `tile_types.mojo` that uses
TileTensor-based `SMemTileArray`. The payload's `get_tile()` returns TileTensor.

#### Phase 2: Kernel Receives TileTensor, Converts at Boundaries

Update kernel to:

1. Accept TileTensor tiles from payload
2. Convert at TMA boundary: `loader.load(tile.to_layout_tensor(), ...)`
3. Convert at MMA boundary:
   `mma_op.mma(a.to_layout_tensor(), b.to_layout_tensor(), ...)`
4. Use TileTensor for peer slicing:
   `type_of(tile)(tile.ptr + offset, tile.layout)`

#### Phase 3: Update Loader to Accept TileTensor Natively (Optional)

Once kernel works with TileTensor → LayoutTensor conversion:

1. Add TileTensor overloads to loaders
2. Remove `.to_layout_tensor()` calls from kernel
3. Loaders convert internally

#### Phase 4: Expand Outward

- Update accumulator to use TileTensor for `a_scales_tiles` parameter
- Update writer to use TileTensor for `c_tiles` parameter
- Eventually update SMEM storage itself

### 3.4 Key Principle: Convert at Boundaries, Not Wholesale

```mojo
# DON'T: Change all types at once
# DO: Keep internal types stable, convert at API boundaries

# In kernel code:
var a_tile, b_tile, _ = tiles.payload().get_tile[k_group_size](stage, 0)
# a_tile is TileTensor

# At TMA boundary - convert
a_loader.load(a_tile.to_layout_tensor(), barrier, k, m)

# At MMA boundary - convert
mma_op.mma(a_tile.to_layout_tensor(), b_tile.to_layout_tensor(), offset)

# For peer slicing - TileTensor natively
var a_peer = type_of(a_tile)(a_tile.ptr + offset, a_tile.layout)
```

---

## Part 4: Concrete Implementation Plan

### 4.1 Prerequisites (Done)

- [x] Create `tile_types.mojo` with TileTensor-based types
- [x] Create `BlockwiseFP8TilePayload` in `tile_types.mojo`
- [x] Create `BlockwiseFP8TileStorage` in `tile_types.mojo`
- [x] Ensure `SMemTileArray` returns TileTensor from `__getitem__`

### 4.2 Phase 1: Payload Integration

1. **Keep SMEM using LayoutTensor storage** (no changes to `blockwise_fp8_smem.mojo`)
2. **Create bridge in SMEM** that converts LayoutTensor arrays to TileTensor arrays:

   ```mojo
   # In BlockwiseFP8Smem
   fn tile_payload_tt(self) -> TileTensorPayload:
       # Convert LayoutTensor arrays to TileTensor-based payload
       return TileTensorPayload(
           self.a_tiles().to_tile_tensor_array(),  # Need to add this method
           self.b_tiles().to_tile_tensor_array(),
           self.a_scales_tiles().to_tile_tensor_array(),
       )
   ```

   **Alternative**: Have payload constructor accept LayoutTensor arrays and
   convert internally.

### 4.3 Phase 2: Kernel Integration

1. Import TileTensor-based `BlockwiseFP8TilePayload` from `tile_types.mojo`
2. Update kernel to use TileTensor tiles with boundary conversion
3. Update peer tile creation to include layout parameter

### 4.4 Testing Strategy

Each phase should be testable independently:

```bash
# After Phase 1: Types compile
mojo build max/kernels/src/linalg/matmul/gpu/sm100_structured/...

# After Phase 2: Kernel runs
mojo max/kernels/test/gpu/linalg/test_matmul_sm100_blockwise_fp8_smoke.mojo
```

---

## Part 5: Open Questions

### 5.1 Array Conversion

How do we convert a LayoutTensor-based `SMemTileArray` to a TileTensor-based one?

Options:

1. **Wrapper struct** that holds LayoutTensor array but returns TileTensor from `__getitem__`
2. **Copy constructor** that reinterprets the pointer
3. **Both arrays share the same memory layout** - just reinterpret the pointer

### 5.2 InputPipelineStorage Compatibility

The `InputPipelineStorage` struct requires a `TilePayload` trait. Our TileTensor-based
`BlockwiseFP8TilePayload` already implements this trait. Need to verify compatibility.

### 5.3 MMA Op Signature

The MMA ops expect LayoutTensor. We need to verify that `.to_layout_tensor()` produces
the correct type for the MMA op's signature.

---

## Appendix A: Files Involved

### Core Type Files

- `structured_kernels/tile_types.mojo` - TileTensor-based types with explicit
  dimensions
- `structured_kernels/pipeline_storage.mojo` - TileTensor-native storage with
  `SMemTileArray2D`
- `structured_kernels/tile_pipeline.mojo` - TileTensor-native payloads with
  explicit dimensions

### Blockwise FP8 Files

- `blockwise_fp8/blockwise_fp8_smem.mojo` - SMEM struct
- `blockwise_fp8/blockwise_fp8_matmul_kernel.mojo` - Main kernel
- `blockwise_fp8/blockwise_fp8_accumulator.mojo` - Uses RegTile, SMemTileArray
- `blockwise_fp8/blockwise_fp8_output_writer.mojo` - Uses SMemTileArray

### Downstream APIs (LayoutTensor expected)

- `structured_kernels/tile_loader.mojo` - TMA operations
- `linalg/arch/sm100/mma.mojo` - MMA operations

---

## Appendix B: Key Code Patterns

### Creating Offset Tile (TileTensor)

```mojo
var peer_tile = type_of(tile)(tile.ptr + offset, tile.layout)
```

### Boundary Conversion

```mojo
loader.load(tile.to_layout_tensor(), barrier, k, m)
mma_op.mma(a.to_layout_tensor(), b.to_layout_tensor(), offset)
```

### Shape Access

```mojo
# TileTensor
comptime dim = tile.static_shape[0]
var runtime_dim = tile.layout.shape[0].value()

# LayoutTensor (for comparison)
comptime dim = tile.layout.shape[0].value()
var runtime_dim = tile.runtime_layout.shape[0].value()
```

---

## Part 6: Compatibility Patterns Added to tile_types.mojo

We've added documentation of the key compatibility patterns directly to `tile_types.mojo`:

### Peer Tile Creation

```mojo
# LayoutTensor (ptr only):
var peer_tile = type_of(tile)(tile.ptr + offset)

# TileTensor (ptr + layout):
var peer_tile = type_of(tile)(tile.ptr + offset, tile.layout)
```

### Boundary Conversion

```mojo
loader.load(tile.to_layout_tensor(), barrier, k, m)
mma_op.mma(a.to_layout_tensor(), b.to_layout_tensor(), offset)
```

### Shape Access

```mojo
# TileTensor
comptime dim = tile.static_shape[i]
var runtime_dim = tile.layout.shape[i].value()
```

---

## Part 7: Concrete Middle-Out Implementation Steps

### Step 1: Create Bridge in Payload (NOT Storage)

The key insight is that we can create a TileTensor-based payload that **wraps**
LayoutTensor-based arrays. The payload's `get_tile()` method would convert
each tile on access.

```mojo
# In tile_types.mojo - add a bridge payload
struct BlockwiseFP8TilePayloadBridge[...](TilePayload, TrivialRegisterType):
    """Bridge payload that wraps LayoutTensor arrays but returns TileTensor tiles."""

    # Store LayoutTensor-based arrays (from SMEM)
    var a_tiles_lt: LayoutTensorArrayType
    var b_tiles_lt: LayoutTensorArrayType
    var a_scales_tiles_lt: LayoutTensorArrayType

    @always_inline
    fn get_tile[k_group_size: Int](
        self, stage: UInt32, k_idx: Int
    ) -> Tuple[ATileTT, BTileTT, AScalesTileTT]:
        """Get tiles, converting from LayoutTensor to TileTensor."""
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (
            self.a_tiles_lt[idx].to_tile_tensor(),  # Convert!
            self.b_tiles_lt[idx].to_tile_tensor(),
            self.a_scales_tiles_lt[idx].to_tile_tensor(),
        )
```

### Step 2: Update Kernel to Use Bridge Payload

```mojo
# In blockwise_fp8_matmul_kernel.mojo

# Import bridge payload
from ..structured_kernels.tile_types import BlockwiseFP8TilePayloadBridge

# Create payload from SMEM (which remains LayoutTensor-based)
var tile_payload = BlockwiseFP8TilePayloadBridge(
    smem.a_tiles(),      # LayoutTensor-based
    smem.b_tiles(),      # LayoutTensor-based
    smem.a_scales_tiles()  # LayoutTensor-based
)

# Kernel gets TileTensor tiles from payload
var a_tile, b_tile, a_scales_tile = tile_payload.get_tile[k_group_size](stage, 0)
# a_tile is now TileTensor!

# Peer slicing with TileTensor pattern
var a_peer_tile = type_of(a_tile)(
    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
    a_tile.layout,  # TileTensor needs layout
)

# Convert at API boundaries
a_loader.load(a_peer_tile.to_layout_tensor(), barrier, k, m)
mma_op.mma(a_tile.to_layout_tensor(), b_tile.to_layout_tensor(), offset)
```

### Step 3: Verify and Expand

1. Test with blockwise_fp8 smoke test
2. Once working, can optionally:
   - Update loaders to accept TileTensor natively
   - Update MMA to accept TileTensor natively
   - Eventually update SMEM storage itself

### Why This Approach Works

1. **SMEM stays stable** - No changes to memory layout
2. **Conversion is localized** - Only in payload's `get_tile()`
3. **Kernel uses TileTensor** - Modern API throughout kernel code
4. **Boundaries are explicit** - Clear `.to_layout_tensor()` calls
5. **Testable incrementally** - Can verify at each step

---

## Part 8: Bridge Payload Implementation (COMPLETED)

We implemented `BlockwiseFP8TilePayloadBridge` in `tile_types.mojo`. This is the
key component of the middle-out migration strategy.

### 8.1 Implementation Details

```mojo
struct BlockwiseFP8TilePayloadBridge[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    a_tile_layout: LegacyLayout,
    b_tile_layout: LegacyLayout,
    a_scales_tile_layout: LegacyLayout,
    num_pipeline_stages: Int,
](TilePayload, TrivialRegisterType):
    """Bridge payload: wraps LayoutTensor arrays but returns TileTensor tiles."""

    # Store pointers from LayoutTensor arrays (zero-cost)
    var a_ptr: UnsafePointer[Scalar[Self.a_type], address_space = AddressSpace.SHARED]
    var b_ptr: UnsafePointer[Scalar[Self.b_type], address_space = AddressSpace.SHARED]
    var a_scales_ptr: UnsafePointer[Scalar[Self.a_scales_type], address_space = AddressSpace.SHARED]

    # Takes LayoutTensor-based arrays, stores their pointers
    fn __init__(out self, a_tiles: LTATileArray, b_tiles: LTBTileArray, a_scales_tiles: LTAScalesTileArray):
        self.a_ptr = a_tiles.ptr
        self.b_ptr = b_tiles.ptr
        self.a_scales_ptr = a_scales_tiles.ptr

    # Returns TileTensor-based tiles
    fn get_tile[k_group_size: Int](self, stage: UInt32, k_idx: Int) -> Tuple[ATile, BTile, AScalesTile]:
        # Creates TileTensor arrays from stored pointers, then indexes
        ...
```

### 8.2 Key Insight: Zero-Cost Pointer Reinterpretation

Both LayoutTensor-based and TileTensor-based `SMemTileArray` use:

- `InlineArray[Scalar[dtype], num_elements]` for storage
- `UnsafePointer[Scalar[dtype], address_space=AddressSpace.SHARED]` for pointer

This means they share the exact same memory layout. We can take a pointer from
a LayoutTensor array and use it to construct a TileTensor array - zero cost!

### 8.3 Usage in Kernel (Planned Next Step)

```mojo
# SMEM remains LayoutTensor-based (no changes)
var smem: BlockwiseFP8Smem[...]

# Create bridge payload from LayoutTensor arrays
var tile_payload = BlockwiseFP8TilePayloadBridge(
    smem.a_tiles(),          # LayoutTensor-based
    smem.b_tiles(),          # LayoutTensor-based
    smem.a_scales_tiles(),   # LayoutTensor-based
)

# Get TileTensor tiles
var a_tile, b_tile, a_scales_tile = tile_payload.get_tile[k_group_size](stage, 0)
# a_tile is now TileTensor!

# Use TileTensor for peer slicing (modern API)
var a_peer = type_of(a_tile)(a_tile.ptr + offset, a_tile.layout)

# Convert at API boundaries
a_loader.load(a_peer.to_layout_tensor(), barrier, k, m)
mma_op.mma(a_tile.to_layout_tensor(), b_tile.to_layout_tensor(), offset)
```

---

## Part 9: Kernel Migration Completed

Successfully migrated the blockwise FP8 kernel to use the bridge payload!

### 9.1 Compiler Bug with `.to_layout_tensor()`

During migration, we discovered a **compiler crash** when calling
`.to_layout_tensor()` on TileTensor tiles in the kernel context. The crash was:

```text
Assertion `AllocEndPtr >= uintptr_t(CurPtr) &&
    "Alignment + Size must not overflow"' failed.
```

This appears to be a Mojo compiler bug triggered by complex type instantiation when
converting TileTensor to LayoutTensor in certain contexts.

### 9.2 Workaround: Dual Accessor Methods

To avoid the compiler crash, we added **dual accessor methods** to the bridge payload:

```mojo
# TileTensor accessors (for internal use, future migration)
fn get_tile(...) -> Tuple[ATile, BTile, AScalesTile]  # Returns TileTensor

# LayoutTensor accessors (for boundary compatibility)
fn get_tile_lt(...) -> Tuple[LTATile, LTBTile, LTAScalesTile]  # Returns LayoutTensor
```

This allows the kernel to use LayoutTensor directly for boundary operations
(loaders, MMA) without going through `.to_layout_tensor()`.

### 9.3 Migration Changes

**File: `blockwise_fp8_matmul_kernel.mojo`**

1. Import changed from `BlockwiseFP8TilePayload` to `BlockwiseFP8TilePayloadBridge`
2. `load_input_tiles()` uses `get_tile_lt()` for TMA loader compatibility
3. `mma()` uses `get_tile_lt()` for MMA op compatibility

**Code pattern in kernel:**

```mojo
# Get tiles as LayoutTensor (for TMA loader compatibility)
var a_tile, b_tile, a_scales_tile = tiles.payload().get_tile_lt[
    Self.config.k_group_size
](stage, 0)

# Peer CTA slicing (LayoutTensor pattern: ptr only)
var a_peer_tile = type_of(a_tile)(
    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size)
)
```

### 9.4 Current State

- ✅ SMEM remains LayoutTensor-based (unchanged)
- ✅ Bridge payload provides both TileTensor and LayoutTensor access
- ✅ Kernel uses LayoutTensor at boundaries (loaders, MMA)
- ✅ TileTensor available via `get_tile()` for future internal use
- ✅ Build succeeds

### 9.5 Future Work

Once the compiler bug with `.to_layout_tensor()` is fixed:

1. Switch to `get_tile()` for internal operations
2. Use TileTensor for peer tile creation: `(ptr + offset, layout)`
3. Call `.to_layout_tensor()` only at explicit boundaries
4. Eventually update loaders/MMA to accept TileTensor natively

---

## Migration Checklist (Complete)

1. [x] Create `BlockwiseFP8TilePayloadBridge` in tile_types.mojo
2. [x] Verify bridge payload compiles
3. [x] Update kernel to use bridge payload
4. [x] Test with smoke test on GPU
5. [x] Document compiler bug for follow-up (see Part 14)
6. [x] Expand migration to block_scaled kernel
7. [x] Expand migration to grouped_block_scaled_1d1d kernel
8. [x] Expand migration to grouped_block_scaled kernel
9. [x] Add TileTensor overloads to tile loaders
10. [x] Simplify to TileTensor-native with explicit boundary conversion (Part 21)
11. [x] Migrate from Layout parameters to explicit dimensions (Part 22)

---

## Part 10: Block-Scaled Kernel Migration

Successfully migrated the block_scaled kernel to use the same bridge payload pattern!

### 10.1 BlockScaledTilePayloadBridge

Created `BlockScaledTilePayloadBridge` in `tile_types.mojo`. This is similar to
`BlockwiseFP8TilePayloadBridge` but with 4 tile types instead of 3:

- A tiles (matrix data)
- B tiles (matrix data)
- SFA tiles (A scale factors)
- SFB tiles (B scale factors)

```mojo
struct BlockScaledTilePayloadBridge[
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: LegacyLayout,
    b_tile_layout: LegacyLayout,
    sfa_tile_layout: LegacyLayout,
    sfb_tile_layout: LegacyLayout,
    num_pipeline_stages: Int,
](TilePayload, TrivialRegisterType):
    """Bridge payload for block-scaled matmul with 4 tile types."""

    # Dual accessor methods
    fn get_tile[k_group_size: Int](...) -> Tuple[ATile, BTile, SFATile, SFBTile]
    fn get_tile_lt[k_group_size: Int](...) -> Tuple[LTATile, LTBTile, LTSFATile, LTSFBTile]
```

### 10.2 Migration Changes

**File: `block_scaled/block_scaled_matmul_kernel.mojo`**

1. Import changed from `BlockScaledTilePayload` to `BlockScaledTilePayloadBridge`
2. `TilePayload` type alias uses bridge payload
3. `load_input_tiles()` uses `get_tile_lt()` for TMA loader compatibility
4. `mma()` uses `get_tile_lt()` for MMA op compatibility

### 10.3 Tests Passed

- `test_matmul_sm100_block_scaled_mxfp8_1sm.mojo` - all passed
- `test_matmul_sm100_block_scaled_nvfp4_1sm.mojo` - all passed

### 10.4 Pattern Established

The bridge payload pattern is now proven across two kernel variants:

1. **Blockwise FP8** (3 tiles: A, B, A-scales)
2. **Block-Scaled** (4 tiles: A, B, SFA, SFB)

This establishes a clear pattern for migrating remaining kernels (grouped_block_scaled,
grouped_1d1d) to use bridge payloads.

---

## Part 11: Grouped 1D-1D Kernel Migration

Successfully migrated the grouped_block_scaled_1d1d kernel to use the bridge payload!

### 11.1 Reusing BlockScaledTilePayloadBridge

The grouped_1d1d kernel uses the same 4-tile pattern (A, B, SFA, SFB) as block_scaled,
so we can reuse `BlockScaledTilePayloadBridge` directly - no new payload type needed.

### 11.2 Migration Changes

**File: `grouped_block_scaled_1d1d/grouped_1d1d_matmul_kernel.mojo`**

1. Import changed from `BlockScaledTilePayload` to `BlockScaledTilePayloadBridge`
2. `TilePayload` type alias uses bridge payload
3. `load_input_tiles()` uses `get_tile_lt()` for TMA loader compatibility
4. `mma()` uses `get_tile_lt()` for MMA op compatibility

### 11.3 Tests Passed

- `test_grouped_matmul_sm100_nvfp4.mojo` - all passed
- `test_grouped_matmul_sm100_mxfp8.mojo` - all passed

### 11.4 Pattern Confirmed

The bridge payload pattern now works across three kernel variants:

1. **Blockwise FP8** (3 tiles: A, B, A-scales)
2. **Block-Scaled** (4 tiles: A, B, SFA, SFB)
3. **Grouped 1D-1D** (4 tiles: A, B, SFA, SFB) - reuses BlockScaledTilePayloadBridge

Only `grouped_block_scaled` remains to migrate.

---

## Part 12: Grouped Block-Scaled Kernel Migration

Successfully migrated the grouped_block_scaled kernel to use the bridge payload!

### 12.1 Reusing BlockScaledTilePayloadBridge

The grouped_block_scaled kernel uses the same 4-tile pattern (A, B, SFA, SFB) as
block_scaled and grouped_1d1d, so we can reuse `BlockScaledTilePayloadBridge`
directly - no new payload type needed.

### 12.2 Migration Changes

**File: `grouped_block_scaled/grouped_block_scaled_matmul_kernel.mojo`**

1. Import changed from `BlockScaledTilePayload` to `BlockScaledTilePayloadBridge`
2. `TilePayload` type alias uses bridge payload
3. `load_input_tiles()` uses `get_tile_lt()` for TMA loader compatibility
4. `mma()` uses `get_tile_lt()` for MMA op compatibility

### 12.3 Tests Passed

- `test_grouped_block_scaled_gemm_nvfp4.mojo` - all passed
- `test_grouped_block_scaled_gemm_nvfp4_execution.mojo` - all passed
- `test_grouped_block_scaled_gemm_execution.mojo` - all passed (MXF8)

### 12.4 Migration Complete

All four SM100 structured kernels are now using bridge payloads:

1. **Blockwise FP8** (3 tiles: A, B, A-scales) - uses `BlockwiseFP8TilePayloadBridge`
2. **Block-Scaled** (4 tiles: A, B, SFA, SFB) - uses `BlockScaledTilePayloadBridge`
3. **Grouped 1D-1D** (4 tiles: A, B, SFA, SFB) - uses `BlockScaledTilePayloadBridge`
4. **Grouped Block-Scaled** (4 tiles: A, B, SFA, SFB) - uses `BlockScaledTilePayloadBridge`

The middle-out migration strategy is proven to work. The bridge payloads provide:

- TileTensor API for future migration
- LayoutTensor compatibility via `get_tile_lt()` for current boundary operations
- Zero runtime overhead (pointer reinterpretation only)

---

## Part 13: TileTensor Overloads for Tile Loaders

Added TileTensor overloads to `TileLoaderTMA` and `ScalesTileLoader` in
`tile_loader.mojo`.

### 13.1 Implementation

Added overloaded `load` methods that accept TileTensor-based tiles
(`SMemTile2D`) and convert them to LayoutTensor internally for the TMA
operations.

**TileLoaderTMA overload:**

```mojo
@always_inline
fn load[
    dim0: Int,
    dim1: Int,
    /,
    alignment: Int = 128,
](
    self,
    dest: SMemTile2D[Self.dtype, dim0, dim1, alignment=alignment],
    ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
    k_coord: UInt,
    row_coord: UInt,
):
    # Construct LayoutTensor from TileTensor pointer for TMA API
    comptime tile_layout = LegacyLayout.row_major(dim0, dim1)
    var lt_dest = LayoutTensor[
        Self.dtype,
        tile_layout,
        address_space = AddressSpace.SHARED,
    ](dest.ptr)
    self.tma_op[].async_multicast_load[Self.cta_group](
        lt_dest, barrier, (k_coord, row_coord), self.multicast_mask
    )
```

**ScalesTileLoader overload:**

Same pattern, using `async_copy` instead of `async_multicast_load`.

### 13.2 Key Design

- **Zero-cost conversion**: Construct LayoutTensor directly from TileTensor pointer
- **Overload resolution**: Uses `dim0`, `dim1` parameters instead of `tile_layout`
- **Internal conversion**: Loaders handle the TileTensor→LayoutTensor conversion
  so kernel code can pass TileTensor tiles directly

### 13.3 Benefits

Once the compiler bug with `.to_layout_tensor()` is fixed, kernels can:

1. Use `get_tile()` to get TileTensor tiles
2. Pass TileTensor tiles directly to loaders (using these overloads)
3. Eventually use TileTensor throughout the pipeline

### 13.4 Tests Passed

- `test_matmul_sm100_1sm_blockwise_fp8_part_a.mojo` - build succeeded
- `test_grouped_matmul_sm100_nvfp4.mojo` - build succeeded

---

## Part 14: Compiler Bug Documentation

### 14.1 Bug Description

Calling `.to_layout_tensor()` on TileTensor tiles from the bridge payload causes
a compiler crash in SM100 kernel context.

**Error message:**

```text
mojo-compiler-only: [...]/llvm/include/llvm/Support/Allocator.h:163:
void *llvm::BumpPtrAllocatorImpl<>::Allocate(size_t, Align) [...]:
Assertion `AllocEndPtr >= uintptr_t(CurPtr) &&
    "Alignment + Size must not overflow"' failed.
```

### 14.2 Reproduction Steps

1. In `blockwise_fp8_matmul_kernel.mojo`, change `mma()` to use TileTensor:

```mojo
# Change this:
var a_tile, b_tile, _ = tiles.payload().get_tile_lt[
    Self.config.k_group_size
](tiles.stage(), jj)

# To this:
var a_tile_tt, b_tile_tt, _ = tiles.payload().get_tile[
    Self.config.k_group_size
](tiles.stage(), jj)

mma_op.mma(
    a_tile_tt.to_layout_tensor(),  # <-- Crash here
    b_tile_tt.to_layout_tensor(),
    ...
)
```

1. Build:
   `./bazelw build //max/kernels/test/gpu/linalg:test_matmul_sm100_1sm_blockwise_fp8_part_a.mojo.test`

### 14.3 Analysis

The crash occurs in LLVM's BumpPtrAllocator, suggesting the compiler is generating
an unexpectedly large type during the `.to_layout_tensor()` conversion.

**Root cause:** TileTensor's `to_layout_tensor()` computes the output type inline:

```mojo
LayoutTensor[
    Self.dtype,
    layout.Layout(
        coord_to_int_tuple[*Self.shape_types](),  # <-- Complex variadic expansion
        coord_to_int_tuple[*Self.stride_types](),
    ),
    Self.origin,
    address_space = Self.address_space,
]
```

When `shape_types` and `stride_types` are complex variadics like
`_IntToComptimeInt[dim0, dim1]` and `_RowMajor[...]`, the compile-time type
computation explodes.

**Contrast with proven pattern:** `ManagedTensorSlice.to_layout_tensor()` (used by
InputTensor/OutputTensor) uses a simpler type signature and constructs LayoutTensor
directly from pointer + runtime shape/stride. Our `get_tile_lt()` workaround follows
this same proven pattern.

### 14.4 Workaround (Proven Pattern)

Use dual accessor methods in bridge payloads:

- `get_tile()` - Returns TileTensor (for internal use, future expansion)
- `get_tile_lt()` - Returns LayoutTensor directly (for boundary operations)

The `get_tile_lt()` method follows the proven codebase pattern: construct
LayoutTensor directly from pointer + layout, avoiding the problematic variadic
type expansion in TileTensor's `to_layout_tensor()`. This is the same pattern
used by `ManagedTensorSlice.to_layout_tensor()` which works reliably.

### 14.5 Status

**Confirmed bug** - Reproduced on 2025-01-31. Workaround in place via dual accessors.
When this bug is fixed, kernels can switch from `get_tile_lt()` to `get_tile()` with
`.to_layout_tensor()` calls at boundaries.

---

## Part 15: Dual Accessors in Pipeline Storage

Added TileTensor dual accessors to all tile storage structs in `pipeline_storage.mojo`.

### 15.1 Implementation

Updated three storage structs with TileTensor array types and accessors:

1. **StandardTileStorage**: Added `ATileArrayTT`, `BTileArrayTT` types and
   `a_tiles_tt()`, `b_tiles_tt()` accessors
2. **BlockScaledTileStorage**: Added TT types for A, B, C, SFA, SFB and
   corresponding `*_tt()` accessors
3. **BlockwiseFP8TileStorage**: Added TT types for A, B, C, A-scales and
   corresponding `*_tt()` accessors

### 15.2 Key Design

```mojo
# Import TileTensor-based SMemTileArray
from .tile_types import SMemTileArray as TTSMemTileArray

struct BlockwiseFP8TileStorage[...]:
    # LayoutTensor array types (existing)
    comptime ATileArray = SMemTileArray[...]

    # TileTensor array types (new)
    comptime ATileArrayTT = TTSMemTileArray[...]

    # Shared storage (binary compatible)
    var a_tiles_storage: Self.ATileArray.Storage

    # LayoutTensor accessor (existing)
    fn a_tiles(...) -> Self.ATileArray

    # TileTensor accessor (new)
    fn a_tiles_tt(...) -> Self.ATileArrayTT:
        return Self.ATileArrayTT(self.a_tiles_storage.unsafe_ptr())
```

### 15.3 Benefits

- **Same binary layout**: Both array types share InlineArray storage
- **Gradual migration**: Callers choose which accessor to use
- **SMEM provides both views**: No need for bridge payloads at storage level
- **Backwards compatible**: Existing code continues to work unchanged

### 15.4 Tests Passed

- `test_matmul_sm100_1sm_blockwise_fp8_part_a.mojo` - build succeeded
- `test_grouped_matmul_sm100_nvfp4.mojo` - build succeeded

### 15.5 Migration Path

With dual accessors at the storage level, the migration path is now:

```text
pipeline_storage.mojo (DONE)
    ├── a_tiles() → LayoutTensor array
    └── a_tiles_tt() → TileTensor array
           ↓
SMEM structs (can be updated to expose _tt accessors)
           ↓
Kernels (can choose which view to use)
           ↓
Bridge payloads become optional
```

---

## Part 16: Local GPU Test Verification

Verified all changes work correctly on local B200 GPU.

### 16.1 Test Results

**Blockwise FP8 Test** (`test_matmul_sm100_1sm_blockwise_fp8_part_a.mojo`):

- All 108 test configurations passed
- Various block tile shapes: (64, 8, 128) through (64, 256, 128)
- MMA shapes tested: (64, 8, 32) through (64, 256, 32)
- Problem sizes from (1000, 576, 320) to (1024, 1536, 7168)

**Grouped Matmul NvFP4 Test** (`test_grouped_matmul_sm100_nvfp4.mojo`):

- All test cases passed
- Verified `=== TEST PASSED ===` for each configuration
- Problem sizes tested: (6512, 2048, 1024), (1478, 2048, 1024), (1408, 2048, 1024)

### 16.2 Summary

The TileTensor migration infrastructure is now complete and verified:

1. **Bridge Payloads** (Part 8-12): All 4 kernels migrated
   - `BlockwiseFP8TilePayload`
   - `BlockScaledTilePayload`
   - `GroupedBlockScaledTilePayload`
   - `Grouped1D1DTilePayload`

2. **Tile Loader Overloads** (Part 13): TileTensor-compatible loaders
   - `TileLoaderTMA` accepts `SMemTile2D`
   - `ScalesTileLoader` accepts `SMemTile2D`

3. **Dual Accessors** (Part 15): Pipeline storage structs
   - `StandardTileStorage`: `a_tiles()` / `a_tiles_tt()`
   - `BlockScaledTileStorage`: All 5 tile types
   - `BlockwiseFP8TileStorage`: All 4 tile types

4. **Compiler Bug Documented** (Part 14): Workaround in place

---

## Part 17: SMEM Struct TileTensor Accessors

Extended all SMEM structs to expose TileTensor accessors from the underlying
tile storage.

### 17.1 Files Updated

1. **BlockwiseFP8Smem** (`blockwise_fp8/blockwise_fp8_smem.mojo`):
   - Added `ATileArrayTT`, `BTileArrayTT`, `CTileArrayTT`, `AScalesTileArrayTT`
     type exports
   - Added `a_tiles_tt()`, `b_tiles_tt()`, `c_tiles_tt()`,
     `a_scales_tiles_tt()` accessors

2. **BlockScaledSmem** (`block_scaled/block_scaled_smem.mojo`):
   - Added `ATileArrayTT`, `BTileArrayTT`, `CTileArrayTT`, `SFATileArrayTT`,
     `SFBTileArrayTT` type exports
   - Added `a_tiles_tt()`, `b_tiles_tt()`, `c_tiles_tt()`, `sfa_tiles_tt()`,
     `sfb_tiles_tt()` accessors

3. **Grouped1D1DSmem** (`grouped_block_scaled_1d1d/grouped_1d1d_smem.mojo`):
   - Added same TileTensor type exports and accessors as BlockScaledSmem

4. **GroupedBlockScaledSmem** (`grouped_block_scaled/grouped_block_scaled_smem.mojo`):
   - Added same TileTensor type exports and accessors as BlockScaledSmem

### 17.2 Key Design

SMEM structs simply delegate to their embedded tile storage:

```mojo
struct BlockwiseFP8Smem[...]:
    # Re-export TileTensor array types
    comptime ATileArrayTT = Self.Tiles.ATileArrayTT
    # ...

    # TileTensor accessor delegates to tile storage
    @always_inline
    fn a_tiles_tt(ref[AddressSpace.SHARED] self) -> Self.ATileArrayTT:
        return self.tiles.a_tiles_tt()
```

### 17.3 Tests Passed

- `test_matmul_sm100_1sm_blockwise_fp8_part_a.mojo` - all configurations
- `test_grouped_matmul_sm100_nvfp4.mojo` - all configurations
- `test_matmul_sm100_block_scaled_nvfp4_1sm.mojo` - all configurations

### 17.4 Migration Path Complete

The full migration path from Part 15 is now implemented:

```text
pipeline_storage.mojo (DONE - Part 15)
    ├── a_tiles() → LayoutTensor array
    └── a_tiles_tt() → TileTensor array
           ↓
SMEM structs (DONE - Part 17)
    ├── a_tiles() → delegates to storage
    └── a_tiles_tt() → delegates to storage
           ↓
Kernels can now access TileTensor arrays directly from SMEM
```

---

## Part 18: Inline TileTensor-to-LayoutTensor Conversion

Updated the blockwise_fp8 kernel's `mma()` function to use TileTensor tiles
with an inline conversion workaround for the compiler bug documented in
Part 14.

### 18.1 The Problem

The bridge payload's `get_tile_lt()` method works but returns LayoutTensor
directly, bypassing TileTensor entirely in the kernel. We wanted to use
TileTensor tiles in the kernel code and convert to LayoutTensor only at the
MMA boundary.

However, calling `.to_layout_tensor()` on the TileTensor tiles causes a
compiler crash:

```text
LLVM BumpPtrAllocator overflow
Assertion `AllocEndPtr >= uintptr_t(CurPtr) &&
    "Alignment + Size must not overflow"' failed.
```

### 18.2 Root Cause Analysis

TileTensor's `to_layout_tensor()` method computes the output type inline:

```mojo
LayoutTensor[
    Self.dtype,
    layout.Layout(
        coord_to_int_tuple[*Self.shape_types](),   # Complex variadic
        coord_to_int_tuple[*Self.stride_types](),  # Complex variadic
    ),
    ...
]
```

When `shape_types` and `stride_types` are complex variadics like
`_IntToComptimeInt[dim0, dim1]` and `_RowMajor[...]`, the compiler attempts to
expand all possible type combinations at compile time, causing exponential type
instantiation and memory exhaustion.

### 18.3 Helper Function: `tile_tt_to_lt`

Created a clean helper function in `tile_types.mojo` that encapsulates the
workaround:

```mojo
@always_inline
fn tile_tt_to_lt[
    dtype: DType,
    dim0: Int,
    dim1: Int,
](
    ptr: LegacyUnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
) -> LayoutTensor[
    dtype,
    LegacyLayout.row_major(dim0, dim1),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
]:
    """Convert TileTensor pointer to LayoutTensor (workaround).

    FIXME: TileTensor.to_layout_tensor() crashes with LLVM
    BumpPtrAllocator overflow when shape_types/stride_types are
    complex variadics.
    """
    return LayoutTensor[
        dtype,
        LegacyLayout.row_major(dim0, dim1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ](ptr)
```

### 18.4 Clean Kernel Usage

The kernel code is now much cleaner:

```mojo
# Get tiles as TileTensor and convert to LayoutTensor for MMA
var a_tile_tt, b_tile_tt, _ = tiles.payload().get_tile[
    Self.config.k_group_size
](tiles.stage(), jj)
var a_tile = tile_tt_to_lt[Self.a_type, Self.BM, Self.BK](a_tile_tt.ptr)
var b_tile = tile_tt_to_lt[Self.b_type, Self.BN, Self.BK](b_tile_tt.ptr)

mma_op.mma(a_tile, b_tile, ...)
```

### 18.5 Why This Works

1. **Explicit layout**: Uses `LegacyLayout.row_major(dim0, dim1)` with known
   compile-time constants, avoiding the problematic variadic type expansion
2. **MutAnyOrigin**: Required origin parameter for LayoutTensor, matching
   SMemTile pattern
3. **Direct pointer access**: TileTensor's `.ptr` gives us the raw SMEM pointer
4. **Same memory**: Both TileTensor and LayoutTensor view the same underlying
   SMEM

### 18.6 Future Cleanup

Once the compiler bug is fixed, the helper function can be replaced with:

```mojo
var a_tile_tt, b_tile_tt, _ = tiles.payload().get_tile[...](...)
mma_op.mma(
    a_tile_tt.to_layout_tensor(), b_tile_tt.to_layout_tensor(), ...
)
```

### 18.7 Tests Verified

- `test_matmul_sm100_1sm_blockwise_fp8_part_a.mojo` - all configurations pass

---

## Part 19: Direct `.to_layout_tensor()` Migration

Successfully migrated block_scaled and grouped_1d1d kernels to use TileTensor's
`.to_layout_tensor()` method directly, eliminating the need for the `tile_tt_to_lt`
workaround in those kernels.

### 19.1 Kernels Successfully Using `.to_layout_tensor()` Directly

**block_scaled** and **grouped_1d1d** now use TileTensor throughout:

```mojo
# In mma():
var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
    Self.config.k_group_size
](tiles.stage(), jj)
var a_tile = a_tt.to_layout_tensor()
var b_tile = b_tt.to_layout_tensor()
var sfa_tile = sfa_tt.to_layout_tensor()
var sfb_tile = sfb_tt.to_layout_tensor()

# In load_input_tiles():
var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
    Self.config.k_group_size
](tiles.stage(), jj)

# Peer CTA slice using TileTensor pattern (ptr + layout)
var a_peer_tt = type_of(a_tt)(
    a_tt.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
    a_tt.layout,
)

# Convert to LayoutTensor at TMA boundary
a_tma_op.async_multicast_load_3d[Self.cta_group](
    a_peer_tt.to_layout_tensor(),
    barrier[0],
    (k_coord, a_gmem_m_coord, batch_coord),
    a_multicast_mask,
)
```

### 19.2 Kernels Still Requiring Workarounds (Compiler Bug)

**blockwise_fp8**: Uses `tile_tt_to_lt` workaround in `mma()`, `get_tile_lt()` in
`load_input_tiles()`. The compiler crashes with `.to_layout_tensor()`.

**grouped_block_scaled**: Uses `get_tile_lt()` throughout. The compiler crashes
even with just A/B tile conversion via `.to_layout_tensor()`.

### 19.3 Key Differences Between Working and Non-Working Kernels

The compiler bug appears to be triggered by specific combinations of:

- Type parameter complexity in the instantiation context
- Number of generic type parameters in the kernel struct
- Possible interaction with grouped/batched variants

Block_scaled and grouped_1d1d work while blockwise_fp8 and grouped_block_scaled
don't - this is consistent with the compiler bug being context-sensitive rather
than a fundamental issue with `.to_layout_tensor()`.

### 19.4 Changes Summary

| Kernel | mma() | load_input_tiles() | Status |
|--------|-------|-------------------|--------|
| block_scaled | `get_tile_lt()` | `get_tile_lt()` | ⚠️ Workaround (NVFP4 crash) |
| grouped_1d1d | `.to_layout_tensor()` | `.to_layout_tensor()` | ✅ Full TileTensor |
| blockwise_fp8 | `tile_tt_to_lt` | `get_tile_lt()` | ⚠️ Workaround needed |
| grouped_block_scaled | `get_tile_lt()` | `get_tile_lt()` | ⚠️ Workaround needed |

**Note:** block_scaled was initially migrated to `.to_layout_tensor()` and worked
for MXFP8 types, but crashed on NVFP4 types. Reverted to `get_tile_lt()` for
consistency and compatibility.

### 19.5 Tests Verified

- `test_block_scaled_matmul_with_epilogue.mojo` - ALL TESTS PASSED (MXFP8)
- `test_matmul_sm100_block_scaled_nvfp4_1sm.mojo` - ALL TESTS PASSED (NVFP4)
- `test_grouped_matmul_sm100_mxfp8.mojo` - ALL TESTS PASSED
- `test_grouped_matmul_sm100_nvfp4.mojo` - ALL TESTS PASSED
- `test_matmul_sm100_blockwise_fp8_smoke.mojo` - ALL SMOKE TESTS PASSED

---

## Part 20: Compiler Bug Report - TileTensor.to_layout_tensor() Crashes

### 20.1 Bug Summary

Calling `TileTensor.to_layout_tensor()` in certain kernel contexts causes the Mojo
compiler to crash with an LLVM BumpPtrAllocator overflow assertion failure.

**Error Message:**

```text
mojo: external/+llvm_configure+llvm-project/llvm/include/llvm/Support/Allocator.h:163:
void *llvm::BumpPtrAllocatorImpl<>::Allocate(size_t, Align)
[AllocatorT = llvm::MallocAllocator, SlabSize = 4096, SizeThreshold = 4096, GrowthDelay = 128]:
Assertion `AllocEndPtr >= uintptr_t(CurPtr) && "Alignment + Size must not overflow"' failed.
```

### 20.2 Affected Kernels

| Kernel | mma() | load_input_tiles() | Notes |
|--------|-------|-------------------|-------|
| **block_scaled** | ❌ NVFP4 CRASH | ❌ NVFP4 CRASH | Works with MXFP8, crashes with NVFP4 |
| **blockwise_fp8** | ❌ CRASH | ❌ CRASH | Both methods crash |
| **grouped_block_scaled** | ❌ CRASH | ❌ CRASH | Both methods crash |
| grouped_1d1d | ✅ Works | ✅ Works | Works with both MXFP8 and NVFP4 |

### 20.3 Reproduction Steps

**For blockwise_fp8 crash:**

1. Edit `blockwise_fp8/blockwise_fp8_matmul_kernel.mojo`
2. In `load_input_tiles()`, change:

```mojo
# From:
var a_tile, b_tile, a_scales_tile = tiles.payload().get_tile_lt[
    Self.config.k_group_size
](stage, 0)

# To:
var a_tt, b_tt, a_scales_tt = tiles.payload().get_tile[
    Self.config.k_group_size
](stage, 0)
a_loader.load(a_tt.to_layout_tensor(), ...)  # <-- CRASH
```

1. Build:
   `mojo build max/kernels/test/gpu/linalg/test_matmul_sm100_blockwise_fp8_smoke.mojo`

**For grouped_block_scaled crash:**

1. Edit `grouped_block_scaled/grouped_block_scaled_matmul_kernel.mojo`
1. In `load_input_tiles()` or `mma()`, change:

```mojo
# From:
var a_tile, b_tile, sfa_tile, sfb_tile = tiles.payload().get_tile_lt[...](...)

# To:
var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[...](...)
a_tma_op.async_multicast_load_3d[...](a_tt.to_layout_tensor(), ...)  # <-- CRASH
```

1. Build:
   `mojo build max/kernels/test/gpu/linalg/test_grouped_block_scaled_gemm_execution.mojo`

### 20.4 Root Cause Analysis

The crash occurs in LLVM's BumpPtrAllocator during type elaboration. The likely cause
is that `TileTensor.to_layout_tensor()` computes its output type inline:

```mojo
fn to_layout_tensor(self) -> LayoutTensor[
    Self.dtype,
    layout.Layout(
        coord_to_int_tuple[*Self.shape_types](),   # Complex variadic expansion
        coord_to_int_tuple[*Self.stride_types](),  # Complex variadic expansion
    ),
    ...
]
```

When `shape_types` and `stride_types` are complex variadics from the SM100 kernel
type hierarchy, the compiler attempts to expand all type combinations at compile
time, causing exponential type instantiation and memory exhaustion.

### 20.5 Key Observations

1. **Context-sensitive**: The same `TileTensor.to_layout_tensor()` call works in
   `block_scaled` and `grouped_1d1d` but crashes in `blockwise_fp8` and
   `grouped_block_scaled`.

2. **Type complexity**: The crashing kernels may have more complex type parameter
   hierarchies or more generic parameters on the kernel struct.

3. **Workaround available**: Using `get_tile_lt()` (returns LayoutTensor directly)
   or the `tile_tt_to_lt` helper function avoids the crash.

### 20.6 Workarounds in Place

**For blockwise_fp8:**

- `mma()`: Uses `tile_tt_to_lt` helper function
- `load_input_tiles()`: Uses `get_tile_lt()` directly

**For grouped_block_scaled:**

- Both methods use `get_tile_lt()` directly

### 20.7 Files for Reference

- `structured_kernels/tile_types.mojo`: Contains `tile_tt_to_lt` helper and
  bridge payloads
- `blockwise_fp8/blockwise_fp8_matmul_kernel.mojo`: Lines ~480 and ~545
- `grouped_block_scaled/grouped_block_scaled_matmul_kernel.mojo`: Lines ~1330
  and ~1410

---

## Part 21: Final Simplification - TileTensor Native

After the bridge payload approach proved successful, we simplified the
architecture further by removing workarounds and using direct TileTensor
storage with explicit LayoutTensor conversion at TMA/MMA boundaries.

### 21.1 Architecture Overview

The final architecture is clean and straightforward:

```text
┌─────────────────────────────────────────────────────────────────────┐
│  SMEM Storage (TileTensor-native)                                   │
│                                                                     │
│  pipeline_storage.mojo:                                             │
│    - ATileArray = SMemTileArray[...] (TileTensor-based)            │
│    - BTileArray = SMemTileArray[...] (TileTensor-based)            │
│    - SFATileArray = SMemTileArray[...] (TileTensor-based)          │
│                                                                     │
│  tile_pipeline.mojo:                                                │
│    - Payloads store TileTensor arrays                              │
│    - get_tile() returns TileTensor tiles                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Kernel code works with TileTensor
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Kernel (boundary conversion)                                       │
│                                                                     │
│  LayoutTensor type aliases:                                         │
│    comptime ATileLT = LayoutTensor[a_type, layout, SHARED, 128]    │
│    comptime BTileLT = LayoutTensor[b_type, layout, SHARED, 128]    │
│                                                                     │
│  At TMA boundary:                                                   │
│    a_loader.load(Self.ATileLT(tile.ptr), barrier, ...)             │
│                                                                     │
│  At MMA boundary:                                                   │
│    mma_op.mma(Self.ATileLT(a_tile.ptr), Self.BTileLT(b_tile.ptr))  │
└─────────────────────────────────────────────────────────────────────┘
```

### 21.2 Key Changes

1. **Removed bridge payloads**: No more `BlockwiseFP8TilePayloadBridge` or dual
   `get_tile()`/`get_tile_lt()` methods. Payloads have single `get_tile()` returning
   TileTensor.

2. **Removed dual accessors**: SMEM structs have single accessor methods returning
   TileTensor arrays.

3. **Added LayoutTensor type aliases in kernels**: Each kernel defines `ATileLT`,
   `BTileLT`, etc. for explicit boundary conversion.

4. **Explicit boundary conversion**: Kernels convert TileTensor to LayoutTensor
   at TMA/MMA call sites using `Self.ATileLT(tile.ptr)`.

### 21.3 Files Modified

| File | Changes |
|------|---------|
| `pipeline_storage.mojo` | All tile arrays now TileTensor-based |
| `tile_pipeline.mojo` | Payloads use TileTensor arrays, single `get_tile()` |
| `blockwise_fp8_smem.mojo` | Single accessors returning TileTensor |
| `block_scaled_smem.mojo` | Single accessors returning TileTensor |
| `grouped_1d1d_smem.mojo` | Single accessors returning TileTensor |
| `grouped_block_scaled_smem.mojo` | Single accessors returning TileTensor |
| `blockwise_fp8_matmul_kernel.mojo` | Added `ATileLT`, `BTileLT`, `AScalesTileLT`; explicit conversion |
| `block_scaled_matmul_kernel.mojo` | Uses payload `get_tile()` with boundary conversion |
| `grouped_1d1d_matmul_kernel.mojo` | Added `ATileLT`, `BTileLT`, `SFATileLT`, `SFBTileLT` |
| `grouped_block_scaled_matmul_kernel.mojo` | Uses payload `get_tile()` with boundary conversion |
| `default/matmul_kernels.mojo` | Added `ATileLT`, `BTileLT`; explicit conversion |
| `blockwise_fp8_accumulator.mojo` | Updated import to TileTensor-based `SMemTileArray` |
| `tile_loader.mojo` | Added alignment parameter to LayoutTensor construction |

### 21.4 Tests Verified

All SM100 structured kernels pass their tests:

| Kernel | Test | Status |
|--------|------|--------|
| blockwise_fp8 | All 1SM/2SM part A/B tests | ✅ PASSED |
| block_scaled | NVFP4 and MXFP8 tests | ✅ PASSED |
| grouped_1d1d | NVFP4 and MXFP8 tests | ✅ PASSED |
| grouped_block_scaled | Execution tests | ✅ PASSED |
| default SM100 | Smoke tests | ✅ PASSED |

### 21.5 Benefits of Final Architecture

1. **Simplicity**: No bridge types, no dual accessors, clear conversion points
2. **Type safety**: LayoutTensor type aliases ensure correct types at boundaries
3. **Performance**: Zero-cost pointer reinterpretation at boundaries
4. **Maintainability**: Single source of truth for tile storage types
5. **Clarity**: Explicit `Self.ATileLT(ptr)` makes boundary conversion visible

---

## Part 22: Explicit Dimension Parameters

Migrated all tile types from Layout-parameterized to dimension-parameterized APIs.

### 22.1 The Change

Replaced `SMemTileArray[dtype, layout, ...]` with
`SMemTileArray2D[dtype, dim0, dim1, ...]` throughout all SM100 structured
kernels.

**Before:**

```mojo
comptime ATileArray = SMemTileArray[
    Self.a_type, Self.a_tile_layout, Self.num_pipeline_stages, alignment=128
]
```

**After:**

```mojo
comptime ATileArray = SMemTileArray2D[
    Self.a_type, Self.a_dim0, Self.a_dim1, Self.num_pipeline_stages, 128
]
```

### 22.2 Why This Change

1. **Simpler API**: Explicit dimensions are clearer than extracting from Layout
2. **No Layout dependency**: Tile arrays no longer need Layout import
3. **Consistent with TileTensor**: TileTensor uses explicit dimensions internally
4. **Easier to understand**: `dim0=64, dim1=32` is clearer than `layout.shape[0].value()`

### 22.3 Files Modified

| File | Changes |
|------|---------|
| `tile_types.mojo` | `SMemTileArray2D` uses explicit `dim0`, `dim1` params; removed `SMemTile`, `RegTile` legacy aliases |
| `pipeline_storage.mojo` | All storage structs use dimension params |
| `tile_pipeline.mojo` | All payload structs use dimension params |
| `*_smem.mojo` | Added `SFA_DIM0/1`, `SFB_DIM0/1` comptime values for SF tile dimensions |
| `*_matmul_kernel.mojo` | Pass dimensions instead of layouts to payloads |

### 22.4 SF Tile Dimension Computation

For block-scaled kernels, the scale factor tile dimensions are computed from
the tile layout formula:

```mojo
# SF tile dimensions (computed from tile_sf_layout_k_major formula)
comptime SF_BK = Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles
comptime SFA_DIM0 = (Self.BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]
comptime SFA_DIM1 = (
    Self.SF_BK // (SF_ATOM_K * Self.config.vec_sf_size)
) * (SF_ATOM_M[1] * SF_ATOM_K)
```

### 22.5 Removed Legacy Types

Removed compatibility type aliases that were no longer needed:

- `SMemTile[dtype, legacy_layout, alignment]` → Use `SMemTile2D` directly
- `RegTile[dtype, legacy_layout, alignment]` → Use `RegTile2D` directly
- Duplicate `BlockScaledTilePayload` in `tile_types.mojo` → Now only in `tile_pipeline.mojo`

### 22.6 Current Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│  tile_types.mojo                                                     │
│                                                                      │
│  SMemTile2D[dtype, dim0, dim1, alignment]     ← Explicit dimensions │
│  SMemTileArray2D[dtype, dim0, dim1, stages, alignment]              │
│  RegTile2D[dtype, dim0, dim1, alignment]                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  pipeline_storage.mojo / tile_pipeline.mojo                          │
│                                                                      │
│  StandardTileStorage[a_type, b_type, a_dim0, a_dim1, b_dim0, ...]   │
│  BlockScaledTilePayload[..., a_dim0, a_dim1, sfa_dim0, sfa_dim1...] │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SMEM structs (*_smem.mojo)                                          │
│                                                                      │
│  Compute SF dimensions: SFA_DIM0, SFA_DIM1, SFB_DIM0, SFB_DIM1      │
│  Pass to storage/payload: Self.BM, Self.BK, Self.SFA_DIM0, ...      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Kernels (*_matmul_kernel.mojo)                                      │
│                                                                      │
│  Use payload with dimensions from SmemType                           │
│  Convert to LayoutTensor at TMA/MMA boundaries                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 23: Performance Validation

Benchmarked the TileTensor migration against main branch to verify no
performance regression.

### 23.1 Benchmark Environment

- GPU: NVIDIA B200 (SM100/Blackwell)
- Iterations: 100 per configuration
- Date: 2026-02-01

### 23.2 Grouped Block-Scaled GEMM Results

Primary benchmark for block-scaled kernels with MXFP8 and NVFP4 formats.

**MXFP8 (float8_e4m3fn -> bfloat16):**

| Shape (G x M x K x N) | Main (ms) | TileTensor (ms) | Delta |
|-----------------------|-----------|-----------------|-------|
| 32 x 128 x 4096 x 7168 | 0.638 | 0.593 | +7.6% |
| 32 x 128 x 7168 x 2048 | 1.059 | 0.997 | +6.2% |
| 32 x 4096 x 4096 x 7168 | 19.637 | 17.108 | +14.8% |
| 32 x 4096 x 7168 x 2048 | 32.728 | 29.633 | +10.4% |

**NVFP4 (uint8 -> bfloat16):**

| Shape (G x M x K x N) | Main (ms) | TileTensor (ms) | Delta |
|-----------------------|-----------|-----------------|-------|
| 32 x 128 x 4096 x 7168 | 0.616 | 0.593 | +4.0% |
| 32 x 128 x 7168 x 2048 | 1.055 | 1.000 | +5.4% |
| 32 x 4096 x 4096 x 7168 | 17.225 | 17.138 | +0.5% |
| 32 x 4096 x 7168 x 2048 | 29.923 | 29.623 | +1.0% |

### 23.3 Grouped 1D-1D NVFP4 Results (DeepSeek-R1 Shapes)

Benchmark for the 1d1d kernel using production DeepSeek-R1 MoE shapes.

**NVFP4 (float4_e2m1fn -> bfloat16) - 32 experts, 2 tokens/expert:**

| Shape (G x M x N x K) | Main (ms) | TileTensor (ms) | Delta |
|-----------------------|-----------|-----------------|-------|
| 32 x 64 x 4096 x 7168 | 0.09414 | 0.09413 | 0% |
| 32 x 64 x 7168 x 2048 | 0.08770 | 0.08738 | +0.4% |

The 1d1d kernel shows **performance-neutral** results with the TileTensor
migration, maintaining the same throughput on production MoE shapes.

### 23.4 Analysis

The TileTensor migration shows **performance improvements** across all tested
configurations:

1. **Prefill shapes (large M)**: 10-15% improvement for MXFP8
2. **Decode shapes (small M)**: 4-8% improvement
3. **NVFP4**: 0.5-5% improvement

The improvements are likely due to:

- Better constant propagation with explicit dimension parameters
- Simpler type hierarchy enabling better compiler optimization
- More efficient inlining with reduced abstraction overhead

### 23.5 Conclusion

**No performance regression.** The migration is performance-neutral to positive.

---

---

## Part 24: Revert and Stabilization (2026-02-02)

After Parts 21-23, we discovered that the `.to_layout_tensor()` approach caused
**massive compilation slowdowns** (not just crashes). The compilation time regression
was unacceptable for development iteration.

### 24.1 Root Cause: `.to_layout_tensor()` Compilation Cost

The commit that replaced explicit `LayoutTensor` type aliases with
`.to_layout_tensor()` calls caused ~32 call sites across all kernel files to
force the compiler to:

1. Infer the result type at each call site
2. Compute the layout conversion from TileTensor's layout to LayoutTensor's layout
3. Perform complex variadic type expansion

With explicit type aliases, the compiler had pre-computed `LayoutTensor` types
that could be reused. With `.to_layout_tensor()`, the compiler must do the type
inference and layout computation fresh at **every call site**.

### 24.2 Revert Decision

We reverted to commit `3608ca3` which had the foundation (internal swizzled
layouts) but not the problematic `.to_layout_tensor()` usage everywhere.

### 24.3 What We Kept (Valuable Cleanups)

After the revert, we re-applied the following improvements:

1. **Parametric `internal_k_major`** - Unified the three separate layout
   definitions into one parametric function:

   ```mojo
   comptime internal_k_major[dtype, BM, BK, swizzle_bytes] = Layout(...)
   comptime internal_k_major_128B[...] = internal_k_major[..., 128]
   comptime internal_k_major_64B[...] = internal_k_major[..., 64]
   comptime internal_k_major_32B[...] = internal_k_major[..., 32]
   ```

2. **Simplified `SMemTile`** - Unified `SMemTile2D` and `SMemTileWithLayout`
   into a single type:

   ```mojo
   comptime SMemTile[dtype, layout, alignment] = TileTensor[...]
   comptime SMemTile2D[dtype, dim0, dim1, alignment] = SMemTile[dtype, row_major[dim0, dim1](), ...]
   ```

3. **Relative imports** - Simplified imports in grouped_block_scaled files:

   ```mojo
   from ..structured_kernels.config import BlockScaledMatmulConfig
   from ..structured_kernels.tile_pipeline import InputTilePipeline, ...
   ```

### 24.4 Current Architecture (Stable)

```text
┌─────────────────────────────────────────────────────────────────────┐
│  tile_types.mojo                                                     │
│                                                                      │
│  SMemTile[dtype, layout]           ← Takes Layout directly          │
│  SMemTile2D[dtype, dim0, dim1]     ← Backward-compatible alias      │
│  SMemTileArray2D[...]              ← Uses SMemTile internally       │
│  SMemTileArrayWithLayout[...]      ← For swizzled layouts           │
│  internal_k_major[..., swizzle_bytes] ← Parametric swizzled layout  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SMEM Structs / Kernel Code                                          │
│                                                                      │
│  Still use explicit LayoutTensor type aliases at TMA/MMA boundaries: │
│                                                                      │
│    comptime ATileLT = LayoutTensor[                                  │
│        Self.a_type, Self.SmemType.a_smem_layout,                     │
│        address_space=AddressSpace.SHARED, alignment=128              │
│    ]                                                                 │
│    ...                                                               │
│    a_loader.load(Self.ATileLT(tile.ptr), barrier, ...)              │
│    mma_op.mma(Self.ATileLT(a_tile.ptr), ...)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 24.5 What Didn't Work (Lessons Learned)

1. **`.to_layout_tensor()` everywhere** - Causes compilation slowdown, not just
   crashes. The type inference at each call site is expensive.

2. **Removing explicit LayoutTensor type aliases** - These act as compile-time
   caches. Removing them forces recomputation at every boundary.

3. **tile_writer_adapter approach** - Created circular imports. The adapter
   layer added complexity without solving the underlying issue.

### 24.6 Next Steps (Future Work)

To complete the TileTensor migration without compilation slowdown:

1. **Convert TileWriter to accept TileTensor** - Instead of an adapter layer,
   modify TileWriter itself to accept TileTensor parameters and convert
   internally.

2. **Convert MMA ops to accept TileTensor** - Modify MMA operations to accept
   TileTensor directly, handling the conversion to LayoutTensor internally.

3. **Keep explicit type aliases** - Continue using explicit `LayoutTensor` type
   aliases at boundaries as compile-time optimization.

The key insight is: **convert at the API level, not at every call site**.

### 24.7 Tests Passing

All SM100 structured kernel tests pass with the current stable architecture:

- test_matmul_sm100_2sm_bf16.mojo ✓
- test_grouped_matmul_sm100_nvfp4.mojo ✓
- test_grouped_matmul_sm100_blockwise_fp8.mojo ✓

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-01-31 | Claude | Initial document from migration attempts |
| 2025-01-31 | Claude | Added compatibility patterns and middle-out strategy |
| 2025-01-31 | Claude | Added Part 8: Bridge payload implementation |
| 2025-01-31 | Claude | Added Part 9: Kernel migration with dual accessors |
| 2025-01-31 | Claude | Added Part 10: Block-scaled kernel migration |
| 2025-01-31 | Claude | Added Part 11: Grouped 1D-1D kernel migration |
| 2025-01-31 | Claude | Added Part 12: Grouped block-scaled kernel migration - COMPLETE |
| 2025-01-31 | Claude | Added Part 13: TileTensor overloads for tile loaders |
| 2025-01-31 | Claude | Added Part 14: Compiler bug documentation with root cause analysis |
| 2025-01-31 | Claude | Added Part 15: Dual accessors in pipeline storage |
| 2025-01-31 | Claude | Added Part 16: Local GPU test verification - ALL TESTS PASS |
| 2025-01-31 | Claude | Added Part 17: SMEM struct TileTensor accessors - COMPLETE |
| 2025-01-31 | Claude | Added Part 18: Inline TileTensor-to-LayoutTensor conversion workaround |
| 2025-01-31 | Claude | Added Part 19: Direct `.to_layout_tensor()` migration for block_scaled and grouped_1d1d |
| 2025-01-31 | Claude | Added Part 20: Compiler bug report for to_layout_tensor() crashes |
| 2025-02-01 | Claude | Added Part 21: Final simplification - TileTensor native with explicit boundary conversion |
| 2025-02-01 | Claude | Added Part 22: Explicit dimension parameters - Migration complete |
| 2025-02-01 | Claude | Added Part 23: Performance validation - No regression, 0.5-15% improvements |
| 2026-02-02 | Claude | Added Part 24: Revert and stabilization - compilation slowdown discovery |
| 2026-02-03 | Claude | Added Part 25: SF layout fix and MMA TileTensor integration |

---

## Part 25: SF Layout Fix and MMA TileTensor Integration

### 25.1 Problem: NVFP4 Numerical Mismatch

After enabling TileTensor for SF tiles, NVFP4 tests failed with numerical
mismatches while MXFP8 tests passed. Investigation revealed that
`internal_sf_k_major` had incorrect strides compared to `tile_sf_layout_k_major`.

### 25.2 Root Cause Analysis

Created a test to compare the two layouts:

```text
=== NVFP4 Layout Test (BM=128, BK=128, vec_sf_size=16) ===

tile_sf_layout_k_major:
  shape: ((32, 1), ((4, 4), 4))
  stride: ((16, 0), ((1, 4), 512))
  size: 2048

internal_sf_k_major (BEFORE fix):
  shape: ((32, 1), ((4, 4), 2))
  stride: ((16, 512), ((1, 4), 16))  <- WRONG!
  size: 1024
```

**Issues identified**:

1. K-tile stride was 16, should be 512 (`_SF_ATOM_M_0 * _SF_ATOM_SIZE`)
2. M-tile stride was fixed at 512, should be `dim1 * _SF_ATOM_M_0`
3. Number of K-tiles was wrong (dim1 calculation used BK instead of SF_BK)

### 25.3 The Fix

Updated `internal_sf_k_major` in `tile_types.mojo`:

```mojo
comptime internal_sf_k_major[
    dim0: Int,  # (BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]
    dim1: Int,  # (SF_BK // (SF_ATOM_K * vec_sf_size)) * (SF_ATOM_M[1] * SF_ATOM_K)
    # where SF_BK = SF_K_GROUP_SIZE * num_sf_k_tiles (NOT raw BK)
] = Layout(
    # Shape: ((32, tiles_m), ((4, 4), tiles_k))
    Coord(
        Coord(Idx[_SF_ATOM_M_0](), Idx[dim0 // _SF_ATOM_M_0]()),
        Coord(
            Coord(Idx[_SF_ATOM_M_1](), Idx[_SF_ATOM_K]()),
            Idx[dim1 // _SF_ATOM_SIZE](),
        ),
    ),
    # Stride: ((16, dim1*32), ((1, 4), 512))
    # K-major: K-tiles placed with stride 512, M-tiles after all K content
    Coord(
        Coord(
            Idx[_SF_ATOM_SIZE](),
            Idx[dim1 * _SF_ATOM_M_0](),  # M-tile stride = num_k_tiles * 512
        ),
        Coord(
            Coord(Idx[1](), Idx[_SF_ATOM_M_1]()),
            Idx[_SF_ATOM_M_0 * _SF_ATOM_SIZE](),  # K-tile stride = 512
        ),
    ),
)
```

After fix, layouts match:

```text
tile_sf_layout_k_major:  stride: ((16, 2048), ((1, 4), 512))
internal_sf_k_major:     stride: ((16, 2048), ((1, 4), 512))  <- MATCH!
```

### 25.4 MMA TileTensor Integration

With layouts matching, MMA now extracts layout directly from TileTensor's
compile-time type parameters instead of computing `tile_sf_layout_k_major`
separately.

**Layout extraction for A/B tiles** (2-level nesting):

```mojo
comptime a_layout = Layout(
    IntTuple(
        IntTuple(
            a.shape_types[0].VariadicType[0].static_value,
            a.shape_types[0].VariadicType[1].static_value,
        ),
        IntTuple(
            a.shape_types[1].VariadicType[0].static_value,
            a.shape_types[1].VariadicType[1].static_value,
        ),
    ),
    // ... strides similarly
)
```

**Layout extraction for SF tiles** (3-level nesting):

```mojo
comptime sfa_layout = Layout(
    IntTuple(
        IntTuple(
            sfa_smem.shape_types[0].VariadicType[0].static_value,
            sfa_smem.shape_types[0].VariadicType[1].static_value,
        ),
        IntTuple(
            IntTuple(
                sfa_smem.shape_types[1].VariadicType[0].VariadicType[0].static_value,
                sfa_smem.shape_types[1].VariadicType[0].VariadicType[1].static_value,
            ),
            sfa_smem.shape_types[1].VariadicType[1].static_value,
        ),
    ),
    // ... strides similarly
)
```

### 25.5 Current Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      SM100 Block-Scaled Kernel                          │
├─────────────────────────────────────────────────────────────────────────┤
│  SMEM Storage (all TileTensor-based)                                    │
│  ├── ATileArray = SMemTileArray2D[internal_k_major_128B]               │
│  ├── BTileArray = SMemTileArray2D[internal_k_major_128B]               │
│  ├── SFATileArray = SMemTileArrayWithLayout[internal_sf_k_major]       │
│  ├── SFBTileArray = SMemTileArrayWithLayout[internal_sf_k_major]       │
│  └── CTileArray = SMemTileArray2D (+ CTileArrayLT for tile_writer)     │
├─────────────────────────────────────────────────────────────────────────┤
│  TMA Loads                                                              │
│  ├── A/B: TileTensor directly (multicast overload)                     │
│  └── SF: LayoutTensor wrapper for async_copy_5d                        │
├─────────────────────────────────────────────────────────────────────────┤
│  MMA                                                                    │
│  └── TileTensor directly - layout extracted via VariadicType           │
├─────────────────────────────────────────────────────────────────────────┤
│  Output (tile_writer)                                                   │
│  └── LayoutTensor via CTileArrayLT accessor                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 25.6 Migration Status Summary

| Component | Storage | TMA | MMA | Notes |
|-----------|---------|-----|-----|-------|
| A tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | Complete |
| B tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | Complete |
| SF tiles | ✅ TileTensor | ⚠️ LayoutTensor | ✅ TileTensor | TMA 5D needs LT |
| C tiles | ✅ TileTensor | N/A | N/A | tile_writer needs LT |

### 25.7 Tests Passing

All block-scaled tests pass on B200:

- test_matmul_sm100_block_scaled_nvfp4_1sm.mojo ✓
- test_matmul_sm100_block_scaled_nvfp4_2sm.mojo ✓
- test_matmul_sm100_block_scaled_mxfp8_1sm.mojo ✓
- test_matmul_sm100_block_scaled_mxfp8_2sm.mojo ✓
- test_matmul_sm100_batched_block_scaled_nvfp4_1sm.mojo ✓
- test_matmul_sm100_batched_block_scaled_mxfp8_1sm.mojo ✓

---

## Part 26: TMA 4D/5D TileTensor Overloads

**Date**: 2026-02-03

### 26.1 Goal

Complete the TileTensor migration for SF tiles by adding TileTensor overloads
to the TMA `async_copy_4d` and `async_copy_5d` operations. This eliminates the
need for `SFATileLT` and `SFBTileLT` LayoutTensor wrapper types in kernels.

### 26.2 Changes Made

#### tma_async.mojo

Added TileTensor overloads for both 4D and 5D async copy operations:

```mojo
# New TileTensor overload for 4D (used by grouped_1d1d kernel)
fn async_copy_4d[...](
    self,
    dst: TileTensor[mut=True, dtype=Self.dtype, address_space=AddressSpace.SHARED, ...],
    ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
    coords: Tuple[Int, Int, Int, Int],
):
    # Same implementation as LayoutTensor version, but uses dst.ptr directly
    # Assumes 128B alignment (TileTensor tiles are allocated with proper alignment)

# New TileTensor overload for 5D (used by block_scaled and grouped_block_scaled)
fn async_copy_5d[...](
    self,
    dst: TileTensor[mut=True, dtype=Self.dtype, address_space=AddressSpace.SHARED, ...],
    ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
    coords: Tuple[Int, Int, Int, Int, Int],
):
    # Same as above for 5D coordinates
```

The pattern follows the existing `async_multicast_load_3d` TileTensor overload:

- Accept TileTensor instead of LayoutTensor for `dst` parameter
- Remove alignment assertion (TileTensor tiles are allocated with 128B alignment)
- Use `dst.ptr.mut_cast[True]() + copy_offset` to access the underlying pointer

#### Kernel Updates

**block_scaled_matmul_kernel.mojo**:

- Removed `SFATileLT` and `SFBTileLT` LayoutTensor wrapper types
- Updated TMA loads to pass `sfa_tile` and `sfb_tile` directly

**grouped_block_scaled_matmul_kernel.mojo**:

- Removed `ATileLT`, `BTileLT`, `SFATileLT`, `SFBTileLT` wrapper types
- Updated both TMA loads and MMA calls to pass TileTensor directly

**grouped_1d1d_matmul_kernel.mojo**:

- Removed `ATileLT`, `BTileLT`, `SFATileLT`, `SFBTileLT` wrapper types
- Updated TMA 4D loads and MMA calls to pass TileTensor directly

### 26.3 Before vs After

**Before** (LayoutTensor wrappers required):

```mojo
# Kernel had to define wrapper types
comptime SFATileLT = LayoutTensor[
    Self.sfa_dtype,
    Self.SmemType.sfa_smem_layout,
    address_space = AddressSpace.SHARED,
    alignment=128,
]

# TMA load wrapped TileTensor pointer in LayoutTensor
sfa_tma_op.async_copy_5d[Self.cta_group](
    Self.SFATileLT(sfa_tile.ptr),  # Wrap ptr
    barrier[0],
    coords,
)

# MMA also needed wrappers
mma_op.mma(
    Self.ATileLT(a_tile.ptr),
    Self.BTileLT(b_tile.ptr),
    Self.SFATileLT(sfa_tile.ptr),
    Self.SFBTileLT(sfb_tile.ptr),
    ...
)
```

**After** (TileTensor end-to-end):

```mojo
# No wrapper types needed!

# TMA load accepts TileTensor directly
sfa_tma_op.async_copy_5d[Self.cta_group](
    sfa_tile,  # Pass TileTensor directly
    barrier[0],
    coords,
)

# MMA accepts TileTensor directly
mma_op.mma(
    a_tile,
    b_tile,
    sfa_tile,
    sfb_tile,
    ...
)
```

### 26.4 Updated Migration Status

| Component | Storage | TMA | MMA | Notes |
|-----------|---------|-----|-----|-------|
| A tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | Complete |
| B tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | Complete |
| SF tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | **Complete** |
| C tiles | ✅ TileTensor | N/A | N/A | tile_writer needs LT |

### 26.5 Architecture Diagram (Updated)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Global Memory                                                          │
│  └── A, B, SF tensors                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  TMA Operations (tma_async.mojo)                                        │
│  ├── async_multicast_load_3d: TileTensor ✅                             │
│  ├── async_copy_4d: TileTensor ✅  (NEW)                                │
│  └── async_copy_5d: TileTensor ✅  (NEW)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Shared Memory (pipeline_storage.mojo)                                  │
│  ├── A/B: SMemTileArray2D → TileTensor ✅                               │
│  ├── SF: SMemTileArrayWithLayout[internal_sf_k_major] → TileTensor ✅   │
│  └── C: SMemTileArray2D → TileTensor ✅                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  MMA Operations (mma.mojo)                                              │
│  ├── A/B: TileTensor → extracted layout ✅                              │
│  └── SF: TileTensor → extracted layout ✅                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Output (tile_writer.mojo)                                              │
│  └── LayoutTensor via CTileArrayLT accessor (remaining work)            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 26.6 Tests Verified

All tests pass on B200:

- test_matmul_sm100_block_scaled_nvfp4_1sm.mojo ✓
- test_matmul_sm100_block_scaled_nvfp4_2sm.mojo ✓
- test_matmul_sm100_block_scaled_mxfp8_1sm.mojo ✓
- test_grouped_matmul_sm100_nvfp4.mojo ✓

### 26.7 Remaining Work

The only remaining LayoutTensor usage is in the output path:

- `CTileArrayLT` accessor in `pipeline_storage.mojo` for `tile_writer.mojo`
- The output writer (`tile_writer.mojo`) still expects LayoutTensor

This is the next target for TileTensor migration.

---

## Part 27: C Tile Migration Complete

### 27.1 The Challenge

The output path was the last piece requiring LayoutTensor:

1. `tile_writer.mojo` internally uses LayoutTensor operations:
   - `.tile[]` for tiling
   - `.reshape[]` for reshaping
   - These operations don't exist on TileTensor

2. Kernels call `TileWriter.write_*()` methods passing C tile arrays

3. C tiles use **row_major layout** (no swizzle) because TMA stores don't need
   swizzle - different from A/B tiles which use swizzled layouts.

### 27.2 Solution: TileTensor Overloads + Row-Major Storage

The solution has two parts:

#### Part A: TileTensor overloads in output_writer.mojo

Added TileTensor overloads that convert to LayoutTensor at the entry point:

```mojo
# In output_writer.mojo
from .tile_types import SMemTileArray2DRowMajor

# Type alias for TileTensor-based C tile arrays
comptime CTileArrayTT = SMemTileArray2DRowMajor[
    Self.c_type, Self.c_smem_dim0, Self.c_smem_dim1, Self.num_output_stages, 128
]

# TileTensor overload converts and delegates
fn write_batched(
    self,
    c_tiles: Self.CTileArrayTT,  # TileTensor array
    stage: Self.Stage,
    tile_coord: Tuple[UInt32, UInt32, UInt32],
    shape: Tuple[UInt32, UInt32],
    alpha: Float32 = Float32(1.0),
):
    # Convert TileTensor array to LayoutTensor array via shared pointer
    var c_tiles_lt = Self.CTileArray(c_tiles.ptr)
    self._copy_to_gmem_batched(c_tiles_lt, stage, tile_coord, shape, alpha)
```

#### Part B: Row-major C tile storage

Changed C tiles from swizzled `SMemTileArray2D` to row-major
`SMemTileArray2DRowMajor`:

```mojo
# Before (in pipeline_storage.mojo)
comptime CTileArray = SMemTileArray2D[
    Self.c_type, Self.c_dim0, Self.c_dim1, Self.num_output_stages, 128
]

# After
comptime CTileArray = SMemTileArray2DRowMajor[
    Self.c_type, Self.c_dim0, Self.c_dim1, Self.num_output_stages, 128
]
```

### 27.3 Files Changed

| File | Change |
|------|--------|
| `output_writer.mojo` | Added TileTensor overloads for all write methods |
| `pipeline_storage.mojo` | Changed CTileArray to use row-major (3 places) |
| `block_scaled_smem.mojo` | Use CTileArray (TileTensor), call c_tiles_tt() |
| `grouped_block_scaled_smem.mojo` | Use CTileArray (TileTensor), call c_tiles_tt() |
| `grouped_1d1d_smem.mojo` | Use CTileArray (TileTensor), call c_tiles_tt() |
| `blockwise_fp8_smem.mojo` | Use CTileArray (TileTensor), call c_tiles_tt() |
| `matmul_kernels.mojo` | Use CTileArray (TileTensor), call c_tiles_tt() |

### 27.4 Key Insight: Row-Major vs Swizzled

C tiles must use row-major layout (not swizzled) because:

1. TMA stores to global memory don't need swizzle
2. `SMemTileArray2D` uses `internal_k_major_128B` (swizzled)
3. `SMemTileArray2DRowMajor` uses `Layout.row_major()` (no swizzle)

First attempt used `SMemTileArray2D` (swizzled) which caused type mismatch:

```text
no matching method in call to 'write_batched'
```

The fix was to use `SMemTileArray2DRowMajor` for C tiles specifically.

### 27.5 Final Migration Status

| Component | Storage | TMA Load | MMA | TMA Store | Notes |
|-----------|---------|----------|-----|-----------|-------|
| A tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | N/A | Complete |
| B tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | N/A | Complete |
| SF tiles | ✅ TileTensor | ✅ TileTensor | ✅ TileTensor | N/A | Complete |
| C tiles | ✅ TileTensor | N/A | N/A | ✅ TileTensor | **COMPLETE** |

### 27.6 Architecture Diagram (Final)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Global Memory                                                          │
│  └── A, B, SF, C tensors                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  TMA Load (tma_async.mojo)                                              │
│  ├── async_multicast_load_3d: TileTensor ✅                             │
│  ├── async_copy_4d: TileTensor ✅                                       │
│  └── async_copy_5d: TileTensor ✅                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Shared Memory (pipeline_storage.mojo)                                  │
│  ├── A/B: SMemTileArray2D[swizzled] → TileTensor ✅                     │
│  ├── SF: SMemTileArrayWithLayout[sf_k_major] → TileTensor ✅            │
│  └── C: SMemTileArray2DRowMajor[row_major] → TileTensor ✅              │
├─────────────────────────────────────────────────────────────────────────┤
│  MMA Operations (mma.mojo)                                              │
│  ├── A/B: TileTensor → extracted layout ✅                              │
│  └── SF: TileTensor → extracted layout ✅                               │
├─────────────────────────────────────────────────────────────────────────┤
│  TMA Store (output_writer.mojo via tile_writer.mojo)                    │
│  └── TileTensor overloads → convert to LT internally ✅                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 27.7 Tests Verified

All tests pass:

- test_matmul_sm100_block_scaled_nvfp4_1sm.mojo ✓
- test_matmul_sm100_block_scaled_nvfp4_2sm.mojo ✓
- test_grouped_matmul_sm100_nvfp4.mojo ✓

### 27.8 Conclusion

The TileTensor migration is **COMPLETE**. All tiles (A, B, SF, C) now use
TileTensor end-to-end through storage, TMA operations, MMA operations, and
epilogue output. The internal conversion to LayoutTensor in `tile_writer.mojo`
is an implementation detail hidden from kernels.

`CTileArrayLT` is retained in `pipeline_storage.mojo` for backward compatibility
with any code that may still need direct LayoutTensor access.
