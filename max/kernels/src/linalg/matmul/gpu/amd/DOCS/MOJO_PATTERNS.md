<!-- markdownlint-disable MD013 MD036 MD040 MD052 -->

# Mojo Programming Patterns for GPU Kernels

This document captures Mojo programming patterns, idioms, and environment
tips discovered during kernel development. Reference this for future
development.

---

## 1. Generic Type Patterns

### 1.1 Parameterized Type Aliases

Mojo allows type aliases that depend on compile-time parameters:

```mojo
# Define a generic tile type parameterized by row count
alias HalfTile[rows: Int] = SMemTileType[in_type, Layout.row_major(rows, BK)]

# Use it in function signatures
fn _load_tile_to_lds[
    half_data_rows: Int, //,  # Note: // makes it inferred
    which: Int,
](
    self,
    resource: AMDBufferResource,
    dst_tile: Self.HalfTile[half_data_rows],  # Generic over rows!
):
    ...
```

**Key insight**: The `//` after a parameter makes it **inferred** from
arguments, so you don't need to specify it:

```mojo
# Caller doesn't need to specify half_data_rows - inferred from tile type
self._load_tile_to_lds[which=which](resource, self.a_load_tiles[stage][which])
```

### 1.2 Unifying Functions with Different Tile Types

**Problem**: Two functions doing the same thing but with different tile types.

**Solution**: Use a parameterized type alias:

```mojo
# Before: Two nearly-identical functions
fn _load_tile_to_lds_a(self, dst_tile: Self.AHalfTile): ...
fn _load_tile_to_lds_b(self, dst_tile: Self.BHalfTile): ...

# After: One generic function
alias HalfTile[rows: Int] = SMemTileType[in_type, Layout.row_major(rows, BK)]

fn _load_tile_to_lds[half_data_rows: Int, which: Int](
    self, dst_tile: Self.HalfTile[half_data_rows]
): ...
```

---

## 2. Layout and RuntimeLayout Patterns

### 2.1 Compile-Time Offset Computation

Use `RuntimeLayout` to compute offsets at compile time when the layout is
fully known:

```mojo
from layout import Layout, RuntimeLayout, IntTuple

# Define layout with shape and stride
alias mma_access_layout = Layout(
    IntTuple(16, 4),   # Shape: decompose index as col % 16, row // 16
    IntTuple(32, 8)    # Stride: col * 32 + row * 8
)

# RuntimeLayout evaluates at compile time - no GPU heap allocation!
var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))
```

**When to use**:

- `Layout(idx)` - compile-time only (can cause heap allocation on GPU)
- `RuntimeLayout[layout]()(idx)` - safe for GPU, compile-time when static

### 2.2 LayoutTensor.distribute for Thread Mapping

Map thread IDs to tensor positions:

```mojo
alias thread_layout = Layout.row_major(16, 4)  # 64 threads as 16×4 grid

# distribute() maps thread index to tensor element
var dist_tensor = subtile_tensor.vectorize[1, load_width]()
    .distribute[thread_layout](UInt(effective_lane))

# Extract offset from pointer difference
var offset = (Int(dist_tensor.ptr) - Int(subtile_tensor.ptr)) // elem_size
```

### 2.3 Tile-Based Addressing

Pass tiles to functions, derive pointers inside:

```mojo
fn load_tile(self, dst_tile: LayoutTensor):
    # Compute warp's subtile via tile indexing
    var warp_subtile = dst_tile.tile[rows_per_warp, cols](tile_idx, 0)
    
    # Get scalar pointer for hardware instruction
    var ptr = readfirstlane(warp_subtile.ptr)
```

### 2.4 Compile-Time Evaluation with `eval[]`

GPU kernels cannot heap-allocate. When accessing layout properties that
might not be compile-time known, use `eval[]` to force compile-time
evaluation:

```mojo
# Define helper alias (in structuring.mojo)
comptime eval[T: AnyType, //, val: T] = val

# Use in __init__ to derive stride from layout
fn __init__[layout: Layout](
    out self,
    src: LayoutTensor[_, layout, *_, **_],
    ...
):
    # BAD: May cause heap allocation on GPU
    # self.stride = src.layout.shape[1].value()
    
    # GOOD: eval[] forces compile-time evaluation
    alias stride_value = src.layout.shape[1].value()
    self.stride = stride_value
    
    # Alternative: inline with eval[]
    self.stride = eval[src.layout.shape[1].value()]
```

**Key insight**: `eval[]` is a compile-time alias that returns its argument
unchanged, but forces the compiler to evaluate the expression at compile
time rather than runtime.

### 2.5 GPU-Safe Buffer Creation with `make_amd_buffer_resource`

AMD kernels need `AMDBufferResource` for global memory access. Use the
helper function to create buffers safely on GPU:

```mojo
from layout._utils import make_amd_buffer_resource

# In TileLoader.__init__
fn __init__(out self, src: LayoutTensor, ...):
    # GPU-safe: uses readfirstlane internally
    self.buffer = make_amd_buffer_resource(src)
    
    # What it does internally:
    # var ptr = src.ptr
    # var size = _get_bounds(src)
    # return AMDBufferResource(readfirstlane(ptr), readfirstlane(size))
```

**Why it works**: `readfirstlane()` broadcasts a scalar value to all lanes,
converting potentially lane-divergent values to uniform scalars that don't
require heap allocation.

### 2.6 Parameter Inference for Layouts

Mojo's parameter inference lets you extract compile-time values from
argument types automatically. See the
[docs](https://docs.modular.com/mojo/manual/parameters/#parameter-inference).

**Pattern 1: Infer-only parameters with `//`**

Use `//` to mark parameters as infer-only—they're extracted from arguments,
not passed explicitly:

```mojo
struct Buffers[
    in_type: DType,
    a_layout: Layout,
    b_layout: Layout, //,  # <-- infer-only: extracted from __init__ args
    BM: Int,
    ...
]:
    # K derived at compile time from inferred layout
    alias K = Self.a_layout.shape[1].value()

    fn __init__(
        out self,
        # Layouts inferred from these tensor arguments
        a: LayoutTensor[Self.in_type, Self.a_layout, *_, **_],
        b: LayoutTensor[_, Self.b_layout, *_, **_],
        ...
    ):
        ...
```

#### Pattern 2: Struct parameter on TileLoader

Pass layout as a struct parameter to derive stride at compile time:

```mojo
struct TileLoaderLDS[
    dtype: DType,
    src_layout: Layout,  # Full tensor layout
    ...
]:
    # Stride as compile-time alias (no runtime storage)
    alias stride = Self.src_layout.shape[1].value()
```

#### Pattern 3: Parameterized type alias

Create type aliases parameterized on layout:

```mojo
alias TileLoader[src_layout: Layout] = TileLoaderLDS[
    Self.in_type,
    src_layout,
    Self.half_tile_layout,
    ...
]
alias ATileLoader = Self.TileLoader[Self.a_layout]
alias BTileLoader = Self.TileLoader[Self.b_layout]
```

**Benefits:**

- No runtime `K` argument needed—derived from layout
- Compile-time constants enable optimizations
- Type-safe: tensor layout checked at compile time

---

## 3. Swizzle Patterns

### 3.1 Using the Swizzle Class

```mojo
from layout.swizzle import Swizzle

# Swizzle(bits, base, shift): XOR bits from position (base+shift) into position base
alias byte_swizzle = Swizzle(1, 5, 4)   # XOR bit 9 into bit 5
alias elem_swizzle = Swizzle(1, 4, 4)   # XOR bit 8 into bit 4

# Apply swizzle
var swizzled = byte_swizzle(byte_offset)
```

### 3.2 Deriving Swizzle from Tile Geometry

The swizzle pattern breaks LDS bank conflicts by XORing row bits into
column bits. Parameters are derived from the **loading thread layout**
(owned by TileBuffers):

```mojo
from stdlib.bit import log2_floor

# Loading thread layout determines subtile dimensions
# For 16×4 thread layout, each thread loads load_width elements:
alias subtile_rows = 16                    # Thread layout rows
alias subtile_cols = 4 * load_width        # 4 thread cols × SIMD width

# Swizzle formula:
#   elem_base = log2(subtile_cols // 2)    (distinguishes subtile halves)
#   byte_base = elem_base + log2(elem_size) (same pattern in byte space)
#   shift = log2(subtile_rows)             (covers all rows in subtile)

alias elem_size = size_of[in_type]()
alias swizzle_elem_base = log2_floor(subtile_cols // 2)
alias swizzle_byte_base = swizzle_elem_base + log2_floor(elem_size)
alias swizzle_shift = log2_floor(subtile_rows)

# Buffers uses byte_swizzle for global→LDS writes
alias byte_swizzle = Swizzle(1, swizzle_byte_base, swizzle_shift)

# MmaOp receives elem_base and shift as parameters, creates elem_swizzle
alias elem_swizzle = Swizzle(1, swizzle_elem_base, swizzle_shift)
```

### 3.3 Architectural Separation

The swizzle configuration follows a clear ownership model:

| Component | Owns | Receives |
|-----------|------|----------|
| **Kernel** | Loading thread layout (16×4) | - |
| **Buffers** | byte_swizzle (for writing) | load_width, enable_swizzle |
| **MmaOp** | elem_swizzle (for reading) | swizzle_elem_base, swizzle_shift |

```mojo
# Kernel computes swizzle params from loading pattern
alias swizzle_subtile_rows = 16
alias swizzle_subtile_cols = 4 * load_width
alias swizzle_elem_base = log2_floor(swizzle_subtile_cols // 2)
alias swizzle_shift = log2_floor(swizzle_subtile_rows)

# Pass to MmaOp (consumer receives parameters, doesn't compute them)
alias MmaOpType = MmaOp[..., swizzle_elem_base, swizzle_shift]
```

### 3.4 Example: bf16 with load_width=8

| Parameter | Value | Derivation |
|-----------|-------|------------|
| subtile_rows | 16 | Thread layout |
| subtile_cols | 32 | 4 × 8 |
| elem_size | 2 | sizeof(bf16) |
| swizzle_elem_base | 4 | log2(32 / 2) = log2(16) |
| swizzle_byte_base | 5 | 4 + log2(2) |
| swizzle_shift | 4 | log2(16) |
| **byte_swizzle** | `Swizzle(1, 5, 4)` | For global→LDS |
| **elem_swizzle** | `Swizzle(1, 4, 4)` | For LDS→reg |

### 3.5 Why These Parameters?

**AMD LDS Bank Structure**: 64 banks × 4 bytes = 256 bytes/cycle

**Without swizzle**: Threads in the same MMA quadrant access elements with
the same column offset, causing bank conflicts (4-way for 4×16 MMA pattern).

**With swizzle**: XORing bit (base+shift) into bit base means:

- `base` determines which bit position in the address gets modified
- `shift` determines how far "up" we look for the XOR source
- Together they break the conflict pattern by making row position affect
  bank selection

**Coordination requirement**: Both loading (byte_swizzle) and reading
(elem_swizzle) must use the same pattern. They differ only in base
(byte vs element indexing).

### 3.6 Other Swizzle Patterns

| Pattern | Use | Notes |
|---------|-----|-------|
| `Swizzle(1, 5, 4)` | Byte offsets | 16×32 subtile, bf16 |
| `Swizzle(1, 4, 4)` | Element offsets | Same pattern |
| `Swizzle(3, 0, 1)` | matmul.mojo | 3 bits |

---

## 4. GPU Intrinsics and Synchronization

### 4.1 AMD-Specific Intrinsics

```mojo
from gpu.sync import s_waitcnt, schedule_barrier
from sys.intrinsics import readfirstlane, llvm_intrinsic

# Wait for N or fewer operations in flight
s_waitcnt[vmcnt=0]()      # Global memory operations
s_waitcnt[lgkmcnt=0]()    # LDS operations

# Workgroup barrier (control-flow only, NOT memory fence)
fn s_barrier():
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

# Workgroup barrier WITH memory fences
from gpu.sync import barrier
barrier()  # Includes RELEASE + ACQUIRE fences

# Prevent compiler reordering
schedule_barrier()
```

### 4.2 Direct Global→LDS Transfer

```mojo
from gpu.intrinsics import AMDBufferResource

# Create buffer resource from tensor pointer
var resource = AMDBufferResource(tensor.ptr)

# Direct transfer bypassing VGPRs
resource.load_to_lds[width=8](
    buf_offset,           # Per-thread global offset
    smem_ptr,             # Destination LDS pointer (must be scalar/SGPR)
    scalar_offset=offset  # Additional scalar offset
)
```

### 4.3 Scalar Pointer for Hardware Instructions

```mojo
# readfirstlane broadcasts lane 0's value to all lanes (creates SGPR)
var scalar_ptr = readfirstlane(warp_subtile.ptr)

# Required for load_to_lds which needs uniform pointer across warp
resource.load_to_lds[width=8](offset, scalar_ptr, ...)
```

---

## 5. Struct and Alias Patterns

### 5.1 Nested Type Aliases

```mojo
struct Buffers[in_type: DType, BM: Int, BN: Int, BK: Int, ...]:
    # Type aliases that depend on struct parameters
    alias SMemTile[rows: Int, cols: Int] = SMemTileType[
        in_type, Layout.row_major(rows, cols), alignment=alignment
    ]
    
    alias AHalfTile = Self.SMemTile[Self.half_BM, Self.BK]
    alias BHalfTile = Self.SMemTile[Self.half_BN, Self.BK]
    
    # Tuple types for indexing: [stage][which]
    alias AHalfTilePair = Tuple[Self.AHalfTile, Self.AHalfTile]
    var a_load_tiles: Tuple[Self.AHalfTilePair, Self.AHalfTilePair]
```

### 5.2 Constrained Parameters

```mojo
fn __init__(out self):
    # Compile-time assertions
    constrained[Self.WM % Self.MMA_M == 0]()
    constrained[Self.BK % Self.MMA_K == 0]()
    constrained[(Self.MMA_M * Self.MMA_N) % WARP_SIZE == 0]()
```

---

## 6. Tile Type System Architecture

### 6.1 The Type Compatibility Problem

**Problem**: When two structs define types with the same layout but
different derivation paths, Mojo treats them as different types:

```mojo
struct MmaOp[WM: Int, BK: Int, ...]:
    alias ASMemMmaTile = SMemTileType[in_type, Layout.row_major(WM // 2, BK)]

struct Buffers[half_BM: Int, BK: Int, ...]:
    # Even though half_BM == WM // 2 at runtime, these are DIFFERENT types!
    alias AMmaTile = SMemTileType[in_type, Layout.row_major(half_BM, BK)]
```

**Solution**: Create a **shared source of truth** struct that both derive from.

### 6.2 MmaTileTypes: Shared Source of Truth

```mojo
struct MmaTileTypes[
    in_type: DType,
    WM: Int,        # From KernelConfig.warp_shape[0]
    WN: Int,        # From KernelConfig.warp_shape[1]
    BK: Int,        # From KernelConfig.block_shape[2]
    half_BN: Int,   # = BN // 2
    alignment: Int,
]:
    """Single source of truth for MMA tile types."""
    
    # Derived dimensions
    alias mma_tile_m = Self.WM // 2
    alias mma_tile_n = Self.WN // 2
    
    # Base tile type
    alias SMemTile[rows: Int, cols: Int] = SMemTileType[
        Self.in_type, Layout.row_major(rows, cols), alignment = Self.alignment
    ]
    
    # A tiles
    alias AHalfTile = Self.SMemTile[Self.WM, Self.BK]
    alias ASMemMmaTile = Self.AHalfTile.TileType[Self.mma_tile_m, Self.BK]
    
    # B tiles  
    alias BHalfTile = Self.SMemTile[Self.half_BN, Self.BK]
    alias BSMemMmaTile = Self.BHalfTile.TileType[Self.WN, Self.BK].TileType[
        Self.mma_tile_n, Self.BK
    ]
```

### 6.3 Entity Relationship Model

```text
┌─────────────────────────────────────────────────────────────────┐
│                       KernelConfig                               │
│  block_shape: [BM, BN, BK]                                       │
│  warp_shape:  [WM, WN, WK]                                       │
│  mma_shape:   [MMA_M, MMA_N, MMA_K]                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ derives parameters
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MmaTileTypes                                │
│  Parameters: in_type, WM, WN, BK, half_BN, alignment            │
│                                                                  │
│  Defines:                                                        │
│   ├── mma_tile_m, mma_tile_n (dimensions)                       │
│   ├── AHalfTile, BHalfTile (warp-region tiles)                  │
│   └── ASMemMmaTile, BSMemMmaTile (MMA-sized tiles)              │
└─────────────────────────────────────────────────────────────────┘
                    ▲                           ▲
                    │                           │
         uses TileTypes alias          uses TileTypes alias
                    │                           │
┌───────────────────┴───────┐     ┌─────────────┴───────────────┐
│          MmaOp            │     │          TileBuffers         │
│                           │     │                              │
│  TileTypes = MmaTileTypes │     │  TileTypes = MmaTileTypes   │
│  ASMemMmaTile (consumer)  │     │  AMmaTile (producer)        │
│  BSMemMmaTile (consumer)  │     │  BMmaTile (producer)        │
│                           │     │                              │
│  load_a(smem: ASMemMmaTile)│    │  a_mma_tiles: AMmaTilePair  │
│  load_b(smem: BSMemMmaTile)│    │  b_mma_tiles: BMmaTilePair  │
└───────────────────────────┘     └──────────────────────────────┘
         │                                      │
         │        Type Compatibility            │
         └──────────────────────────────────────┘
              Both derive from same MmaTileTypes
              → Types are identical at compile-time!
```

### 6.4 Using the Shared Types

```mojo
struct MmaOp[in_type: DType, WM: Int, WN: Int, BK: Int, half_BN: Int, ...]:
    # Create alias to shared type source
    alias TileTypes = MmaTileTypes[
        Self.in_type, Self.WM, Self.WN, Self.BK, Self.half_BN, Self.alignment
    ]
    
    # Use shared types
    alias ASMemMmaTile = Self.TileTypes.ASMemMmaTile
    alias BSMemMmaTile = Self.TileTypes.BSMemMmaTile
    
    # Now functions accept the EXACT same type as Buffers produces
    fn load_a[which: Int](self, smem_tile: Self.ASMemMmaTile): ...
    fn load_b[which: Int](self, smem_tile: Self.BSMemMmaTile): ...

struct Buffers[in_type: DType, WM: Int, WN: Int, BK: Int, BN: Int, ...]:
    alias half_BN = Self.BN // 2
    
    # Same TileTypes with same parameters
    alias TileTypes = MmaTileTypes[
        Self.in_type, Self.WM, Self.WN, Self.BK, Self.half_BN, Self.alignment
    ]
    
    # Types are IDENTICAL to MmaOp's types
    alias AMmaTile = Self.TileTypes.ASMemMmaTile
    alias BMmaTile = Self.TileTypes.BSMemMmaTile
```

### 6.5 Compile-Time Validation with Inferred Layouts

Use inferred layout parameters for powerful compile-time checks:

```mojo
@staticmethod
fn _load_fragment[
    smem_layout: Layout,
    smem_element_layout: Layout,
    frag_layout: Layout,
    frag_element_layout: Layout, //,  # All inferred from arguments!
](
    smem_tile: SMemTileType[
        dtype, smem_layout, element_layout=smem_element_layout, **_
    ],
    reg_frag: RegTileType[
        dtype, frag_layout, element_layout=frag_element_layout, **_
    ],
):
    alias num_iterations = frag_layout.size()
    alias frag_width = frag_element_layout.size()
    
    # Compile-time validation using inferred layouts
    alias smem_elements = smem_layout.size() * smem_element_layout.size()
    alias required_elements = num_iterations * WARP_SIZE * frag_width
    
    constrained[
        smem_elements >= required_elements,
        "smem tile too small for fragment load pattern",
    ]()
    constrained[
        smem_element_layout.size() == frag_element_layout.size(),
        "smem and frag element layouts must match",
    ]()
```

### 6.6 Vectorize Both Tiles for Consistent Element Layouts

```mojo
fn load_a[which: Int](self, smem_tile: Self.ASMemMmaTile):
    # Vectorize BOTH to match element layouts
    var smem_frag = smem_tile.vectorize[1, Self.load_width]()
    var reg_frag = self.a_reg_tile.tile[...](which, 0).vectorize[1, Self.load_width]()
    
    # Now smem_element_layout.size() == frag_element_layout.size() == load_width
    Self._load_fragment(smem_frag, reg_frag)
```

### 6.7 Key Insights

| Problem | Solution |
|---------|----------|
| Types with same values, different derivations | Shared `MmaTileTypes` |
| Producer/consumer type mismatch | Both derive from same source |
| Runtime type validation | Use `constrained[]` |
| Catching layout mismatches | Infer layouts in signature |

**Golden Rule**: When two structs need to exchange typed values, define
the types in a shared location that both can import.

---

## 7. Parameter Patterns

### 7.1 Inferred Parameters (`//`)

```mojo
# The // makes half_data_rows inferred from the argument type
fn _load_tile_to_lds[
    half_data_rows: Int, //,  # Inferred!
    which: Int,               # Must be specified
](self, dst_tile: Self.HalfTile[half_data_rows]):
    ...

# Caller only specifies 'which', half_data_rows comes from tile type
self._load_tile_to_lds[which=0](resource, tile)
```

### 7.2 Compile-Time Loops with @parameter

```mojo
@parameter
for i in range(num_iterations):
    # Loop is unrolled at compile time
    # Can use i in compile-time expressions
    alias offset = i * stride  # Compile-time constant
```

### 7.3 Conditional Compilation

```mojo
@parameter
if Self.enable_swizzle:
    # This code only exists if swizzle is enabled
    full_offset = elem_swizzle(iter_base + lane_offset)
else:
    full_offset = iter_base + lane_offset
```

---

## 8. Testing and Running

### 8.1 Running Kernel Tests

```bash
# Source the environment first
source ~/start-modular.sh

# Run specific test file
mojo -D use_vendor_blas=False max/kernels/test/gpu/linalg/test_ping_pong.mojo

# Run with output truncated
mojo -D use_vendor_blas=False max/kernels/test/gpu/linalg/test_ping_pong.mojo 2>&1 | tail -20

# Run AMD matmul tests
mojo max/kernels/test/gpu/linalg/test_matmul_amd.mojo
```

### 8.2 Common Test Patterns

```mojo
from testing import assert_equal, assert_almost_equal

fn test_kernel():
    # Setup
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    
    # Run kernel
    ctx.enqueue_function[kernel_fn](grid, block, args...)
    ctx.synchronize()
    
    # Verify
    assert_almost_equal(result, expected, rtol=1e-3)
```

### 8.3 Debug Tips

```mojo
# Print from first thread only (avoid spam)
if thread_idx.x == 0 and block_idx.x == 0:
    print("Debug:", value)

# Check for linter errors
# Run: read_lints tool on the file

# Enable stack traces
# Set: MOJO_ENABLE_STACK_TRACE_ON_ERROR=1
```

---

## 9. Type Unification and Casting

### 9.1 rebind for Type Unification

When the compiler can't prove two types are the same (even though they
will be at runtime):

```mojo
# Problem: smem_half_tile._dtype vs Self.in_type
# Mojo can't prove they're equal even though SMemTileType uses in_type

# Solution: Define concrete alias and rebind
alias _SharedPtr = UnsafePointer[Scalar[Self.in_type], address_space = AddressSpace.SHARED]

fn load(self, smem_half_tile: SMemTileType):
    # rebind tells compiler "trust me, these are the same type"
    self._helper(rebind[Self._SharedPtr](smem_half_tile.ptr))
```

### 9.2 bitcast for Pointer Type Conversion

Convert pointer element types without changing the underlying address:

```mojo
# Convert LayoutTensor's element pointer to SIMD pointer for indexed access
var frag = self.a_reg_tile.tile[...](which, 0).vectorize[1, load_width]()

# frag.ptr is UnsafePointer[SIMD[type, element_size]]
# We want UnsafePointer[SIMD[type, load_width]] for indexed stores
var frag_ptr = frag.ptr.bitcast[SIMD[Self.in_type, Self.load_width]]()
frag_ptr[i] = loaded_value
```

### 9.3 Address Space Annotations

GPU pointers live in different memory spaces:

```mojo
# Shared memory (LDS)
UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED]

# Register file (LOCAL)
UnsafePointer[SIMD[type, width], address_space = AddressSpace.LOCAL]

# Global memory (default/GENERIC)
UnsafePointer[Scalar[type]]  # Defaults to GENERIC

# Example: Helper function accepting both pointer types
fn _load_fragment[num_iterations: Int, type: DType](
    self,
    smem_ptr: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED],
    frag_ptr: UnsafePointer[SIMD[type, Self.load_width], address_space = AddressSpace.LOCAL],
):
    ...
```

### 9.4 Complete Pattern: Refactoring Duplicate Load Functions

```mojo
# Before: Two functions with nearly identical bodies
fn load_a[which: Int](self, tile: TileA): ... 
fn load_b[which: Int](self, tile: TileB): ...

# After: One helper + thin wrappers
alias _SharedPtr = UnsafePointer[Scalar[Self.in_type], address_space = AddressSpace.SHARED]

fn _load_fragment[num_iterations: Int, type: DType](
    self,
    smem_ptr: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED],
    frag_ptr: UnsafePointer[SIMD[type, Self.load_width], address_space = AddressSpace.LOCAL],
): ...

fn load_a[which: Int](self, smem_half_tile: SMemTileType):
    var frag = self.a_reg_tile.tile[...](which, 0).vectorize[...]()
    self._load_fragment[Self.half_m_mmas * Self.num_k_mmas, Self.in_type](
        rebind[Self._SharedPtr](smem_half_tile.ptr),
        frag.ptr.bitcast[SIMD[Self.in_type, Self.load_width]](),
    )
```

---

## 10. Common Gotchas

### 10.1 GPU Heap Allocation

**Problem**: `Layout(idx)` can cause "heap allocation not supported on GPU"

**Solution**: Use `RuntimeLayout`:

```mojo
# Bad - may allocate on GPU
var offset = some_layout(lane_id)

# Good - compile-time evaluation
var offset = Int(RuntimeLayout[some_layout]()(lane_id))
```

### 10.2 readfirstlane for Scalar Pointers

**Problem**: `load_to_lds` needs uniform (SGPR) pointer

**Solution**: Use `readfirstlane`:

```mojo
# Bad - pointer may be in VGPR (divergent)
resource.load_to_lds(offset, tile.ptr, ...)

# Good - broadcasts lane 0's value, creates SGPR
var scalar_ptr = readfirstlane(tile.ptr)
resource.load_to_lds(offset, scalar_ptr, ...)
```

### 10.3 Barrier vs s_barrier

| Function | Memory Fence | Use When |
|----------|--------------|----------|
| `s_barrier()` | No | Control-flow sync only |
| `barrier()` | Yes (release+acquire) | Need memory visibility |

### 10.4 vmcnt is Per-Wave

`s_waitcnt[vmcnt=N]()` only waits for **this wave's** operations, not the
whole workgroup. For cross-wave visibility:

```mojo
s_waitcnt[vmcnt=0]()  // My loads complete
s_barrier()            // Sync with other waves
// Now all data visible to all waves
```

---

## 11. File Organization

### 11.1 Kernel File Structure

```text
pingpong_kernel.mojo:
├── Imports and aliases
├── Header comments (architecture overview)
├── Helper functions (barriers, loads)
├── MmaOp struct (MMA operations)
├── Buffers struct (memory management)  
├── Main kernel function
└── Store results
```

### 11.2 Related Files

| File | Purpose |
|------|---------|
| `pingpong_kernel.mojo` | Main kernel |
| `matmul.mojo` | Standard AMD matmul |
| `test_ping_pong.mojo` | Tests |
| `PINGPONG_KERNEL_ARCHITECTURE.md` | Architecture doc |
| `AMD_PINGPONG_KERNEL_DESIGN.md` | Detailed design history |

---

## 12. Quick Reference

### Layout Creation

```mojo
Layout.row_major(rows, cols)           # Row-major
Layout.col_major(rows, cols)           # Column-major
Layout(IntTuple(shape), IntTuple(stride))  # Custom
```

### Common Intrinsics

```mojo
lane_id()                    # Lane within warp (0-63)
warp_id()                    # Warp within block
thread_idx.x                 # Thread within block
block_idx.x                  # Block within grid
readfirstlane(val)           # Broadcast lane 0's value
```

### Synchronization

```mojo
s_waitcnt[vmcnt=N]()         # Wait for global mem
s_waitcnt[lgkmcnt=N]()       # Wait for LDS
s_barrier()                  # Control-flow barrier
barrier()                    # Barrier + memory fence
schedule_barrier()           # Compiler fence
```

---

*Document created: December 2024*
*Based on pingpong_kernel.mojo development*
