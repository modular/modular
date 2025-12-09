<!-- markdownlint-disable MD013 MD036 MD040 MD052 -->

# AMD Ping-Pong Matmul Kernel Architecture

## Executive Summary

This document describes the architecture of `pingpong_kernel.mojo`, a high-performance BF16 matrix multiplication kernel for AMD MI300X/MI355X GPUs. The kernel achieves **~1550 GFLOPS/s** using an 8-warp ping-pong pattern inspired by the HipKittens framework.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Memory Transfer** | Direct `load_to_lds` | Bypasses registers, ~10% bandwidth gain |
| **Warp Organization** | 8 warps, 2 groups of 4 | Enables compute/memory overlap via stagger |
| **Double Buffering** | Explicit 2-stage ping-pong | Clear load/compute separation |
| **Layout System** | `Layout`, `RuntimeLayout`, `LayoutTensor` | Type-safe, compile-time optimized offsets |
| **Synchronization** | `s_barrier()` + `s_waitcnt` | Fine-grained control, matches HipKittens |

### Performance Summary

| Configuration | GFLOPS/s | Notes |
|---------------|----------|-------|
| 8K×8K×8K BF16 with swizzle | ~1550 | Optimized schedule |
| 8K×8K×8K BF16 simplified | ~1000 | Conservative schedule |

---

## 1. AMD GPU Architecture

### 1.1 Hardware Hierarchy

```
GPU (MI300X: 8 XCDs, MI355X: 32 XCDs)
 └── XCD (Accelerator Complex Die)
      └── CU (Compute Unit, 32-38 per XCD)
           └── SIMD (4 per CU)
                └── Wave/Wavefront (64 threads, up to 8 per SIMD)
```

### 1.2 Memory Hierarchy

| Memory | Size | Latency | Notes |
|--------|------|---------|-------|
| VGPRs | 512×32-bit per SIMD | ~1 cycle | Vector registers |
| SGPRs | ~100 per SIMD | ~1 cycle | Scalar registers (uniform) |
| LDS | 160 KB per CU | ~20 cycles | Shared memory (64 banks × 4 bytes) |
| L1 Cache | 32 KB per CU | ~50 cycles | |
| L2 Cache | 4 MB per XCD | ~300 cycles | |
| HBM | 128-192 GB | ~800 cycles | High Bandwidth Memory |

### 1.3 Key Differences from NVIDIA

| Feature | NVIDIA (H100/B200) | AMD (MI300X/MI355X) |
|---------|-------------------|---------------------|
| Warp/Wave size | 32 threads | **64 threads** |
| Register allocation | Dynamic | **Static** (evenly partitioned) |
| Async primitives | TMA, wgmma, mbarriers | **`load_to_lds`** (buffer loads) |
| MMA operands | From shared/tensor memory | **From registers only** |
| Shared memory | ~228KB (B200) | ~160KB |
| Register file | Smaller | **2× larger** |

**Critical Insight**: AMD's **static register allocation** means all waves on a SIMD share registers equally. Producer-consumer patterns waste 50% of register capacity.

---

## 2. The 8-Warp Ping-Pong Pattern

### 2.1 Why Not Producer-Consumer on AMD?

From HipKittens paper (Table 2):

| Configuration | Output Tile | TFLOPs |
|---------------|-------------|--------|
| 0 producers, 8 consumers | 256×256 | **1570** |
| 4 producers, 8 consumers | 256×256 | 1507 |
| 4 producers, 4 consumers | 192×256 | 1291 |

> "AMD hardware statically divides registers across all waves, meaning **producers consume registers without contributing to output computation**."

### 2.2 Ping-Pong Solution

All 8 waves do **both compute AND memory** operations:

```
SIMD Layout (2 waves per SIMD):
  SIMD 0: Wave 0 (Group 0), Wave 4 (Group 1)
  SIMD 1: Wave 1 (Group 0), Wave 5 (Group 1)
  SIMD 2: Wave 2 (Group 0), Wave 6 (Group 1)
  SIMD 3: Wave 3 (Group 0), Wave 7 (Group 1)

Phase Alternation (via stagger):
  Phase A: Group 0 computes, Group 1 loads
  Phase B: Group 0 loads, Group 1 computes
```

### 2.3 The Stagger Mechanism

```mojo
// PROLOGUE: Create 1-barrier offset
if warp_id_m == 1:
    s_barrier()  // Group 1 waits here, Group 0 skips

// Now Group 0 is one phase ahead of Group 1
// This offset persists through the entire K-loop
```

### 2.4 K-Loop Compute Schedule

Each K-loop iteration processes `BK × 2` elements (two stages):

```
PING-PONG COMPUTE SCHEDULE:
┌─────────────────────────────────────────────────────────────┐
│  K dimension: ──k=0────k=BK────k=2BK────k=3BK────...        │
│                  │      │       │       │                   │
│  compute_stage:  │  [0] │  [1]  │  [0]  │  [1]  │ ...      │
│                  │      │       │       │                   │
│  Stage 0 bufs:   │ COMP │ LOAD  │ COMP  │ LOAD  │ ...      │
│  Stage 1 bufs:   │ LOAD │ COMP  │ LOAD  │ COMP  │ ...      │
└─────────────────────────────────────────────────────────────┘
```

**Simplified Schedule (per iteration)**:

1. Load stage 1 buffers
2. `compute_stage[0]()` - compute from stage 0
3. Load stage 0 buffers (overlaps with next step)
4. `compute_stage[1]()` - compute from stage 1
5. Barrier, advance K pointer

### 2.5 Quadrant Processing in compute_stage

Each warp's output tile (WM×WN = 128×64) is divided into 4 quadrants for MMA scheduling:

```
WARP OUTPUT TILE (128×64):
┌───────────────────────────────────────┐
│  Warp Output (WM=128 × WN=64)         │
│  ┌─────────────────┬─────────────────┐│
│  │ mma[0,0]        │ mma[0,1]        ││ quadrant_m=0
│  │ (64×32 output)  │ (64×32 output)  ││ (quadrant_m_mmas × MMA_M rows)
│  ├─────────────────┼─────────────────┤│
│  │ mma[1,0]        │ mma[1,1]        ││ quadrant_m=1
│  │ (64×32 output)  │ (64×32 output)  ││
│  └─────────────────┴─────────────────┘│
│     quadrant_n=0      quadrant_n=1    │
└───────────────────────────────────────┘
```

**compute_stage[stage]** executes:

1. `load_a[0]`, `load_b[0]` - LDS → registers for quadrant 0
2. `load_a[1]`, `load_b[1]` - LDS → registers for quadrant 1
3. `mma[0,0]()` - A quadrant 0 × B quadrant 0
4. `mma[0,1]()` - A quadrant 0 × B quadrant 1
5. `mma[1,0]()` - A quadrant 1 × B quadrant 0
6. `mma[1,1]()` - A quadrant 1 × B quadrant 1

Each `mma[qa,qb]` executes `quadrant_m_mmas × quadrant_n_mmas` hardware MMA instructions.

---

## 3. Memory Organization

### 3.1 Tile Dimensions

```
Block tile:  BM=256 × BN=256 × BK=64
Warp tile:   WM=128 × WN=64
MMA tile:    MMA_M=16 × MMA_N=16 × MMA_K=32
```

### 3.2 Warp Layout in Block Tile

The 8 warps are arranged as a 2×4 grid covering the block output tile:

```
BLOCK TILE (BM=256 × BN=256):
┌─────────────────────────────────────────────────────────────┐
│                    BN = 256 columns                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Warp 0    │   Warp 1    │   Warp 2    │   Warp 3    │  │
│  │  WM×WN      │  WM×WN      │  WM×WN      │  WM×WN      │  │
│  │  128×64     │  128×64     │  128×64     │  128×64     │  │ half_BM
│  │ (group 0)   │ (group 0)   │ (group 0)   │ (group 0)   │  │ = 128
│  ├─────────────┼─────────────┼─────────────┼─────────────┤  │ rows
│  │   Warp 4    │   Warp 5    │   Warp 6    │   Warp 7    │  │
│  │  WM×WN      │  WM×WN      │  WM×WN      │  WM×WN      │  │
│  │  128×64     │  128×64     │  128×64     │  128×64     │  │ half_BM
│  │ (group 1)   │ (group 1)   │ (group 1)   │ (group 1)   │  │ = 128
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │ rows
└─────────────────────────────────────────────────────────────┘
   warp_id_m = warp_id // 4    (0 or 1, selects row group)
   warp_id_n = warp_id % 4     (0-3, selects column)
```

### 3.3 LDS Buffer Layout

```
Total LDS: 128KB (fits in 160KB limit)

A buffers (64KB):
  Stage 0: a_s0_g0 (128×64), a_s0_g1 (128×64)  // Per warp-group
  Stage 1: a_s1_g0 (128×64), a_s1_g1 (128×64)

B buffers (64KB):
  Stage 0: b_s0_h0 (128×64), b_s0_h1 (128×64)  // SHARED across groups
  Stage 1: b_s1_h0 (128×64), b_s1_h1 (128×64)
```

```
DOUBLE BUFFERING:
┌─────────────────────────────────────────────────────────────┐
│  Stage 0 LDS Buffers        │  Stage 1 LDS Buffers          │
│  ┌─────────┐ ┌─────────┐    │  ┌─────────┐ ┌─────────┐     │
│  │ A_s0[0] │ │ B_s0[0] │    │  │ A_s1[0] │ │ B_s1[0] │     │
│  │ 128×64  │ │ 128×64  │    │  │ 128×64  │ │ 128×64  │     │
│  ├─────────┤ ├─────────┤    │  ├─────────┤ ├─────────┤     │
│  │ A_s0[1] │ │ B_s0[1] │    │  │ A_s1[1] │ │ B_s1[1] │     │
│  │ 128×64  │ │ 128×64  │    │  │ 128×64  │ │ 128×64  │     │
│  └─────────┘ └─────────┘    │  └─────────┘ └─────────┘     │
│  [0] = group 0's region     │  [1] = group 1's region      │
└─────────────────────────────────────────────────────────────┘
```

**Important**: B buffers are shared across warp groups. This creates synchronization constraints.

### 3.4 Double Buffering (Ping-Pong)

```
K-iteration N:
  Compute from Stage 0 ←→ Load into Stage 1

K-iteration N+1:
  Compute from Stage 1 ←→ Load into Stage 0
```

---

## 4. Layout-Based Memory Access

Our implementation uses Mojo's `Layout`, `RuntimeLayout`, and `LayoutTensor` for structured memory access.

### 4.1 Global → LDS Loading

**Pattern**: `LayoutTensor.distribute` + `Swizzle`

```mojo
// Thread layout: 16 rows × 4 col-groups per warp (64 threads)
alias thread_layout = Layout.row_major(16, 4)

// Swizzle for bank conflict avoidance: XOR bit 9 into bit 5
alias byte_swizzle = Swizzle(1, 5, 4)

// Compute effective lane with swizzle applied
var lds_write_bytes = lane_id * load_width * 2  // T × 16 bytes
var swizzled_bytes = byte_swizzle(lds_write_bytes)
var effective_lane = swizzled_bytes // (load_width * 2)

// Use distribute() to compute global memory position
var dist_tensor = subtile_tensor.vectorize[1, load_width]()
    .distribute[thread_layout](UInt(effective_lane))

// Extract offset for global load
var buf_offset = (Int(dist_tensor.ptr) - Int(subtile_tensor.ptr)) // elem_size
```

### 4.2 LDS → Register Loading (MMA Pattern)

**Pattern**: `RuntimeLayout` for compile-time offset computation

```mojo
// MMA access layout: maps 64 lanes to LDS offsets
// Shape (16, 4): decompose lane as col = lane % 16, row = lane // 16
// Stride (32, 8): compute offset as col × 32 + row × 8
alias mma_access_layout = Layout(IntTuple(16, 4), IntTuple(32, 8))

// RuntimeLayout evaluates at compile time (no GPU heap allocation)
var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))

// Apply element swizzle: Swizzle(1, 4, 4)
alias elem_swizzle = Swizzle(1, 4, 4)
var full_offset = elem_swizzle(iter_base + lane_offset)
```

### 4.3 Tile-Based Loading Functions

**Pattern**: Pass `LayoutTensor` tiles, derive pointers via tile indexing

```mojo
fn _load_tile_to_lds_a[which: Int](
    self, resource: AMDBufferResource, dst_tile: Self.AHalfTile
):
    // Compute warp's subtile position via tile indexing
    var tile_idx = i * loading_warps + warp_id
    var warp_subtile = dst_tile.tile[rows_per_warp, BK](tile_idx, 0)
    
    // readfirstlane ensures scalar (SGPR) pointer for load_to_lds
    var smem_ptr = readfirstlane(warp_subtile.ptr)
    
    // Direct global→LDS transfer (bypasses VGPRs)
    resource.load_to_lds[width=load_width](buf_offset, smem_ptr, ...)
```

---

## 5. Comparison with HipKittens

### 5.1 Architecture Comparison

| Aspect | HipKittens (C++) | Our Implementation (Mojo) |
|--------|------------------|---------------------------|
| **Language** | C++ with inline ASM | Mojo with intrinsics |
| **Memory Layout** | Manual pointer arithmetic | `LayoutTensor` + `RuntimeLayout` |
| **Swizzle** | Bit manipulation macros | `Swizzle` class |
| **MMA Operation** | Inline assembly | `mma()` intrinsic |
| **Synchronization** | Raw `s_barrier`, `s_waitcnt` | Same (wrapped in functions) |
| **Thread Mapping** | Manual coordinate calculation | `distribute()` API |

### 5.2 Data Loading Comparison

**HipKittens (C++)**:

```cpp
// Manual coordinate calculation
const int swizzle = ((offset % 1024) >> 9) << 5;
const int swizzled_offset = offset ^ swizzle;

// Direct load with inline asm
G::load(Bs[stage][half], B_ptr + offset, ...);
```

**Our Implementation (Mojo)**:

```mojo
// Layout-based with Swizzle class
alias byte_swizzle = Swizzle(1, 5, 4)
var swizzled_bytes = byte_swizzle(lds_write_bytes)

// Type-safe LayoutTensor API
var dist_tensor = subtile_tensor.distribute[thread_layout](effective_lane)
resource.load_to_lds[width=load_width](buf_offset, smem_ptr, ...)
```

### 5.3 LDS Read Comparison

**HipKittens (C++)**:

```cpp
// Manual lane offset calculation
int lane_offset = (lane / 16) * 8 + (lane % 16) * 32;

// Inline assembly LDS read
asm volatile("ds_read_b128 %0, %1" : "=v"(result) : "v"(ptr + offset));
```

**Our Implementation (Mojo)**:

```mojo
// Layout-based (same math, type-safe)
alias mma_access_layout = Layout(IntTuple(16, 4), IntTuple(32, 8))
var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))

// Intrinsic-based LDS read with alias annotations
var loaded = _load_from_lds[width=load_width](smem_ptr.offset(offset))
```

### 5.4 Synchronization Comparison

| Primitive | HipKittens | Our Implementation |
|-----------|------------|-------------------|
| Workgroup barrier | `__builtin_amdgcn_s_barrier()` | `s_barrier()` |
| Vector memory wait | `asm("s_waitcnt vmcnt(N)")` | `s_waitcnt[vmcnt=N]()` |
| LDS memory wait | `asm("s_waitcnt lgkmcnt(N)")` | `s_waitcnt[lgkmcnt=N]()` |
| Schedule fence | `asm("s_sched_barrier")` | `schedule_barrier()` |

### 5.5 Key vmcnt Values

Both implementations use similar vmcnt patterns:

| Location | HipKittens | Our Implementation |
|----------|------------|-------------------|
| Prologue | `vmcnt(4)` | `vmcnt(0)` (conservative) |
| Main loop | `vmcnt(6)` | `vmcnt(6)` (aggressive) |
| Epilogue | `vmcnt(2)`, `vmcnt(1)` | `vmcnt(0)` (critical fix) |

**Lesson Learned**: The epilogue `vmcnt` values were the source of race conditions. Using `vmcnt(0)` in the epilogue fixed correctness with minimal performance impact.

---

## 6. Synchronization Model

### 6.1 AMD Memory Counters (Per-Wave)

| Counter | Tracks | Ordering |
|---------|--------|----------|
| `vmcnt` | Global memory ops (`load_to_lds`) | FIFO |
| `lgkmcnt` | LDS ops (`ds_read`, `ds_write`) | FIFO for same-type |

**Critical**: These are **per-wave**, not shared across workgroup!

### 6.2 Synchronization Primitives

| Primitive | Effect | Use Case |
|-----------|--------|----------|
| `s_waitcnt[vmcnt=N]()` | Wait until ≤N global ops in flight | Before reading loaded data |
| `s_waitcnt[lgkmcnt=N]()` | Wait until ≤N LDS ops in flight | Before MMA (data in registers) |
| `s_barrier()` | Control-flow barrier (all waves sync) | Phase transitions |
| `barrier()` | Barrier + memory fences | When memory visibility needed |

### 6.3 The vmcnt/Barrier Pattern

```mojo
// Safe pattern: wait for YOUR loads, then sync with others
s_waitcnt[vmcnt=0]()   // My loads complete
s_barrier()             // All waves reach here
// Now all data is visible to all waves
```

### 6.4 Race Condition Root Cause

The race occurred because:

1. **B buffers are shared** across warp groups
2. **Stagger** puts groups at different K iterations
3. **Aggressive vmcnt** (e.g., `vmcnt=6`) left loads in flight across barriers
4. Result: Group 0 writes B while Group 1 reads B

**Solution**: Conservative `vmcnt=0` in prologue and epilogue; aggressive `vmcnt=6` only in the main loop where the stagger provides natural separation.

---

## 7. Swizzle Patterns

### 7.1 Why Swizzle?

AMD LDS has **64 banks × 4 bytes** = 256 bytes per cycle. Without swizzle, MMA's 4×16 read pattern causes **4-way bank conflicts**.

### 7.2 Swizzle Derived from Tile Geometry

The swizzle pattern is **derived from the loading subtile dimensions**, not hardcoded:

```mojo
# Loading uses 16×4 thread layout, each thread loads load_width elements
alias subtile_rows = 16                    # Thread layout rows
alias subtile_cols = 4 * load_width        # 4 thread cols × SIMD width (32 for bf16)

# Swizzle parameters derived from geometry:
alias swizzle_bits = 1
alias swizzle_elem_base = log2_floor(subtile_cols // 2)  # log2(16) = 4
alias swizzle_byte_base = swizzle_elem_base + log2_floor(elem_size)  # 4 + 1 = 5
alias swizzle_shift = log2_floor(subtile_rows)  # log2(16) = 4

# Resulting swizzle instances
alias byte_swizzle = Swizzle(1, swizzle_byte_base, swizzle_shift)  # Swizzle(1, 5, 4)
alias elem_swizzle = Swizzle(1, swizzle_elem_base, swizzle_shift)  # Swizzle(1, 4, 4)
```

### 7.3 Swizzle Parameters for bf16

| Parameter | Value | Derivation |
|-----------|-------|------------|
| subtile_rows | 16 | Thread layout |
| subtile_cols | 32 | 4 × load_width (8) |
| elem_size | 2 | sizeof(bf16) |
| swizzle_elem_base | 4 | log2(32 / 2) |
| swizzle_byte_base | 5 | 4 + log2(2) |
| swizzle_shift | 4 | log2(16) |
| **byte_swizzle** | `Swizzle(1, 5, 4)` | `B ^ ((B >> 9) & 1) << 5` |
| **elem_swizzle** | `Swizzle(1, 4, 4)` | `E ^ ((E >> 8) & 1) << 4` |

### 7.4 How Swizzle Works

Within each 16×32 bf16 subtile (1024 bytes):

- Bit 9 (bytes) / Bit 8 (elements) indicates upper/lower half
- XORing into bit 5/4 remaps bank indices
- Result: **~6% performance improvement**

```
Without swizzle:     With swizzle:
Lane 0 → Bank 0      Lane 0 → Bank 0
Lane 1 → Bank 0      Lane 1 → Bank 32  (different!)
Lane 2 → Bank 0      Lane 2 → Bank 0
Lane 3 → Bank 0      Lane 3 → Bank 32  (different!)
(4-way conflict)     (no conflict)
```

### 7.5 Architectural Separation

The swizzle follows a clear ownership model:

| Component | Responsibility | Swizzle |
|-----------|----------------|---------|
| **Kernel** | Defines loading thread layout (16×4) | Computes swizzle_elem_base, swizzle_shift |
| **Buffers** | Owns loading logic | Uses byte_swizzle for writing |
| **MmaOp** | Owns MMA logic | Receives swizzle params, uses elem_swizzle |

MmaOp doesn't know about block-level tile organization - it just processes warp-sized
tiles using the swizzle pattern that matches how Buffers wrote the data.

---

## 8. Code Organization

### 8.1 Key Structs

| Struct | Purpose |
|--------|---------|
| `TileLoaderLDS` | Cooperative global→LDS loader (encapsulates buffer + thread positions) |
| `Buffers` | Double-buffered LDS tiles, loaders for A and B, byte_swizzle |
| `MmaOp` | Register tiles, MMA operations, elem_swizzle |
| `AMDBufferResource` | Global memory descriptor for `load_to_lds` |

### 8.2 Parameter Inference Design

Layouts are inferred from tensor arguments using Mojo's parameter inference:

```mojo
struct Buffers[
    in_type: DType,
    a_layout: Layout,
    b_layout: Layout, //,  # infer-only (note //)
    BM: Int, ...
]:
    alias K = Self.a_layout.shape[1].value()  # compile-time

    fn __init__(
        out self,
        a: LayoutTensor[Self.in_type, Self.a_layout, *_, **_],
        b: LayoutTensor[_, Self.b_layout, *_, **_],
        ...
    ): ...
```

`TileLoaderLDS` derives stride from its `src_layout` parameter:

```mojo
struct TileLoaderLDS[
    dtype: DType,
    src_layout: Layout,
    src_tile_layout: Layout,
    num_loading_warps: Int,
    swizzle: OptionalReg[Swizzle],
    load_width: Int,
]:
    alias stride = Self.src_layout.shape[1].value()  # compile-time
    var buffer: AMDBufferResource
    var thread_row: Int  # Pre-computed with swizzle
    var thread_col: Int
```

Key patterns:

- Infer-only parameters (`//`) extract layouts from tensor arguments
- `alias` for compile-time derived values (no runtime storage)
- `make_amd_buffer_resource()` for GPU-safe buffer creation
- Swizzle inversion: load from `swizzle(T)` → write to `T`

### 8.3 Key Functions

| Function | Purpose |
|----------|---------|
| `TileLoader.load_tile` | Cooperative global→LDS using stored buffer |
| `Buffers.load_a/b` | 8-warp loading via TileLoader |
| `Buffers.load_a/b_as_group` | 4-warp loading for overlap |
| `MmaOp.load_a/b` | LDS → register with swizzle |
| `MmaOp.mma` | Matrix multiply-accumulate |
| `compute_stage` | Full compute phase (loads + MMAs) |

### 8.4 Schedule Structure

```
PROLOGUE:
  Load A+B stage 0 → barrier
  Group 1 extra barrier (create stagger)
  
MAIN LOOP (per 2×BK):
  barrier → Load stage 1 → vmcnt(6) → barrier
  Compute stage 0 (LDS→reg + MMA)
  barrier
  Compute stage 1
  barrier → Load stage 0 → vmcnt(0) → advance_k

EPILOGUE:
  Final compute stages
  Group 0 extra barrier (re-sync)
  Store results
```

---

## 9. Performance Tuning

### 9.1 vmcnt Tuning

| Location | Conservative | Aggressive | Notes |
|----------|--------------|------------|-------|
| Prologue | vmcnt=0 | vmcnt=4 | Conservative safer |
| Main loop | vmcnt=0 | vmcnt=6 | Aggressive OK here |
| Epilogue | vmcnt=0 | vmcnt=2,1 | **Must be 0** (race fix) |

### 9.2 Bottleneck Analysis

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| Memory-bound | Low MMA utilization | More aggressive vmcnt |
| LDS-bound | Bank conflicts | Swizzle pattern |
| Compute-bound | Full MMA utilization | Already optimal |
| Sync-bound | Too many barriers | Reduce barrier count |

---

## 10. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| CU | Compute Unit (AMD's SM equivalent) |
| SIMD | Single Instruction Multiple Data unit (4 per CU) |
| Wave | 64-thread execution unit (AMD's warp) |
| LDS | Local Data Share (shared memory) |
| VGPR | Vector General Purpose Register |
| SGPR | Scalar General Purpose Register |
| MFMA | Matrix Fused Multiply Add |
| vmcnt | Global memory operation counter |
| lgkmcnt | LDS operation counter |
| Stagger | 1-barrier offset between warp groups |
| Ping-pong | Alternating between 2 buffer stages |

### B. References

1. HipKittens Paper: Hu et al., "HipKittens: Fast and Furious AMD Kernels", arXiv:2511.08083, 2025
2. AMD CDNA3 ISA Reference
3. AMD CDNA4 Architecture Whitepaper

### C. File Locations

| Component | File |
|-----------|------|
| Kernel implementation | `pingpong_kernel.mojo` |
| Previous design doc | `AMD_PINGPONG_KERNEL_DESIGN.md` |
| Standard matmul | `matmul.mojo` |
| Test file | `test_ping_pong.mojo` |

---

*Document created: December 2024*
*Architecture version: Layout-based with RuntimeLayout*
