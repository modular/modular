<!-- markdownlint-disable MD013 MD036 MD040 MD052 -->

# Structured GPU Kernels: A Case Study in High-Performance Matrix Multiplication

*A deep dive into the pingpong_kernel.mojo architecture, achieving state-of-the-art performance while maintaining software clarity*

---

## TL;DR

We built a BF16 matrix multiplication kernel for AMD MI300X/MI355X GPUs that:

- **Outperforms vendor BLAS by 7%** at 8K×8K×8K (1552 vs 1451 GFLOPS/s)
- **Matches HipKittens' hand-optimized assembly** (~1570 GFLOPS/s) without sacrificing code clarity
- **Uses structured programming patterns** that make the code maintainable and understandable
- **Shows the trade-off**: optimized for specific tile sizes, not universally faster

This paper explores how we achieved this through careful software architecture, not just raw optimization.

---

## 1. The Challenge: Why GPU Kernels Are Usually Ugly

Let's be honest—most high-performance GPU kernels are write-only code. You write them once, profile them obsessively, and pray you never have to modify them. The typical pattern looks like:

```
// Actual kernel code I've seen in production:
// 2000 lines of inline assembly
// Magic numbers everywhere
// Comments that say "don't touch this, it breaks performance"
```

The HipKittens framework from AMD proved that you can match hand-tuned assembly with high-level abstractions. But their implementation still has a lot of complexity hidden in template metaprogramming and intricate scheduling logic.

**Our question**: Can we build a kernel that's both fast AND readable?

---

## 2. Performance Results

Let's start with the numbers that matter.

### 2.1 Scaling Across Problem Sizes

| Problem Size | pingpong_kernel | Vendor BLAS | Winner |
|--------------|-----------------|-------------|--------|
| 2K × 2K × 2K | 316 GFLOPS/s | - | - |
| 4K × 4K × 4K | 1234 GFLOPS/s | 1314 GFLOPS/s | Vendor (+6%) |
| **8K × 8K × 8K** | **1552 GFLOPS/s** | 1451 GFLOPS/s | **Us (+7%)** |
| 16K × 16K × 16K | 1559 GFLOPS/s | 1686 GFLOPS/s | Vendor (+8%) |

### 2.2 The Sweet Spot Story

We **beat vendor BLAS by 7% at 8K×8K×8K**, but lose at other sizes:

- **4K**: Launch overhead dominates; vendor has better small-tile handling
- **8K**: Our 256×256 tile size is optimal; full GPU utilization
- **16K**: Vendor's memory system tuning wins at scale

**Key insight**: We're optimized for a specific tile configuration (256×256×64), and it shows. At 8K, we're also within 1% of HipKittens' reported ~1570 GFLOPS/s.

*All measurements: MI355X, BF16, transpose_b=True, cache_busting=True, 100 iterations.*

---

## 3. Architecture Overview: The Structured Approach

### 3.1 The Core Insight

Instead of a monolithic kernel with interleaved concerns, we decomposed the problem into **three distinct abstractions**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        KERNEL LAYER                              │
│  - Orchestrates the ping-pong schedule                          │
│  - Defines tile dimensions and thread layout                    │
│  - Owns the swizzle configuration (derived from loading pattern)│
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│        BUFFERS          │     │         MMAOP           │
│                         │     │                         │
│  - LDS tile management  │     │  - Register tiles       │
│  - Global→LDS loading   │     │  - LDS→Register loads   │
│  - Double buffering     │     │  - MMA execution        │
│  - byte_swizzle         │     │  - elem_swizzle         │
└─────────────────────────┘     └─────────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
              ┌─────────────────────────┐
              │     COMPUTE_STAGE       │
              │                         │
              │  Coordinates loads and  │
              │  MMA within a stage     │
              └─────────────────────────┘
```

### 3.2 Entity-Relationship Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           KernelConfig                                   │
│  block_shape: [BM=256, BN=256, BK=64]                                   │
│  warp_shape:  [WM=128, WN=64, WK=64]                                    │
│  mma_shape:   [MMA_M=16, MMA_N=16, MMA_K=32]                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                     derives configuration parameters
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Kernel      │         │    Buffers      │         │     MmaOp       │
│               │         │                 │         │                 │
│ swizzle_      │ passes  │ swizzle_subtile │ matches │ swizzle_elem_   │
│ elem_base,    │ ──────> │ _rows=16        │ <────── │ base,           │
│ swizzle_shift │         │ _cols=32        │         │ swizzle_shift   │
│               │         │ byte_swizzle    │         │ elem_swizzle    │
│               │         │                 │         │                 │
│ load_width    │ passes  │ load_width      │ matches │ load_width      │
│               │ ──────> │                 │ <────── │                 │
└───────────────┘         └─────────────────┘         └─────────────────┘
                                    │                           │
                          owns LDS tiles                owns register tiles
                                    │                           │
                                    ▼                           ▼
                          ┌─────────────────┐         ┌─────────────────┐
                          │ a_load_tiles    │         │ a_reg_tile      │
                          │ b_load_tiles    │         │ b_reg_tile      │
                          │ a_mma_tiles     │ ──────> │ out_quadrants   │
                          │ b_mma_tiles     │  feeds  │                 │
                          └─────────────────┘         └─────────────────┘
```

### 3.3 Separation of Concerns

Each component has a **single responsibility**:

| Component | Responsibility | Doesn't Know About |
|-----------|----------------|-------------------|
| **Kernel** | Schedule orchestration, config | How tiles are actually loaded |
| **Buffers** | Memory management, loading | How data is consumed |
| **MmaOp** | Compute execution | Block-level organization |

This separation means you can modify the loading pattern without touching compute, or change the MMA scheduling without affecting memory management.

---

## 4. The Warp Layout: How 8 Warps Cover 256×256

### 4.1 Block Tile Organization

The kernel uses 8 warps arranged as a 2×4 grid:

```
BLOCK OUTPUT TILE (BM=256 × BN=256):
┌─────────────────────────────────────────────────────────────┐
│                    BN = 256 columns                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Warp 0    │   Warp 1    │   Warp 2    │   Warp 3    │  │
│  │  WM×WN      │  WM×WN      │  WM×WN      │  WM×WN      │  │
│  │  128×64     │  128×64     │  128×64     │  128×64     │  │  Group 0
│  │             │             │             │             │  │  (128 rows)
│  ├─────────────┼─────────────┼─────────────┼─────────────┤  │
│  │   Warp 4    │   Warp 5    │   Warp 6    │   Warp 7    │  │
│  │  WM×WN      │  WM×WN      │  WM×WN      │  WM×WN      │  │
│  │  128×64     │  128×64     │  128×64     │  128×64     │  │  Group 1
│  │             │             │             │             │  │  (128 rows)
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Indexing**:

```mojo
warp_id_m = warp_id // 4    // 0 or 1 (selects row group)
warp_id_n = warp_id % 4     // 0-3 (selects column)
```

### 4.2 Why This Layout?

AMD's MI300X has 4 SIMDs per CU, each capable of running 2 waves. With 8 warps:

- **2 warps per SIMD**: Maximum utilization
- **2 groups of 4**: Enables ping-pong overlap
- **Equal register pressure**: No wasted capacity

The HipKittens paper showed that producer-consumer patterns waste registers on AMD:

> "AMD hardware statically divides registers across all waves, meaning producers consume registers without contributing to output computation."

Our solution: **All 8 waves do both compute AND memory**.

---

## 5. The Ping-Pong Schedule: Overlapping Memory and Compute

### 5.1 Double Buffering Structure

```
LDS DOUBLE BUFFER ORGANIZATION:
┌─────────────────────────────┬─────────────────────────────┐
│     Stage 0 Buffers         │     Stage 1 Buffers         │
│  ┌─────────┐ ┌─────────┐    │  ┌─────────┐ ┌─────────┐   │
│  │ A_s0[0] │ │ B_s0[0] │    │  │ A_s1[0] │ │ B_s1[0] │   │
│  │ 128×64  │ │ 128×64  │    │  │ 128×64  │ │ 128×64  │   │
│  ├─────────┤ ├─────────┤    │  ├─────────┤ ├─────────┤   │
│  │ A_s0[1] │ │ B_s0[1] │    │  │ A_s1[1] │ │ B_s1[1] │   │
│  │ 128×64  │ │ 128×64  │    │  │ 128×64  │ │ 128×64  │   │
│  └─────────┘ └─────────┘    │  └─────────┘ └─────────┘   │
│  [0]=Group0  [1]=Group1     │  [0]=Group0  [1]=Group1    │
└─────────────────────────────┴─────────────────────────────┘
```

**Total LDS**: 128KB (fits in 160KB limit)

### 5.2 The K-Loop Schedule

Each iteration processes `2×BK` elements (two complete stages):

```
PING-PONG TIMELINE:
┌─────────────────────────────────────────────────────────────┐
│  K dimension: ──k=0────k=BK────k=2BK────k=3BK────...        │
│                  │      │       │       │                   │
│  compute_stage:  │  [0] │  [1]  │  [0]  │  [1]  │ ...      │
│                  │      │       │       │                   │
│  Stage 0 bufs:   │ COMP │ LOAD  │ COMP  │ LOAD  │ ...      │
│  Stage 1 bufs:   │ LOAD │ COMP  │ LOAD  │ COMP  │ ...      │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 The Stagger Mechanism

The magic that enables overlap:

```mojo
// PROLOGUE: Create 1-barrier offset
if warp_id_m == 1:
    s_barrier()  // Group 1 waits here, Group 0 skips

// Now: Group 0 is one phase ahead of Group 1
// This offset persists through the entire K-loop!
```

After the stagger:

- **Phase A**: Group 0 computes, Group 1 loads
- **Phase B**: Group 0 loads, Group 1 computes

---

## 6. Quadrant Processing: MmaOp's Internal Structure

### 6.1 Warp Tile Decomposition

Each warp's 128×64 output is divided into 4 quadrants:

```
WARP OUTPUT (WM=128 × WN=64):
┌───────────────────────────────────────┐
│  ┌─────────────────┬─────────────────┐│
│  │ mma[0,0]        │ mma[0,1]        ││  quadrant_m=0
│  │ (64×32 output)  │ (64×32 output)  ││
│  ├─────────────────┼─────────────────┤│
│  │ mma[1,0]        │ mma[1,1]        ││  quadrant_m=1
│  │ (64×32 output)  │ (64×32 output)  ││
│  └─────────────────┴─────────────────┘│
│     quadrant_n=0      quadrant_n=1    │
└───────────────────────────────────────┘
```

### 6.2 compute_stage Implementation

```mojo
fn compute_stage[stage: Int]():
    // Issue all LDS loads asynchronously
    mma_op.load_b[0](buffers.b_mma_tiles[stage][0])  // +4 lgkm
    mma_op.load_a[0](buffers.a_mma_tiles[stage][0])  // +8 lgkm
    mma_op.load_b[1](buffers.b_mma_tiles[stage][1])  // +4 lgkm
    mma_op.load_a[1](buffers.a_mma_tiles[stage][1])  // +8 lgkm = 24 total

    // Fine-grained waits for maximum overlap
    s_waitcnt[lgkmcnt=12]()  // b[0], a[0] ready
    mma_op.mma[0, 0]()       // A[0] × B[0]
    
    s_waitcnt[lgkmcnt=8]()   // b[1] ready
    mma_op.mma[0, 1]()       // A[0] × B[1]
    
    s_waitcnt[lgkmcnt=0]()   // a[1] ready
    mma_op.mma[1, 0]()       // A[1] × B[0]
    mma_op.mma[1, 1]()       // A[1] × B[1]
```

**Why this order?**

The `lgkmcnt` (LDS/GDS Kernel Memory count) tracks in-flight LDS operations. By carefully ordering loads and waits, we overlap LDS transfers with MMA execution.

---

## 7. Swizzle: From Magic Numbers to Derived Parameters

### 7.1 The Problem

LDS has 64 banks × 4 bytes. Without swizzle, MMA's 4×16 access pattern causes 4-way bank conflicts.

### 7.2 The Old Way (Magic Numbers)

```mojo
// What you see in most kernels:
alias byte_swizzle = Swizzle(1, 5, 4)  // Why these values?
alias elem_swizzle = Swizzle(1, 4, 4)  // No one knows!
```

### 7.3 The Structured Way (Derived from Geometry)

```mojo
// Swizzle determined by loading thread layout (16×4 per warp)
alias subtile_rows = 16
alias subtile_cols = 4 * load_width  // 32 for bf16

// Derived parameters:
alias swizzle_elem_base = log2_floor(subtile_cols // 2)  // = 4
alias swizzle_byte_base = swizzle_elem_base + log2_floor(elem_size)  // = 5
alias swizzle_shift = log2_floor(subtile_rows)  // = 4

// Now the "magic" has meaning:
alias byte_swizzle = Swizzle(1, swizzle_byte_base, swizzle_shift)
alias elem_swizzle = Swizzle(1, swizzle_elem_base, swizzle_shift)
```

### 7.4 Architectural Ownership

| Component | Owns | Why |
|-----------|------|-----|
| **Kernel** | Subtile dimensions (16×4) | Defines loading thread layout |
| **Buffers** | byte_swizzle | Writes to LDS |
| **MmaOp** | elem_swizzle (received) | Reads from LDS |

MmaOp doesn't compute the swizzle—it receives parameters from the kernel because the loading pattern determines the swizzle.

---

## 8. Comparison: Three Approaches to GPU Matmul

### 8.1 Architecture Comparison

| Aspect | pingpong_kernel (AMD) | matmul.mojo (AMD) | sm90 (NVIDIA) |
|--------|----------------------|-------------------|---------------|
| **Warp Organization** | 8 warps, all compute+memory | N warps, all compute | Producer-consumer split |
| **Double Buffering** | Explicit 2-stage | Multi-stage pipeline | Ring buffer abstraction |
| **Memory Transfer** | `load_to_lds` intrinsic | Copy through registers | TMA hardware |
| **Synchronization** | `s_barrier` + `s_waitcnt` | `barrier()` + `schedule_group_barrier` | mbarriers |
| **Abstraction Level** | Buffers + MmaOp | MmaOpAMD + MMATileBuffers | RingBuffer + TileLoader |

### 8.2 Code Structure Comparison

**pingpong_kernel.mojo (1976 lines)**:

```
├── MmaOp struct (270 lines: 313-582)
│   ├── Register tile management
│   ├── _load_fragment (generic LDS→reg)
│   ├── load_a, load_b (public API)
│   └── mma (compute)
├── Buffers struct (469 lines: 583-1051)
│   ├── LDS tile allocation
│   ├── _load_tile_to_lds (generic global→LDS)
│   └── load_a, load_b variants
├── AMDPingPongMatmul struct (823 lines: 1107-1929)
│   ├── Configuration and validation
│   ├── compute_stage helper
│   └── Main kernel loop
└── Helper functions and entry point
```

**sm90/matmul_kernels.mojo (1362 lines) + support files**:

```
├── HopperMatmulSM90Kernel_SMem struct
│   ├── TileArray management
│   └── PipelineBarrier coordination
├── HopperMatmulSM90Kernel struct
│   ├── Uses RingBuffer (462 lines, separate file)
│   ├── Uses TileLoader (407 lines, separate file)
│   └── Uses MatmulTileWriter (separate file)
└── Main kernel logic
```

**matmul.mojo (AMD, 823 lines)**:

```
├── MmaOpAMD struct
│   ├── TiledTensorCore integration
│   └── Register tile management
├── MMATileBuffers struct
│   └── Memory region management
└── gemm_kernel_amd function
```

### 8.3 Abstraction Trade-offs

| Kernel | Abstraction Style | Pros | Cons |
|--------|-------------------|------|------|
| **pingpong** | Monolithic with helpers | Clear data flow, minimal indirection | Larger single file |
| **sm90** | Decomposed modules | Reusable components, clean separation | More files to understand |
| **matmul.mojo** | Inherited abstractions | Leverages existing infra | Harder to customize |

---

## 9. The Structured Kernel Principles

From building pingpong_kernel, we extracted these principles:

### 9.1 Principle: Derived, Not Hardcoded

**Bad**: Magic numbers with no explanation

```mojo
alias byte_swizzle = Swizzle(1, 5, 4)  // Why?
```

**Good**: Values derived from configuration

```mojo
alias swizzle_elem_base = log2_floor(subtile_cols // 2)
alias swizzle_shift = log2_floor(subtile_rows)
```

### 9.2 Principle: Clear Ownership

**Bad**: Parameters computed in multiple places

```mojo
// In Buffers:
alias load_width = simd_width_of[in_type]()
// In MmaOp:
alias load_width = simd_width_of[in_type]()  // Duplicated!
```

**Good**: Single source of truth with explicit passing

```mojo
// Kernel defines:
alias load_width = simd_width_of[in_type]()
// Passes to both Buffers and MmaOp
alias MmaOpType = MmaOp[..., load_width]
alias BuffersType = Buffers[..., load_width]
```

### 9.3 Principle: Separation of Concerns

**Bad**: One struct does everything

```mojo
struct MonolithicKernel:
    fn load_global_to_lds(): ...
    fn load_lds_to_registers(): ...
    fn execute_mma(): ...
    fn store_output(): ...
```

**Good**: Each struct has one job

```mojo
struct Buffers:      // Memory management only
struct MmaOp:        // Compute only
fn compute_stage():  // Coordination only
```

### 9.4 Principle: Type-Safe Contracts

**Bad**: Accept anything, hope it works

```mojo
fn load(tile: SMemTileType): ...  // No validation
```

**Good**: Compile-time validation

```mojo
fn _load_fragment[..., frag_element_layout: Layout, //](
    smem_tile: SMemTileType[...],
    reg_frag: RegTileType[...],
):
    constrained[
        frag_element_layout.size() == Self.load_width,
        "fragment width must match MmaOp's load_width",
    ]()
```

### 9.5 Principle: Diagrams as Documentation

Every non-trivial structure should have an ASCII diagram:

```mojo
// ================================================================
// TILE ORGANIZATION AND PING-PONG SCHEDULE
// ================================================================
//
// BLOCK TILE (BM=256 × BN=256):
// ┌─────────────┬─────────────┬─────────────┬─────────────┐
// │   Warp 0    │   Warp 1    │   Warp 2    │   Warp 3    │
// │  (group 0)  │  (group 0)  │  (group 0)  │  (group 0)  │
// ├─────────────┼─────────────┼─────────────┼─────────────┤
// │   Warp 4    │   Warp 5    │   Warp 6    │   Warp 7    │
// │  (group 1)  │  (group 1)  │  (group 1)  │  (group 1)  │
// └─────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 10. Lessons from HipKittens

### 10.1 What HipKittens Got Right

1. **All waves compute**: No wasted register capacity
2. **Stagger mechanism**: Simple but effective overlap
3. **Large tiles**: 256×256 output per block
4. **bulk operations**: Avoid fine-grained instruction interleaving

### 10.2 What We Added

1. **Structured abstractions**: MmaOp, Buffers, compute_stage
2. **Derived parameters**: Swizzle from geometry, not magic
3. **Type-safe contracts**: Compile-time validation
4. **Documentation**: Inline diagrams and clear ownership

### 10.3 What We Preserved

The core scheduling insights from HipKittens remain:

- 8-wave ping-pong with stagger
- Double buffering with explicit stage management
- Fine-grained `lgkmcnt` waits
- `s_barrier` synchronization

We didn't try to outsmart the scheduling—we wrapped it in better abstractions.

---

## 11. Performance Analysis

### 11.1 Where the Performance Comes From

| Optimization | How We Achieve It |
|--------------|-------------------|
| **Warp efficiency** | All 8 warps compute (no dedicated producers wasting registers) |
| **Memory overlap** | Stagger mechanism + interleaved A loads during compute |
| **Bank conflict avoidance** | Swizzle patterns derived from tile geometry |
| **Large tiles** | 256×256 output amortizes launch and sync overhead |
| **lgkmcnt tuning** | Fine-grained waits in compute_stage overlap LDS loads with MMA |
| **Loop unrolling** | `@parameter` for loop enables compiler optimization |

*Note: We don't have ablation studies measuring individual contribution of each optimization. The table shows what we believe matters based on the HipKittens paper and our experimentation.*

### 11.2 Basic Analysis (8K×8K×8K)

- **Compute ops**: 2 × 8192³ = 1.1 TFLOPs
- **Achieved**: 1552 GFLOPS/s
- **Time**: 0.71 ms

For comparison, HipKittens reports ~1570 GFLOPS/s in their paper, though direct comparison requires running on the same system due to hardware variability between installations.

*We don't have verified peak theoretical numbers for MI355X bf16 MMA throughput to do a proper roofline analysis.*

### 11.3 Why We Beat Vendor BLAS at 8K (But Not Elsewhere)

*Note: These are hypotheses based on the tile configuration, not verified through profiling.*

**At 8K (we win by 7%)**:

- Our 256×256 tile may be well-matched to the problem size
- Block count may align well with GPU resources

**At 4K (vendor wins by 6%)**:

- Possibly fewer blocks leading to underutilization
- Vendor likely has dynamic tile selection

**At 16K (vendor wins by 8%)**:

- Memory system behavior may differ at scale
- Vendor may have better large-problem optimizations

**The honest truth**: We're a specialized kernel optimized for a specific configuration. Vendor BLAS is tuned across many problem sizes.

---

## 12. Future Directions

### 12.1 Immediate Improvements

1. **Quadrant-based B loading**: Enable true overlap for B matrix
2. **Atomic barriers**: Replace `s_barrier` with explicit phase counters
3. **Variable tile sizes**: Support problem-specific configurations

### 12.2 Longer-term Goals

1. **Fusion**: Integrate epilogue operations (bias, activation)
2. **Quantization**: FP8 and INT8 support
3. **Sparse matmul**: Structured sparsity patterns

### 12.3 Architectural Evolution

The structured approach enables incremental improvement:

- Swap out Buffers for a TMA-like abstraction
- Replace MmaOp's swizzle with hardware-assisted patterns
- Add new compute_stage variants for different schedules

---

## 13. Conclusion

We built a GPU matmul kernel that:

1. **Performs at state-of-the-art levels** (1552 GFLOPS/s, beating vendor BLAS)
2. **Uses structured programming** (clear ownership, derived parameters)
3. **Remains maintainable** (documented, diagrams, type-safe)
4. **Follows HipKittens insights** without copying their implementation

The key insight: **Performance and clarity are not opposed**. By understanding the underlying algorithm deeply, we can express it in structured code that the compiler optimizes well.

### Key Takeaways for Kernel Developers

1. **Derive, don't hardcode**: Magic numbers should come from somewhere
2. **Separate concerns**: Memory management ≠ compute ≠ orchestration
3. **Own your parameters**: Each struct should know what it owns vs. receives
4. **Document with diagrams**: ASCII art is underrated
5. **Trust the compiler**: Well-structured code optimizes well

---

## Appendix A: Code Structure Reference

```
pingpong_kernel.mojo
├── Module header (documentation, imports)
├── Helper functions
│   ├── atomic_barrier_sync
│   └── _load_from_lds
├── KernelConfig struct
├── MmaOp struct
│   ├── Type aliases (RegTileType, OutQuadrantType)
│   ├── Swizzle configuration (received from kernel)
│   ├── lgkmcnt tracking aliases
│   ├── __init__, reset_accumulator
│   ├── _load_fragment (generic helper)
│   ├── load_a, load_b (public API)
│   └── mma (compute)
├── Buffers struct
│   ├── Swizzle configuration (derived from geometry)
│   ├── Tile type aliases
│   ├── LDS allocations
│   ├── _load_tile_to_lds (generic global→LDS)
│   ├── _load_tile_4warp (4-warp variant)
│   └── load_a, load_b variants
├── compute_stage_params struct
├── PingPongMatmul struct
│   ├── Configuration aliases
│   ├── validate_config
│   └── matmul_demo_ping_pong (main kernel)
└── demo_pingpong_matmul (entry point)
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **BM, BN, BK** | Block tile dimensions (M×N×K) |
| **WM, WN** | Warp tile dimensions |
| **MMA_M, MMA_N, MMA_K** | Hardware MMA instruction dimensions |
| **LDS** | Local Data Share (AMD's shared memory) |
| **lgkmcnt** | LDS/GDS Kernel Memory counter |
| **vmcnt** | Vector Memory counter |
| **SIMD** | AMD's vector processing unit (4 per CU) |
| **Wave** | AMD's term for warp (64 threads) |
| **Swizzle** | Memory access pattern to avoid bank conflicts |

---

*Document version: 1.0*
*Date: December 2024*
*Authors: Modular Kernel Team*
