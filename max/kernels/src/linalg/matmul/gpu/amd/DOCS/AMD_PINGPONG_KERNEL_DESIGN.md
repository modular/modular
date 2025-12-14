<!-- markdownlint-disable MD013 MD036 MD040 MD052 -->
<!-- Disable: line-length, emphasis-as-heading, code-language, reference-links -->

# AMD Ping-Pong Matmul Kernel: Design Philosophy and Architecture

## Executive Summary

This document provides a comprehensive analysis of the AMD ping-pong
matrix multiplication kernel, its design philosophy derived from the
HipKittens framework, the trade-offs involved, and important semantic
considerations for future development.

**Key References:**

- HipKittens Paper: [arXiv:2511.08083](https://arxiv.org/abs/2511.08083)
- HipKittens Repository: `~/HipKittens/`
- Reference C++ Implementation: `~/HipKittens/analysis/bf16_gemm/mi350x/kernel_4096.cpp`

---

## Table of Contents

1. [AMD GPU Architecture Background](#1-amd-gpu-architecture-background)
2. [The Problem: Overlapping Compute and Memory](#2-the-problem-overlapping-compute-and-memory)
3. [Scheduling Patterns for AMD](#3-scheduling-patterns-for-amd)
4. [The Barrier Semantics Issue](#4-the-barrier-semantics-issue)
5. [Compiler Interactions and Race Conditions](#5-compiler-interactions-and-race-conditions)
6. [Proposed Solution: Explicit Phase Synchronization](#6-proposed-solution-explicit-phase-synchronization)
7. [Memory Layout and Swizzle Patterns](#7-memory-layout-and-swizzle-patterns)
8. [Implementation Details](#8-implementation-details)
9. [Performance Characteristics](#9-performance-characteristics)
10. [Alternative Approaches](#10-alternative-approaches)
11. [Future Directions](#11-future-directions)
12. [**Current Progress: A-Interleaving Schedule (~1000 GFLOPS/s)**](#12-current-progress-a-interleaving-schedule-1000-gflopss)
13. [**Future Direction: Quadrant-Based Computation**](#13-future-direction-quadrant-based-computation)

---

## 1. AMD GPU Architecture Background

### 1.1 Hardware Hierarchy

AMD MI300X/MI355X GPUs organize compute resources as follows:

| Level | Component | Count | Description |
|-------|-----------|-------|-------------|
| GPU | XCDs (Chiplets) | 8 (MI300X) / 32 (MI355X) | Accelerator Complex Dies |
| XCD | CUs | 32-38 per XCD | Compute Units |
| CU | SIMDs | 4 per CU | Single Instruction Multiple Data units |
| SIMD | Waves | Up to 8 per SIMD | 64-thread execution units (wavefronts) |

### 1.2 Memory Hierarchy

| Memory Type | Size per CU | Latency | Description |
|-------------|-------------|---------|-------------|
| Registers (VGPRs) | 512 × 32-bit per SIMD | ~1 cycle | Vector General Purpose Registers |
| Registers (SGPRs) | ~100 per SIMD | ~1 cycle | Scalar registers (uniform values) |
| LDS (Shared Memory) | 160 KB | ~20 cycles | Local Data Share |
| L1 Cache | 32 KB | ~50 cycles | Per-CU cache |
| L2 Cache | 4 MB per XCD | ~300 cycles | Per-XCD cache |
| LLC (L3) | 256 MB | ~500 cycles | Last-level cache |
| HBM | 128-192 GB | ~800 cycles | High Bandwidth Memory |

### 1.3 Key Architectural Differences from NVIDIA

| Feature | NVIDIA (e.g., H100/B200) | AMD (MI300X/MI355X) |
|---------|--------------------------|---------------------|
| Warp/Wave size | 32 threads | 64 threads |
| Register allocation | Dynamic (can reallocate) | **Static** (evenly partitioned) |
| Async memory primitives | TMA, wgmma, mbarriers | `load_to_lds` (buffer loads) |
| Matrix instruction operands | From shared/tensor memory | From registers only |
| Shared memory per SM/CU | Larger (~228KB on B200) | Smaller (~160KB) |
| Register file size | Smaller | **2× larger** |

**Critical Insight**: AMD's **static register allocation** means all waves on a SIMD share registers equally. If you have 8 waves but only 4 do useful computation (producer-consumer pattern), you waste 50% of register capacity.

---

## 2. The Problem: Overlapping Compute and Memory

### 2.1 The Fundamental Challenge

High-performance matrix multiplication requires hiding memory latency by overlapping:

- **Global → LDS loads**: Moving data from HBM to shared memory
- **LDS → Register loads**: Moving data from shared memory to registers
- **MMA compute**: Matrix multiply-accumulate operations

### 2.2 NVIDIA's Solution: Wave Specialization (Producer-Consumer)

On NVIDIA GPUs, the dominant pattern is **wave specialization**:

- **Producer warps**: Handle memory operations (TMA loads)
- **Consumer warps**: Handle computation (wgmma/mma)
- Synchronization via hardware mbarriers

This works well because:

1. TMA bypasses registers (producers need few registers)
2. wgmma accepts operands from shared memory (consumers don't load to registers)
3. Hardware can reallocate registers from producers to consumers

### 2.3 Why Producer-Consumer Underperforms on AMD

From the HipKittens paper (Table 2):

| Configuration | Output Tile | TFLOPs |
|---------------|-------------|--------|
| 0 producers, 8 consumers | 256×256 | **1570** |
| 4 producers, 8 consumers | 256×256 | 1507 |
| 4 producers, 4 consumers | 192×256 | 1291 |

> "AMD hardware statically divides registers across all waves, meaning **producers consume registers without contributing to output computation**. This limits the usable output tile size when using wave specialization."

---

## 3. Scheduling Patterns for AMD

The HipKittens paper identifies two patterns that achieve peak AMD performance:

### 3.1 Pattern 1: 8-Wave Ping-Pong (Balanced Workloads)

**Configuration:**

- 8 waves per thread block (2 per SIMD)
- Waves split into 2 groups of 4 (one wave per SIMD per group)
- All waves do BOTH compute AND memory (no specialization)

**Mechanism:**

```text
SIMD Layout:
  SIMD 0: Wave 0 (Group 0), Wave 4 (Group 1)
  SIMD 1: Wave 1 (Group 0), Wave 5 (Group 1)
  SIMD 2: Wave 2 (Group 0), Wave 6 (Group 1)
  SIMD 3: Wave 3 (Group 0), Wave 7 (Group 1)

Phase Alternation:
  Phase A: Group 0 computes, Group 1 loads
  Phase B: Group 0 loads, Group 1 computes
  (repeat)
```

**Advantages:**

- All 8 waves contribute to computation (no wasted registers)
- Large tile primitives (256×256 output tiles)
- Simple, readable code with bulk operations
- Matches AMD's hand-optimized assembly performance

**Disadvantages:**

- Requires "conditional barrier trick" to create phase offset
- Best for balanced compute/memory workloads

### 3.2 Pattern 2: 4-Wave Interleave (Imbalanced Workloads)

**Configuration:**

- 4 waves per thread block (1 per SIMD)
- Each wave issues both compute AND memory instructions
- Fine-grained instruction interleaving

**Mechanism:**

```text
Each wave independently interleaves:
  Issue MFMA → Issue global load → Issue LDS load → Issue MFMA → ...
```

**Advantages:**

- Better for imbalanced (compute-heavy or memory-heavy) workloads
- Single wave per SIMD can adapt instruction mix dynamically

**Disadvantages:**

- Requires small tile primitives
- Much larger code size (fine-grained instruction issues)
- Harder to write and maintain

### 3.3 Performance Comparison (from HipKittens paper)

| Kernel | Pattern | Lines of Code | TFLOPs |
|--------|---------|---------------|--------|
| FP8 GEMM | 8-wave | 48 | 3222 |
| FP8 GEMM | 4-wave | 183 | 3327 |
| MHA backwards | 8-wave | 331 | 894 |
| MHA backwards | 4-wave | 989 | 1091 |

> "Remarkably, the simple 8-wave pattern is sufficient to match AMD's hand-optimized assembly kernels across BF16 GEMM, FP8 GEMM, and attention forward."

---

## 4. The Barrier Semantics Issue

### 4.1 Intended Semantics of `s_barrier`

The `s_barrier` instruction (workgroup barrier) has specific semantics defined by AMD:

> **Intended purpose**: Synchronize all wavefronts in a workgroup, ensuring all threads have completed prior memory operations and have a consistent view of memory.

This is a **memory consistency primitive**, analogous to:

- CUDA `__syncthreads()`
- OpenCL `barrier(CLK_LOCAL_MEM_FENCE)`

### 4.2 How Ping-Pong Abuses Barrier Semantics

The 8-wave ping-pong pattern uses `s_barrier` for **scheduling**, not memory consistency:

```cpp
// From HipKittens paper, Section E.1:
// "The kernel inserts a conditional barrier to stall half the waves
// (one wave per SIMD) while the other half begins performing additional loads."

if (warp_row == 1) {
    __builtin_amdgcn_s_barrier();  // ← NOT for memory visibility!
}                                   // ← Used to DELAY group 1 by one phase
```

**What this code actually does:**

1. Group 0 (warp_row == 0) skips the barrier, proceeds to load
2. Group 1 (warp_row == 1) hits the barrier, must wait for Group 0
3. This creates a **permanent phase offset** between the groups

**The semantic abuse:**
| Aspect | Proper Usage | Ping-Pong Usage |
|--------|--------------|-----------------|
| Purpose | Memory visibility | Scheduling/delay |
| Meaning | "Data is ready" | "Wait for other group" |
| Barrier semantics | All-to-all sync | Asymmetric stall |
| Intent | Correctness | Performance |

### 4.3 Why This Is Problematic

1. **Violates principle of least surprise**: Future maintainers will misunderstand the code
2. **Fragile to compiler changes**: Different compilers may reorder around barriers differently
3. **Hardware-dependent**: Future AMD architectures might change barrier behavior
4. **Hidden state machine**: The phase relationship is implicit, not explicit in code

### 4.4 The Paper's Acknowledgment

The HipKittens paper explicitly describes this as "a conditional barrier controls the alternation" - they are aware this is unconventional but accept it for performance.

---

## 5. Compiler Interactions and Race Conditions

### 5.1 The LLVM Backend Problem

Both Mojo and AMD's HIP compiler use the same LLVM backend for code generation. However, there's a critical difference in how the `s_barrier` semantic abuse interacts with compiler optimizations.

**The fundamental issue**: The LLVM optimizer sees `s_barrier()` as a synchronization point but doesn't understand the **intent** of creating phase offsets between warp groups. The compiler's optimization passes may:

1. **Reorder memory operations** around barriers that appear "safe" to move
2. **Hoist loads/stores** across what we intended as phase boundaries
3. **Sink operations** past barriers when it sees no data dependency
4. **Eliminate "redundant" barriers** in its analysis

### 5.2 Observed Race Conditions

The ping-pong kernel has exhibited **intermittent race conditions** that are extremely difficult to debug:

| Symptom | Frequency | Pattern |
|---------|-----------|---------|
| Incorrect output values | ~10-20% of runs | Small deltas from expected |
| Tile boundary errors | Sporadic | Mismatches at row/column 2048, 384, etc. |
| Size-dependent failures | More frequent with larger matrices | 8K×8K fails more than 4K×4K |

**Root cause hypothesis**: The LLVM backend performs register data flow optimizations that don't respect the implicit phase dependencies we're trying to create with barrier staggering.

### 5.3 Why `schedule_barrier()` Is Insufficient

We attempted to fix race conditions using `schedule_barrier()` (maps to `__builtin_amdgcn_sched_barrier(0)`):

```mojo
s_waitcnt[vmcnt=X]()
schedule_barrier()    # Prevent reordering before
s_barrier()
schedule_barrier()    # Prevent reordering after
```

**Finding**: This helps but doesn't fully solve the problem because:

1. `schedule_barrier()` is a **hint** to the instruction scheduler, not a hard constraint
2. It doesn't prevent **register allocation** optimizations
3. The LLVM backend may still perform **value numbering** or **dead store elimination** that breaks our assumptions
4. Different LLVM versions may behave differently

### 5.4 The Evolving Backend Problem

The AMD LLVM backend is under **constant development**:

- New optimization passes are added regularly
- Existing passes are tuned for better performance
- What works this week may break next week

This makes the `s_barrier` abuse approach **fundamentally unstable** for production use.

### 5.5 The Semantic Gap

The core problem is a **semantic gap** between what we want and what we express:

| What We Want | What We Write | What Compiler Sees |
|--------------|---------------|-------------------|
| "Group 1 waits for Group 0 to complete phase N" | `if (group == 1) s_barrier()` | "Half the threads hit a barrier... weird but ok" |
| "Don't move loads past this point" | `s_barrier()` | "Sync point, but no data dependency I can see" |
| "This phase depends on previous phase" | (implicit in barrier sequence) | "Independent operations, let me optimize!" |

The compiler cannot violate **explicit data dependencies**, but our dependencies are implicit in the barrier timing, which the compiler doesn't understand.

---

## 6. Proposed Solution: Explicit Phase Synchronization

### 6.1 Design Principle

Replace implicit barrier-based phase offsets with **explicit atomic counter-based synchronization**. The key insight:

> **Atomic operations have well-defined acquire/release semantics that the compiler MUST respect.**

Unlike `s_barrier()` which the compiler sees as an opaque sync point, atomic operations create **explicit data dependencies** that prevent reordering.

### 6.2 Reference Implementation

The warp-specialized matmul kernel (`warp_spec_matmul.mojo`) successfully uses atomic counter-based synchronization via the ring buffer traits:

```mojo
# From ring_buffer_traits.mojo:
@always_inline
fn wait_for_counter(
    counter: UnsafePointer[Int32, address_space=AddressSpace.SHARED],
    threshold: Int32,
):
    """Spin-wait until counter reaches threshold."""
    while Atomic.load(counter) < threshold:
        inlined_assembly[
            "s_sleep 0", NoneType, constraints="", has_side_effect=True
        ]()

@always_inline
fn increment_counter_if_first_thread(
    counter: UnsafePointer[Int32, address_space=AddressSpace.SHARED],
    increment: Int32,
):
    """Atomically increment counter, but only from first thread in warp."""
    if thread_idx.x % UInt(WARP_SIZE) == 0:
        _ = Atomic.fetch_add(counter, increment)
```

### 6.3 Two-Counter Design

For the ping-pong kernel, we need a **minimal** synchronization structure with just two counters.

**Group composition:**

- Group 0: 4 warps (warp_id_m == 0, i.e., warps 0, 1, 2, 3)
- Group 1: 4 warps (warp_id_m == 1, i.e., warps 4, 5, 6, 7)

**Counter semantics (monotonically increasing):**

- `counter_a`: Each of the 4 warps in Group 0 increments by 1 when it completes a phase
- `counter_b`: Each of the 4 warps in Group 1 increments by 1 when it completes a phase

**Wait condition:**

- Group 1 waits for `counter_a >= 4 * target_phase` (all 4 Group 0 warps done)
- Group 0 waits for `counter_b >= 4 * target_phase` (all 4 Group 1 warps done)

```mojo
struct PingPongSync:
    """Two-counter synchronization for phase-offset between warp groups.
    
    Design:
    - 8 warps split into 2 groups of 4 (Group 0: warp_id_m=0, Group 1: warp_id_m=1)
    - counter_a: Incremented by each Group 0 warp after completing work
    - counter_b: Incremented by each Group 1 warp after completing work
    - Counters are monotonically increasing (never reset)
    
    Protocol:
    - Each warp increments its group's counter when done with a phase
    - Waiting checks for threshold: 4 * phase_number (all 4 warps must signal)
    - Group 0 (leader): signals first, waits later
    - Group 1 (follower): waits first, signals after
    """
    alias WARPS_PER_GROUP: Int = 4
    
    var counter_a: UnsafePointer[Int32, address_space=AddressSpace.SHARED]
    var counter_b: UnsafePointer[Int32, address_space=AddressSpace.SHARED]
    
    fn __init__(out self):
        """Allocate counters in shared memory, initialize to 0."""
        self.counter_a = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
        self.counter_b = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
        # Initialize only from first thread
        if thread_idx.x == 0:
            self.counter_a.store(0)
            self.counter_b.store(0)
        # Use a regular barrier here since we're initializing
        barrier()
    
    @always_inline
    fn signal_a(self):
        """One warp in Group 0 signals completion. Call from all Group 0 warps."""
        # Only first lane in each warp increments (avoid redundant atomics)
        if thread_idx.x % UInt(WARP_SIZE) == 0:
            _ = Atomic.fetch_add(self.counter_a, 1)
    
    @always_inline
    fn signal_b(self):
        """One warp in Group 1 signals completion. Call from all Group 1 warps."""
        # Only first lane in each warp increments
        if thread_idx.x % UInt(WARP_SIZE) == 0:
            _ = Atomic.fetch_add(self.counter_b, 1)
    
    @always_inline
    fn wait_a(self, phase: Int32):
        """Wait for all 4 Group 0 warps to complete the specified phase."""
        var threshold = phase * Self.WARPS_PER_GROUP
        while Atomic.load(self.counter_a) < threshold:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()
    
    @always_inline
    fn wait_b(self, phase: Int32):
        """Wait for all 4 Group 1 warps to complete the specified phase."""
        var threshold = phase * Self.WARPS_PER_GROUP
        while Atomic.load(self.counter_b) < threshold:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()
```

### 6.4 Synchronization Protocol

The two groups follow different protocols to maintain the phase offset. Each warp participates in signaling.

**Counter progression example:**

```text
Phase 1:
  Group 0: 4 warps each call signal_a() → counter_a = 4
  Group 1: wait_a(1) blocks until counter_a >= 4
           then 4 warps each call signal_b() → counter_b = 4

Phase 2:
  Group 0: wait_b(1) blocks until counter_b >= 4
           then 4 warps each call signal_a() → counter_a = 8
  Group 1: wait_a(2) blocks until counter_a >= 8
           then 4 warps each call signal_b() → counter_b = 8

Phase 3:
  Group 0: wait_b(2) blocks until counter_b >= 8
           then 4 warps each call signal_a() → counter_a = 12
  Group 1: wait_a(3) blocks until counter_a >= 12
           then 4 warps each call signal_b() → counter_b = 12
...
```

**Timeline visualization:**

```text
counter_a:  0 ──→ 4 ──────→ 8 ──────→ 12 ──────→ ...
               ↑         ↑          ↑
            G0 done   G0 done    G0 done
            phase 1   phase 2    phase 3

counter_b:  0 ──────→ 4 ──────→ 8 ──────→ 12 ──→ ...
                   ↑         ↑          ↑
                G1 done   G1 done    G1 done
                phase 1   phase 2    phase 3
```

**Key invariant**: Group 1 always waits for `counter_a >= 4 * (phase + 1)` before proceeding, ensuring Group 0 stays one phase ahead.

**Benefits of this model:**

1. **Every warp participates** - not just one designated thread
2. **Natural threshold** - wait for `4 * phase` is intuitive  
3. **Monotonic** - counters only increase, easy to reason about
4. **Robust** - if one warp is slow, the wait correctly blocks until all 4 are done

### 6.5 Detailed Mapping to Simplified Schedule

Here is the **complete transformation** of the simplified schedule from `pingpong_kernel.mojo`:

#### Current Code Structure (with s_barrier abuse)

```text
PROLOGUE (k=0 only):
  Lines 1200-1204: All warps load stage 0
  Lines 1207-1208: if (warp_id_m == 1) s_barrier()  ← STAGGER
  Line 1211: s_barrier()                            ← SYNC POINT 1

MAIN LOOP BODY:
  Lines 1215-1220: All warps load stage 1
  Line 1223: s_barrier()                            ← SYNC POINT 2
  Lines 1227-1234: All warps compute stage 0
  Line 1237: s_barrier()                            ← SYNC POINT 3
  Lines 1240-1247: All warps compute stage 1
  Lines 1251-1261: if not last: s_barrier() + load stage 0  ← SYNC POINT 4

EPILOGUE:
  Lines 1264-1265: if (warp_id_m == 0) s_barrier()  ← RE-ALIGN
```

#### Proposed Code Structure (with explicit counters)

```mojo
var sync = PingPongSync()
var phase: Int32 = 0

@parameter
for _k in range(0, K, BK * 2):
    @parameter
    if _k == 0:
        # ============================================================
        # PROLOGUE: Load stage 0
        # ============================================================
        buffers.load_b[0, 0](b_resource)
        buffers.load_a[0, 0](a_resource)
        buffers.load_b[0, 1](b_resource)
        buffers.load_a[0, 1](a_resource)
        buffers.advance_k()
        
        # STAGGER: Create phase offset
        # Group 0 signals phase 1, Group 1 waits then signals
        phase = 1
        if warp_id_m == 0:
            sync.signal_a()      # Each G0 warp: counter_a += 1 → total = 4
        else:
            sync.wait_a(1)       # Wait for counter_a >= 4 (all G0 warps done)
            sync.signal_b()      # Each G1 warp: counter_b += 1 → total = 4
    
    # ============================================================
    # SYNC POINT 1: Before stage 1 loads
    # ============================================================
    phase += 1
    if warp_id_m == 0:
        sync.signal_a()          # Each G0 warp signals (counter_a += 1)
        sync.wait_b(phase - 1)   # Wait for G1 to finish previous phase
    else:
        sync.wait_a(phase)       # Wait for G0 to finish current phase
        sync.signal_b()          # Each G1 warp signals (counter_b += 1)
    
    # ============================================================
    # LOAD: Issue stage 1 loads
    # ============================================================
    buffers.load_b[1, 0](b_resource)
    buffers.load_a[1, 0](a_resource)
    buffers.load_b[1, 1](b_resource)
    buffers.load_a[1, 1](a_resource)
    s_waitcnt[vmcnt=0]()
    buffers.advance_k()
    
    # ============================================================
    # SYNC POINT 2: Before compute stage 0
    # ============================================================
    phase += 1
    if warp_id_m == 0:
        sync.signal_a()
        sync.wait_b(phase - 1)
    else:
        sync.wait_a(phase)
        sync.signal_b()
    
    # ============================================================
    # COMPUTE: Process stage 0 from LDS
    # ============================================================
    mma_op.load_b[0](buffers.b_mma_tiles[0][0])
    mma_op.load_a[0](buffers.a_mma_tiles[0][0])
    mma_op.load_b[1](buffers.b_mma_tiles[0][1])
    mma_op.load_a[1](buffers.a_mma_tiles[0][1])
    mma_op.mma[0, 0]()
    mma_op.mma[0, 1]()
    mma_op.mma[1, 0]()
    mma_op.mma[1, 1]()
    
    # ============================================================
    # SYNC POINT 3: Before compute stage 1
    # ============================================================
    phase += 1
    if warp_id_m == 0:
        sync.signal_a()
        sync.wait_b(phase - 1)
    else:
        sync.wait_a(phase)
        sync.signal_b()
    
    # ============================================================
    # COMPUTE: Process stage 1 from LDS
    # ============================================================
    mma_op.load_b[0](buffers.b_mma_tiles[1][0])
    mma_op.load_a[0](buffers.a_mma_tiles[1][0])
    mma_op.load_b[1](buffers.b_mma_tiles[1][1])
    mma_op.load_a[1](buffers.a_mma_tiles[1][1])
    mma_op.mma[0, 0]()
    mma_op.mma[0, 1]()
    mma_op.mma[1, 0]()
    mma_op.mma[1, 1]()
    
    # ============================================================
    # SYNC POINT 4 + LOAD: Prefetch next stage 0 (if not last)
    # ============================================================
    @parameter
    if _k < K - BK * 2:
        phase += 1
        if warp_id_m == 0:
            sync.signal_a()
            sync.wait_b(phase - 1)
        else:
            sync.wait_a(phase)
            sync.signal_b()
        
        buffers.load_b[0, 0](b_resource)
        buffers.load_a[0, 0](a_resource)
        buffers.load_b[0, 1](b_resource)
        buffers.load_a[0, 1](a_resource)
        s_waitcnt[vmcnt=0]()
        buffers.advance_k()

# ============================================================
# EPILOGUE: Re-align warp groups
# ============================================================
if warp_id_m == 0:
    sync.wait_b(phase)   # Group 0: wait for Group 1 to finish final phase
# Group 1 is already done, no wait needed
```

#### Counter Value Trace (for K=256, BK=64, so 2 loop iterations)

```
Iteration 0 (_k=0):
  Prologue:     G0 signals → counter_a=4,  G1 waits(4), signals → counter_b=4
  Sync point 1: G0 signals → counter_a=8,  G1 waits(8), signals → counter_b=8
  Sync point 2: G0 signals → counter_a=12, G1 waits(12), signals → counter_b=12
  Sync point 3: G0 signals → counter_a=16, G1 waits(16), signals → counter_b=16
  Sync point 4: G0 signals → counter_a=20, G1 waits(20), signals → counter_b=20

Iteration 1 (_k=128):
  Sync point 1: G0 signals → counter_a=24, G1 waits(24), signals → counter_b=24
  Sync point 2: G0 signals → counter_a=28, G1 waits(28), signals → counter_b=28
  Sync point 3: G0 signals → counter_a=32, G1 waits(32), signals → counter_b=32
  (no sync point 4 - last iteration)

Epilogue: G0 waits for counter_b >= 32
```

### 6.6 Sync Point Helper Function

To reduce code duplication, we can extract the sync pattern into a helper:

```mojo
@always_inline
fn sync_point(sync: PingPongSync, warp_id_m: Int, phase: Int32):
    """Execute a sync point between the two warp groups.
    
    Protocol:
    - Group 0 (leader): signal first, then wait for Group 1's previous phase
    - Group 1 (follower): wait for Group 0's current phase, then signal
    
    Args:
        sync: The PingPongSync instance
        warp_id_m: 0 for Group 0, 1 for Group 1
        phase: Current phase number (1-indexed)
    """
    if warp_id_m == 0:
        sync.signal_a()           # G0: I'm done with this phase
        sync.wait_b(phase - 1)    # G0: Wait for G1 to finish previous phase
    else:
        sync.wait_a(phase)        # G1: Wait for G0 to finish this phase
        sync.signal_b()           # G1: I'm done with this phase
```

This simplifies the main loop to:

```mojo
phase += 1
sync_point(sync, warp_id_m, phase)  # Before stage 1 loads
# ... load stage 1 ...

phase += 1
sync_point(sync, warp_id_m, phase)  # Before compute stage 0
# ... compute stage 0 ...

phase += 1
sync_point(sync, warp_id_m, phase)  # Before compute stage 1
# ... compute stage 1 ...
```

**Note**: The phase must be incremented *before* calling `sync_point`, so the wait conditions use the correct threshold.

### 6.7 Why This Works

1. **Explicit data dependency**: `wait_a()` and `wait_b()` create **load dependencies** on the counter values. The compiler CANNOT move code past a load that the code depends on.

2. **Atomic semantics**: `Atomic.load()` and `Atomic.fetch_add()` have defined memory ordering semantics. Even with relaxed ordering, the spin-wait loop creates a control dependency.

3. **No reordering possible**:
   - Code after `wait_a(N)` depends on `counter_a >= N`
   - Code before `signal_a()` must complete before the atomic increment
   - The compiler must respect these dependencies

4. **Self-documenting**: The code explicitly states "Group 1 waits for Group 0" rather than relying on implicit barrier counting.

5. **Deterministic**: Unlike `s_barrier` which relies on barrier counting across different code locations, the counters have explicit values that can be inspected and reasoned about.

### 6.8 Comparison with Current Approach

| Aspect | s_barrier Abuse | Two-Counter Sync |
|--------|-----------------|------------------|
| Compiler understanding | Opaque sync point | Explicit data dependency |
| Reordering risk | High (compiler may optimize) | None (atomic semantics) |
| Future-proof | No (LLVM changes may break) | Yes (atomic semantics are stable) |
| Debuggability | Hard (implicit state) | Easy (print counter values) |
| Code clarity | Low (hidden intent) | High (explicit signal/wait) |
| Performance overhead | ~0 | Small (~10-20 cycles per sync) |
| Correctness guarantee | Fragile | Robust |

### 6.9 Performance Considerations

The atomic counter approach has some overhead compared to hardware barriers:

1. **Atomic operations**: ~10-20 cycles per atomic on AMD MI355X
2. **Spin-wait loop**: Consumes some SIMD cycles, but `s_sleep` reduces power and allows other waves to execute
3. **Memory traffic**: Counter updates go through LDS, but minimal (8 bytes total)
4. **Sync points per iteration**: 4 sync points × 2 operations (signal + wait) = 8 atomic ops per K-iteration

**Expected impact**: <5% performance overhead, which is acceptable for:

- Eliminating race conditions that cause incorrect results
- Future-proofing against compiler changes
- Improved maintainability and debuggability

### 6.10 Detailed Implementation Plan

#### Step 1: Create `PingPongSync` Struct

**File**: `pingpong_kernel.mojo`  
**Location**: Near the top of the file, after other struct definitions

```mojo
struct PingPongSync:
    """Two-counter sync for 8-warp ping-pong (4 warps per group)."""
    alias WARPS_PER_GROUP: Int = 4
    
    var counter_a: UnsafePointer[Int32, address_space=AddressSpace.SHARED]
    var counter_b: UnsafePointer[Int32, address_space=AddressSpace.SHARED]
    
    fn __init__(out self):
        # Allocate and zero-initialize counters
        ...
    
    fn signal_a(self):
        # First lane of each G0 warp increments counter_a
        if thread_idx.x % WARP_SIZE == 0:
            Atomic.fetch_add(self.counter_a, 1)
    
    fn signal_b(self):
        # First lane of each G1 warp increments counter_b
        if thread_idx.x % WARP_SIZE == 0:
            Atomic.fetch_add(self.counter_b, 1)
    
    fn wait_a(self, phase: Int32):
        # Spin until counter_a >= 4 * phase
        while Atomic.load(self.counter_a) < phase * WARPS_PER_GROUP:
            s_sleep()
    
    fn wait_b(self, phase: Int32):
        # Spin until counter_b >= 4 * phase
        while Atomic.load(self.counter_b) < phase * WARPS_PER_GROUP:
            s_sleep()
```

**Key invariant**: `wait_a(N)` blocks until all 4 Group 0 warps have signaled N times total.

#### Step 2: Add `sync_point` Helper Function

**File**: `pingpong_kernel.mojo`  
**Location**: Inside the kernel function, before the main loop

```mojo
@always_inline
fn sync_point(sync: PingPongSync, warp_id_m: Int, phase: Int32):
    """Sync point: G0 leads, G1 follows."""
    if warp_id_m == 0:
        sync.signal_a()           # G0: signal completion
        sync.wait_b(phase - 1)    # G0: wait for G1's previous phase
    else:
        sync.wait_a(phase)        # G1: wait for G0's current phase
        sync.signal_b()           # G1: signal completion
```

#### Step 3: Replace Prologue Stagger

**Current code** (lines 1206-1208):

```mojo
if warp_id_m == 1:
    s_barrier()
```

**Replace with**:

```mojo
var phase: Int32 = 1
if warp_id_m == 0:
    sync.signal_a()       # G0: 4 warps signal → counter_a = 4
else:
    sync.wait_a(1)        # G1: wait for counter_a >= 4
    sync.signal_b()       # G1: 4 warps signal → counter_b = 4
```

#### Step 4: Replace Main Loop Barriers

**Current code** (lines 1211, 1223, 1237, 1253):

```mojo
s_barrier()  # 4 occurrences in simplified schedule
```

**Replace each with** (note: increment phase BEFORE the sync_point):

```mojo
phase += 1
sync_point(sync, warp_id_m, phase)
```

#### Step 5: Replace Epilogue Re-align

**Current code** (lines 1264-1265):

```mojo
if warp_id_m == 0:
    s_barrier()
```

**Replace with**:

```mojo
if warp_id_m == 0:
    sync.wait_b(phase)    # G0: wait for G1 to finish final phase
# G1 doesn't need to wait - it's already done
```

#### Step 6: Remove `schedule_barrier()` Calls

With explicit counters, we no longer need `schedule_barrier()` sandwiches since the atomic operations themselves prevent reordering.

#### Step 7: Testing

1. **Correctness tests**: Run existing matmul tests 50+ times to verify no race conditions
2. **Dimension sweep**: Test various matrix sizes (1K, 2K, 4K, 8K, 16K)
3. **Performance benchmark**: Compare against original implementation
4. **Stress test**: Run continuously for extended period

#### Step 8: Cleanup

1. Remove old `s_barrier()` code paths
2. Update comments to reflect new synchronization model
3. Consider removing `USE_SIMPLIFIED_SCHEDULE` toggle once validated

### 6.11 Verification Checklist

Before considering the implementation complete:

- [ ] `PingPongSync` struct compiles and allocates correctly in shared memory
- [ ] Counter initialization is visible to all threads (initial barrier after init)
- [ ] Prologue creates correct phase offset:
  - [ ] Group 0 signals phase 1 (counter_a = 4)
  - [ ] Group 1 waits for counter_a >= 4, then signals (counter_b = 4)
- [ ] Each sync point correctly signals and waits:
  - [ ] Group 0: signal_a() then wait_b(phase-1)
  - [ ] Group 1: wait_a(phase) then signal_b()
- [ ] Epilogue correctly re-aligns: Group 0 waits for Group 1's final phase
- [ ] Counter values match expected trace (see 6.5)
- [ ] No race conditions in 50+ consecutive test runs
- [ ] Performance within 5% of original (or better)
- [ ] Code is cleaner and more maintainable

---

## 7. Memory Layout and Swizzle Patterns

### 7.1 LDS Bank Conflicts on AMD

AMD MI300X/MI355X have:

- **64 LDS banks** × 4 bytes each = 256 bytes per bank cycle
- Bank index = `(byte_address / 4) % 64`

Without swizzling, MMA's 4×16 thread read pattern causes **4-way bank conflicts**.

### 7.2 Layout-Based Memory Access (Current Implementation)

The kernel uses Mojo's `Layout`, `RuntimeLayout`, and `LayoutTensor` for structured memory access:

#### 7.2.1 Global → LDS Loading (`Buffers.__init__`, `_load_tile_to_lds_*`)

```mojo
# Thread layout: 16 rows × 4 col-groups per warp
alias thread_layout = Layout.row_major(16, 4)

# Swizzle for byte offsets: XOR bit 9 into bit 5
alias byte_swizzle = Swizzle(1, 5, 4)

# Compute effective lane with swizzle
var lds_write_bytes = lane_id * load_width * 2  # T * 16 bytes
var swizzled_bytes = byte_swizzle(lds_write_bytes)
var effective_lane = swizzled_bytes // (load_width * 2)

# Use distribute() to compute global position
var dist_tensor = subtile_tensor.vectorize[1, load_width]()
    .distribute[thread_layout](UInt(effective_lane))
var buf_offset = (Int(dist_tensor.ptr) - Int(subtile_tensor.ptr)) // elem_size
```

#### 7.2.2 LDS → Register Loading (`MmaOp.load_a/b`)

```mojo
# MMA access pattern: Layout((16, 4), (32, 8))
# - Shape (16, 4) decomposes lane as: col = lane % 16, row = lane // 16
# - Stride (32, 8) computes offset: col * 32 + row * 8
# - Equivalent to: (lane // 16) * 8 + (lane % 16) * 32
alias mma_access_layout = Layout(IntTuple(16, 4), IntTuple(32, 8))

# RuntimeLayout enables compile-time offset computation
var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))

# Apply element swizzle: Swizzle(1, 4, 4)
alias elem_swizzle = Swizzle(1, 4, 4)
var full_offset = elem_swizzle(iter_base + lane_offset)
```

#### 7.2.3 Tile-Based Addressing (`_load_tile_to_lds_*`, `_load_tile_4warp_*`)

```mojo
# Pass LayoutTensor tiles directly to loading functions
fn _load_tile_to_lds_a[which: Int](
    self, resource: AMDBufferResource, dst_tile: Self.AHalfTile
):
    # Compute warp's subtile position via tile indexing
    var tile_idx = i * loading_warps + warp_id
    var warp_subtile = dst_tile.tile[rows_per_warp, BK](tile_idx, 0)
    
    # readfirstlane ensures scalar (SGPR) pointer for load_to_lds
    var smem_ptr = readfirstlane(warp_subtile.ptr)
    
    resource.load_to_lds[width=load_width](buf_offset, smem_ptr, ...)
```

### 7.3 Swizzle Pattern Details

| Type | Pattern | Formula | Use Case |
|------|---------|---------|----------|
| Byte swizzle | `Swizzle(1, 5, 4)` | `offset ^ ((offset >> 9) & 1) << 5` | Global → LDS |
| Element swizzle | `Swizzle(1, 4, 4)` | `offset ^ ((offset >> 8) & 1) << 4` | LDS → Register |

**How it works:**

- Each 16×32 subtile is 1024 bytes (16 rows × 32 cols × 2 bytes/bf16)
- Bit 9 (bytes) / Bit 8 (elements) indicates upper/lower half
- XORing remaps bank indices to avoid 4-way conflicts

### 7.4 Thread Layout for 8-Warp Cooperative Loading

```
Thread layout within warp (64 threads):
  - Layout.row_major(16, 4): 16 rows × 4 col-groups
  - Each thread loads 8 bf16 elements (128 bits)
  - Warp covers 16 × 32 = 512 elements per load

Warp layout within 8-warp block:
  - warp_row = warp_id // 2    (0-3)
  - warp_col = warp_id % 2     (0-1)
  - 4×2 grid of 16×32 subtiles = 64×64 elements per iteration
```

---

## 8. Implementation Details

### 8.1 Double Buffering (Ping-Pong)

The kernel maintains two sets of shared memory buffers:

```
Stage 0 buffers: A_s0, B_s0  (currently being computed)
Stage 1 buffers: A_s1, B_s1  (currently being loaded)

K-loop iteration:
  While computing from stage 0 → Load into stage 1
  While computing from stage 1 → Load into stage 0
  (swap stages each iteration)
```

### 8.2 Synchronization Primitives

| Primitive | Purpose | Wait Counter |
|-----------|---------|--------------|
| `s_waitcnt[vmcnt=N]` | Wait for global memory ops | At most N outstanding |
| `s_waitcnt[lgkmcnt=N]` | Wait for LDS operations | At most N outstanding |
| `s_barrier()` | Workgroup barrier | All waves synchronized |
| `schedule_barrier()` | Compiler scheduling fence | Prevent instruction reordering |

### 8.3 The Schedule Barrier Sandwich

Due to Mojo's LLVM backend being more aggressive at instruction reordering than HIP's compiler:

```mojo
# Required pattern for correct synchronization:
s_waitcnt[vmcnt=X]()
schedule_barrier()    # Prevent reordering before barrier
s_barrier()
schedule_barrier()    # Prevent reordering after barrier
```

### 8.4 Pointer Hoisting with `readfirstlane`

LDS pointers are hoisted to scalar registers for efficiency:

```mojo
self.lds_ptr[0][0][0] = readfirstlane(
    a_s0_g0.tile[Self.rows_per_warp, Self.BK](warp_id, 0).ptr
)
```

This moves the pointer from vector registers (VGPRs) to scalar registers (SGPRs), since all lanes in a warp share the same base pointer.

---

## 9. Performance Characteristics

### 9.1 Achieved Performance

| Matrix Size | Configuration | TFLOPs | Notes |
|-------------|---------------|--------|-------|
| 8192×8192×8192 | BF16, swizzle enabled | ~1.5 | Simplified schedule |
| 8192×8192×8192 | BF16, optimized schedule | ~1.5-1.6 | With tuned vmcnt |

### 9.2 Bottleneck Analysis

The kernel can be:

1. **Compute-bound**: MMA units fully utilized, memory prefetched in time
2. **Memory-bound**: Waiting for global loads or LDS operations
3. **LDS-bound**: Bank conflicts limiting shared memory bandwidth

Swizzle patterns address (3), double-buffering addresses (2).

### 9.3 Register Pressure

With 8 waves per CU and 512 VGPRs per SIMD:

- Each wave gets ~64 VGPRs
- Must fit: A fragments, B fragments, C accumulators, address calculations

The 256×256 output tile with 8 warps = 128×64 per warp = manageable register pressure.

---

## 10. Alternative Approaches

### 10.1 True Producer-Consumer (Cleaner Semantics)

From `~/HipKittens/kernels/gemm/bf16fp32/mi350x/micros/producer_consumer/16x32/micro_02_2stage_8c4p.cpp`:

```cpp
bool is_producer = (warp_group_id == 0);
bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
```

**Advantages:**

- Barriers mean what they're supposed to mean
- Control flow makes roles explicit
- Easier to understand and maintain

**Disadvantages:**

- Wastes registers on producer waves
- Smaller output tiles → lower arithmetic intensity
- ~5-10% lower performance on AMD

### 10.2 4-Wave Interleave

Single wave per SIMD with fine-grained instruction interleaving:

**Advantages:**

- No barrier abuse
- Can achieve highest peak performance
- Better for imbalanced workloads

**Disadvantages:**

- 3-4× more lines of code
- Much harder to write and maintain
- Requires deep understanding of instruction scheduling

### 10.3 Potential Future: Explicit Phase Primitives

A cleaner solution would be explicit phase/scheduling primitives:

```mojo
# Hypothetical API:
with phase_offset(group=1, delay=1):
    # Group 1 starts one phase behind
    pass

# Or explicit role switching:
@alternating_roles(compute_group=0, memory_group=1)
fn kernel_body():
    ...
```

This would make the scheduling intent explicit without abusing synchronization primitives.

---

## 11. Future Directions

### 11.1 Short-Term Improvements

1. **vmcnt Tuning**: Profile to find optimal wait counts between conservative (current) and HipKittens' aggressive values
2. **Tile Size Exploration**: Test 192×256 or other configurations
3. **K-dimension Unrolling**: Experiment with different K-step sizes

### 11.2 Medium-Term Considerations

1. **Refactor for Clarity**: Consider producer-consumer pattern if 5-10% performance loss is acceptable
2. **Abstract the Barrier Pattern**: Create a wrapper that documents the scheduling intent
3. **Profiling Infrastructure**: Build tools to identify bottlenecks

### 11.3 Long-Term Architecture

1. **DSL for Scheduling**: Create abstractions that express intent without low-level barrier manipulation
2. **Compiler Support**: Work with LLVM/Mojo team on better scheduling primitives
3. **Hardware Evolution**: Monitor AMD's future primitives (potential mbarrier equivalents)

---

---

## 12. Current Progress: A-Interleaving Schedule (~1000 GFLOPS/s)

*Updated: December 2024*

### 12.1 Key Discoveries

Through iterative experimentation, we discovered several important insights:

#### 12.1.1 Self-Loading vs Cross-Loading

**Cross-loading (original):** Each group loads data for the OTHER group

- WG0 loads A into WG1's buffer (g1)
- WG1 loads A into WG0's buffer (g0)
- Creates coupling that prevents overlap with stagger

**Self-loading (new):** Each group loads data for ITSELF

- WG0 loads A into WG0's buffer (g0)
- WG1 loads A into WG1's buffer (g1)
- Groups are independent → enables overlap!

#### 12.1.2 B Matrix Sharing Constraint

The B matrix is fundamentally shared across groups:

- h0 (columns 0-127) used by warp_id_n 0,1 from BOTH groups
- h1 (columns 128-255) used by warp_id_n 2,3 from BOTH groups

This sharing prevents B from being overlapped with compute in the current architecture.

#### 12.1.3 `s_barrier()` Works Well

Surprisingly, the simpler `s_barrier()` works well with the A-interleaved schedule:

- No need for complex atomic counter synchronization
- The timing/stagger effect is implicit in the barrier sequence
- Cleaner code, good performance

### 12.2 Current Schedule (~1000 GFLOPS/s)

```
PROLOGUE:
  Load A+B stage 0 → advance_k → barrier
  WG1 extra barrier (creates stagger)

MAIN LOOP (per 2×BK of K):
  ┌─────────────────────────────────────────────────────────┐
  │ STEP 1: barrier → load A+B stage 1 → waitcnt → advance  │
  │ STEP 2: barrier → compute[0] (uses stage 0)             │
  │ STEP 3: load A stage 0 (NO WAIT - OVERLAPPED!)          │  ← Key optimization
  │ STEP 4: barrier → compute[1] (uses stage 1)             │
  │ STEP 5: barrier → load B stage 0 → waitcnt → advance    │
  └─────────────────────────────────────────────────────────┘

EPILOGUE:
  WG0 extra barrier (resync)
```

**Key optimization:** A stage 0 load is issued DURING compute[0], overlapping memory latency with computation. This works because:

1. A is self-loaded (each group independent)
2. The barrier + compute[1] provides enough time for A load to complete
3. By next iteration's compute[0], A data is ready

**B cannot be moved** because it's shared across groups - moving it would create race conditions.

### 12.3 Performance Results

| Configuration | GFLOPS/s | Notes |
|---------------|----------|-------|
| Original cross-loading, symmetric | ~820 | Baseline |
| Self-loading A, symmetric | ~816 | Slight regression |
| Self-loading A, A-interleaved | **~1000** | 23% improvement |
| With extra barriers (MMA conflict avoidance) | ~772 | Barrier overhead too high |

### 12.4 Remaining Limitation

The **B matrix sharing** prevents true ping-pong overlap:

- Both groups need both h0 and h1 for compute
- Cannot overlap B load with compute without race conditions
- MMA conflict remains: WG0 compute[1] + WG1 compute[0] = 8 MMAs for 4 cores

---

## 13. Future Direction: Quadrant-Based Computation

### 13.1 The Core Insight

The current limitation stems from having a **single MmaOp** that needs all of B. What if we split the computation into **four independent quadrants**?

### 13.2 Quadrant Layout

```
Output 256×256:
┌─────────────────┬─────────────────┐
│ Q0: g0 × h0     │ Q1: g0 × h1     │  ← Uses WG0's A (g0)
│ (128×128)       │ (128×128)       │
├─────────────────┼─────────────────┤
│ Q2: g1 × h0     │ Q3: g1 × h1     │  ← Uses WG1's A (g1)
│ (128×128)       │ (128×128)       │
└─────────────────┴─────────────────┘
  Uses h0            Uses h1
```

Each quadrant is computed as: `Q[i,j] = A_half[i] × B_half[j]`

### 13.3 Diagonal Execution Strategy

**Diagonal 1 (fully independent):**

- WG0: Q0 = g0 × h0 (own A, own B)
- WG1: Q3 = g1 × h1 (own A, own B)
- **True overlap possible!** No shared data.

**Diagonal 2 (cross-access):**

- WG0: Q1 = g0 × h1 (own A, WG1's B)
- WG1: Q2 = g1 × h0 (own A, WG0's B)
- Requires barrier for B visibility, but data already in LDS.

### 13.4 Proposed Schedule

```
Per K iteration:
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Load (can overlap with prev Phase 3!)            │
│   WG0 loads g0, h0                                        │
│   WG1 loads g1, h1                                        │
│   waitcnt, barrier                                        │
├────────────────────────────────────────────────────────────┤
│ Phase 2: Diagonal 1 - TRUE OVERLAP POSSIBLE!              │
│   WG0: Q0 += g0 × h0    (with stagger)                   │
│   WG1: Q3 += g1 × h1                                      │
│   barrier (B visibility for phase 3)                      │
├────────────────────────────────────────────────────────────┤
│ Phase 3: Diagonal 2 (no new loads!)                       │
│   WG0: Q1 += g0 × h1                                      │
│   WG1: Q2 += g1 × h0                                      │
└────────────────────────────────────────────────────────────┘
```

### 13.5 LDS Memory: Same 128KB

The quadrant approach uses the **same LDS layout**:

- A: g0 + g1 = 2 × (128×64) × 2 stages × 2 bytes = **64KB**
- B: h0 + h1 = 2 × (128×64) × 2 stages × 2 bytes = **64KB**
- **Total: 128KB** (unchanged!)

### 13.6 Implementation Requirements

1. **Smaller MmaOp:** 4 warps computing 128×128 output (vs 8 warps computing 256×256)

2. **Two accumulators per group:** Each group maintains 2 quadrant accumulators
   - WG0: Q0 and Q1
   - WG1: Q2 and Q3

3. **Modified store logic:** Each group stores its two quadrants to the correct output locations

4. **Diagonal scheduling:** Within each K iteration:
   - Diagonal 1: Independent computation (overlap possible)
   - Diagonal 2: Cross-access computation (barrier needed)

### 13.7 Expected Benefits

| Aspect | Current | Quadrant |
|--------|---------|----------|
| Independence | A only | A AND B (for diagonal 1) |
| True overlap | Partial (A load only) | Full (diagonal 1) |
| MMA conflicts | Yes (8 MMAs) | No (4 MMAs per diagonal) |
| Target performance | ~1000 GFLOPS/s | ~1500-2000 GFLOPS/s |

### 13.8 Detailed Development Plan

#### Phase 1: Memory Footprint Analysis

**LDS Memory (UNCHANGED - 128KB total):**

| Buffer | Size per half | Stages | Total |
|--------|---------------|--------|-------|
| A: g0 | 128 × 64 × 2 bytes | 2 | 16KB |
| A: g1 | 128 × 64 × 2 bytes | 2 | 16KB |
| B: h0 | 64 × 128 × 2 bytes | 2 | 16KB |
| B: h1 | 64 × 128 × 2 bytes | 2 | 16KB |
| **Total LDS** | | | **64KB** |

*Note: Same layout as current implementation - no LDS changes required!*

**Register Memory per Warp:**

| Configuration | Accumulator Size | Per Lane | Notes |
|---------------|------------------|----------|-------|
| Current (8 warps, 256×256) | 128×64 = 8192 fp32 | 128 fp32 | Single quadrant |
| Quadrant (4 warps, 2×128×128) | 2×(64×64) = 8192 fp32 | 128 fp32 | Two quadrants |

*Register pressure is UNCHANGED because each warp handles smaller quadrants but has two of them.*

#### Phase 2: 4-Warp MmaOp Design

**Warp Layout for 128×128 Quadrant:**

```
Quadrant 128×128:
┌─────────────┬─────────────┐
│ Warp 0      │ Warp 1      │  warp_id_n=0,1
│ (64×64)     │ (64×64)     │
├─────────────┼─────────────┤
│ Warp 2      │ Warp 3      │  (local warp IDs within group)
│ (64×64)     │ (64×64)     │
└─────────────┴─────────────┘
  warp_id_m=0   warp_id_m=1
```

**MmaOp Parameters:**

```mojo
# Current 8-warp MmaOp:
WM = 128, WN = 64   # Per warp tile
BM = 256, BN = 256  # Block tile (8 warps)

# New 4-warp MmaOp:
WM = 64, WN = 64    # Per warp tile (halved both)
QM = 128, QN = 128  # Quadrant tile (4 warps)
```

**Fragment Layout:**

- MMA instruction: 16×16×16
- Fragments per warp: (64/16) × (64/16) = 4 × 4 = 16 fragments
- Each warp maintains 16 accumulator fragments per quadrant
- Two quadrants = 32 fragments per warp

#### Phase 3: Detailed Schedule with Overlap Analysis

**PROLOGUE:**

```
┌────────────────────────────────────────────────────────────────┐
│ WG0: load g0, h0 stage 0                                       │
│ WG1: load g1, h1 stage 0                                       │
│ waitcnt                                                        │
│ advance_k                                                      │
│ barrier                                                        │
│ if WG1: barrier (create stagger)                               │
└────────────────────────────────────────────────────────────────┘
```

**MAIN LOOP (per 2×BK of K):**

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Load stage 1 data                                           │
│ ─────────────────────────────────────────────────────────────────── │
│   barrier                                                           │
│   WG0: load g0, h0 stage 1   (4 loads)                             │
│   WG1: load g1, h1 stage 1   (4 loads)                             │
│   waitcnt, advance_k                                               │
│   Latency: ~800 cycles (HBM access)                                │
├─────────────────────────────────────────────────────────────────────┤
│ STEP 2: Diagonal 1, Stage 0 - FULLY INDEPENDENT!                    │
│ ─────────────────────────────────────────────────────────────────── │
│   barrier                                                           │
│   WG0: Q0 += g0 × h0   [16 MMAs]  ← Uses OWN data only             │
│   WG1: Q3 += g1 × h1   [16 MMAs]  ← Uses OWN data only             │
│                                                                     │
│   ★ TRUE OVERLAP POSSIBLE! ★                                       │
│   With stagger: WG0 computes while WG1 finishes loads              │
│                 WG1 computes while WG0 starts next loads           │
│   Compute: ~200 cycles (16 MMAs)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ STEP 3: Diagonal 2, Stage 0 - Cross-access                          │
│ ─────────────────────────────────────────────────────────────────── │
│   barrier (B visibility across groups)                              │
│   WG0: Q1 += g0 × h1   [16 MMAs]  ← Uses WG1's h1                  │
│   WG1: Q2 += g1 × h0   [16 MMAs]  ← Uses WG0's h0                  │
│   Compute: ~200 cycles                                             │
│                                                                     │
│   ★ OVERLAP OPPORTUNITY ★                                          │
│   Can issue A loads for next stage here (data already in LDS)      │
├─────────────────────────────────────────────────────────────────────┤
│ STEP 4: Diagonal 1, Stage 1 - FULLY INDEPENDENT!                    │
│ ─────────────────────────────────────────────────────────────────── │
│   barrier                                                           │
│   WG0: Q0 += g0 × h0   [16 MMAs]                                   │
│   WG1: Q3 += g1 × h1   [16 MMAs]                                   │
│                                                                     │
│   ★ TRUE OVERLAP! ★                                                │
│   Issue stage 0 A loads here (overlapped)                          │
│   WG0: load g0 stage 0 (no wait)                                   │
│   WG1: load g1 stage 0 (no wait)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ STEP 5: Diagonal 2, Stage 1 - Cross-access                          │
│ ─────────────────────────────────────────────────────────────────── │
│   barrier                                                           │
│   WG0: Q1 += g0 × h1   [16 MMAs]                                   │
│   WG1: Q2 += g1 × h0   [16 MMAs]                                   │
├─────────────────────────────────────────────────────────────────────┤
│ STEP 6: Load B stage 0 (if not last K)                              │
│ ─────────────────────────────────────────────────────────────────── │
│   barrier                                                           │
│   WG0: load h0 stage 0                                             │
│   WG1: load h1 stage 0                                             │
│   waitcnt, advance_k                                               │
└─────────────────────────────────────────────────────────────────────┘
```

#### Phase 4: Overlap Comparison

**Current Schedule (A-interleaved):**

```
Timeline per K iteration:
│ Load A+B │ Compute[0] │ Load A │ Compute[1] │ Load B │
│  stage1  │  stage0    │ stage0 │  stage1    │ stage0 │
│   800cy  │   400cy    │ (ovlp) │   400cy    │  400cy │

Overlap: Only A stage 0 load (partially hidden)
Effective memory latency: ~1200 cycles visible
```

**Quadrant Schedule:**

```
Timeline per K iteration:
│ Load    │ Diag1  │ Diag2  │ Diag1  │ Diag2  │ Load B │
│ A+B s1  │ stg0   │ stg0   │ stg1   │ stg1   │ stg0   │
│  800cy  │ 200cy  │ 200cy  │ 200cy  │ 200cy  │ 400cy  │
          │ ↑ OVERLAP between groups! ↑       │

With stagger:
- WG0 Diag1 overlaps with WG1 finishing loads
- WG1 Diag1 overlaps with WG0 starting next loads
- A loads can be issued during Diag1 (independent)

Effective memory latency: ~600-800 cycles visible (40-50% reduction!)
```

**Overlap Improvement:**

| Metric | Current | Quadrant | Improvement |
|--------|---------|----------|-------------|
| Diagonal 1 overlap | None | Full (between groups) | +100% |
| A load overlap | Partial | Full | +50% |
| Compute phases per K | 2 | 4 | 2× more granular |
| True overlap cycles | ~200 | ~600 | **3× more overlap** |
| Expected GFLOPS/s | ~1000 | ~1500-2000 | **50-100% gain** |

#### Phase 5: Implementation Checklist

**Step 1: Create QuadrantMmaOp struct**

```mojo
struct QuadrantMmaOp[...]:
    # 4 warps computing 128×128 quadrant
    # Each warp holds 64×64 accumulator
    # Parameters: WM=64, WN=64, same MMA_M/N/K
    
    # Two accumulator sets (for 2 quadrants)
    var accum_diag1: ...  # Q0 for WG0, Q3 for WG1
    var accum_diag2: ...  # Q1 for WG0, Q2 for WG1
    
    fn compute_diag1[stage](self, a_tile, b_tile): ...
    fn compute_diag2[stage](self, a_tile, b_tile): ...
```

**Step 2: Modify Buffers for quadrant access**

```mojo
# Add quadrant-specific MMA tile pointers
var a_quadrant_tiles: ...  # g0 for WG0, g1 for WG1
var b_own_tiles: ...       # h0 for WG0, h1 for WG1  
var b_cross_tiles: ...     # h1 for WG0, h0 for WG1
```

**Step 3: Implement quadrant schedule**

- Prologue: Load stage 0, create stagger
- Main loop: 6-step schedule per K iteration
- Epilogue: Re-sync and store

**Step 4: Update store logic**

```mojo
# WG0 stores:
#   Q0 → output[0:128, 0:128]
#   Q1 → output[0:128, 128:256]
# WG1 stores:
#   Q2 → output[128:256, 0:128]
#   Q3 → output[128:256, 128:256]
```

**Step 5: Testing milestones**

- [ ] QuadrantMmaOp computes single quadrant correctly
- [ ] Two quadrants per group accumulate correctly
- [ ] Diagonal 1 produces correct results (independent)
- [ ] Diagonal 2 produces correct results (cross-access)
- [ ] Full K-loop produces correct output
- [ ] Store writes to correct output locations
- [ ] Performance benchmark vs current

#### Phase 6: Risk Assessment

| Risk | Mitigation |
|------|------------|
| Register pressure | Pre-calculated: same as current (128 fp32/lane) |
| LDS pressure | Unchanged (128KB) |
| Barrier overhead (6 per K vs 4) | Offset by 3× more overlap |
| Code complexity | Modular design with clear diagonal separation |
| MMA instruction compatibility | Use same MMA_M/N/K, just smaller tiles |

---

## 14. Deep Dive: Race Condition Analysis and Counter Semantics

*Updated: December 2024*

### 14.1 AMD Memory Counter Semantics (from CDNA4 ISA)

AMD GPUs use **per-wave counters** to track outstanding memory operations:

#### VM_CNT (Vector Memory Count)

- **Scope**: Per-wave (not shared across workgroup)
- **Incremented**: When vector-memory read/write (MUBUF, MTBUF, FLAT) is issued
- **Decremented**: When data written to VGPRs (reads) or L2 cache (writes)
- **Ordering**: Memory reads and writes return **in order** (FIFO)
- **Usage**: `s_waitcnt[vmcnt=N]()` waits until ≤N operations remain in flight

#### LGKM_CNT (LDS/GDS/K-constant/Message Count)

- **Scope**: Per-wave (not shared across workgroup)
- **Incremented**: +1 for each LDS instruction, +1 for each FLAT instruction
- **Decremented**: When LDS read returns to VGPRs, or LDS write completes
- **Ordering**: Same-type instructions return **in order** (FIFO)
- **Usage**: `s_waitcnt[lgkmcnt=N]()` waits until ≤N operations remain in flight

#### Critical Insight: Per-Wave Counters

```
s_waitcnt[vmcnt=0]() on WG0 ensures WG0's writes complete.
It says NOTHING about WG1's state!

For cross-wave visibility, you need:
1. Each wave waits for its own operations (s_waitcnt)
2. All waves synchronize (s_barrier)
3. Then data is visible across waves
```

### 14.2 Async Load Counts for Our Configuration

**Kernel Configuration:**

```
block_shape = (BM=256, BN=256, BK=64)
warp_shape  = (WM=128, WN=64, WK=64)
mma_shape   = (MMA_M=16, MMA_N=16, MMA_K=32)
```

**Derived Values:**

```
num_m_mmas = WM / MMA_M = 128 / 16 = 8
num_n_mmas = WN / MMA_N = 64 / 16 = 4
num_k_mmas = BK / MMA_K = 64 / 32 = 2
half_m_mmas = 4, half_n_mmas = 2
load_width = simd_width_of[BFloat16] = 8
```

**LDS → Registers (lgkmcnt):**
| Operation | Formula | Count |
|-----------|---------|-------|
| `mma_op.load_a[which]` | half_m_mmas × num_k_mmas | **8 ds_read ops** |
| `mma_op.load_b[which]` | half_n_mmas × num_k_mmas | **4 ds_read ops** |
| Combined load_a + load_b | | **12 ds_read ops** |

**Global → LDS (vmcnt) with 8-warp loading:**
| Operation | Formula | Count |
|-----------|---------|-------|
| `buffers.load_a[stage, which]` | (BM/2) / rows_per_iter | **2 load_to_lds ops** |
| `buffers.load_b[stage, which]` | (BN/2) / rows_per_iter | **2 load_to_lds ops** |

**Global → LDS (vmcnt) with 4-warp loading:**
| Operation | Formula | Count |
|-----------|---------|-------|
| `buffers.load_a_as_group` | (BM/2) / 32 | **4 load_to_lds ops** |
| `buffers.load_b_as_group` | (BN/2) / 32 | **4 load_to_lds ops** |

### 14.3 The Shared B Buffer Problem

#### Buffer Allocation Structure

```
A buffers: SEPARATE per warp group
  a_s0_g0 - stage 0, warp_id_m=0 (WG0 only)
  a_s0_g1 - stage 0, warp_id_m=1 (WG1 only)
  a_s1_g0 - stage 1, warp_id_m=0 (WG0 only)
  a_s1_g1 - stage 1, warp_id_m=1 (WG1 only)

B buffers: SHARED across warp groups!
  b_s0_h0 - stage 0, warp_n 0-1 (BOTH groups read)
  b_s0_h1 - stage 0, warp_n 2-3 (BOTH groups read)
  b_s1_h0 - stage 1, warp_n 0-1 (BOTH groups read)
  b_s1_h1 - stage 1, warp_n 2-3 (BOTH groups read)
```

#### The Race Condition with Stagger

With the warp stagger (WG1 one barrier behind WG0):

```
Timeline with stagger:
  WG0: [Load B stage 0] → [Compute from stage 0] → [Load B stage 0 for K+1]
                                                          ↑ WRITES to B
  WG1:        [barrier] → [Load B stage 0] → [Compute from stage 0]
                                                    ↑ READS from B

  RACE: WG0 writes to B while WG1 reads from B!
```

The aggressive optimized schedule interleaves loads and compute:

```mojo
mma_op.load_b[1](b_mma_tiles[0][1])  // WG1 reading from stage 0
buffers.load_b[0, 0](b_resource)     // WG0 writing to stage 0 - RACE!
```

### 14.4 Why the Simplified Schedule Works

The simplified schedule is safe despite using the same stagger because:

1. **`vmcnt=0` after every batch of global loads**

   ```mojo
   buffers.load_b_as_group[1, 0](...)
   buffers.load_b_as_group[1, 1](...)
   s_waitcnt[vmcnt=0]()  // ALL loads complete before proceeding
   ```

2. **`barrier_sync()` between every major phase**

   ```mojo
   s_waitcnt[vmcnt=0]()
   buffers.advance_k()
   barrier_sync()        // ALL waves sync before compute
   compute_stage[0]()
   ```

3. **Grouped operations - no interleaving across phases**

   ```
   Load Phase:    [All loads] → vmcnt=0 → barrier
   Compute Phase: [All compute] → barrier
   ```

Both groups finish ALL their loads (vmcnt=0) and sync (barrier) BEFORE any group starts reading from LDS. The stagger offsets when groups enter each phase, but within each phase, everything is synchronized.

### 14.5 Why the Optimized Schedule Fails

The optimized schedule fails because:

1. **Interleaved loads and compute** - B writes happen during compute phase
2. **Aggressive vmcnt values** - `vmcnt=6` leaves writes in flight across barriers
3. **Shared B buffers** - both groups access the same LDS regions
4. **Stagger** - groups are at different iterations, causing write/read conflicts

Attempted fixes that didn't work:

- `vmcnt=0` before barriers: Doesn't help because the issue is cross-group, not within-wave
- `barrier()` with memory fences: Destroys performance, still has race
- Extra barriers before B writes: Still fails because stagger puts groups at different phases

### 14.6 Path Forward: Gradual Optimization

**Strategy**: Start from working simplified schedule, gradually add optimizations while maintaining correctness.

#### Phase 1: Optimize compute_stage (SAFE - intra-phase)

Fine-grained lgkmcnt waits within compute:

```mojo
fn compute_stage[stage: Int]():
    mma_op.load_b[0](...)  // +4 lgkm
    mma_op.load_a[0](...)  // +8 lgkm = 12 total
    s_waitcnt[lgkmcnt=8]() // Wait for load_b only
    mma_op.mma[0, 0]()     // Can start with just b[0], a[0] partial
    
    mma_op.load_b[1](...)  // +4 lgkm
    s_waitcnt[lgkmcnt=4]() // Wait for load_a[0] complete
    mma_op.mma[0, 1]()
    // ... etc
```

#### Phase 2: Interleave A loads with compute (SAFE - A is per-group)

```mojo
compute_stage[0]()
buffers.load_a_as_group[0, 0](...)  // Safe: A is separate per group
barrier_sync()
compute_stage[1]()
```

#### Phase 3: Interleave B loads with compute (CAREFUL - B is shared)

Options:

- **Option A**: Duplicate B buffers (doubles B LDS usage: 64KB → 128KB)
- **Option B**: Remove stagger, use lockstep execution
- **Option C**: Add explicit B protection barriers (current approach, failed)

#### Phase 4: Consider alternative scheduling patterns

- 4-wave interleave (no stagger needed)
- Producer-consumer with dedicated load waves
- Quadrant-based with independent B regions

### 14.7 Key Learnings

| Finding | Implication |
|---------|-------------|
| vmcnt/lgkmcnt are per-wave | Cross-wave sync needs barrier + waitcnt |
| B buffers are shared | Stagger + B interleaving = race |
| s_barrier is control-flow only | Memory visibility needs vmcnt=0 BEFORE barrier |
| Simplified schedule works | Conservative sync (vmcnt=0 + barrier) is correct |
| Optimized schedule fails | Aggressive interleaving breaks with shared B |

---

## 15. Detailed Schedule Trace Analysis

*Updated: December 2024*

This section provides a step-by-step trace of both the HipKittens reference schedule and our Mojo optimized schedule, tracking barrier counts and buffer accesses with the stagger.

### 15.1 Notation

- **WG0**: Warp group 0 (warp_id_m == 0, warps 0-3)
- **WG1**: Warp group 1 (warp_id_m == 1, warps 4-7)
- **B[n]**: Barrier number n (all 8 warps must hit barrier n before any proceeds past it)
- **As[stage][half]**: A buffer, stage 0 or 1, half 0 or 1 (g0/g1)
- **Bs[stage][half]**: B buffer, stage 0 or 1, half 0 or 1 (h0/h1)
- **→LDS**: Global to LDS write (tracked by vmcnt)
- **LDS→**: LDS to register read (tracked by lgkmcnt)

### 15.2 HipKittens Reference Schedule (256_256_64_32_with16x32.cpp)

#### Buffer Layout

```cpp
ST_A (&As)[2][2]  // As[stage][row_half] - 2 stages × 2 row halves
ST_B (&Bs)[2][2]  // Bs[stage][col_half] - 2 stages × 2 col halves
```

#### Prologue

```
Line 117-120: All warps load stage 0
  Bs[0][0] →LDS  (vmcnt +1)
  As[0][0] →LDS  (vmcnt +1)
  Bs[0][1] →LDS  (vmcnt +1)
  As[0][1] →LDS  (vmcnt +1)
  Total: vmcnt = 4

Line 122-124: STAGGER
  if (warp_row == 1) s_barrier()  // WG1 hits B[0], WG0 skips

Line 126-127:
  WG0: s_waitcnt vmcnt(4)  // Wait for 0 loads (4 in flight, wait until ≤4)
       s_barrier()          // Hits B[0] - releases WG1
  
  WG1: [already past B[0]]
       s_waitcnt vmcnt(4)
       s_barrier()          // Hits B[1]

Line 129-131: Load stage 1 (first 3)
  Bs[1][0] →LDS  (vmcnt +1)
  As[1][0] →LDS  (vmcnt +1)
  Bs[1][1] →LDS  (vmcnt +1)
  Total: vmcnt = 3 (from this batch)

Line 133-134:
  s_waitcnt vmcnt(6)  // Wait until ≤6 in flight
  s_barrier()         // B[1] for WG0, B[2] for WG1
```

**After prologue barrier state:**

- WG0: passed B[1]
- WG1: passed B[2] (one ahead due to stagger)

#### Main Loop Iteration (tile 0→1)

```
STEP 1: LDS reads + global load As[1][1]
Line 140-142:
  load B_tile_0 LDS→ from Bs[0][0]  (lgkmcnt +4)
  load A_tile   LDS→ from As[0][0]  (lgkmcnt +8)

Line 143:
  As[1][1] →LDS  (vmcnt +1)

Line 144-145:
  s_waitcnt lgkmcnt(8)  // A_tile loaded
  s_barrier()           // B[2] for WG0, B[3] for WG1

STEP 2: MMA + barrier
Line 147-151:
  s_waitcnt lgkmcnt(0)
  mma C[0][0] += A_tile × B_tile_0
  s_barrier()           // B[3] for WG0, B[4] for WG1

STEP 3: LDS read B[0][1] + global load Bs[0][0]
Line 154-157:
  load B_tile_1 LDS→ from Bs[0][1]  (lgkmcnt +4)
  Bs[0][0] →LDS  (vmcnt +1)  *** WRITES TO Bs[0][0]! ***
  s_barrier()           // B[4] for WG0, B[5] for WG1

  ┌─────────────────────────────────────────────────────────────┐
  │ CRITICAL OBSERVATION:                                        │
  │ WG0 at B[4]: writes Bs[0][0]                                │
  │ WG1 at B[5]: already past reading Bs[0][0] (was at B[2])    │
  │ The stagger ensures WG1 reads BEFORE WG0 writes!            │
  └─────────────────────────────────────────────────────────────┘

STEP 4: MMA
Line 159-163:
  s_waitcnt lgkmcnt(0)
  mma C[0][1] += A_tile × B_tile_1
  s_barrier()           // B[5] for WG0, B[6] for WG1

STEP 5: LDS read As[0][1] + global load As[0][0]
Line 165-168:
  load A_tile LDS→ from As[0][1]  (lgkmcnt +8)
  As[0][0] →LDS  (vmcnt +1)
  s_barrier()           // B[6] for WG0, B[7] for WG1

STEP 6: MMA
Line 170-174:
  s_waitcnt lgkmcnt(0)
  mma C[1][0] += A_tile × B_tile_0
  s_barrier()           // B[7] for WG0, B[8] for WG1

STEP 7: LDS read Bs[1][0] + global load Bs[0][1] + vmcnt wait
Line 177-181:
  load B_tile_0 LDS→ from Bs[1][0]  (lgkmcnt +4)
  Bs[0][1] →LDS  (vmcnt +1)  *** WRITES TO Bs[0][1]! ***
  s_waitcnt vmcnt(6)    // Critical: wait for prior writes
  s_barrier()           // B[8] for WG0, B[9] for WG1

STEP 8: MMA
Line 183-186:
  mma C[1][1] += A_tile × B_tile_1
  s_barrier()           // B[9] for WG0, B[10] for WG1

... (continues with stage 1 reads and stage 0 writes)
```

### 15.3 Key Insight: How HipKittens Avoids Races

The stagger creates a **2-barrier offset**:

- WG1 is always 1 barrier AHEAD of WG0 (from initial conditional barrier)
- When WG0 writes to Bs[0][0] at B[4], WG1 already read it at B[2]

**Timeline visualization:**

```
Barrier:     B[0]  B[1]  B[2]  B[3]  B[4]  B[5]  B[6]  B[7]  B[8]
             ─────────────────────────────────────────────────────
WG0:         skip  ───●───●───●───●───●───●───●───●
                      ↑           ↑
                      │           └─ WG0 WRITES Bs[0][0]
                      └─ WG0 reads Bs[0][0]

WG1:         ───●───●───●───●───●───●───●───●───●
                 ↑       ↑
                 │       └─ WG1 reads Bs[0][0] (BEFORE WG0 writes!)
                 └─ WG1 stagger barrier

Key: The 2-barrier gap ensures reads complete before writes begin.
```

### 15.4 Mojo Optimized Schedule Comparison

Let me now trace through our Mojo optimized schedule:

#### Prologue

```mojo
Line 1739-1742: All warps load stage 0
  buffers.load_b[0, 0]  // Bs[0][0] →LDS
  buffers.load_a[0, 0]  // As[0][0] →LDS
  buffers.load_b[0, 1]  // Bs[0][1] →LDS
  buffers.load_a[0, 1]  // As[0][1] →LDS
  vmcnt = 4

Line 1747-1748: STAGGER
  if (warp_id_m == 1) s_barrier()  // WG1 hits B[0]

Line 1750-1753:
  s_waitcnt vmcnt(0)   *** DIFFERENCE: vmcnt(0) not vmcnt(4)! ***
  s_barrier()          // B[0] for WG0, B[1] for WG1

Line 1756-1758: Load stage 1 (first 3)
  buffers.load_b[1, 0]
  buffers.load_a[1, 0]
  buffers.load_b[1, 1]
  vmcnt = 3

Line 1760-1763:
  s_waitcnt vmcnt(0)   *** DIFFERENCE: vmcnt(0) not vmcnt(6)! ***
  s_barrier()          // B[1] for WG0, B[2] for WG1
```

#### Main Loop Differences

**HipKittens uses `vmcnt(4)` and `vmcnt(6)` - ALLOWS loads to overlap:**

```cpp
asm volatile("s_waitcnt vmcnt(6)");  // 6 loads can still be in flight!
```

**Mojo uses `vmcnt(0)` - BLOCKS until all complete:**

```mojo
s_waitcnt[vmcnt=0]()  // All loads must complete!
```

### 15.5 Verified vmcnt Counts

**Per-load vmcnt counts (both HipKittens and Mojo):**

- Half-tile size: 128 × 64 × 2 bytes = 16,384 bytes
- bytes_per_thread = 16 (128-bit load)
- threads = 512 (8 warps × 64)
- bytes_per_memcpy = 16 × 512 = 8,192
- **loads_per_half_tile = 16,384 / 8,192 = 2**

Each `G::load` (HipKittens) or `buffers.load_*[stage, which]` (Mojo) issues **2 vmcnt ops**.

### 15.6 Corrected vmcnt Analysis

**HipKittens Prologue:**

```cpp
G::load(Bs[0][0], ...)  // vmcnt +2 = 2
G::load(As[0][0], ...)  // vmcnt +2 = 4
G::load(Bs[0][1], ...)  // vmcnt +2 = 6
G::load(As[0][1], ...)  // vmcnt +2 = 8

s_waitcnt vmcnt(4)      // Wait until ≤4 in flight
                        // → First 4 loads (Bs[0][0], As[0][0]) complete
                        // → Bs[0][1], As[0][1] still in flight!
```

**Mojo Prologue:**

```mojo
buffers.load_b[0, 0]    // vmcnt +2 = 2
buffers.load_a[0, 0]    // vmcnt +2 = 4
buffers.load_b[0, 1]    // vmcnt +2 = 6
buffers.load_a[0, 1]    // vmcnt +2 = 8

s_waitcnt[vmcnt=0]()    // Wait for ALL 8
                        // → No overlap possible!
```

**After prologue barrier + 3 more loads (stage 1):**

```cpp
// HipKittens:
G::load(Bs[1][0], ...)  // vmcnt +2
G::load(As[1][0], ...)  // vmcnt +2
G::load(Bs[1][1], ...)  // vmcnt +2
// With 4 still in flight from prologue: vmcnt = 4 + 6 = 10?

s_waitcnt vmcnt(6)      // Wait until ≤6 in flight
```

**CRITICAL INSIGHT**: HipKittens allows **4-6 loads to remain in flight across barriers**:

- After prologue: vmcnt(4) keeps 4 loads in flight
- After stage 1 setup: vmcnt(6) keeps 6 loads in flight

This creates **overlap**: while the barrier ensures control-flow sync, the global→LDS transfers continue asynchronously in the background!

### 15.7 Why Our vmcnt(0) is Necessary (For Now)

When we tried matching HipKittens' vmcnt values, we got race conditions because:

1. **Compiler reordering**: LLVM may reorder instructions around our `s_waitcnt`
2. **Missing schedule_barrier()**: HipKittens uses raw `asm volatile` which prevents reordering; our Mojo intrinsics may not
3. **Different LDS load semantics**: Our `_load_from_lds` may interact differently with the outstanding global loads

### 15.8 Proposed Fix: Match HipKittens with Proper Guards

```mojo
// Pattern: schedule_barrier before AND after waitcnt
schedule_barrier()
s_waitcnt[vmcnt=4]()
schedule_barrier()
s_barrier()
schedule_barrier()
```

This ensures:

1. No instructions reorder past the schedule_barrier
2. vmcnt is respected
3. Barrier happens after vmcnt

### 15.9 Experimental Results (December 2024)

We tested various vmcnt configurations:

| Schedule | vmcnt Config | GFLOPS/s | Status |
|----------|--------------|----------|--------|
| Simplified | vmcnt=0 everywhere | ~1000 | ✅ Works |
| Optimized | vmcnt=0 everywhere | ~1025 | ❌ Race (intermittent) |
| Optimized | vmcnt=4/6 (HipKittens) | ~1550 | ❌ Race (intermittent) |
| Optimized | vmcnt=6 main loop only | ~1540 | ❌ Race (intermittent) |

**Critical Finding**: The optimized schedule has race conditions **regardless of vmcnt values**!

### 15.10 Root Cause Hypothesis

The race is NOT from vmcnt values alone. It's from the **fundamental interleaving pattern**:

```
// Optimized schedule interleaves:
mma_op.load_b[0](...)     // LDS→reg read from stage 0
buffers.load_b[0, 1](...)  // global→LDS write to stage 0!
s_waitcnt[vmcnt=N]()
s_barrier()
```

The `mma_op.load_*` reads from LDS while `buffers.load_*` writes to the same LDS buffers (with stagger, potentially the same stage). Even with vmcnt=0 and barriers, the compiler may:

1. **Reorder LDS reads** past the vmcnt wait
2. **Hoist LDS reads** before global→LDS completes
3. **Schedule LDS reads** based on alias analysis that doesn't account for async writes

### 15.11 Why Simplified Schedule Works

The simplified schedule has **strict phase separation**:

```
// Phase 1: ALL loads complete
load_a_as_group[1, 0](...)
load_a_as_group[1, 1](...)
load_b_as_group[1, 0](...)
load_b_as_group[1, 1](...)
s_waitcnt[vmcnt=0]()       // WAIT before anything else
barrier_sync()

// Phase 2: ALL compute (no interleaving with loads)
compute_stage[0]()         // LDS reads + MMA, no global writes
```

No LDS reads happen while global→LDS writes are in flight.

### 15.12 Potential Fixes for Optimized Schedule

1. **Use inline assembly for LDS reads**: `ds_read_b128` with proper constraints
2. **Add memory fences**: Use `barrier()` instead of `s_barrier()` at critical points
3. **Restructure interleaving**: Ensure LDS reads don't overlap with writes to same stage
4. **Inspect generated ISA**: Compare Mojo output with HipKittens assembly

### 15.13 Root Cause Found: Epilogue vmcnt Values (December 2024)

**The race condition was in the EPILOGUE, not the main loop!**

#### The Bug: Incorrect vmcnt Assumptions

**EPILOGUE BLOCK 1** (line ~1835):

```mojo
// OLD (wrong):
buffers.load_a[1, 1](a_resource)  // +2 vmcnt
// ... LDS loads and MMA ...
s_waitcnt[vmcnt=2]()              // ❌ Assumed only 2 in flight!
```

**Problem**: At main loop exit, `vmcnt=6` left up to 6 loads in flight. After epilogue's `load_a[1,1]` (+2), vmcnt could be **up to 8**, not 2!

**EPILOGUE BLOCK 2** (line ~1850):

```mojo
// OLD (wrong):
s_waitcnt[vmcnt=1]()              // ❌ Leaves 1 load potentially incomplete!
```

**Problem**: `vmcnt=1` means "wait until ≤1 in flight" - but that 1 load might be the one we're about to read from!

#### The Fix

```mojo
// NEW (correct):
s_waitcnt[vmcnt=0]()              // ✅ ALL loads complete, data in LDS
```

#### Why vmcnt=0 is Necessary

1. **Per-wave counters**: vmcnt is per-wave, not shared across warp groups
2. **Stagger interference**: With stagger, one group's vmcnt state affects data availability for the other
3. **Conservative is correct**: `vmcnt=0` makes no assumptions about in-flight count
4. **Main loop contamination**: The `vmcnt=6` in the main loop leaves loads in flight that persist into the epilogue

#### Performance Impact

| Configuration | GFLOPS/s | Status |
|---------------|----------|--------|
| Epilogue vmcnt=2,1 (old) | ~1550 | ❌ Race condition |
| Epilogue vmcnt=0 (new) | ~1550 | ✅ Correct |

**Key insight**: The epilogue vmcnt fix has **minimal performance impact** because the epilogue only runs once per block, not in the hot K-loop.

### 15.14 Final Working Configuration

The optimized schedule now works correctly with:

- **Prologue**: vmcnt=0 (conservative)
- **Main loop**: vmcnt=6 (aggressive, provides overlap)
- **Epilogue**: vmcnt=0 (conservative, critical for correctness)

---

## Appendix A: Key Code Locations

| Component | File |
|-----------|------|
| Mojo Ping-Pong Kernel | `pingpong_kernel.mojo` |
| Mojo Warp-Specialized Kernel | `warp_spec_matmul.mojo` |
| Ring Buffer Traits (Atomic Sync) | `ring_buffer_traits.mojo` |
| Ring Buffer Implementation | `ring_buffer.mojo` |
| Reference C++ (HipKittens) | `~/HipKittens/analysis/bf16_gemm/mi350x/kernel_4096.cpp` |
| Producer-Consumer Example | `~/HipKittens/kernels/gemm/bf16fp32/mi350x/micros/producer_consumer/16x32/micro_02_2stage_8c4p.cpp` |
| Swizzle Patterns | `~/HipKittens/include/types/shared/st_shape.cuh` |
| LDS Load/Store | `~/HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh` |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| CU | Compute Unit - AMD's equivalent of NVIDIA SM |
| SIMD | Single Instruction Multiple Data unit within a CU |
| Wave/Wavefront | 64-thread execution unit (AMD's warp) |
| XCD | Accelerator Complex Die - chiplet within AMD GPU |
| LDS | Local Data Share - AMD's shared memory |
| VGPR | Vector General Purpose Register |
| SGPR | Scalar General Purpose Register |
| MFMA | Matrix Fused Multiply Add instruction |
| vmcnt | Vector memory operation counter (per-wave, tracks global memory ops) |
| lgkmcnt | LDS/GDS/Scalar memory operation counter (per-wave, tracks LDS ops) |
| s_waitcnt | Instruction to wait for counters to reach threshold |
| s_barrier | Workgroup control-flow barrier (NOT a memory fence) |
| barrier() | Workgroup barrier WITH memory fences (release + acquire) |
| load_to_lds | Direct global→LDS transfer, tracked by vmcnt |
| ds_read | LDS→register transfer, tracked by lgkmcnt |
| Stagger | Technique where one warp group starts one barrier ahead |
| Double buffering | Using two buffer sets to overlap load and compute |
| Ping-pong | Alternating between two buffer stages across K iterations |

## Appendix C: References

1. HipKittens Paper: Hu et al., "HipKittens: Fast and Furious AMD Kernels", arXiv:2511.08083, 2025
2. AMD CDNA3 ISA: [AMD Instinct MI300 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
3. AMD CDNA4 Architecture: [CDNA4 Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
4. ThunderKittens: [Stanford HAI ThunderKittens](https://github.com/HazyResearch/ThunderKittens)

---

*Document created: December 2024*
*Based on analysis of HipKittens framework and pingpong_kernel.mojo implementation*
