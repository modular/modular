# SM100 Matmul Kernel Refactoring Plan

## Executive Summary

This document outlines a detailed plan for refactoring the SM100 (NVIDIA B200
GPU) matmul kernel to follow the modular, structured approach successfully
implemented in the SM90 (NVIDIA H100 GPU) kernel. The goal is to improve code
organization, maintainability, testability, and reusability by isolating
responsibilities into well-defined components.

---

## 1. Current Architecture Comparison

### 1.1 SM90 Kernel Structure (Target Architecture)

The SM90 kernel has been refactored into a clean, modular architecture with
the following components:

```text
sm90/
├── matmul_kernels.mojo    # Main kernel struct and entry points
├── matmul_output.mojo     # Output tile writing orchestration
├── ring_buffer.mojo       # Producer-consumer synchronization
├── tile_loader.mojo       # Tile loading abstractions (TMA, CPAsync)
├── tile_writer.mojo       # Tile writing abstractions (TMA, threadwise)
├── dispatch.mojo          # Dispatch logic
└── grouped_matmul.mojo    # Grouped/MoE variant
```

**Key Design Patterns in SM90:**

1. **Shared Memory Management (`HopperMatmulSM90Kernel_SMem`)**
   - Declarative struct defining all shared memory allocations
   - Manages A/B tile arrays, C tile, and pipeline barriers
   - Provides storage size calculations

2. **Ring Buffer Abstraction (`RingBuffer`)**
   - Context manager-based producer/consumer API
   - Encapsulates barrier synchronization logic
   - Supports both TMA and CPAsync transfers
   - Clean separation between producer and consumer views

3. **Tile Loader Trait (`TileLoader`)**
   - Abstract interface for tile loading
   - Two implementations: `TileLoaderTMA`, `TileLoaderCPAsync`
   - Handles multicast, partitioned multicast, and single-block modes

4. **Tile Writer Abstractions (`SMemTileWriter`, `RegTileWriter`)**
   - `TileWriterTMA`: Hardware-accelerated TMA stores
   - `TileWriterThreadwise`: Thread-distributed stores with swizzling
   - `FragmentToSMemWriter`: Register → shared memory via st.matrix
   - `RegisterToGMemWriter`: Register → global memory with epilogue support

5. **Main Kernel Struct (`HopperMatmulSM90Kernel`)**
   - Template-heavy configuration
   - Factory methods for building loaders, ring buffers
   - Clear separation of producer/consumer main loops

### 1.2 SM100 Kernel Structure (Current State)

The SM100 kernel is currently a monolithic implementation in a single large
file (~3400 lines):
file (~3400 lines):

```text
sm100/
├── matmul.mojo           # Everything: kernel, helpers, epilogue, stores
├── config.mojo           # Configuration struct
├── dispatch.mojo         # Dispatch logic
├── pipeline.mojo         # ProducerConsumerPipeline struct
├── tile_scheduler.mojo   # CLC-based tile scheduling
└── tile_scheduler_splitk.mojo  # Split-K scheduling
```

**Current Issues in SM100:**

1. **Monolithic Structure**: The main `matmul.mojo` file contains:
   - Shared memory struct (`B200MatmulSmem`)
   - Tile loading function (`load_AB`)
   - Consumer main loop (`consumer_main_loop`)
   - Multiple output functions (`multi_stage_store_C`, `multi_stage_store_C_split_k`)
   - Register/shared memory epilogue logic
   - Fragment conversion helpers
   - Multiple kernel variants

2. **Tight Coupling**:
   - Loading, computing, and storing logic are intertwined
   - No abstraction over different loading mechanisms
   - Epilogue logic duplicated across variants

3. **Hard to Extend**:
   - Adding new features requires modifying the core kernel
   - Testing individual components is difficult

---

## 2. Key Architectural Differences

### 2.1 Warp Specialization Model

| Aspect | SM90 (H100) | SM100 (B200) |
|--------|-------------|--------------|
| Producer | 1 warp group (128 threads) | 1 warp (32 threads) |
| Consumer | N-1 warp groups | 1 warp (MMA) + 4 warps (Epilogue) |
| Scheduler | Implicit in cluster | 1 warp (CLC-based) |
| Total Threads | 128-384 | 224 (7 warps) |

### 2.2 Memory Architecture

| Aspect | SM90 (H100) | SM100 (B200) |
|--------|-------------|--------------|
| Accumulators | Registers | Tensor Memory (TMEM) |
| MMA Type | WGMMA (async) | UMMA (via TMEM) |
| Output Path | Reg → SMem → GMem | TMEM → Reg → SMem → GMem |
| Pipeline | Full/Empty barriers | Full/Empty + CLC pipeline |

### 2.3 Pipeline Stages

| Pipeline | SM90 | SM100 |
|----------|------|-------|
| TMA → MMA | `num_pipeline_stages` | `num_pipeline_stages // k_group_size` |
| MMA → Output | N/A (synchronous) | `num_accum_pipeline_stages` |
| CLC Throttle | N/A | `num_clc_pipeline_stages` |
| Output Double-Buffer | Yes | `num_output_stages` |

---

## 3. Proposed Refactoring Plan

### Phase 1: Extract Shared Memory Organization

**Goal**: Create a clean shared memory management module.

**Files to Create**:

- `sm100/smem.mojo` - Shared memory struct and helpers

**Changes**:

```mojo
# sm100/smem.mojo

@register_passable("trivial")
struct B200MatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Shared memory organization for B200 matmul kernel.
    
    Manages:
    - A/B tile storage (multi-stage pipeline)
    - C tile storage (double-buffered output)
    - TMA-MMA barriers
    - Accumulator barriers  
    - CLC scheduling barriers
    - TMEM allocation storage
    """
    
    # [Keep existing fields]
    
    @staticmethod
    fn a_smem_size() -> Int: ...
    @staticmethod  
    fn b_smem_size() -> Int: ...
    @staticmethod
    fn c_smem_size() -> Int: ...
    @staticmethod
    fn total_size() -> Int: ...
    
    fn get_a_smem_iter(self) -> LayoutTensorIter[...]: ...
    fn get_b_smem_iter(self) -> LayoutTensorIter[...]: ...
    fn get_c_smem_iter(self) -> LayoutTensorIter[...]: ...
```

### Phase 2: Extract Pipeline Abstractions

**Goal**: Generalize the pipeline management for multiple use cases.

**Files to Modify**:

- `sm100/pipeline.mojo` - Extend with SM100-specific features

**New Abstractions**:

```mojo
# sm100/pipeline.mojo

@register_passable("trivial")
struct TMAMmaPipeline[num_stages: Int, k_group_size: Int]:
    """Pipeline for TMA load → MMA synchronization.
    
    Groups k_group_size TMA loads per barrier synchronization
    for reduced overhead on small tiles.
    """
    var inner: ProducerConsumerPipeline[num_stages // k_group_size]
    
    fn producer_acquire(mut self) -> Tuple[UInt32, MbarPtr]: ...
    fn producer_release(mut self): ...
    fn consumer_acquire(mut self) -> UInt32: ...
    fn consumer_release(mut self, mbar: MbarPtr): ...

@register_passable("trivial")
struct AccumPipeline[num_stages: Int]:
    """Pipeline for MMA → Epilogue synchronization.
    
    Manages TMEM accumulator handoff between MMA and output warps.
    """
    var inner: ProducerConsumerPipeline[num_stages]
    
    fn get_tmem_offset(self, stage: UInt32, base_addr: UInt32) -> UInt32: ...
```

### Phase 3: Extract Tile Loading Module

**Goal**: Create a unified tile loading interface matching SM90's design.

**Files to Create**:

- `sm100/tile_loader.mojo`

**Design**:

```mojo
# sm100/tile_loader.mojo

@register_passable("trivial")
trait TileLoaderSM100:
    """Base trait for SM100 tile loading mechanisms."""
    
    fn load_tile(
        self,
        a_smem: LayoutTensorIter[...],
        b_smem: LayoutTensorIter[...],
        pipeline: TMAMmaPipeline,
        coords: Tuple[UInt, UInt, UInt],  # (m, n, k)
        k_iter: UInt32,
    ): ...

@register_passable("trivial")
struct TileLoaderTMA_SM100[
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    k_group_size: UInt,
](TileLoaderSM100):
    """TMA-based tile loader for SM100.
    
    Features:
    - CTA group coordination for 2-SM MMA
    - K-grouping for reduced sync overhead
    - Partitioned multicast for cluster distribution
    """
    
    var a_tma_op: TMATensorTile[...]
    var b_tma_op: TMATensorTile[...]
    var a_multicast_mask: UInt16
    var b_multicast_mask: UInt16
    var peer_cta_coord: Tuple[UInt, UInt, UInt]
    
    fn __init__(out self, ...): ...
    
    fn load_tile(
        self,
        a_smem: LayoutTensorIter[...],
        b_smem: LayoutTensorIter[...],
        pipeline: TMAMmaPipeline,
        work_coord: Tuple[UInt, UInt],
        k_iter: UInt32,
        elect_one_cta: Bool,
    ): ...
```

### Phase 4: Extract MMA Operation Module

**Goal**: Encapsulate UMMA operations and TMEM management.

**Files to Create**:

- `sm100/mma_op.mojo`

**Design**:

```mojo
# sm100/mma_op.mojo

@register_passable("trivial")
struct UmmaMmaOp[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    cluster_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
]:
    """UMMA operation wrapper for SM100.
    
    Manages:
    - TMEM allocation/deallocation
    - MMA instruction dispatch
    - Accumulator pipeline coordination
    """
    
    var mma_op: MmaOpSM100_SS[...]
    var tmem_addr: UInt32
    var tmem_dealloc_mbar: MbarPtr
    
    @staticmethod
    fn allocate_tmem(ptr: UnsafePointer[UInt32], cols: Int): ...
    
    @staticmethod
    fn deallocate_tmem(addr: UInt32, cols: Int): ...
    
    fn execute_mma(
        self,
        a_smem: LayoutTensor[...],
        b_smem: LayoutTensor[...],
        tmem_offset: UInt32,
        init_c: Bool,
    ): ...
    
    fn signal_completion(self, mbar: MbarPtr, mask: Int): ...
```

### Phase 5: Extract Output/Epilogue Module

**Goal**: Create a unified output path matching SM90's design.

**Files to Create**:

- `sm100/tile_writer.mojo`
- `sm100/epilogue.mojo`

**Design**:

```mojo
# sm100/tile_writer.mojo

@register_passable("trivial")
trait TileWriterSM100:
    """Base trait for SM100 tile writing mechanisms."""
    
    fn write_tile(
        self,
        src: LayoutTensor[...],
        coords: Tuple[UInt, UInt],
    ): ...

@register_passable("trivial")
struct TMEMToSMemWriter[
    c_type: DType,
    c_smem_layout: Layout,
    accum_type: DType,
    mma_shape: IndexList[3],
    cta_group: Int,
    swizzle: TensorMapSwizzle,
](TileWriterSM100):
    """Writes TMEM accumulator to shared memory via registers.
    
    Uses tcgen05_ld to load from TMEM, then st.matrix to store
    to shared memory with proper swizzling.
    """
    
    fn load_from_tmem(self, stage: Int, tmem_offset: UInt32) -> Tuple[SIMD, SIMD]: ...
    fn store_to_smem(self, upper: SIMD, lower: SIMD, c_smem: LayoutTensor[...]): ...

@register_passable("trivial")
struct SMemToGMemWriter[
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    swizzle: TensorMapSwizzle,
    transpose_c: Bool,
](TileWriterSM100):
    """Writes shared memory tile to global memory via TMA."""
    
    var tma_op: TMATensorTile[...]
    
    fn write_tile(self, c_smem: LayoutTensor[...], coords: Tuple[UInt, UInt]): ...
```

```mojo
# sm100/epilogue.mojo

@register_passable("trivial")
struct EpilogueSM100[
    c_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    output_tile_shape: IndexList[2],
    c_swizzle: TensorMapSwizzle,
    cta_group: Int,
    num_output_warps: Int,
    elementwise_compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type],
    register_based_epilogue: Bool,
    transpose_c: Bool,
]:
    """Orchestrates the full output pipeline for SM100.
    
    Flow:
    1. Wait for MMA completion (accum pipeline)
    2. Load accumulators from TMEM
    3. Apply compute lambda (if any, either register or smem based)
    4. Store to shared memory (double buffered)
    5. TMA store to global memory
    """
    
    fn execute(
        self,
        c_iter: LayoutTensorIter[...],
        c_tma_op: TMATensorTile[...],
        mma_output_pipeline: AccumPipeline,
        tmem_addr: UInt32,
        work_coord: Tuple[UInt32, UInt32],
        M: UInt32,
        N: UInt32,
    ): ...
    
    fn _apply_register_epilogue[stage: Int](
        self,
        mut upper_frag: SIMD,
        mut lower_frag: SIMD,
        c_row: UInt32,
        c_col: UInt32,
    ): ...
    
    fn _apply_smem_epilogue[stage: Int](
        self,
        c_smem: LayoutTensor[...],
        c_row: UInt32,
        c_col: UInt32,
        M: UInt32,
        N: UInt32,
    ): ...
```

### Phase 6: Restructure Main Kernel

**Goal**: Create a clean main kernel struct similar to SM90.

**Files to Modify**:

- `sm100/matmul.mojo` - Simplify to orchestration logic

**Design**:

```mojo
# sm100/matmul.mojo (refactored)

struct BlackwellMatmulKernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
]:
    """Blackwell SM100 Matrix Multiplication kernel.
    
    Implements a warp-specialized GEMM using:
    - TMA for efficient global-to-shared memory transfers
    - UMMA instructions with Tensor Memory (TMEM) accumulators
    - CLC-based tile scheduling for dynamic load balancing
    - Multi-stage pipelining for compute/memory overlap
    
    Warp Roles:
    - Warp 5 (MainLoad): TMA loads for A and B tiles
    - Warp 6 (Mma): UMMA instruction dispatch
    - Warp 4 (Scheduler): CLC query and response handling
    - Warps 0-3 (Epilogue): TMEM → SMEM → GMEM output path
    """
    
    # Compile-time derived types
    comptime SMem = B200MatmulSmem[...]
    comptime TileLoader = TileLoaderTMA_SM100[...]
    comptime MmaOp = UmmaMmaOp[...]
    comptime Epilogue = EpilogueSM100[...]
    comptime Scheduler = TileScheduler[...]
    
    @staticmethod
    fn validate_constraints(): ...
    
    @staticmethod
    fn build_smem() -> SMem: ...
    
    @staticmethod
    fn build_tile_loader(...) -> TileLoader: ...
    
    @staticmethod
    fn build_mma_op(...) -> MmaOp: ...
    
    @staticmethod
    fn build_epilogue(...) -> Epilogue: ...
    
    @staticmethod
    fn run[...](
        a_tma_op: TMATensorTile[...],
        b_tma_op: TMATensorTile[...],
        c_tma_op: TMATensorTile[...],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Main kernel entry point."""
        var smem = Self.build_smem()
        var scheduler = Scheduler(...)
        var work_info = scheduler.initial_work_info()
        
        # Warp specialization
        if WarpRole.is_main_load():
            Self._producer_loop(...)
        elif WarpRole.is_scheduler():
            Self._scheduler_loop(...)
        elif WarpRole.is_mma():
            Self._mma_loop(...)
        elif WarpRole.is_epilogue():
            Self._epilogue_loop(...)
    
    @staticmethod
    fn _producer_loop(...): ...
    
    @staticmethod
    fn _scheduler_loop(...): ...
    
    @staticmethod
    fn _mma_loop(...): ...
    
    @staticmethod
    fn _epilogue_loop(...): ...
```

---

## 4. Detailed File Structure After Refactoring

```text
sm100/
├── __init__.mojo           # Module exports
├── config.mojo             # MatmulConfig (unchanged)
├── dispatch.mojo           # Dispatch logic (minor changes)
├── smem.mojo               # NEW: Shared memory organization
├── pipeline.mojo           # EXTENDED: TMAMmaPipeline, AccumPipeline
├── tile_loader.mojo        # NEW: TileLoaderTMA_SM100
├── mma_op.mojo             # NEW: UmmaMmaOp wrapper
├── tile_writer.mojo        # NEW: TMEMToSMemWriter, SMemToGMemWriter
├── epilogue.mojo           # NEW: EpilogueSM100 orchestration
├── tile_scheduler.mojo     # (unchanged)
├── tile_scheduler_splitk.mojo  # (unchanged)
├── matmul.mojo             # REFACTORED: BlackwellMatmulKernel
└── matmul_splitk.mojo      # OPTIONAL: Split-K variant (if separated)
```

---

## 5. Migration Strategy (Incremental, Test-Driven)

The key principle is **maintaining functional integrity at every step**.
Each phase must:

1. Pass all existing tests
2. Show no performance regression (within 1% tolerance)
3. Be independently verifiable before proceeding

### Phase 1: Kernel Struct Refactoring

**Goal**: Unify the kernel functions into a single parameterized struct to
consolidate the parameter space and identify refactoring opportunities.

**Why First**:

- Creates a single source of truth for all kernel parameters
- Makes the relationship between parameters explicit
- Identifies redundant or conflicting parameters
- Provides a foundation for subsequent refactoring

**Deliverables**:

1. `BlackwellMatmulSM100Kernel` struct containing all compile-time parameters
2. Static methods wrapping existing kernel functions
3. Compile-time validation of parameter constraints
4. Unchanged runtime behavior

**Testing Checkpoint**:

- [x] All matmul tests pass
- [x] Performance benchmark shows no regression (~1811 TFLOPS)
- [x] Kernel binary size unchanged (validates no code bloat)

**Status**: ✅ **COMPLETED** (December 2024)

Completed sub-steps:

1. Created `BlackwellMatmulSM100Kernel` struct with all compile-time parameters
2. Added derived constants (`BM`, `BN`, `BK`, layouts, etc.)
3. Added `run()` static method with full kernel body
4. Added `run_splitk()` static method with full split-K kernel body
5. Updated dispatch code to use struct methods directly
6. Removed legacy wrapper functions
7. Created `BlackwellMatmulSM100FallbackKernel` struct for fallback kernel
8. All parameters use `Self.` prefix per Mojo conventions

### Phase 2: Shared Memory Reorganization

**Goal**: Reorganize `B200MatmulSmem` to use consistent tile types similar to
SM90's approach.

**Why Second**:

- Establishes consistent memory types used throughout the kernel
- Enables type-safe access to shared memory regions
- Prerequisite for RingBuffer which needs typed tile access

**Deliverables**:

1. Typed tile accessors (A tiles, B tiles, C tiles)
2. Barrier accessor methods
3. LayoutTensorIter-based tile iteration
4. Size calculation methods for each component

**Testing Checkpoint**:

- [ ] All matmul tests pass
- [ ] Shared memory layout verified via debug prints
- [ ] Performance benchmark shows no regression

### Phase 3: RingBuffer Implementation

**Goal**: Build a producer-consumer ring buffer abstraction for SM100, adapting
SM90's design to SM100's warp specialization model.

**Why Third**:

- Decouples synchronization logic from kernel body
- Enables cleaner producer/consumer code paths
- Facilitates future optimizations to synchronization

**Deliverables**:

1. `RingBufferSM100` struct with producer/consumer views
2. Context manager API matching SM90's pattern
3. Support for k-grouping (multiple K tiles per sync)
4. Integration with existing pipeline structs

**Testing Checkpoint**:

- [ ] All matmul tests pass
- [ ] Barrier correctness verified via synchronization tests
- [ ] Performance benchmark shows no regression

### Phase 4: TileLoader Abstraction

**Goal**: Extract tile loading logic into a TileLoader trait/struct.

**Why Fourth**:

- Clean separation of memory transfer concerns
- Enables future support for different loading strategies
- Makes the producer loop more readable

**Deliverables**:

1. `TileLoaderTMA_SM100` struct
2. Support for multicast and k-grouping
3. Integration with RingBuffer

**Testing Checkpoint**:

- [ ] All matmul tests pass
- [ ] TMA transfer correctness verified
- [ ] Performance benchmark shows no regression

### Phase 5: TileWriter/Epilogue Abstraction

**Goal**: Extract output path logic into TileWriter traits/structs.

**Why Last**:

- Most complex component with multiple variants
- Depends on clean shared memory types from Phase 2
- Benefits from RingBuffer's clean synchronization

**Deliverables**:

1. `TMEMToSMemWriter` for TMEM → shared memory
2. `SMemToGMemWriter` for shared → global via TMA
3. `EpilogueSM100` orchestration struct
4. Clean handling of register vs. smem-based epilogue

**Testing Checkpoint**:

- [ ] All matmul tests pass
- [ ] Epilogue correctness verified across all configurations
- [ ] Performance benchmark shows no regression
- [ ] Split-K variant works correctly

---

## 6. Phase 1 Detailed Implementation: Kernel Struct Refactoring

### 6.1 Current State Analysis

Currently, SM100 has **three kernel functions** with overlapping parameters:

```mojo
// 1. Main warp-specialized kernel
fn blackwell_tma_umma_warp_specialized_kernel[
    a_type, b_type, c_type,
    a_layout, b_layout, c_layout,
    a_desc_layout, b_desc_layout, c_desc_layout,
    transpose_b,
    config: MatmulConfig[...],
    cluster_shape: StaticTuple[Int32, 3],
    elementwise_compute_lambda_fn,
    register_based_epilogue,
    pdl_level,
    max_profiled_tiles_per_SM,
](...)

// 2. Split-K variant
fn blackwell_tma_umma_warp_specialized_split_k_kernel[
    // Same parameters + reduction_layout
](...)

// 3. Fallback kernel
fn matmul_sm100_fallback_kernel[
    // Subset of parameters + different structure
](...)
```

**Problems**:

1. Parameter duplication across functions
2. No compile-time validation of parameter combinations
3. Derived types computed inline in each function
4. Hard to see which parameters affect which behavior

### 6.2 Target Structure

```mojo
struct BlackwellMatmulSM100Kernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    
    # Layouts (from TMA ops)
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    
    # Configuration
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    
    # Cluster configuration (must match config but needed for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3],
    
    # Optional features
    elementwise_compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
]:
    """Blackwell SM100 GEMM kernel with warp specialization.
    
    This struct unifies all parameters and derived types for the SM100
    matmul kernel, providing:
    - Compile-time parameter validation
    - Centralized derived type computation
    - Factory methods for kernel components
    - Multiple kernel entry points (standard, split-k, fallback)
    """
    
    # ========== Derived Constants ==========
    
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    
    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]
    
    comptime accum_type = get_accum_type[Self.a_type]()
    comptime cta_group = Self.config.cta_group
    
    comptime CLUSTER_M = Int(Self.config.cluster_shape[0])
    comptime CLUSTER_N = Int(Self.config.cluster_shape[1])
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N
    
    # Thread/warp organization
    comptime num_output_warps = 4
    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = Self.num_output_warps * WARP_SIZE
    
    # TMEM configuration
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = Self.NUM_TMEM_COLS // Int(
        Self.config.num_accum_pipeline_stages
    )
    
    # ========== Derived Types ==========
    
    comptime SmemType = B200MatmulSmem[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b,
        config = Self.config
    ]
    
    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, swizzle_mode = Self.config.a_swizzle
    ]()
    
    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]()
    
    comptime MmaOpType = MmaOpSM100_SS[
        Self.c_type, Self.a_type, Self.b_type,
        Self.config.block_tile_shape, Self.config.mma_shape,
        accum_type = Self.accum_type,
        cta_group = Self.cta_group,
        cluster_shape = Self.config.cluster_shape,
        a_swizzle = Self.config.a_swizzle,
        b_swizzle = Self.config.b_swizzle,
        transpose_b = Self.transpose_b,
    ]
    
    comptime SchedulerType = TileScheduler[
        num_stages = Int(Self.config.num_clc_pipeline_stages),
        cluster_shape = Index[dtype = DType.uint32](
            Self.config.cluster_shape[0],
            Self.config.cluster_shape[1],
            Self.config.cluster_shape[2],
        ),
        block_swizzle_size = Self.config.block_swizzle_size,
        rasterize_order = Self.config.raster_order,
    ]
    
    # ========== Compile-Time Validation ==========
    
    @staticmethod
    fn validate_constraints():
        """Validate parameter constraints at compile time."""
        constrained[
            Self.c_type is not DType.float32, 
            "c_type cannot be float32"
        ]()
        constrained[
            Self.transpose_b, 
            "Only support transposed B (K-major)"
        ]()
        constrained[
            Self.cta_group in (1, 2), 
            "Only support cta_group == 1 or 2"
        ]()
        
        @parameter
        if Self.cta_group == 2:
            constrained[
                Self.MMA_M in (128, 256),
                "cta_group=2 requires MMA_M == 128 or 256"
            ]()
            constrained[
                Self.MMA_M != 256 or Self.MMA_N % 16 == 0,
                "MMA_N must be multiple of 16 when MMA_M=256"
            ]()
        else:
            constrained[
                Self.MMA_M in (64, 128),
                "cta_group=1 requires MMA_M == 64 or 128"
            ]()
    
    # ========== Factory Methods ==========
    
    @staticmethod
    fn build_mma_op() -> Self.MmaOpType:
        """Create the MMA operation instance."""
        return Self.MmaOpType()
    
    @staticmethod
    fn build_scheduler(
        cluster_dim: StaticTuple[Int32, 3],
        clc_response: UnsafePointer[UInt128, address_space = AddressSpace.SHARED],
        clc_full_mbar: UnsafePointer[SharedMemBarrier, address_space = AddressSpace.SHARED],
        clc_empty_mbar: UnsafePointer[SharedMemBarrier, address_space = AddressSpace.SHARED],
    ) -> Self.SchedulerType:
        """Create the tile scheduler instance."""
        return Self.SchedulerType(
            cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar
        )
    
    @staticmethod
    fn compute_multicast_masks(
        rank_m: UInt, rank_n: UInt, peer_cta_coord: Tuple[UInt, UInt, UInt]
    ) -> Tuple[UInt16, UInt16]:
        """Compute A and B multicast masks for cluster distribution."""
        var a_multicast_mask: UInt16 = 0x0
        var b_multicast_mask: UInt16 = 0x0
        
        @parameter
        for i in range(Self.CLUSTER_N):
            a_multicast_mask |= 1 << (i * Self.CLUSTER_M)
        
        @parameter
        for i in range(Self.CLUSTER_M // Self.cta_group):
            b_multicast_mask |= 1 << (i * Self.cta_group)
        
        a_multicast_mask <<= rank_m
        b_multicast_mask <<= peer_cta_coord[0]
        b_multicast_mask <<= rank_n * UInt(Self.CLUSTER_M)
        
        return (a_multicast_mask, b_multicast_mask)
    
    # ========== Kernel Entry Points ==========
    
    @staticmethod
    @__llvm_metadata(...)
    fn run(
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Main kernel entry point."""
        Self.validate_constraints()
        # ... kernel body (initially just calls existing logic)
    
    @staticmethod
    @__llvm_metadata(...)
    fn run_splitk[
        reduction_layout: Layout,
        splits: Int,
    ](
        a_tma_op: TMATensorTile[...],
        b_tma_op: TMATensorTile[...],
        c_tma_op: TMATensorTile[...],
        reduction_tensor: LayoutTensor[Self.accum_type, reduction_layout, MutAnyOrigin],
        lock_ptr: UnsafePointer[UInt8],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Split-K kernel entry point."""
        Self.validate_constraints()
        # ... kernel body
```

### 6.3 Implementation Steps

**Step 1.1**: Create the kernel struct shell (empty methods)

```mojo
# Add at the end of matmul.mojo, before existing kernel functions
struct BlackwellMatmulSM100Kernel[...]:
    # All comptime declarations
    # Empty method stubs
    pass
```

**Step 1.2**: Add compile-time derived constants

- Move `comptime` declarations from kernel functions into struct
- Verify they compute the same values

**Step 1.3**: Add factory methods that return existing types

- `build_mma_op()` → returns `MmaOpSM100_SS`
- `build_scheduler()` → returns `TileScheduler`

**Step 1.4**: Create `run()` that delegates to existing kernel

```mojo
@staticmethod
fn run(...):
    Self.validate_constraints()
    # For now, just call the existing function
    blackwell_tma_umma_warp_specialized_kernel[
        Self.a_type, Self.b_type, Self.c_type,
        Self.a_layout, Self.b_layout, Self.c_layout,
        # ... map all parameters from Self
    ](a_tma_op, b_tma_op, c_tma_op, cluster_dim, mnk, workspace)
```

**Step 1.5**: Update dispatch to use new struct

```mojo
# In dispatch.mojo or _blackwell_matmul_tma_umma_warp_specialized
comptime Kernel = BlackwellMatmulSM100Kernel[
    a_type, b_type, c_type,
    a_tma_op.layout, b_tma_op.layout, c_tma_op.layout,
    a_tma_op.desc_layout, b_tma_op.desc_layout, c_tma_op.desc_layout,
    transpose_b,
    config=config,
    cluster_shape=cluster_shape,
    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    register_based_epilogue=register_based_epilogue,
    pdl_level=pdl_level,
    max_profiled_tiles_per_SM=max_profiled_tiles,
]

ctx.enqueue_function_checked[Kernel.run, Kernel.run](...)
```

**Step 1.6**: Gradually move kernel body into struct methods

- Move helper functions as static methods
- Move warp-specific loops as private static methods
- Keep existing kernel as compatibility shim until fully migrated

### 6.4 Testing Plan for Phase 1

```bash
# Run after each sub-step:

# 1. Unit tests for matmul
bazel test //max/kernels/test/gpu:matmul_test --config=b200

# 2. Performance benchmark (capture baseline first!)
bazel run //max/kernels/benchmark:matmul_benchmark --config=b200 -- \
    --dtype=bf16 --m=4096 --n=4096 --k=4096

# 3. Verify no binary size regression (large increase = template bloat)
bazel build //max/kernels:matmul_kernel --config=b200
ls -la bazel-bin/max/kernels/...

# 4. Full test suite
bazel test //max/... --config=b200
```

### 6.5 Rollback Plan

If issues are found:

1. The existing kernel functions remain unchanged
2. The new struct is purely additive
3. Remove the struct and dispatch changes to restore original behavior

---

## 7. Phase 2 Detailed Implementation: Shared Memory Reorganization

### 7.1 Current State Analysis

The current `B200MatmulSmem` struct uses raw `InlineArray` for storage:

```mojo
struct B200MatmulSmem[...]:
    var a_smem: InlineArray[Self.AType, Self.a_smem_size]
    var b_smem: InlineArray[Self.BType, Self.b_smem_size]
    var c_smem: InlineArray[Self.CType, Self.c_smem_size]
    var tma_mma_mbars: InlineArray[SharedMemBarrier, ...]
    var accum_mbars: InlineArray[SharedMemBarrier, ...]
    # ... more barriers
```

The kernel then manually creates `LayoutTensorIter` from these:

```mojo
var a_smem = LayoutTensorIter[a_type, a_smem_layout, ...](
    a_smem_storage.unsafe_ptr(), SmemType.a_smem_size
)
```

**Problems**:

1. Type safety lost - raw pointers passed around
2. Layout information separate from storage
3. Manual size calculations duplicated
4. No encapsulation of tile access patterns

### 7.2 Target Structure (Matching SM90 Pattern)

```mojo
struct B200MatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Shared memory organization for B200 matmul kernel.
    
    Provides typed access to all shared memory regions with proper
    layouts and alignment guarantees.
    """
    
    # ========== Layout Definitions ==========
    
    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.config.block_tile_shape[0], 
        Self.config.block_tile_shape[2],
        swizzle_mode = Self.config.a_swizzle
    ]()
    
    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.config.block_tile_shape[1],
        Self.config.block_tile_shape[2],
        swizzle_mode = Self.config.b_swizzle
    ]() if Self.transpose_b else ...
    
    comptime c_smem_layout = Layout.row_major(
        Self.config.output_tile_shape[0],
        Self.config.output_tile_shape[1],
    )
    
    # ========== Tile Array Types (like SM90) ==========
    
    comptime SMM = NVIDIASharedMemoryManager[]
    
    comptime ATileArray = Self.SMM.TileArray[
        Self.a_type, Self.a_smem_layout, 
        Int(Self.config.num_pipeline_stages)
    ]
    comptime BTileArray = Self.SMM.TileArray[
        Self.b_type, Self.b_smem_layout,
        Int(Self.config.num_pipeline_stages)
    ]
    comptime CTileArray = Self.SMM.TileArray[
        Self.c_type, Self.c_smem_layout,
        Int(Self.config.num_output_stages)
    ]
    
    comptime PipelineBarrier = PipelineBarrier[
        Int(Self.config.num_pipeline_stages // Self.config.k_group_size)
    ]
    comptime AccumBarrier = PipelineBarrier[
        Int(Self.config.num_accum_pipeline_stages)
    ]
    
    # ========== Storage ==========
    
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var c_tiles: Self.CTileArray
    
    var tma_mma_mbar: Self.PipelineBarrier
    var accum_mbar: Self.AccumBarrier
    
    var clc_mbars_full: InlineArray[SharedMemBarrier, Int(config.num_clc_pipeline_stages)]
    var clc_mbars_empty: InlineArray[SharedMemBarrier, Int(config.num_clc_pipeline_stages)]
    var clc_throttle_mbars: InlineArray[SharedMemBarrier, Int(config.num_clc_pipeline_stages) * 2]
    var clc_response: InlineArray[UInt128, Int(config.num_clc_pipeline_stages)]
    
    var tmem_dealloc_mbar: InlineArray[SharedMemBarrier, 1]
    var tmem_addr: InlineArray[UInt32, 1]
    
    # ========== Accessors ==========
    
    fn __init__(out self):
        """Initialize shared memory with proper layouts."""
        var smem_mgr = Self.SMM()
        self.a_tiles = smem_mgr.build[T = Self.ATileArray]()
        self.b_tiles = smem_mgr.build[T = Self.BTileArray]()
        self.c_tiles = smem_mgr.build[T = Self.CTileArray]()
        self.tma_mma_mbar = smem_mgr.build[T = Self.PipelineBarrier]()
        self.accum_mbar = smem_mgr.build[T = Self.AccumBarrier]()
        # ... initialize other fields
    
    fn get_a_tile(self, stage: UInt32) -> Self.ATileArray.Tile:
        """Get A tile for a specific pipeline stage."""
        return self.a_tiles[stage]
    
    fn get_b_tile(self, stage: UInt32) -> Self.BTileArray.Tile:
        """Get B tile for a specific pipeline stage."""
        return self.b_tiles[stage]
    
    fn get_c_tile(self, stage: UInt32) -> Self.CTileArray.Tile:
        """Get C tile for a specific output stage."""
        return self.c_tiles[stage]
    
    fn get_a_iter(self) -> LayoutTensorIter[
        Self.a_type, Self.a_smem_layout, MutAnyOrigin,
        address_space = AddressSpace.SHARED, alignment=128
    ]:
        """Get iterator over A tiles."""
        return LayoutTensorIter[...](
            self.a_tiles.ptr, Self.ATileArray.storage_size
        )
    
    fn get_b_iter(self) -> LayoutTensorIter[...]:
        """Get iterator over B tiles."""
        return LayoutTensorIter[...](
            self.b_tiles.ptr, Self.BTileArray.storage_size
        )
    
    fn get_c_iter(self) -> LayoutTensorIter[...]:
        """Get iterator over C tiles."""
        return LayoutTensorIter[...](
            self.c_tiles.ptr, Self.CTileArray.storage_size
        )
    
    # ========== Size Calculations ==========
    
    @staticmethod
    fn ab_pipeline_size() -> Int:
        """Size of A+B tiles for all pipeline stages."""
        return Self.ATileArray.storage_size + Self.BTileArray.storage_size
    
    @staticmethod
    fn c_output_size() -> Int:
        """Size of C tiles for all output stages."""
        return Self.CTileArray.storage_size
    
    @staticmethod
    fn barrier_size() -> Int:
        """Size of all barrier storage."""
        return (
            Self.PipelineBarrier.storage_size
            + Self.AccumBarrier.storage_size
            + Int(config.num_clc_pipeline_stages) * 3 * size_of[SharedMemBarrier]()
            + Int(config.num_clc_pipeline_stages) * size_of[UInt128]()
            + size_of[SharedMemBarrier]()  # tmem_dealloc
            + size_of[UInt32]()  # tmem_addr
        )
    
    @staticmethod
    fn total_size() -> Int:
        """Total shared memory size."""
        return Self.ab_pipeline_size() + Self.c_output_size() + Self.barrier_size()
```

### 7.3 Implementation Steps

**Step 2.1**: Add typed tile accessors to existing struct

- Keep existing `InlineArray` storage
- Add `get_a_tile()`, `get_b_tile()`, `get_c_tile()` methods
- Add `get_a_iter()`, `get_b_iter()`, `get_c_iter()` methods
- Update kernel to use accessors instead of manual construction

**Step 2.2**: Add `NVIDIASharedMemoryManager`-based TileArray types

- Import from SM90's structuring module
- Create `ATileArray`, `BTileArray`, `CTileArray` type aliases
- Verify size calculations match

**Step 2.3**: Replace InlineArray with TileArray

- Update storage to use TileArray types
- Update `__init__` to use `smem_mgr.build`
- Verify layout consistency

**Step 2.4**: Add barrier accessors

- Create typed barrier accessor methods
- Encapsulate CLC barrier access
- Update kernel to use accessors

### 7.4 Testing Plan for Phase 2

```bash
# After each sub-step:

# 1. Verify shared memory layout hasn't changed
bazel test //max/kernels/test/gpu:matmul_test --config=b200 --test_arg=--debug_smem

# 2. Run full matmul test suite
bazel test //max/kernels/test/gpu:matmul_test --config=b200

# 3. Performance verification
bazel run //max/kernels/benchmark:matmul_benchmark --config=b200 -- \
    --dtype=bf16 --m=4096 --n=4096 --k=4096
```

---

## 8. Quick Validation Commands

### Functional Smoke Test

Run the quick functional smoke test (~10 seconds) to verify basic correctness:

```bash
cd ~ && source start-modular.sh && cd ~/modular
mojo ./max/kernels/test/gpu/linalg/test_matmul_sm100_smoke.mojo
```

This tests 10 configurations covering all major code paths:

- 1SM and 2SM kernels
- Single CTA and multi-CTA clusters
- swapAB variants
- k_group_size variations
- Split-K kernel
- Misaligned dimensions

### Performance Smoke Test

Run the performance benchmark (~5 seconds) to check for performance regressions:

```bash
cd ~ && source start-modular.sh && cd ~/modular
mojo -D use_vendor_blas=False -D N=8192 -D K=8192 ./max/kernels/benchmarks/gpu/bench_matmul.mojo --M=8192
```

**Baseline Performance (B200)**: ~1746 TFLOPS for 8192x8192x8192 BF16 matmul

A regression of >5% should be investigated before proceeding.

### Full Bazel Test (for CI or final validation)

```bash
bazel test //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test --config=b200
bazel test //max/kernels/test/gpu/linalg:test_matmul_sm100_1sm_bf16.mojo.test --config=b200
# ... other comprehensive tests
```

---

## 9. Full Testing Strategy

### Unit Tests

1. **Shared Memory Tests**: Verify size calculations and layout correctness
2. **Pipeline Tests**: Verify barrier synchronization patterns
3. **Tile Loader Tests**: Verify TMA loads with various configurations
4. **Epilogue Tests**: Verify register/smem epilogue computations

### Integration Tests

1. **Full Kernel Tests**: Run against existing matmul test suite
2. **Performance Tests**: Ensure no regression vs. current implementation
3. **Configuration Coverage**: Test all config combinations
   (cta_group, mma_shape, etc.)

### Benchmarks

1. **Latency**: Compare against current implementation
2. **Throughput**: Verify TFLOPS across problem sizes
3. **Memory Efficiency**: Profile shared memory and TMEM usage

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance regression | Profile each change, keep old code paths until verified |
| Breaking existing functionality | Maintain backward-compatible wrappers |
| Increased complexity | Clear documentation, consistent naming conventions |
| Template bloat | Careful factoring of common patterns |

---

## 11. Dependencies and Prerequisites

1. **Existing Infrastructure**:
   - `ProducerConsumerPipeline` (already extracted)
   - `MatmulConfig` (already clean)
   - `TileScheduler` (already modular)

2. **Required Reviews**:
   - SM90 architecture review with original authors
   - Performance baseline establishment
   - API design review for new abstractions

---

## 12. Success Criteria

1. **Code Organization**:
   - Main kernel file < 500 lines
   - Each module has single responsibility
   - Clear interfaces between components

2. **Maintainability**:
   - New features can be added without modifying core kernel
   - Bug fixes isolated to specific modules
   - Easy to understand data flow

3. **Performance**:
   - No regression vs. current implementation (< 1% allowed)
   - Maintained or improved compile times

4. **Extensibility**:
   - Easy to add new epilogue operations
   - Easy to support new MMA shapes
   - Easy to add profiling/debugging hooks

---

## 13. Revised Timeline (Incremental Approach)

| Week | Phase | Milestone | Testing Gate |
|------|-------|-----------|--------------|
| 1 | 1a | Create kernel struct shell with derived constants | Compiles, tests pass |
| 1 | 1b | Add factory methods, delegate to existing kernel | Tests pass, perf verified |
| 2 | 1c | Move kernel body into struct methods | All tests pass, perf ±1% |
| 2 | 2a | Reorganize `B200MatmulSmem` with typed accessors | Tests pass |
| 3 | 2b | Add LayoutTensorIter-based tile iteration | Tests pass, perf verified |
| 3 | 3a | Create `RingBufferSM100` producer view | Tests pass |
| 4 | 3b | Create `RingBufferSM100` consumer view | Tests pass, perf verified |
| 4 | 4 | Extract `TileLoaderTMA_SM100` | Tests pass, perf verified |
| 5 | 5a | Extract `TMEMToSMemWriter`, `SMemToGMemWriter` | Tests pass |
| 5 | 5b | Create `EpilogueSM100` orchestration | Tests pass, perf verified |
| 6 | - | Code review, documentation, cleanup | Final sign-off |

**Key Principle**: Never proceed to the next phase until the current phase
passes all tests with no performance regression.

---

## Appendix A: SM90 vs SM100 Component Mapping

| SM90 Component | SM100 Equivalent | Notes |
|----------------|------------------|-------|
| `HopperMatmulSM90Kernel_SMem` | `B200MatmulSmem` | Similar, SM100 has CLC storage |
| `RingBuffer` | `ProducerConsumerPipeline` + CLC | SM100 separates load/mma/accum pipelines |
| `TileLoaderTMA` | `TileLoaderTMA_SM100` | SM100 has k-grouping |
| `WgmmaOp` | `UmmaMmaOp` | TMEM vs register accumulators |
| `MatmulTileWriter` | `EpilogueSM100` | SM100 has TMEM→Reg→SMem→GMem |
| `FragmentToSMemWriter` | `TMEMToSMemWriter` | tcgen05_ld vs direct copy |
| N/A | `TileScheduler` | SM100-specific CLC scheduling |

---

## Appendix B: Key SM100-Specific Considerations

### B.1 Tensor Memory (TMEM)

SM100 uses dedicated Tensor Memory for MMA accumulators:

- Allocated via `tcgen05_alloc`
- Loaded via `tcgen05_ld`
- Different layout than register accumulators
- Requires explicit deallocation

### B.2 CLC (Cluster Launch Control)

SM100 uses hardware-accelerated tile scheduling:

- `clusterlaunchcontrol_try_cancel` for querying next work
- Enables dynamic load balancing across SMs
- Requires dedicated scheduler warp

### B.3 CTA Groups

SM100 supports 2-SM cooperative MMA:

- Two CTAs collaborate on single large MMA
- Requires multicast coordination for B matrix
- Different output layouts for MMA_M=128 vs MMA_M=256

### B.4 K-Grouping

SM100 can group multiple K iterations per synchronization:

- Reduces barrier overhead for small tiles
- `k_group_size` parameter controls grouping factor
- Must be accounted for in pipeline stage calculations

---

## Appendix C: Code Examples

### C.1 Current SM100 Producer Loop (Before)

```mojo
// In blackwell_tma_umma_warp_specialized_kernel
if WarpRole.is_main_load():
    while work_info.is_valid():
        if is_first_cta_in_cluster and required_clc_query:
            load_clc_pipeline.wait_consumer()
            var load_clc_producer_state = load_clc_pipeline.producer_stage()
            _ = load_clc_pipeline.producer_mbar(load_clc_producer_state)[0].arrive()
            load_clc_pipeline.producer_step()

        for i in range(num_iters // config.k_group_size):
            load_AB[...](
                a_tma_op, b_tma_op, a_smem, b_smem, load_mma_pipeline,
                peer_cta_coord, (UInt(work_info.m), UInt(work_info.n)),
                a_multicast_mask, b_multicast_mask, i * config.k_group_size, elect_one_cta,
            )
            load_mma_pipeline.producer_step()

        syncwarp()
        var next_work_info = scheduler.fetch_next_work(work_info, clc_pipe_consumer_state)
        work_info = next_work_info
        clc_pipe_consumer_state.step()
```

### C.2 Proposed SM100 Producer Loop (After)

```mojo
// In BlackwellMatmulKernel._producer_loop
@staticmethod
fn _producer_loop[...](
    tile_loader: Self.TileLoader,
    mut load_mma_pipeline: TMAMmaPipeline,
    mut load_clc_pipeline: ProducerConsumerPipeline,
    scheduler: Self.Scheduler,
    mut clc_state: PipelineState,
    num_k_iters: UInt32,
    is_first_cta: Bool,
    elect_one_cta: Bool,
):
    var work_info = scheduler.initial_work_info()
    
    while work_info.is_valid():
        # CLC throttle coordination
        if is_first_cta:
            load_clc_pipeline.wait_and_advance()
        
        # Load all K tiles for this output tile
        tile_loader.load_work_tile(
            work_info,
            load_mma_pipeline,
            num_k_iters,
            elect_one_cta,
        )
        
        # Fetch next work
        work_info = scheduler.fetch_next_work(work_info, clc_state)
        clc_state.step()
```

---

## Appendix D: Phase 3 Detailed Implementation Plan - RingBufferSM100

### D.1 Overview

**Goal**: Create a `RingBufferSM100` abstraction that encapsulates
producer-consumer synchronization and tile access, similar to SM90's
`RingBuffer`.

**Key Differences from SM90**:

| Aspect | SM90 | SM100 |
|--------|------|-------|
| Barriers | `full_mbar` + `empty_mbar` | `ProducerConsumerPipeline` (already exists) |
| K-grouping | Not supported | Supports `k_group_size` tiles per sync |
| CTA groups | Single CTA | 1-SM or 2-SM (`cta_group`) |
| Context managers | Producer/Consumer views | Same pattern, adapted for SM100 |

### D.2 Implementation Steps

#### Step 1: Create `ring_buffer.mojo` file structure

```mojo
# sm100/ring_buffer.mojo

"""Ring buffer for SM100 producer-consumer synchronization.

This module provides a ring buffer abstraction that wraps:
- ProducerConsumerPipeline (barrier management)
- Typed tile access via B200MatmulSmem types
- K-grouping support for multiple tiles per sync
"""

from .pipeline import ProducerConsumerPipeline
from .matmul import B200MatmulSmem
```

#### Step 2: Define `RingBufferSM100` struct

```mojo
@register_passable("trivial")
struct RingBufferSM100[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Ring buffer for SM100 TMA load → MMA synchronization.

    Parameters:
        config: MatmulConfig containing pipeline stage counts, k_group_size,
                tile shapes, swizzle modes, etc.

    This struct combines:
    - ProducerConsumerPipeline for barrier synchronization
    - Typed tile iterators from B200MatmulSmem
    - K-grouping: loads k_group_size tiles per barrier sync

    Usage:
        var ring_buffer = RingBufferSM100[...](pipeline, a_tiles, b_tiles)

        # Producer side
        with ring_buffer.producer() as slot:
            Self._load_AB(..., slot.a_tile, slot.b_tile, slot.barrier, ...)

        # Consumer side
        with ring_buffer.consumer() as slot:
            mma_op.mma(slot.a_tile, slot.b_tile, ...)
    """

    # Type aliases from shared memory struct
    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime ATileIter = Self.SmemType.ATileIter
    comptime BTileIter = Self.SmemType.BTileIter
    comptime ATile = Self.SmemType.ATile
    comptime BTile = Self.SmemType.BTile

    # Pipeline stages accounting for k_group_size
    comptime num_group_stages = Int(
        config.num_pipeline_stages // config.k_group_size
    )

    # State
    var pipeline: ProducerConsumerPipeline[Self.num_group_stages]
    var a_tiles: Self.ATileIter
    var b_tiles: Self.BTileIter
```

#### Step 3: Define producer context manager structs

```mojo
@register_passable("trivial")
struct ProducerSlot[ring_buffer_type: AnyType]:
    """Context manager for producer access to a ring buffer slot.

    Provides access to:
    - barrier: The mbar to use for TMA expect_bytes/arrive
    - stage: The current pipeline stage index
    - a_tile/b_tile getters for each k in k_group
    """
    var ring_buffer_ptr: Pointer[ring_buffer_type, MutableAnyOrigin]
    var stage: UInt32
    var barrier: MbarPtr

    fn __enter__(mut self) -> Self: ...
    fn __exit__(mut self): ...  # Calls pipeline.producer_step()

    fn get_a_tile(self, k_in_group: Int) -> ring_buffer_type.ATile:
        """Get A tile for k_in_group within current stage."""
        return self.ring_buffer_ptr[].a_tiles.next(
            self.stage * config.k_group_size + k_in_group
        )[]

    fn get_b_tile(self, k_in_group: Int) -> ring_buffer_type.BTile:
        """Get B tile for k_in_group within current stage."""
        ...
```

#### Step 4: Define consumer context manager structs

```mojo
@register_passable("trivial")
struct ConsumerSlot[ring_buffer_type: AnyType]:
    """Context manager for consumer access to a ring buffer slot.

    Waits for producer, provides tile access, signals completion on exit.
    """
    var ring_buffer_ptr: Pointer[ring_buffer_type, MutableAnyOrigin]
    var stage: UInt32

    fn __enter__(mut self) -> Self:
        self.ring_buffer_ptr[].pipeline.wait_producer()
        return self

    fn __exit__(mut self):
        # Signal via consumer_mbar if needed
        ...

    fn get_a_tile(self, k_in_group: Int) -> ring_buffer_type.ATile: ...
    fn get_b_tile(self, k_in_group: Int) -> ring_buffer_type.BTile: ...
```

#### Step 5: Add producer/consumer view methods to RingBufferSM100

```mojo
# In RingBufferSM100:

fn producer(mut self) -> RingBufferProducer[Self]:
    """Get producer view of the ring buffer."""
    return RingBufferProducer[Self](Pointer.address_of(self))

fn consumer(mut self) -> RingBufferConsumer[Self]:
    """Get consumer view of the ring buffer."""
    return RingBufferConsumer[Self](Pointer.address_of(self))

# Producer view with get_slot() method
@register_passable("trivial")
struct RingBufferProducer[ring_buffer_type: AnyType]:
    var ring_buffer_ptr: Pointer[ring_buffer_type, MutableAnyOrigin]

    fn __enter__(mut self) -> Self: ...
    fn __exit__(mut self): ...

    fn get_slot(mut self) -> ProducerSlot[ring_buffer_type]:
        """Get next slot for loading tiles."""
        var stage = self.ring_buffer_ptr[].pipeline.producer_stage()
        self.ring_buffer_ptr[].pipeline.wait_consumer()
        var barrier = self.ring_buffer_ptr[].pipeline.producer_mbar(stage)
        return ProducerSlot[ring_buffer_type](
            self.ring_buffer_ptr, stage, barrier
        )
```

#### Step 6: Integrate with BlackwellMatmulSM100Kernel

Update `run()` and `run_splitk()` to use the new RingBuffer:

```mojo
# In BlackwellMatmulSM100Kernel:

comptime RingBuffer = RingBufferSM100[
    Self.a_type, Self.b_type, Self.c_type, Self.transpose_b,
    config = Self.config,
]

# In run():
var ring_buffer = Self.RingBuffer(
    load_mma_pipeline, a_smem, b_smem
)

# Producer warp
with ring_buffer.producer() as producer:
    while work_info.is_valid():
        with producer.get_slot() as slot:
            Self._load_tiles(slot, a_tma_op, b_tma_op, ...)

# Consumer warp
with ring_buffer.consumer() as consumer:
    for i in range(num_iters):
        with consumer.get_slot() as slot:
            Self._mma_tiles(slot, mma_op, ...)
```

### D.3 Task Checklist

- [x] **Step 1**: Create `sm100/ring_buffer.mojo` file with imports
- [x] **Step 2**: Define `RingBufferSM100` struct (simplified thin wrapper)
- [x] **Step 3**: Implement producer/consumer methods
- [x] **Step 4**: Add factory method to `BlackwellMatmulSM100Kernel`
- [x] **Step 5**: Refactor `_load_AB` to use ring buffer
- [x] **Step 6**: Refactor producer loops in `run()` and `run_splitk()`
- [x] **Step 7**: Refactor `_consumer_main_loop` to use ring buffer
- [x] **Step 8**: Test all kernel paths (standard and split-k)
- [x] **Step 9**: Run performance benchmark (no regression)
- [x] **Step 10**: Format code and update documentation

### D.4 Implementation Notes

**Final Design**: The `RingBufferWithTiles` provides an SM90-style `get_tiles()`
API with full encapsulation:

```mojo
# Producer pattern
with ring_buffer.producer() as producer:
    with producer.get_tiles() as tiles:
        Self._load_AB_tiles(a_tma_op, b_tma_op, tiles, ...)

# Consumer pattern
with ring_buffer.consumer() as consumer:
    with consumer.get_tiles() as tiles:
        Self._mma_tiles(tmem_offset, tiles, mma_op, ...)
```

The `ProducerTiles`/`ConsumerTiles` structs contain:

- `stage: UInt32` - current pipeline stage index
- `barrier/mbar: MbarPtr` - synchronization barrier
- `a_tiles: ATileArray` - full A tile array for k_group access
- `b_tiles: BTileArray` - full B tile array for k_group access

Functions like `_load_AB_tiles` and `_mma_tiles` accept the `tiles` struct
directly, extracting what they need internally:

```mojo
fn _load_AB_tiles[tiles_origin: Origin[True], //](
    a_tma_op: ...,
    b_tma_op: ...,
    tiles: ProducerTiles[tiles_origin, ...],
    peer_cta_coord: ...,
    ...
):
    # Access tiles.stage, tiles.barrier, tiles.a_tiles, tiles.b_tiles
```

### D.5 Expected Benefits (Achieved)

1. **Cleaner kernel code**: Single `tiles` parameter instead of 4+ separate args
2. **Encapsulated sync logic**: Barrier wait/signal hidden in context managers
3. **Consistent interface**: Both producer and consumer use same abstraction
4. **Type safety**: Typed tile arrays with proper layouts
5. **Reusable pattern**: Same abstraction can be used for grouped matmul

### D.6 Performance Results

- **Before refactoring**: ~1750 TFLOPS (8192x8192x8192 bfloat16)
- **After refactoring**: ~1822 TFLOPS (slight improvement due to better
  code organization, within normal variance)

---

## Appendix E: Phase 4 Detailed Implementation Plan - TileLoader

### E.1 Overview

**Goal**: Extract tile loading logic into a `TileLoader` abstraction for SM100,
similar to SM90's design but adapted for SM100's unique requirements.

### E.2 Key Differences: SM90 vs SM100 Tile Loading

| Aspect | SM90 | SM100 |
|--------|------|-------|
| Interface | `load_tile(dst, barrier, coords)` | Needs k_group awareness |
| Tiles per call | 1 | k_group_size (1, 2, or 4) |
| expect_bytes | Per tile | Once for all k_group tiles |
| CTA groups | Single CTA | 1 or 2 CTAs cooperating |
| Multicast | `async_multicast_load()` | `async_multicast_load[cta_group]()` |
| Peer coordination | None | peer_cta_coord for 2-SM slicing |
| Tile slicing | Direct dst | `dst.ptr + peer_offset * size` |

### E.3 Current SM100 Loading Logic (`_load_AB_tiles`)

```mojo
fn _load_AB_tiles(a_tma_op, b_tma_op, tiles, peer_cta_coord, work_tile_coord,
                  a_multicast_mask, b_multicast_mask, iter_idx, elect_one_cta):
    # 1. Compute expected bytes for ALL k_group tiles
    comptime expected_bytes = cta_group * (a_bytes + b_bytes) * k_group_size
    
    # 2. Compute gmem slice coordinates (accounts for peer CTA)
    var a_gmem_coord = peer_cta_coord[2] * a_tma_rows + work_tile_coord[0] * BM
    var b_gmem_coord = peer_cta_coord[1] * b_tma_rows + peer_cta_coord[0] * BN + ...
    
    # 3. Set expect_bytes ONCE for all k_group tiles
    if elect_one_cta:
        tiles.barrier[0].expect_bytes(expected_bytes)
    
    # 4. Loop over k_group_size, loading each tile
    for j in range(k_group_size):
        var a_tile = tiles.a_tiles[stage * k_group_size + j]
        var b_tile = tiles.b_tiles[stage * k_group_size + j]
        
        # Slice for peer CTA offset
        var a_slice = a_tile.ptr + peer_cta_coord[2] * a_tma_load_size
        var b_slice = b_tile.ptr + peer_cta_coord[1] * b_tma_load_size
        
        # TMA multicast load with cta_group
        a_tma_op.async_multicast_load[cta_group](a_slice, barrier, coords, mask)
        b_tma_op.async_multicast_load[cta_group](b_slice, barrier, coords, mask)
```

### E.4 Design Options

#### Option A: SM90-style single-tile interface

```mojo
trait TileLoader:
    fn load_tile(self, dst, barrier, coords): ...
```

- Problem: Can't call expect_bytes once for k_group
- Problem: Peer CTA slicing is complex

#### Option B: K-group aware interface (Recommended)

```mojo
struct TileLoaderTMA_SM100[...]:
    fn load_tiles(
        self,
        tiles: ProducerTiles,
        k_iter: UInt32,
        elect_one_cta: Bool,
    ): ...
```

- Encapsulates k_group loading, expect_bytes, and peer slicing
- Matches SM100 semantics well
- Less polymorphic but simpler

#### Option C: Separate single-tile loader + k_group orchestrator

```mojo
trait TileLoader:
    fn load_tile(self, dst, barrier, coords): ...

fn load_k_group[Loader: TileLoader](loader, tiles, k_iter, elect_one_cta):
    # Set expect_bytes once
    # Loop and call loader.load_tile() for each
```

- More modular but awkward expect_bytes handling

### E.5 Recommended Design: Option B

```mojo
# sm100/tile_loader.mojo

@register_passable("trivial")
struct TileLoaderTMA[
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    k_group_size: Int,
]:
    """TMA-based tile loader for SM100.
    
    Encapsulates the complete tile loading logic including:
    - K-group batching (multiple tiles per barrier)
    - CTA group coordination (1-SM or 2-SM cooperative loading)
    - Peer CTA slicing for 2-SM MMA
    - expect_bytes management
    """
    
    # TMA descriptors (pointers to grid constants)
    var a_tma_op: Pointer[TMATensorTile[...], ...]
    var b_tma_op: Pointer[TMATensorTile[...], ...]
    
    # Multicast configuration
    var a_multicast_mask: UInt16
    var b_multicast_mask: UInt16
    
    # Peer CTA info for 2-SM slicing
    var peer_cta_coord: Tuple[UInt, UInt, UInt]
    
    # Work tile coordinates (set per output tile)
    var work_tile_coord: Tuple[UInt, UInt]
    
    fn __init__(out self, a_tma_op, b_tma_op, a_mask, b_mask, peer_coord): ...
    
    fn set_work_tile(mut self, m_coord: UInt, n_coord: UInt):
        """Set the current output tile being processed."""
        self.work_tile_coord = (m_coord, n_coord)
    
    fn load_tiles[
        tiles_origin: Origin[True], //,
    ](
        self,
        tiles: ProducerTiles[tiles_origin, ...],
        k_iter: UInt32,
        elect_one_cta: Bool,
    ):
        """Load k_group_size A and B tiles using TMA.
        
        Args:
            tiles: ProducerTiles with stage, barrier, a_tiles, b_tiles.
            k_iter: K iteration index (multiplied by k_group_size internally).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        # Compute expected bytes for all k_group tiles
        comptime expected_bytes = Self.cta_group * (
            Self.a_smem_layout.size() * size_of[Self.a_type]() +
            Self.b_smem_layout.size() * size_of[Self.b_type]()
        ) * Self.k_group_size
        
        # ... rest of loading logic
```

### E.6 Integration with Kernel

```mojo
# In BlackwellMatmulSM100Kernel:

comptime TileLoader = TileLoaderTMA[
    Self.a_type, Self.b_type,
    Self.a_layout, Self.b_layout,
    Self.a_desc_layout, Self.b_desc_layout,
    Self.SmemType.a_smem_layout, Self.SmemType.b_smem_layout,
    Self.config.block_tile_shape, Self.config.mma_shape,
    Self.config.cta_group, Int(Self.config.k_group_size),
]

@staticmethod
fn build_tile_loader(
    a_tma_op: ...,
    b_tma_op: ...,
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    peer_cta_coord: Tuple[UInt, UInt, UInt],
) -> Self.TileLoader:
    return Self.TileLoader(
        Pointer(to=a_tma_op),
        Pointer(to=b_tma_op),
        a_multicast_mask,
        b_multicast_mask,
        peer_cta_coord,
    )
```

### E.7 Updated Producer Loop

```mojo
# Before:
with producer.get_tiles() as tiles:
    Self._load_AB_tiles(a_tma_op, b_tma_op, tiles, peer_cta_coord,
                        work_tile_coord, a_mask, b_mask, k_iter, elect_one_cta)

# After:
var tile_loader = Self.build_tile_loader(a_tma_op, b_tma_op, a_mask, b_mask, peer_cta_coord)
tile_loader.set_work_tile(work_info.m, work_info.n)

with producer.get_tiles() as tiles:
    tile_loader.load_tiles(tiles, k_iter, elect_one_cta)
```

### E.8 Implementation Steps

1. **Step 1**: Create `sm100/tile_loader.mojo` with struct definition
2. **Step 2**: Move loading logic from `_load_AB_tiles` into `TileLoaderTMA.load_tiles`
3. **Step 3**: Add `TileLoader` type alias and `build_tile_loader` factory to kernel
4. **Step 4**: Update `run()` to create and use TileLoader
5. **Step 5**: Update `run_splitk()` similarly
6. **Step 6**: Keep `_load_AB_tiles` as a thin wrapper for backward compatibility
7. **Step 7**: Test all kernel paths
8. **Step 8**: Update warp_specialized_blockwise_fp8.mojo if needed

### E.9 Benefits

1. **Encapsulation**: All loading logic in one place
2. **Testability**: TileLoader can be unit tested independently
3. **Reusability**: Same loader for run() and run_splitk()
4. **Clarity**: Producer loop becomes cleaner
5. **Future**: Easy to add new loading strategies (e.g., CPAsync fallback)

### E.10 Future Extensions

- `TileLoaderCPAsync_SM100` for fallback/debug mode
- Bounds-checked loading for edge tiles
- Profiling hooks for load performance analysis

---

## Appendix F: Phase 5 Detailed Implementation Plan - TileWriter

### F.1 Overview

**Goal**: Break down the complex epilogue (store) functions into modular,
reusable components similar to SM90's `TileWriter` abstractions.

### F.2 SM100 Epilogue Pipeline

```text
TMEM (accumulators) → Registers → SMEM → GMEM (via TMA)
     [tcgen05_ld]     [cast]    [st.matrix]  [async_store]
```

### F.3 Component Architecture

| Component | Responsibility | Status |
|-----------|---------------|--------|
| `TMAStoreWriter` | TMA async store SMEM → GMEM | ✅ Implemented |
| `StMatrixCoords` | Coordinate computation for st.matrix | ✅ Implemented |
| `TMEMFragment` | Fragment pair from tensor memory | ✅ Implemented |
| `TMEMReader` | Load from TMEM using tcgen05_ld | ✅ Implemented |
| `AccumBarrier` | Pipeline barrier helper | ✅ Implemented |
| `StMatrixConfig` | st.matrix configuration parameters | ✅ Implemented |
| `StMatrixWriter` | Full st.matrix with swizzling | ✅ Implemented |
| `EpilogueConfig` | Stage/fragment configuration | ✅ Implemented |
| `FragmentCoords` | Fragment coordinate computation | ✅ Implemented |
| `EpilogueApplier` | Element-wise lambda application | ✅ Implemented |
| `OutputStageWriter` | Single stage orchestration | ✅ Implemented |

### F.4 File Structure

```text
sm100/
├── tile_writer.mojo    # NEW - Modular epilogue components
├── tile_loader.mojo    # Phase 4 - TileLoaderTMA
├── ring_buffer.mojo    # Phase 3 - Ring buffer
├── pipeline.mojo       # ProducerConsumerPipeline
└── matmul.mojo         # Main kernel (uses components)
```

### F.5 TMAStoreWriter Usage

```mojo
var tma_writer = TMAStoreWriter[...](Pointer(to=c_tma_op))

# Store with explicit wait
tma_writer.store_tile(c_smem_tile, (n_coord, m_coord))
tma_writer.wait_stores[num_pending]()

# Or combined
tma_writer.store_tile_and_wait[0](c_smem_tile, coords)
```

### F.6 Integration Status

**Completed Integrations**:

1. ✅ `AccumBarrier`: `accum_arrive` function now delegates to `AccumBarrier.arrive()`
2. ✅ `store_fragment_to_smem`: `stsm_helper` now delegates for non-float32 types
3. ✅ `EpilogueConf`: Type alias added to `BlackwellMatmulSM100Kernel`
4. ✅ `TMEMToSMemWriter`: Now used in `copy_accum_to_gmem` for the default path
   (`register_based_epilogue=True` or no epilogue lambda). The SMEM epilogue
   path (`register_based_epilogue=False`) retains existing code for reshaped
   tile access.
   (`register_based_epilogue=True` or no epilogue lambda). The SMEM epilogue
   path (`register_based_epilogue=False`) retains existing code for reshaped
   tile access.

**tile_writer.mojo Component Summary** (1527 lines, 13 components):

- `TMAStoreWriter` - TMA async store operations
- `StMatrixCoords` - Coordinate computation for st.matrix
- `TMEMFragment` - Fragment pair abstraction
- `TMEMReader` - TMEM address computation
- `AccumBarrier` - Pipeline barrier helper
- `StMatrixConfig` - st.matrix configuration
- `StMatrixWriter` - Full st.matrix with swizzling
- `EpilogueConfig` - Stage/fragment configuration
- `FragmentCoords` - Fragment coordinate computation
- `EpilogueApplier` - Element-wise epilogue operations
- `OutputStageWriter` - Single stage orchestration
- `store_fragment_to_smem` - Static helper for st.matrix (matches stsm_helper interface)

**Next Steps**:

1. **register_epilogue delegation**: Delegate to `EpilogueApplier`
2. **copy_accum_to_gmem refactor**: Further decompose using components
3. **Create higher-level TileWriter**: Combine components for clean API

---

## G. Session Handoff Document (December 2024)

This section provides comprehensive context for continuing the refactoring work.

### G.1 Current File Structure

```text
sm100/
├── matmul.mojo         # Main kernel (~3221 lines)
│   ├── BlackwellMatmulSM100Kernel      # Main kernel struct
│   ├── BlackwellMatmulSM100FallbackKernel  # Fallback for edge cases
│   ├── B200MatmulSmem                  # Shared memory management
│   ├── KernelContext                   # Common kernel state
│   ├── WarpRole                        # Warp specialization roles
│   ├── copy_accum_to_gmem              # TMEM→SMEM→GMEM epilogue (~260 lines)
│   ├── multi_stage_store_C             # Alternative store path
│   └── consumer_main_loop              # MMA consumer (standalone)
│
├── matmul_kernels.mojo # 🆕 Reference impl for extraction (~801 lines)
│   ├── WarpRole                        # (duplicate for extraction)
│   ├── KernelContext                   # (duplicate for extraction)
│   ├── accum_arrive()                  # Accumulator arrival signal
│   ├── copy_accum_to_gmem()            # (duplicate for extraction)
│   ├── multi_stage_store_C()           # (duplicate for extraction)
│   ├── multi_stage_store_C_split_k()   # (duplicate for extraction)
│   └── consumer_main_loop()            # (duplicate for extraction)
│
├── tile_writer.mojo    # Output components (~2526 lines)
│   ├── TMAStoreWriter (alias)          # = SM90's TileWriterTMA
│   ├── ThreadwiseStoreWriter (alias)   # = SM90's TileWriterThreadwise
│   ├── TMEMToSMemWriter                # TMEM→SMEM (default path)
│   ├── SMemEpilogueWriter              # SMEM epilogue path
│   ├── AccumTile                       # Upper+lower fragment pair
│   ├── TMAStoreCoords                  # Coordinate computation
│   ├── TMAStoreExecutor                # TMA store orchestration
│   ├── EpilogueApplier                 # Register-based epilogue
│   ├── EpilogueConfig                  # Stage/fragment configuration
│   ├── AccumBarrier                    # MMA completion signal
│   ├── load_tmem_fragments()           # TMEM load helper
│   ├── tma_wait_pipelined()            # TMA wait helper
│   ├── store_fragment_to_smem()        # st.matrix helper
│   ├── shared_memory_epilogue()        # SMEM epilogue (non-transpose)
│   └── shared_memory_epilogue_transpose()  # SMEM epilogue (transpose)
│
├── tile_loader.mojo    # Input components (~239 lines)
│   └── TileLoaderTMA                   # TMA loading with k_group_size
│
├── ring_buffer.mojo    # Synchronization (~445 lines)
│   ├── RingBuffer                      # Main struct
│   ├── Producer/Consumer               # Views with get_tiles()
│   └── ProducerTiles/ConsumerTiles     # Context managers
│
├── config.mojo         # SM100Config struct
├── pipeline.mojo       # ProducerConsumerPipeline
├── dispatch.mojo       # Dispatch logic
└── tile_scheduler*.mojo # CLC scheduling
```

### G.2 What's Integrated vs Available

| Component | Status | Location | Usage in matmul.mojo |
|-----------|--------|----------|---------------------|
| `TileLoaderTMA` | ✅ Integrated | tile_loader.mojo | `run()`, `run_splitk()` |
| `RingBuffer` | ✅ Integrated | ring_buffer.mojo | Producer/consumer loops |
| `load_tmem_fragments` | ✅ Integrated | tile_writer.mojo | `copy_accum_to_gmem` |
| `AccumBarrier.arrive` | ✅ Integrated | tile_writer.mojo | `accum_arrive` delegates |
| `EpilogueApplier` | ✅ Integrated | tile_writer.mojo | Register-based epilogue path |
| `TMAStoreCoords` | ✅ Integrated | tile_writer.mojo | `copy_accum_to_gmem` |
| `TMAStoreExecutor` | ✅ Integrated | tile_writer.mojo | `copy_accum_to_gmem` |
| `tma_wait_pipelined` | ✅ Integrated | tile_writer.mojo | `copy_accum_to_gmem` |
| `store_fragment_to_smem` | ✅ Integrated | tile_writer.mojo | `stsm_helper` delegates |
| `TMEMToSMemWriter` | ✅ Integrated | tile_writer.mojo | Default epilogue path |
| `SMemEpilogueWriter` | ✅ Integrated | tile_writer.mojo | SMEM epilogue path |
| `AccumTile` | ✅ Integrated | tile_writer.mojo | Tile type for writers |
| `EpilogueConfig` | 🔶 Available | tile_writer.mojo | Type alias only |
| `StMatrixWriter` | 🔶 Available | tile_writer.mojo | Not yet used |

### G.3 The `copy_accum_to_gmem` Decomposition

The epilogue function is fully modularized with dedicated components:

```text
copy_accum_to_gmem (lines 620-880)
│
├── [Setup] L687-729: Constants, component initialization
│   ├── SMEMWriter = TMEMToSMemWriter[...]
│   ├── StoreExecutor = TMAStoreExecutor[...]
│   └── epilogue_applier = EpilogueApplier[...]
│
└── [@parameter for stage] L735-875:
    │
    ├── load_tmem_fragments()           ✅ INTEGRATED
    │
    ├── accum_arrive()                  ✅ INTEGRATED
    │
    ├── epilogue_applier.apply_to_both_fragments()  ✅ INTEGRATED
    │   └── Register-based epilogue path
    │
    ├── SMEM_WRITE_SECTION:
    │   ├── DEFAULT PATH (register_based_epilogue=True):
    │   │   └── smem_writer.write_fragments()    ✅ INTEGRATED
    │   │
    │   └── SMEM_EPILOGUE PATH (register_based_epilogue=False):
    │       └── SMemEpilogueWriter.write_tile()  ✅ INTEGRATED
    │
    ├── TMAStoreCoords()                ✅ INTEGRATED
    │
    ├── TMAStoreExecutor.execute()      ✅ INTEGRATED
    │
    ├── tma_wait_pipelined()            ✅ INTEGRATED
    │
    └── named_barrier()                 (sync after TMA read)
```

**All paths now use dedicated components.** The SMEM epilogue path uses
`SMemEpilogueWriter` with Mojo parameter inference for a clean API.

### G.4 SMemEpilogueWriter Design

The SMEM epilogue path is now encapsulated in `SMemEpilogueWriter`, which uses
Mojo's parameter inference to reduce boilerplate:

```mojo
# Call site - clean API with parameter inference
var writer = SMemEpilogueWriter[
    epilogue_dtype, BM, BN, MMA_M, MMA_N, cta_group, num_output_warps,
    c_swizzle, transpose_c, is_lower_frag_required, num_stages,
    simd_size, stage, rep_frag_size, compute_lambda_fn,
](warp_id, c_tiles, c_shape, c_coord)
writer.write_tile(AccumTile(upper_frag_casted, lower_frag_casted))
```

**Parameter inference benefits:**

- 3 inferred from `c_tiles`: `c_type`, `c_smem_layout`, `num_output_stages`
- 2 derived from layout: `stageN`, `stage_contiguous_size`
- Reduced from 21 to 16 explicit parameters

**Key design decisions:**

- `stage` baked into type (created fresh each `@parameter for` iteration)
- Coordinates stored in constructor, not passed to `write_tile`
- `AccumTile` struct holds upper+lower fragments (SM100's TMEM layout)

### G.5 Practical Knowledge / Gotchas

#### Mojo Parameter Inference

1. **Infer-only parameters (`//`)**: Use for types that can be inferred
   from arguments

```mojo
fn foo[a_type: DType, layout: Layout //](
    tensor: LayoutTensor[a_type, layout, _]
):
```

1. **Name collisions**: When struct parameter names conflict with function
   parameter names, rename the function parameters:

```mojo
struct Foo[a_type: DType]:
    fn bar[a_type_: DType](self, x: SIMD[a_type_, _]):
```

1. **`@register_passable("trivial")`**: All fields must also be
   `@register_passable("trivial")`. This prevented storing `Tuple`
   directly in structs - use individual fields instead.

1. **Compile-time vs runtime in templates**: Use `alias` for compile-time
   values in template parameters, `var` for runtime. The compiler will
   error on "cannot use dynamic value in call parameter" if wrong.

1. **Import paths**: From `sm100/tile_writer.mojo` to
   `sm90/tile_writer.mojo`, use:

```mojo
from linalg.matmul.gpu.sm90.tile_writer import TileWriterTMA
```

NOT `from ..sm90.tile_writer import` (relative imports can be tricky).

#### SM100-Specific

1. **k_group_size**: SM100 batches K-tiles to share barriers. Affects
   ring buffer stage count:
   - `num_pipeline_stages` = total stages
   - `num_group_stages` = `num_pipeline_stages // k_group_size`

1. **TMEM (Tensor Memory)**: SM100 accumulators live in TMEM, not
   registers. Use `tcgen05_ld` to load them to registers before SMEM write.

1. **cta_group**: 1 or 2 CTAs cooperating. Affects tile shapes and warp
   election logic.

1. **TMA swizzle**: 128B swizzle limits tile shapes for transposed output.

### G.6 Recommended Next Steps

#### ✅ DONE: Option A - TMEMToSMemWriter Integration (v2.4)

- Integrated for `register_based_epilogue=True` (default path)
- SMEM epilogue path retains existing code (epilogue interleaving problem)
- Saved ~90 lines in the common path

#### ✅ DONE: Option B - TMAStoreExecutor Integration (v2.5)

- Created `TMAStoreExecutor` with static methods
- Encapsulates all TMA store branching (transpose variants, cta_group, MMA_M)
- Saved ~55 additional lines

#### ✅ DONE: Option C - EpilogueApplier Integration (v2.6)

- Integrated `EpilogueApplier.apply_to_both_fragments()` into `copy_accum_to_gmem`
- Replaces standalone `register_epilogue` function call
- Uses correct `elementwise_compute_lambda_type` with width=1 instantiation
- Also benefits `multi_stage_store_C` and `multi_stage_store_C_split_k` via delegation

#### 🔶 DEFERRED: Option D - SMEM Epilogue Full Integration

- Would require complex tile reshaping return/callback mechanism
- Low benefit/cost ratio since `register_based_epilogue=True` is default
- Current inline code works correctly
- Can revisit if SMEM epilogue becomes more commonly used

#### ✅ DONE: Dead Code & Comment Cleanup (v2.10)

- Removed `register_epilogue` function (~110 lines) and helper (~70 lines)
- Removed unused `simd_size` variable
- Moved `shared_memory_epilogue*` functions to tile_writer.mojo
- Added `SMemPtr` type alias usage across kernel files
- Added `@always_inline` annotations to all performance-critical methods
- Condensed verbose docstrings to single-line summaries
- Changed FIXME to NOTE for `consumer_main_loop` (used by blockwise_fp8)

#### 🆕 Phase 6: Extract matmul_kernels.mojo (v2.12)

**Goal**: Match SM90's clean file separation where:

- `matmul.mojo` (~746 lines) = Entry points + TMA setup
- `matmul_kernels.mojo` (~1363 lines) = Kernel struct + helper functions

**Created**: `sm100/matmul_kernels.mojo` as a **reference implementation** with:

```text
matmul_kernels.mojo (801 lines) - Reference Implementation
│
├── WarpRole                      # Warp specialization roles
├── KernelContext                 # Shared kernel state struct
├── accum_arrive()                # Accumulator arrival signal
├── copy_accum_to_gmem()          # TMEM→SMEM→GMEM epilogue
├── multi_stage_store_C()         # Output pipeline orchestration
├── multi_stage_store_C_split_k() # Split-K output pipeline
└── consumer_main_loop()          # MMA consumer (external API)
```

**Status**: ✅ COMPLETED

**What Was Done**:

1. Moved `B200MatmulSmem` to `matmul_kernels.mojo`
2. Moved `BlackwellMatmulSM100Kernel` to `matmul_kernels.mojo`
3. Moved `BlackwellMatmulSM100FallbackKernel` to `matmul_kernels.mojo`
4. Moved helper functions (`stsm_helper`, `f32_frag_to_smem`, `RLayout32Bits`)
5. Updated imports in `matmul.mojo` to import from `matmul_kernels.mojo`
6. Removed duplicate code from `matmul.mojo`
7. Minimized imports in both files
8. All smoke tests pass

**Final Result**:

- `matmul.mojo`: 697 lines (CPU-only: TMA setup, kernel launch wrappers)
- `matmul_kernels.mojo`: 2,105 lines (kernel structs, warp roles, MMA consumer)
- `matmul_output.mojo`: 529 lines (output pipeline: TMEM → SMEM → GMEM)

#### Phase 7: Extract Output Pipeline to matmul_output.mojo

**Status**: ✅ COMPLETED

Following the SM90 pattern (`sm90/matmul_output.mojo`), extracted output pipeline
functions to a dedicated file for cleaner separation of concerns.

**What Was Moved**:

1. `accum_arrive` - Signal accumulator completion
2. `copy_accum_to_gmem` - Core epilogue pipeline (TMEM → Registers → SMEM → GMEM)
3. `multi_stage_store_C` - Standard output orchestration
4. `multi_stage_store_C_split_k` - Split-K output with reduction

**File Organization (SM90-like)**:

```text
sm100/
├── matmul.mojo          # CPU entry points (697 lines)
├── matmul_kernels.mojo  # Kernel structs (2,025 lines)
├── matmul_output.mojo   # Output pipeline (549 lines)
├── tile_loader.mojo     # Input pipeline
├── tile_writer.mojo     # Output components
├── ring_buffer.mojo     # Pipeline synchronization (726 lines)
└── ...
```

#### Phase 8: OutputRingBuffer for MMA→Epilogue Pipeline

**Status**: ✅ COMPLETED

Created `OutputRingBuffer` to encapsulate MMA→Epilogue synchronization with clean
context manager API, matching the input `RingBuffer` pattern.

**New Components in ring_buffer.mojo**:

1. `OutputStage` - Holds stage index and TMEM offset
2. `OutputRingBuffer` - Wrapper around `ProducerConsumerPipeline` + TMEM base
3. `OutputProducerContext` - Context manager for MMA warp (acquire/release)
4. `OutputConsumerContext` - Context manager for Epilogue warp

**Updated Signatures**:

- `multi_stage_store_C` - Now takes `OutputStage` instead of raw `tmem_addr`
- `multi_stage_store_C_split_k` - Same change

**Usage Pattern**:

```mojo
# MMA warp (producer)
var output_rb = Self.OutputRB(mma_output_pipeline, tmem_addr, mma_complete_mask)
with output_rb.producer() as stage:
    Self.mma(stage.tmem_offset, tiles, mma_op, ...)

# Epilogue warp (consumer)
var output_rb = Self.OutputRB(mma_output_pipeline, tmem_addr, mma_complete_mask)
with output_rb.consumer() as stage:
    multi_stage_store_C[...](..., stage, ...)
```

**Benefits**:

- Encapsulated synchronization (wait/signal hidden in context managers)
- Type-safe stage passing (OutputStage vs raw UInt32)
- Consistent pattern with input RingBuffer
- RAII-style resource management

#### Future Improvements

- Add compile-time validation for template parameters
- Consider further modularization of warp role handlers

### G.7 Test Commands

```bash
# Quick smoke test
cd /home/ubuntu/modular
./bazel-bin/KGEN/tools/mojo/mojo run max/kernels/test/gpu/linalg/test_matmul_sm100_smoke.mojo

# Check for linter errors
# (use read_lints tool in Cursor)
```

### G.8 Files to Read First in New Session

1. `sm100/matmul_kernels.mojo` - Reference implementation for extraction (~801 lines)
2. `sm100/tile_writer.mojo` - All output components (~2526 lines)
3. `sm100/matmul.mojo:620-880` - `copy_accum_to_gmem` function
4. `sm100/matmul.mojo:1335-1435` - `KernelContext` struct
5. `sm90/matmul_kernels.mojo` - SM90 kernel struct (pattern to follow)
6. This document section G

#### Phase 9: Work Iteration Context Managers

Encapsulated the brittle implicit work iteration patterns with explicit, self-documenting
context managers that handle the correct sequencing of `fetch_next_work`, work_info
assignment, and `consumer_state.step()` calls.

**Key insight**: Different warps require different timing for the `step()` call:

| Pattern | Used By | Timing |
|---------|---------|--------|
| `advance_after_work` | Load, Scheduler, Epilogue | step in `__exit__` (after work) |
| `prefetch_before_work` | MMA | step on construction (before work) |

**New Components (tile_scheduler.mojo and tile_scheduler_splitk.mojo)**:

1. `AdvanceAfterWorkContext` - Context for warps that do work THEN advance
2. `PrefetchBeforeWorkContext` - Context for MMA warp software pipelining
3. `scheduler.advance_after_work()` - Returns context that fetches/steps in `__exit__`
4. `scheduler.prefetch_before_work()` - Fetches/steps immediately, assigns in `__exit__`

**Usage**:

```mojo
# Load/Scheduler/Epilogue warps
with scheduler.advance_after_work(work_info, clc_state) as current:
    do_work(current)
# fetch, assign, step happen here

# MMA warp (software pipelining)
with scheduler.prefetch_before_work(work_info, clc_state) as _:
    do_mma()  # Prefetch already done
# assign prefetched value to work_info
```

**Benefits**:

- Self-documenting method names make iteration patterns explicit
- Step encapsulated - caller cannot forget `step()` call
- Correct sequencing guaranteed for each warp type
- Both smoke and split-K tests pass

---

*Document Version: 2.18*
*Last Updated: December 2024*
*Authors: AI Assistant (Claude)*

**Changelog**:

- v2.18: Phase 13 - Scheduler owns throttle pipeline.
  - `TileScheduler` now takes `throttle_storage_ptr` in constructor
  - Added `ThrottlePipeline` type alias and `throttle_pipeline` field
  - Added static `init_throttle_barriers()` method for barrier initialization
  - Factory methods `work_iterator()` and `scheduler_iterator()` no longer need
    pipeline argument - throttle pipeline obtained from scheduler
  - `WorkIterator` and `SchedulerWorkIterator` get throttle from scheduler
  - Same changes applied to `TileSchedulerSplitK` and split-K iterators
  - Eliminated `load_clc_pipeline` variable from kernel code
  - Updated `warp_specialized_blockwise_fp8.mojo` for compatibility
  - All smoke and split-K tests pass
- v2.17: Phase 12 - RingBuffer pipeline encapsulation.
  - `RingBuffer` now takes storage pointer directly, creates pipeline internally
  - Added static `init_barriers()` method to `RingBuffer`
  - Eliminated `load_mma_pipeline` variable from kernel code
  - `consumer_main_loop` public API unchanged (still takes pipeline directly)
  - All smoke and split-K tests pass
- v2.16: Phase 11 - CLC throttle encapsulation.
  - `WorkIterator` and `SchedulerWorkIterator` now take `load_clc_pipeline`
  - Added `throttle_signal(is_first_cta)` to `WorkIterator` (Load warp)
  - Renamed `advance_producer()` → `signal_and_advance()` in
    `SchedulerWorkIterator` - combines consumer signal + producer advance
  - Eliminated explicit `load_clc_pipeline.producer/consumer_signal_and_step()`
    calls from kernel code - now encapsulated in iterators
  - Same changes applied to split-K scheduler variants
  - All smoke and split-K tests pass
- v2.15: Phase 10 - Final cleanup and iterator consolidation.
  - Removed unused `required_clc_query` variable (always True) from all warps
  - Added `drain()` method to `Producer` in ring_buffer.mojo for exit sync
  - Consolidated `work_info` into `WorkIterator` and `SchedulerWorkIterator`
  - Added `has_work()` method replacing external `work_info.is_valid()` checks
  - Renamed: `advance_after_work()` → `next()`, `prefetch_before_work()` →
    `next_prefetch()`
  - All work loops now use clean `work_iter.has_work()` / `next()` API
  - `OutputRingBuffer` now takes storage pointer directly, creates pipeline
    internally. Added static `init_barriers()` method. Eliminated
    `mma_output_pipeline` variable from kernel code.
  - All smoke and split-K tests pass
- v2.14: Phase 9 - Work Iteration Context Managers. Added `advance_after_work`
  and `prefetch_before_work` context managers to tile_scheduler.mojo and
  tile_scheduler_splitk.mojo. Encapsulates the different step() timing
  requirements for Load/Scheduler/Epilogue warps (step after work) vs MMA warp
  (step before work - software pipelining). Updated all 8 work loops in
  matmul_kernels.mojo to use new context managers. All smoke and split-K tests pass.
- v2.13: Phase 8 - OutputRingBuffer. Created `OutputRingBuffer` struct to
  encapsulate MMA→Epilogue synchronization with context manager API. Added
  `OutputStage`, `OutputProducerContext`, `OutputConsumerContext`. Updated
  `multi_stage_store_C` and `multi_stage_store_C_split_k` to take `OutputStage`
  parameter. Both MMA and Epilogue warps now use clean `with` statements for
  stage acquire/release. ring_buffer.mojo: 726 lines. All smoke tests pass.
- v2.12: Phase 6 - matmul_kernels.mojo extraction. Created reference
  implementation with WarpRole, KernelContext, accum_arrive, copy_accum_to_gmem,
  multi_stage_store_C, multi_stage_store_C_split_k, consumer_main_loop (~801
  lines). Full integration deferred pending build environment access. Added
  detailed integration plan following SM90 pattern.
- v2.11: SMemEpilogueWriter integration - Created `SMemEpilogueWriter` component
  for SMEM-based epilogue path. Uses Mojo parameter inference (infer-only params
  before `//`) to deduce 3 params from `c_tiles` and derive 2 from layout. Added
  `AccumTile` struct for upper+lower fragment pairs. Stage baked into type for
  clean `write_tile(tile)` API. matmul.mojo: 3,225 lines. tile_writer.mojo:
  2,526 lines. All 10 smoke tests pass.
- v2.10: Final cleanup - Added `SMemPtr` type alias usage across kernel files
  (matmul.mojo, ring_buffer.mojo, tile_scheduler.mojo, tile_scheduler_splitk.mojo).
  Added `@always_inline` annotations to all performance-critical methods.
  Condensed verbose docstrings to single-line summaries for cleaner code.
  matmul.mojo: 3,380 lines. All 10 smoke tests pass.
- v2.9: KernelContext refactoring - Created `KernelContext` struct to
  encapsulate common state shared between `run()` and `run_splitk()`.
  Includes election variables, CTA coordinates, multicast masks, pipeline
  states, and TMEM pointer. Added `init_barriers()` helper. Eliminated
  ~120 lines of duplicated init code. matmul.mojo: 3,566 lines.
- v2.8: SMEM epilogue extraction - Moved `shared_memory_epilogue_transpose`
  (~185 lines) and `shared_memory_epilogue` (~180 lines) to tile_writer.mojo.
  Added necessary imports (WARP_SIZE, lane_id, get_warp_id, named_barrier,
  RLayout32Bits, zipped_divide, upcast, blocked_product, FastDiv, etc.).
  matmul.mojo reduced from 3,829 to 3,428 lines (~400 lines moved).
  tile_writer.mojo increased from 1,775 to 2,219 lines. All 10 smoke tests pass.
- v2.7: Dead code cleanup - Removed `register_epilogue` function (~110 lines)
  and `_compute_register_lambda_fn` helper (~70 lines) which became dead code
  after EpilogueApplier integration. Also removed unused `simd_size` variable
  in `multi_stage_store_C`. Total: ~190 lines removed. matmul.mojo reduced
  from 4,019 to 3,829 lines. All 10 smoke tests pass.
- v2.6: EpilogueApplier integration - Integrated `EpilogueApplier` into
  `copy_accum_to_gmem` to replace the inline `register_epilogue` call.
  Added `apply_to_both_fragments()` method to EpilogueApplier. Updated lambda
  type to use `elementwise_compute_lambda_type` with width=1 instantiation.
  Benefits propagate to `multi_stage_store_C` and `multi_stage_store_C_split_k`
  via delegation. Deferred SMEM epilogue full integration due to low benefit/cost
  ratio. All 10 smoke tests pass, performance at ~1749 TFLOPS.
- v2.5: TMAStoreExecutor integration - Created `TMAStoreExecutor` struct with
  static methods to encapsulate all TMA store logic including SMEM tiling,
  fence, async_store, and commit_group. Handles 3 paths: transpose+cta_group2+MMA128,
  transpose+other (loop over swizzle tiles), and non-transpose. Replaces ~55 lines
  with single `StoreExecutor.execute()` call. All 10 smoke tests pass,
  performance unchanged (~1717 TFLOPS).
- v2.4: TMEMToSMemWriter integration - Integrated `TMEMToSMemWriter` into
  `copy_accum_to_gmem` for the default path (register_based_epilogue=True or
  no lambda). This replaces ~90 lines of SMEM write code with a single
  `smem_writer.write_fragments()` call. The SMEM epilogue path retains
  existing code. All 10 smoke tests pass, performance unchanged (~1713 TFLOPS).
- v2.3: Session handoff - Added comprehensive Section G with decomposition
  tree, epilogue interleaving analysis, practical gotchas, and next steps
- v2.2: Phase 5 SM90 reuse - Imported TileWriterTMA, TileWriterThreadwise,
  SMemTileWriter from SM90; added TMEMToSMemWriter for SM100-specific
  TMEM→SMEM path; tile_writer.mojo now 1121 lines
- v2.1: Phase 5 delegation - stsm_helper now delegates to store_fragment_to_smem,
  added RLayout32Bits and Swizzle imports to tile_writer.mojo
- v2.0: Phase 5 integration - AccumBarrier integrated into accum_arrive function,
  EpilogueConf type alias added to kernel struct, all smoke tests passing
- v1.9: Phase 5 major progress - Added FragmentCoords, EpilogueApplier,
  OutputStageWriter; tile_writer.mojo now at 938 lines with 11 components
- v1.8: Phase 5 progress - Added StMatrixConfig, StMatrixWriter, EpilogueConfig;
  tile_writer.mojo now at 636 lines with 8 complete components
- v1.7: Phase 5 started - Created tile_writer.mojo with TMAStoreWriter,
  StMatrixCoords, TMEMFragment, TMEMReader, AccumBarrier components
- v1.6: Phase 4 completed - TileLoaderTMA extracted with separate A/B origins,
  integrated into run() and run_splitk(), all smoke tests passing
- v1.5: Added Phase 4 detailed implementation plan for TileLoader
- v1.4: Phase 3 enhanced - SM90-style `get_tiles()` API with `ProducerTiles`/
  `ConsumerTiles` containing tile arrays; functions now accept `tiles` directly
- v1.3: Phase 3 completed - RingBufferSM100 implemented and integrated
- v1.2: Added Phase 3 detailed implementation plan for RingBufferSM100
- v1.1: Phase 1 completed - kernel struct refactoring with both main and
  fallback kernels
- v1.0: Initial refactoring plan
