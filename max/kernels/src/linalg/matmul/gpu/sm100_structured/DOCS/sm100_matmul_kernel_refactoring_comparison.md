# SM100 Matmul Kernel Refactoring: Side-by-Side Comparison

This document compares the original `blackwell_tma_umma_warp_specialized_kernel`
function with the refactored `BlackwellMatmulSM100Kernel.run()` method,
highlighting key changes in structure, abstraction, and readability.

---

## Overview

| Aspect | Original | Refactored |
|--------|----------|------------|
| **Location** | `matmul.mojo` (monolithic) | `matmul_kernels.mojo` (kernel struct) |
| **Entry Point** | `blackwell_tma_umma_warp_specialized_kernel()` | `BlackwellMatmulSM100Kernel.run()` |
| **Pipeline Management** | Explicit variables (`load_mma_pipeline`, `mma_output_pipeline`) | Encapsulated in `RingBuffer`, `OutputRingBuffer` |
| **Work Iteration** | Manual `work_info` + `clc_pipe_consumer_state` | `WorkIterator` with `has_work()`, `next()` |
| **Tile Loading** | Inline `load_AB()` function | `TileLoaderTMA` struct with `load_tiles()` |

---

## 1. Initialization

### Original (main branch)

```mojo
ref smem_storage = external_memory[
    Scalar[DType.uint8],
    address_space = AddressSpace.SHARED,
    alignment=128,
]().bitcast[SmemType]()[]

var a_smem = LayoutTensorIter[
    a_type, a_smem_layout, MutAnyOrigin,
    address_space = AddressSpace.SHARED, alignment=128,
](a_smem_storage.unsafe_ptr(), SmemType.a_smem_size)

var b_smem = LayoutTensorIter[
    b_type, b_smem_layout, MutAnyOrigin,
    address_space = AddressSpace.SHARED, alignment=128,
](b_smem_storage.unsafe_ptr(), SmemType.b_smem_size)

# Load warp as producer and mma warp as consumer
var load_mma_pipeline = ProducerConsumerPipeline[
    Int(config.num_pipeline_stages // config.k_group_size)
](tma_mma_mbars_storage.unsafe_ptr())

# MMA warp as producer and Output warp as consumer
var mma_output_pipeline = ProducerConsumerPipeline[
    Int(config.num_accum_pipeline_stages)
](accum_mbars_storage.unsafe_ptr())

# Load warp as producer and scheduler warp as consumer
var load_clc_pipeline = ProducerConsumerPipeline[
    Int(config.num_clc_pipeline_stages)
](clc_throttle_storage.unsafe_ptr())

var ptr_tmem_addr = tmem_addr_storage.unsafe_ptr()

clc_response = clc_response_storage.unsafe_ptr()
clc_full_mbar = clc_mbars_full_storage.unsafe_ptr()
clc_empty_mbar = clc_mbars_empty_storage.unsafe_ptr()
tmem_dealloc_mbar = tmem_dealloc_mbar_storage.unsafe_ptr()

var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
var elect_one_thread = elect_one_sync_with_mask()
var elect_one_cta = (
    block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
)
var is_first_cta_in_cluster = block_rank_in_cluster() == 0

# Pipeline state variables - each warp needs its own copy
var clc_pipe_producer_state = PipelineState[
    Int(config.num_clc_pipeline_stages)
](0, 1, 0)
var clc_pipe_consumer_state = PipelineState[
    Int(config.num_clc_pipeline_stages)
]()

var scheduler = TileScheduler[...](
    cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar
)
var work_info = scheduler.initial_work_info()
```

### Refactored (this branch)

```mojo
# Access shared memory via bitcast (preserves original layout)
ref smem_storage = external_memory[...].bitcast[Self.SmemType]()[]

# Create typed tile arrays using centralized type aliases
var a_smem = Self.SmemType.ATileArray(a_smem_storage.unsafe_ptr())
var b_smem = Self.SmemType.BTileArray(b_smem_storage.unsafe_ptr())
var c_smem_tiles = Self.SmemType.CTileArray(c_smem_storage.unsafe_ptr())

# Create ring buffer for TMA-MMA synchronization (encapsulates pipeline)
var tma_mma_mbars_ptr = tma_mma_mbars_storage.unsafe_ptr()
var load_mma_ring_buffer = Self.RingBuffer(tma_mma_mbars_ptr, a_smem, b_smem)

# Set up pointers for CLC and TMEM management
var clc_response = clc_response_storage.unsafe_ptr()
var clc_full_mbar = clc_mbars_full_storage.unsafe_ptr()
var clc_empty_mbar = clc_mbars_empty_storage.unsafe_ptr()
var clc_throttle_ptr = clc_throttle_storage.unsafe_ptr()
var tmem_dealloc_mbar = tmem_dealloc_mbar_storage.unsafe_ptr()
var accum_mbars_ptr = accum_mbars_storage.unsafe_ptr()

# Create kernel context with election vars, CTA coords, and masks
var ctx = Self.Context(tmem_addr_storage.unsafe_ptr())

# Initialize all barriers (only elect_one_warp && elect_one_thread)
Self.init_barriers(ctx, a_tma_op, b_tma_op, c_tma_op, ...)

# Scheduler owns CLC throttle pipeline internally
var scheduler = Self.Scheduler(
    cluster_dim, clc_response, clc_full_mbar,
    clc_empty_mbar, clc_throttle_ptr,
)

# Per-warp work iterator - owns work_info, pipeline state, and throttle
var work_iter = scheduler.work_iterator()

# Create tile loader for TMA operations
var tile_loader = Self.TileLoaderTMA(
    Pointer(to=a_tma_op), Pointer(to=b_tma_op),
    ctx.a_multicast_mask, ctx.b_multicast_mask, ctx.peer_cta_coord,
)
```

**Key Changes:**

- ❌ Removed: `clc_pipe_producer_state`, `clc_pipe_consumer_state`,
  `work_info` explicit variables
- ❌ Removed: `load_mma_pipeline`, `mma_output_pipeline`,
  `load_clc_pipeline` explicit variables
- ✅ Added: `Self.Context` encapsulates election vars, masks, peer coords
- ✅ Added: `Self.RingBuffer` encapsulates TMA-MMA pipeline + tile storage
- ✅ Added: `scheduler.work_iterator()` returns iterator with owned state
- ✅ Added: `Self.TileLoaderTMA` encapsulates tile loading logic

---

## 2. Load Warp

### Original (main branch)

```mojo
if WarpRole.is_main_load():
    with MatmulProfilerType[0](workspace, 0):
        var required_clc_query = True

        @parameter
        if pdl_level > PDLLevel.OFF:
            wait_on_dependent_grids()

        while work_info.is_valid():
            # CLC throttle prevents each CTA from going a few waves ahead.
            if is_first_cta_in_cluster and required_clc_query:
                load_clc_pipeline.wait_consumer()
                var load_clc_producer_state = load_clc_pipeline.producer_stage()
                _ = load_clc_pipeline.producer_mbar(load_clc_producer_state)[0].arrive()
                load_clc_pipeline.producer_step()

            # DO TMA LOAD
            for i in range(num_iters // config.k_group_size):
                load_AB[
                    block_tile_shape = config.block_tile_shape,
                    mma_shape = config.mma_shape,
                    cta_group = config.cta_group,
                    k_group_size = config.k_group_size,
                ](
                    a_tma_op, b_tma_op,
                    a_smem, b_smem,
                    load_mma_pipeline,
                    peer_cta_coord,
                    (UInt(work_info.m), UInt(work_info.n)),
                    a_multicast_mask, b_multicast_mask,
                    i * config.k_group_size,
                    elect_one_cta,
                )
                load_mma_pipeline.producer_step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        # Prevent CTA to exit when a peer CTA is still working on mma.
        @parameter
        for i in range(config.num_pipeline_stages // config.k_group_size):
            load_mma_pipeline.wait_consumer()
            load_mma_pipeline.producer_step()
```

### Refactored (this branch)

```mojo
if WarpRole.is_main_load():
    with MatmulProfilerType[0](workspace, 0):

        @parameter
        if Self.pdl_level > PDLLevel.OFF:
            wait_on_dependent_grids()

        while work_iter.has_work():
            with work_iter.next() as current:
                # CLC throttle prevents each CTA from going ahead
                work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                # Set work tile coordinates for this iteration
                tile_loader.set_work_tile(UInt(current.m), UInt(current.n))

                # DO TMA LOAD for full K range
                with load_mma_ring_buffer.producer() as producer:
                    for i in range(0, num_iters, Self.config.k_group_size):
                        with producer.get_tiles() as tiles:
                            tile_loader.load_tiles(tiles, i, ctx.elect_one_cta)

                syncwarp()

        # Prevent CTA from exiting while peer CTA is still working on MMA
        with load_mma_ring_buffer.producer() as producer:
            producer.drain()
```

**Key Changes:**

- ❌ Removed: `required_clc_query` variable (always True)
- ❌ Removed: Manual `load_clc_pipeline.wait_consumer() / producer_step()` calls
- ❌ Removed: Manual `scheduler.fetch_next_work()` + `clc_pipe_consumer_state.step()`
- ❌ Removed: Manual pipeline drain loop with explicit step/wait
- ✅ Added: `work_iter.has_work()` for loop termination
- ✅ Added: `with work_iter.next() as current:` context manager auto-advances
- ✅ Added: `work_iter.throttle_signal(is_first_cta)` encapsulates throttle logic
- ✅ Added: `with producer.get_tiles() as tiles:` context manager for synchronization
- ✅ Added: `producer.drain()` method encapsulates drain loop

---

## 3. Scheduler Warp

### Original (main branch)

```mojo
if WarpRole.is_scheduler() and is_first_cta_in_cluster:
    @parameter
    if config.num_clc_pipeline_stages == 0:
        return

    with MatmulProfilerType[1](workspace, 0):
        var required_clc_query = True

        @parameter
        if pdl_level > PDLLevel.OFF:
            wait_on_dependent_grids()

        while work_info.is_valid():
            if required_clc_query:
                load_clc_pipeline.wait_producer()
                var load_clc_consumer_stage = load_clc_pipeline.consumer_stage()
                _ = load_clc_pipeline.consumer_mbar(load_clc_consumer_stage)[0].arrive()
                load_clc_pipeline.consumer_step()

                # advance to next work
                clc_pipe_producer_state = scheduler.advance_to_next_work(
                    clc_pipe_producer_state
                )

            # scheduler fetch next work
            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        # make sure all pipes are empty before kernel exit
        @parameter
        for i in range(config.num_clc_pipeline_stages):
            clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                clc_pipe_producer_state.phase()
            )
            clc_pipe_producer_state.step()
```

### Refactored (this branch)

```mojo
if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
    @parameter
    if Self.config.num_clc_pipeline_stages == 0:
        return

    # Scheduler warp uses its own iterator that manages both
    # producer and consumer state, plus throttle signaling
    var sched_iter = scheduler.scheduler_iterator()

    with MatmulProfilerType[1](workspace, 0):

        @parameter
        if Self.pdl_level > PDLLevel.OFF:
            wait_on_dependent_grids()

        while sched_iter.has_work():
            with sched_iter.next():
                sched_iter.signal_and_advance()

        # Drain all pending CLC requests before kernel exit
        sched_iter.drain()
```

**Key Changes:**

- ❌ Removed: `required_clc_query` variable
- ❌ Removed: Manual `load_clc_pipeline.wait_producer() / consumer_step()` calls
- ❌ Removed: Manual `clc_pipe_producer_state` / `clc_pipe_consumer_state` tracking
- ❌ Removed: Manual drain loop with explicit mbar wait
- ✅ Added: `scheduler.scheduler_iterator()` returns specialized iterator
- ✅ Added: `sched_iter.signal_and_advance()` combines throttle + advance
- ✅ Added: `sched_iter.drain()` encapsulates drain logic

---

## 4. MMA Warp

### Original (main branch)

```mojo
if WarpRole.is_mma():
    with MatmulProfilerType[2](workspace, 0):
        tcgen05_alloc[config.cta_group](ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        named_barrier_arrive[MMA_THREADS + EPILOGUE_THREADS](1)

        tmem_addr = ptr_tmem_addr[0]

        while work_info.is_valid():
            # scheduler fetch next work
            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            clc_pipe_consumer_state.step()

            # DO MMA
            if elect_one_cta:
                var mma_output_mma_stage = mma_output_pipeline.producer_stage()
                mma_output_pipeline.wait_consumer()
                var tmem_offset = tmem_addr + (mma_output_mma_stage * stage_stride_cols)

                for i in range(num_iters // config.k_group_size):
                    consumer_main_loop[
                        block_tile_shape = config.block_tile_shape,
                        mma_shape = config.mma_shape,
                        cta_group = config.cta_group,
                        cluster_shape = config.cluster_shape,
                        k_group_size = config.k_group_size,
                    ](
                        tmem_offset,
                        a_smem, b_smem,
                        load_mma_pipeline,
                        mma_op,
                        elect_one_warp,
                        i * config.k_group_size,
                        0,
                    )
                    load_mma_pipeline.consumer_step()

                # mma arrive multicast
                if elect_one_sync():
                    @parameter
                    if config.cta_group == 1:
                        mma_arrive[config.cta_group](
                            mma_output_pipeline.producer_mbar(mma_output_mma_stage)
                        )
                    else:
                        mma_arrive_multicast[config.cta_group](
                            mma_output_pipeline.producer_mbar(mma_output_mma_stage),
                            mma_complete_mask,
                        )
                mma_output_pipeline.producer_step()
            work_info = next_work_info

        tcgen05_release_allocation_lock[config.cta_group]()
        tmem_dealloc_mbar[].wait()
        tcgen05_dealloc[config.cta_group](tmem_addr, max_tmem_cols)
```

### Refactored (this branch)

```mojo
if WarpRole.is_mma():
    with MatmulProfilerType[2](workspace, 0):
        tcgen05_alloc[Self.config.cta_group](ctx.ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        named_barrier_arrive[Self.MMA_THREADS + Self.EPILOGUE_THREADS](1)

        var tmem_addr = ctx.ptr_tmem_addr[0]

        # Create output ring buffer for MMA→Epilogue synchronization
        var output_rb = Self.OutputRB(
            accum_mbars_ptr, tmem_addr, ctx.mma_complete_mask
        )

        while work_iter.has_work():
            # Prefetch next work BEFORE doing MMA (software pipelining)
            with work_iter.next_prefetch():
                # DO MMA for full K range
                if ctx.elect_one_cta:
                    with output_rb.producer() as stage:
                        with load_mma_ring_buffer.consumer() as consumer:
                            for i in range(0, num_iters, Self.config.k_group_size):
                                with consumer.get_tiles() as tiles:
                                    Self.mma(
                                        stage.tmem_offset,
                                        tiles, mma_op,
                                        ctx.elect_one_warp,
                                        i, 0,
                                    )

        tcgen05_release_allocation_lock[Self.config.cta_group]()
        tmem_dealloc_mbar[].wait()
        tcgen05_dealloc[Self.config.cta_group](tmem_addr, max_tmem_cols)
```

**Key Changes:**

- ❌ Removed: Manual `mma_output_pipeline.producer_stage() / wait_consumer() / producer_step()`
- ❌ Removed: Manual `load_mma_pipeline.consumer_step()`
- ❌ Removed: Manual tmem_offset calculation
- ❌ Removed: Inline `mma_arrive` / `mma_arrive_multicast` code
- ✅ Added: `Self.OutputRB` encapsulates MMA→Epilogue pipeline
- ✅ Added: `work_iter.next_prefetch()` for software pipelining pattern
- ✅ Added: `with output_rb.producer() as stage:` handles acquire/release + mma_arrive
- ✅ Added: `with consumer.get_tiles() as tiles:` handles wait/release
- ✅ Added: `Self.mma()` method encapsulates MMA loop body

---

## 5. Epilogue Warp

### Original (main branch)

```mojo
if WarpRole.is_epilogue():
    named_barrier[MMA_THREADS + EPILOGUE_THREADS](1)
    tmem_addr = ptr_tmem_addr[0]

    var tile_idx = 0

    while work_info.is_valid():
        with MatmulProfilerType[3](workspace, tile_idx):
            # WAIT FOR MMA TO FINISH AND STORE RESULT
            multi_stage_store_C[
                input_type=a_type,
                accum_type=accum_type,
                ...
            ](
                c_smem_iter,
                c_tma_op,
                mma_output_pipeline,  # Passed explicitly
                tmem_addr,            # Passed explicitly
                work_tile_coord=(work_info.m, work_info.n),
                elect_one_warp=elect_one_warp,
                M=mnk[0],
                N=mnk[1],
            )
            mma_output_pipeline.consumer_step()  # Manual step

            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        tile_idx += 1

    @parameter
    if config.cta_group == 2:
        _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
    _ = tmem_dealloc_mbar[].arrive()
```

### Refactored (this branch)

```mojo
if WarpRole.is_epilogue():
    named_barrier[Self.MMA_THREADS + Self.EPILOGUE_THREADS](1)
    var tmem_addr = ctx.ptr_tmem_addr[0]

    # Create output ring buffer for MMA→Epilogue synchronization
    var output_rb = Self.OutputRB(
        accum_mbars_ptr, tmem_addr, ctx.mma_complete_mask
    )

    var tile_idx = 0

    while work_iter.has_work():
        with work_iter.next() as current:
            with MatmulProfilerType[3](workspace, tile_idx):
                # WAIT FOR MMA TO FINISH AND STORE RESULT
                with output_rb.consumer() as stage:
                    multi_stage_store_C[...](
                        c_smem_tiles,
                        c_tma_op,
                        stage,  # Self-contained OutputStage
                        work_tile_coord=(current.m, current.n),
                        elect_one_warp=ctx.elect_one_warp,
                        M=mnk[0],
                        N=mnk[1],
                    )

        tile_idx += 1

    @parameter
    if Self.config.cta_group == 2:
        _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
    _ = tmem_dealloc_mbar[].arrive()
```

**Key Changes:**

- ❌ Removed: `mma_output_pipeline` passed to `multi_stage_store_C`
- ❌ Removed: `tmem_addr` passed to `multi_stage_store_C`
- ❌ Removed: Manual `mma_output_pipeline.consumer_step()`
- ❌ Removed: Manual `scheduler.fetch_next_work()` + `clc_pipe_consumer_state.step()`
- ✅ Added: `Self.OutputRB` encapsulates MMA→Epilogue pipeline
- ✅ Added: `with work_iter.next() as current:` auto-advances work
- ✅ Added: `with output_rb.consumer() as stage:` handles wait/release
- ✅ Added: `stage` is self-contained `OutputStage` with pipeline + tmem_offset

---

## Summary of Benefits

### 1. Reduced Boilerplate

- ~40% fewer lines of code in warp loops
- Pipeline state management is automatic
- No manual `step()` calls scattered throughout

### 2. Clearer Intent

- Context managers (`with ... as`) clearly show acquire/release boundaries
- `work_iter.has_work()` is more readable than `work_info.is_valid()`
- `producer.drain()` is clearer than manual drain loops

### 3. Encapsulation

- Pipeline ownership is clear (iterators own their state)
- `Context` struct groups related election/mask variables
- `TileLoaderTMA` encapsulates TMA loading complexity

### 4. Type Safety

- Iterators are parameterized on scheduler type
- Ring buffers are parameterized on tile types
- Context managers ensure proper cleanup

### 5. Reusability

- Same patterns work for both regular and split-K kernels
- `OutputStage` carries pipeline + offset together
- Iterators can be used in different warp roles
