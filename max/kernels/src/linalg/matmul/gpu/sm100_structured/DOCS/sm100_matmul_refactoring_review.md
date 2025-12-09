# SM100 Matmul Kernel Refactoring Review

This document helps original code authors understand the recent refactoring
changes. It maps original code patterns to their new locations and explains
the design rationale.

## Executive Summary

The refactoring encapsulated pipeline management and work iteration state
into dedicated structs, reducing boilerplate and making synchronization
patterns explicit. **No functional changes** - all original logic preserved.

### Key Transformations

| Original Pattern | New Location |
|------------------|--------------|
| `load_mma_pipeline` variable | Inside `RingBuffer` struct |
| `mma_output_pipeline` variable | Inside `OutputRingBuffer` struct |
| `load_clc_pipeline` variable | Inside `TileScheduler` struct |
| `work_info` + `clc_pipe_consumer_state` | Inside `WorkIterator` struct |
| `clc_pipe_producer_state` | Inside `SchedulerWorkIterator` struct |
| Manual `fetch_next_work` + `step()` | `work_iter.next()` context manager |

---

## 1. Load Warp: Before and After

### BEFORE (Original Code)

```python
# Setup
var load_clc_pipeline = ProducerConsumerPipeline[...](clc_throttle_storage.unsafe_ptr())
var load_mma_pipeline = ProducerConsumerPipeline[...](tma_mma_mbars_storage.unsafe_ptr())
var scheduler = Self.Scheduler(cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)
var work_info = scheduler.initial_work_info()
var clc_pipe_consumer_state = PipelineState[num_stages]()

# Work loop
while work_info.is_valid():
    # CLC throttle signaling (only first CTA)
    if ctx.is_first_cta_in_cluster:
        load_clc_pipeline.wait_consumer()
        _ = load_clc_pipeline.full[load_clc_pipeline.producer_stage()].arrive()
        load_clc_pipeline.producer_step()

    tile_loader.set_work_tile(UInt(work_info.m), UInt(work_info.n))

    # TMA load with manual pipeline management
    for i in range(num_iters // k_group_size):
        load_mma_pipeline.wait_consumer()
        var stage = load_mma_pipeline.producer_stage()
        tile_loader.load_tiles(a_smem[stage], b_smem[stage], i * k_group_size, ...)
        load_mma_pipeline.producer_step()

    syncwarp()

    # Fetch next work and advance state
    var next_work_info = scheduler.fetch_next_work(work_info, clc_pipe_consumer_state)
    work_info = next_work_info
    clc_pipe_consumer_state.step()

# Drain pipeline before exit
for _ in range(num_stages):
    load_mma_pipeline.wait_consumer()
    load_mma_pipeline.producer_step()
```

### AFTER (Refactored Code)

```python
# Setup - pipelines encapsulated in their owners
var scheduler = Self.Scheduler(
    cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar,
    clc_throttle_ptr,  # Scheduler owns throttle pipeline
)
var load_mma_ring_buffer = Self.RingBuffer(
    tma_mma_mbars_ptr, a_smem, b_smem  # RingBuffer owns load_mma pipeline
)
var work_iter = scheduler.work_iterator()  # Owns work_info + consumer state

# Work loop - context managers handle state transitions
while work_iter.has_work():
    with work_iter.next() as current:  # fetch_next_work + step() in __exit__
        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)  # CLC throttle

        tile_loader.set_work_tile(UInt(current.m), UInt(current.n))

        # TMA load with context manager
        with load_mma_ring_buffer.producer() as producer:
            for i in range(num_iters // k_group_size):
                with producer.get_tiles() as tiles:  # wait + step in context
                    tile_loader.load_tiles(tiles, i * k_group_size, ...)

        syncwarp()

# Drain via method call
load_mma_ring_buffer.producer().drain()
```

### Where Did Each Part Go?

| Original Code | New Location | File |
|---------------|--------------|------|
| `load_clc_pipeline` variable | `scheduler.throttle_pipeline` | `tile_scheduler.mojo:469` |
| `load_mma_pipeline` variable | `ring_buffer.pipeline` | `ring_buffer.mojo:115` |
| `work_info` variable | `work_iter.work_info` | `tile_scheduler.mojo:246` |
| `clc_pipe_consumer_state` | `work_iter.consumer_state` | `tile_scheduler.mojo:247` |
| CLC throttle 4-line pattern | `work_iter.throttle_signal()` | `tile_scheduler.mojo:285` |
| `fetch_next_work` + `step()` | `AdvanceAfterWorkContext.__exit__` | `tile_scheduler.mojo:156` |
| Pipeline drain loop | `Producer.drain()` | `ring_buffer.mojo:216` |

---

## 2. MMA Warp: Before and After

### BEFORE (Original Code)

```python
var work_info = scheduler.initial_work_info()
var clc_pipe_consumer_state = PipelineState[num_stages]()

while work_info.is_valid():
    # Prefetch next work BEFORE doing MMA (software pipelining)
    var next_work_info = scheduler.fetch_next_work(work_info, clc_pipe_consumer_state)
    clc_pipe_consumer_state.step()  # Step happens BEFORE work

    # Do MMA work with current work_info
    consumer_main_loop[...](work_info, ...)

    work_info = next_work_info  # Assign prefetched value
```

### AFTER (Refactored Code)

```python
var work_iter = scheduler.work_iterator()

while work_iter.has_work():
    with work_iter.next_prefetch():  # fetch + step happen on ENTRY
        consumer_main_loop[...](work_iter.work_info, ...)
    # Prefetched value assigned in __exit__
```

### Key Insight: Two Different Step Timing Patterns

The MMA warp uses **software pipelining** - it fetches the NEXT work item
before processing the CURRENT one. This requires stepping the pipeline
state BEFORE doing work, not after.

| Warp Type | Pattern | Context Manager |
|-----------|---------|-----------------|
| Load, Scheduler, Epilogue | Step AFTER work | `next()` |
| MMA | Step BEFORE work (prefetch) | `next_prefetch()` |

---

## 3. Scheduler Warp: Before and After

### BEFORE (Original Code)

```python
var work_info = scheduler.initial_work_info()
var clc_pipe_consumer_state = PipelineState[num_stages]()
var clc_pipe_producer_state = PipelineState[num_stages](0, 1, 0)  # Unique to scheduler

while work_info.is_valid():
    # Consumer side: signal throttle consumed
    load_clc_pipeline.wait_producer()
    _ = load_clc_pipeline.empty[load_clc_pipeline.consumer_stage()].arrive()
    load_clc_pipeline.consumer_step()

    # Producer side: issue next CLC request
    clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipe_producer_state)

    var next_work_info = scheduler.fetch_next_work(work_info, clc_pipe_consumer_state)
    work_info = next_work_info
    clc_pipe_consumer_state.step()

# Drain pending CLC requests
for _ in range(num_stages):
    scheduler.empty_mbar[clc_pipe_producer_state.index()].wait(...)
    clc_pipe_producer_state.step()
```

### AFTER (Refactored Code)

```python
var sched_iter = scheduler.scheduler_iterator()  # Owns BOTH producer and consumer state

while sched_iter.has_work():
    with sched_iter.next():
        sched_iter.signal_and_advance()  # Consumer signal + producer advance

sched_iter.drain()  # Drain pending CLC requests
```

### Where Did Each Part Go?

| Original Code | New Location | File |
|---------------|--------------|------|
| `clc_pipe_producer_state` | `sched_iter.producer_state` | `tile_scheduler.mojo:331` |
| `clc_pipe_consumer_state` | `sched_iter.consumer_state` | `tile_scheduler.mojo:330` |
| Consumer signal 4-line pattern | Inside `signal_and_advance()` | `tile_scheduler.mojo:363` |
| `advance_to_next_work()` call | Inside `signal_and_advance()` | `tile_scheduler.mojo:367` |
| Producer drain loop | `sched_iter.drain()` | `tile_scheduler.mojo:397` |

---

## 4. Epilogue Warp: Before and After

### BEFORE (Original Code)

```python
var work_info = scheduler.initial_work_info()
var clc_pipe_consumer_state = PipelineState[num_stages]()

while work_info.is_valid():
    # Acquire output stage from MMA
    mma_output_pipeline.wait_producer()
    var output_stage = mma_output_pipeline.consumer_stage()

    multi_stage_store_C[...](
        output_stage,
        mma_output_pipeline,  # Passed explicitly
        work_info,
        ...
    )

    mma_output_pipeline.consumer_step()

    var next_work_info = scheduler.fetch_next_work(work_info, clc_pipe_consumer_state)
    work_info = next_work_info
    clc_pipe_consumer_state.step()
```

### AFTER (Refactored Code)

```python
var output_rb = Self.OutputRB(accum_mbars_ptr, ...)  # Owns mma_output_pipeline
var work_iter = scheduler.work_iterator()

while work_iter.has_work():
    with work_iter.next() as current:
        with output_rb.acquire_for_epilogue() as output_stage:  # wait + step in context
            multi_stage_store_C[...](
                output_stage,  # Pipeline carried inside OutputStage
                current,
                ...
            )
```

### Where Did Each Part Go?

| Original Code | New Location | File |
|---------------|--------------|------|
| `mma_output_pipeline` variable | `output_rb.pipeline` | `ring_buffer.mojo:602` |
| `wait_producer()` + `consumer_stage()` | `OutputConsumerContext.__enter__` | `ring_buffer.mojo:726` |
| `consumer_step()` | `OutputConsumerContext.__exit__` | `ring_buffer.mojo:736` |

---

## 5. Pipeline Helper Methods

Two new helper methods were added to `ProducerConsumerPipeline` to
encapsulate the common 4-line signal pattern:

### BEFORE

```python
# Producer signaling
load_clc_pipeline.wait_consumer()
_ = load_clc_pipeline.full[load_clc_pipeline.producer_stage()].arrive()
load_clc_pipeline.producer_step()

# Consumer signaling  
load_clc_pipeline.wait_producer()
_ = load_clc_pipeline.empty[load_clc_pipeline.consumer_stage()].arrive()
load_clc_pipeline.consumer_step()
```

### AFTER

```python
# Producer signaling
load_clc_pipeline.producer_signal_and_step()

# Consumer signaling
load_clc_pipeline.consumer_signal_and_step()
```

**Location**: `pipeline.mojo:185-199`

---

## 6. Barrier Initialization

Barrier initialization moved from inline code to static methods on each
struct that owns a pipeline:

### BEFORE

```python
# In init_barriers()
load_mma_pipeline.init_mbars(producer_count, consumer_count)
mma_output_pipeline.init_mbars(producer_count, consumer_count)
load_clc_pipeline.init_mbars(producer_count, consumer_count)
```

### AFTER

```python
# In init_barriers()
Self.RingBuffer.init_barriers(tma_mma_mbars_ptr, producer_count, consumer_count)
Self.OutputRB.init_barriers(accum_mbars_ptr, producer_count, consumer_count)
Self.Scheduler.init_throttle_barriers(clc_throttle_ptr, producer_count, consumer_count)
```

---

## 7. File Structure Summary

| File | What It Contains |
|------|------------------|
| `tile_scheduler.mojo` | `TileScheduler`, `WorkIterator`, `SchedulerWorkIterator`, throttle pipeline |
| `tile_scheduler_splitk.mojo` | Split-K variants of above |
| `ring_buffer.mojo` | `RingBuffer` (TMA→MMA), `OutputRingBuffer` (MMA→Epilogue), all context managers |
| `pipeline.mojo` | `ProducerConsumerPipeline` with helper methods |
| `matmul_kernels.mojo` | Kernel entry points using the above abstractions |

---

## 8. Testing

All changes verified with:

- `test_matmul_sm100_smoke.mojo.test` - 10 test configurations
- `test_matmul_sm100_splitk_2sm_bf16.mojo.test` - Split-K verification

The refactoring is purely structural - no algorithmic changes.

---

## 9. Quick Reference: Context Manager Patterns

```python
# Load warp: advance AFTER work
with work_iter.next() as current:
    do_load_work(current)
# fetch_next_work + step() happen here

# MMA warp: prefetch BEFORE work  
with work_iter.next_prefetch():
    do_mma_work(work_iter.work_info)
# assign prefetched value here

# Scheduler warp: signal + advance
with sched_iter.next():
    sched_iter.signal_and_advance()

# Ring buffer: tile access
with producer.get_tiles() as tiles:
    load_into(tiles)

# Output buffer: stage access
with output_rb.acquire_for_epilogue() as stage:
    store_from(stage)
```

---

*Document created for code review of SM100 matmul refactoring.*
*December 2024*
