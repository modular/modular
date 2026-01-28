# `M::AsyncRT::WorkQueue`

This document introduces the `M::AsyncRT::WorkQueue`, key design points and how
to use it.

## Overview

The `M::AsyncRT::WorkQueue` is an abstract interface for executing work items
concurrently. It is the core abstraction for managing CPU parallelism in
AsyncRT, providing a thread pool that distributes tasks across available CPU
cores.

### Creating a WorkQueue

A WorkQueue is created through factory functions rather than direct
construction:

- **`createSingleThreadWorkQueue`**: Creates a WorkQueue that only uses the
  calling (donor) thread with no synchronization overhead. Useful for
  single-threaded platforms.

- **`createThreadPoolWorkQueue`**: Creates a multi-threaded WorkQueue with the
  following parameters:
  - `numThreads`: Number of worker threads. If 0, defaults based on system
    configuration.
  - `maxThreads`: Upper bound for `numThreads` when auto-detecting.
  - `mainWillDonate`: If true, the creating thread will participate in work
    processing during `await()` calls.
  - `withAffinity`: If true, workers are pinned to specific CPU cores.
  - `threadBusyWaitTime`: Duration to spin before sleeping when idle (default
    1ms).
  - `poolName`: Prefix for thread names (visible in debuggers/profilers).

### Thread Types

The WorkQueue distinguishes between three types of threads:

1. **Worker threads**: Threads created by the WorkQueue that run a dedicated
   work-processing loop. Each worker has a unique `workerID` (0 to N-1).

2. **Main thread**: If `mainWillDonate` is true, the thread that created the
   WorkQueue is designated as the "main" thread (workerID 0). It participates
   in work processing during `await()` and must be the one to call
   `shutdown()`.

3. **Foreign threads**: Any other thread that interacts with the WorkQueue.
   Foreign threads may call `addTask()` and `await()` but do not donate
   themselves to processing work items.

### Worker Allocation and CPU Affinity

When `withAffinity` is enabled, worker threads are pinned to specific CPU cores:

1. **Default thread count**: If `numThreads` is 0:
   - On systems with P-cores and E-cores: uses the number of performance cores.
   - With affinity enabled: uses the number of physical cores.
   - Without affinity: uses the number of logical cores (including
     hyperthreads).

2. **CPU selection**: The `CPUSystemInfo::getPreferredCpuIDs()` function
   determines which CPUs to use, typically preferring:
   - Performance cores over efficiency cores.
   - Physical cores over hyperthreads (when affinity is set).
   - Cores within a single NUMA node when possible.

3. **Cgroup limits**: In containerized environments, thread count is
   automatically capped based on CPU limits (millicores / 1000).

4. **Affinity setting**: Each worker thread calls `setThreadAffinity(cpuID)` at
   startup to pin itself to its assigned CPU.

### Task Queues

The WorkQueue uses a hierarchy of task queues to balance efficiency with work
distribution:

1. **Local task list** (`localTaskList`): A per-worker list with no
   synchronization. Used for `addLocalTask()` calls from the owning thread.
   Work items here take highest priority. Ideal for short-running continuations
   (e.g., AsyncValue waiters) where context-switch overhead would dominate.

2. **Affinity task list** (`affinityTaskList`): A per-worker lock-free ring
   buffer. Used when `addTask()` is called with a non-negative `taskId`
   (typically from `async_parallelize` in Mojo). Tasks are processed by the
   specific worker indicated by `taskId`, enabling cache-friendly execution
   patterns.

3. **Global task list** (`taskList`): A lock-free MPMC queue shared by all
   workers. Used for `addTask()` with `taskId = kDefaultTaskId` (-1). Any
   worker can dequeue and process these tasks.

4. **Overflow task list** (`overflowTaskList`): A mutex-protected fallback
   queue used when the global task list is full. Workers check this before
   going to sleep.

Work items are processed in priority order: local → affinity → global →
overflow.

### Ownership and Lifecycle

The `WorkQueue` is typically owned by an `M::AsyncRT::Runtime` instance, which
creates and manages it based on `RuntimeOptions`. The lifecycle is:

1. **Creation**: Via `createThreadPoolWorkQueue()` or
   `createSingleThreadWorkQueue()`. Worker threads start immediately.

2. **Usage**: Clients add work via `addTask()` / `addLocalTask()` and wait for
   results via `await()`.

3. **Shutdown**: Must call `shutdown()` before destruction. This:
   - Drains remaining work items (main thread helps if `mainWillDonate`).
   - Sets the done flag to signal workers to exit.
   - Posts all worker semaphores to wake sleeping threads.
   - Joins all worker threads.

4. **Destruction**: After `shutdown()` returns, the WorkQueue can be destroyed.

### Idle Behavior and Sleep/Wake

When a worker has no tasks to process:

1. **Busy-wait phase**: Spins with exponential backoff for `busyWaitTime`
   (default 1ms), checking for new tasks.

2. **Overflow check**: Before sleeping, pumps any overflow/spill queues into
   the main queues.

3. **Sleep**: Marks itself as suspended in a shared bitvector and waits on its
   per-worker semaphore.

4. **Wake**: When `addTask()` sees suspended workers, it posts the appropriate
   semaphore(s) to wake them.

For systems with more than 64 worker threads, a multicast scheme groups workers
together in the suspension bitvector, waking all workers in a group when any
might be suspended.

### Key Design Principles

- **Non-blocking assumption**: Work items should not block. See
  [WorkQueueNonblocking.md](WorkQueueNonblocking.md) for rationale and
  strategies.

- **No immediate execution**: `addTask()` never runs work inline; tasks are
  always deferred. This prevents stack overflow and ensures predictable
  behavior.

- **Thread donation**: `await()` from a worker/main thread donates that thread
  to process work items while waiting, avoiding deadlock and maximizing CPU
  utilization.
