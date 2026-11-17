# AsyncRT — agent notes

AsyncRT is the low-level concurrency library everything above it runs on:
GraphRT, the driver, and the compiler's own parallelism. `docs/README.md`
maps the libraries and `docs/AsyncRTRuntime.md` has the design rationale.
This file is the short version: the rules that change what you write.

Invoke the `graph-execution-engineer` skill before you edit anything here.

## Invariants — preserve these

- **It is a library, not the owner of the machine.** Several `CPUDevice`
  instances must be able to coexist in one process, next to whatever else the
  host application is doing with threads. There are no global operations
  acting on an implicit pool, and policy such as the thread pool and the
  allocator is abstracted away from the algorithms
  (`docs/AsyncRTRuntime.md`). A convenience API that reaches a process-wide
  singleton breaks the premise of the library.
- **Work items never block**, and `WorkQueue` implementations are allowed to
  assume it (`docs/WorkQueueNonblocking.md`). A blocking call inside a work
  item takes a thread out of the pool and under-uses the machine.
- **`await()` is for top-level clients.** It donates the calling thread to the
  pool, so calling it from inside a work item runs unrelated queue work on
  that frame's stack. The driver has been bitten by this.
- **`addTask()` never runs work inline.** Tasks are posted and picked up by a
  worker (`docs/WorkQueue.md`). Code that relies on a task having started by
  the time `addTask()` returns is wrong whatever it measures.
- **`Allocator` is for large allocations only.** It takes a size on free,
  which makes some allocator algorithms much cheaper, and it is deliberately
  not shaped for fine-grained C++ allocation (`docs/AsyncRTRuntime.md`). Don't
  route small allocations through it.

## Tests

```bash
./bazelw test //AsyncRT/unittests/... //AsyncRT/test:test
```

One gtest target per file through `modular_cc_test`, flat under `unittests/`
and with no GPU targets; `test/` holds the lit tests for crash reporting. Run
a race under `--runs_per_test`.
