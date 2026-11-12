# MLRT — agent notes

MLRT is the Modular low-level runtime: GraphRT, which executes the MEF
artifacts the graph compiler produces, and the driver, which owns devices,
streams, events and device memory. `docs/README.md` maps the libraries and
`docs/GraphRT/KeyConceptsInGraphRT.md` has the execution model. This file is
the short version: the rules that change what you write.

Invoke the `graph-execution-engineer` skill before you edit anything here.

Symbols move. Find them by name rather than by a line number quoted anywhere.

## Invariants — preserve these

- **GraphRT depends on AsyncRT and MLIR, never on Kernels or KGEN**
  (`docs/README.md`). A dependency edge the other way is the thing to stop,
  not to work around.
- **Execution is dataflow, not lexical.** Side-effect ordering travels on
  `M::AsyncRT::Chain` values, and the executor propagates an error by skipping
  any primitive whose operand is an error
  (`docs/GraphRT/KeyConceptsInGraphRT.md`). Do not add an ordering rule that
  depends on the order primitives appear in.
- **MEF is immutable, mmap-able and versioned**
  (`docs/GraphRT/ModularExecutorFormat.md`). The format trades encoder cost
  for decoder speed, so a change that makes decoding do more work is a change
  to the format's premise.
- **MEF is not a trusted compute format.** GraphRT is dynamically typed and
  looks primitives up by name; type checking happens in the MLIR verifiers,
  not at load. Do not add a validation story the format does not have, and do
  not rely on one.
- **The memory manager has one mutex, and the header says who holds it.**
  `MemoryManager.h` annotates each method "Caller holds `mu_`" or "Caller must
  NOT hold `mu_`". Preserve the annotation when you move code; it is the only
  record of the locking discipline.
- **Allocations are aligned to at least 64 bytes**, forced in the
  `MemoryManager` constructor whatever the caller asked for. The
  large-allocation alignment, 2 MiB on HIP, is best effort and not a
  guarantee.
- **Frees are stream-ordered.** A block freed on a stream stays on that
  stream's list until its recorded event fires, and coalescing merges only
  neighbours with the same owning list. That owner check is what stops one
  stream handing away memory another stream still has in flight.
- **Graph capture has no driver fallback**, and an event recorded into a
  capturing stream becomes a graph node that reports ready before the graph
  has run. Code that polls such an event is reading a lie.
- **A side-stream view's freed pages wait on the base stream too**, because
  the base stream reads the view's results after the region joins. Dropping
  that gate returns memory that is still being read.

There is no "do not expand" exception list for the memory manager, the way
`GraphCompiler/AGENTS.md` has one for the compiler. That is not permission:
the recovery ladder on the virtual-memory miss path is already four deep, and
a fifth rung needs a reproducer and an owner's agreement, not a config knob.

## The driver boundary

`include/MLRT/Driver/DeviceContext/DeviceContext.h` is the legacy C++ driver
and it is frozen while the HAL driver catches up. New driver features compose
in the C interface, `include/MLRT/Driver/CAPI/DriverCAPI.h`. Invoke the
`device-context-boundary` skill before adding a method, hook, enum or field to
either, and the `vendor-api-boundary` skill before any `cu*` or `hip*` call,
`dlopen`, or vendor crate outside `lib/Driver/`. "The driver is frozen" never
means "go around it".

## Multi-stream facts that live only in comments

Streams and events are driver-side. `lib/GraphRT/` makes no stream or event
call at all, and the stream pool is sized by the graph compiler, in
`GraphCompiler/lib/MEFModel/MEFModel.cpp`, which pre-creates three by default
so a host-gated side stream need not share one with compute. These four facts
are written down only in the code, so they are easy to break:

- **Cross-stream ordering travels on buffer guard chains, never the execute
  chain** (`GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp`).
- **A region's inputs are held until the join queues the base-stream wait.** A
  freed buffer goes back to its owner stream's free list knowing nothing about
  the side stream still reading it.
- **`mo.sequence` is single-level per device**, and nesting is rejected in
  `MODialect/MOOpsVerifiers.cpp`, because the stream override installed during
  MOGG to MGP lowering has one level.
- **ROCm graph capture reassigns queues by depth-first order**, giving a
  node's k-th dependent edge the node's queue plus k, which can move the
  forking stream's work onto another queue (`HIP/HIPDeviceContext.cpp`).

## Tests

```bash
./bazelw test //MLRT/unittests/GraphRT/... //MLRT/test/GraphRT:test
./bazelw test --config=remote-b200 //MLRT/unittests/Driver/...
./bazelw test //MLRT/unittests/Driver:MemoryManagerTest --test_output=errors
```

One gtest target per file through `modular_cc_test`. GPU targets carry
`gpu_constraints` and `tags = ["gpu"]`; memory-hungry ones set
`exec_properties = {"test.resources:gpu-memory": ...}`. The OOM tests are
generated one binary per environment condition, because the device manager is
a process-wide singleton.
