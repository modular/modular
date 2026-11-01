# GraphCompiler — agent notes

The graph compiler turns MAX graphs into MEF artifacts for GraphRT. The
architecture it is held to lives in
[docs/ArchitectureSpecification.md](docs/ArchitectureSpecification.md) — 12
principles, 24 quality attributes, 8 constraints, and 8 ADRs. This file is the
short version: the rules that change what you write. Read the spec section
named beside a rule before you argue with it.

Cited symbols move. Find them by name, not by the line number in the spec's
permalinks, which are pinned to revision `ceaf9aea9ed8`.

## How to read the spec's status labels

Every numbered item carries one. They mean different things for a change:

- **Enforced today** — a contract the code keeps. Preserve it.
- **Existing exception (do not expand)** — the code departs from the stated
  requirement here. Scope fixes to the exception; it is not a precedent.
- **Migration target** — not built yet. Don't assume it exists, and don't turn
  an unrelated change into a migration.
- **Measurement required** — the guarantee needs evidence you haven't got yet.
  Don't claim it.

The labels record implementation status. They are not permission to weaken the
requirement. When you change the implementation, update the label and its
evidence in the same PR.

## Invariants — preserve these

- **Buffers do not alias.** Two different `mo.buffer` values never alias
  (C4). Fusion, memory planning, and elision all rely on it. Preserve it when
  you construct or rewrite buffer values.
- **MEF is the boundary with GraphRT** (C2), and there is **one MEF per
  top-level graph** (C6). Subgraphs and callables live inside that artifact.
- **MAX is the only IR producer** (C1). Internal IR tools and tests do not
  extend the public producer contract.
- **Device count is explicit in the graph** (C5). The compiler does no
  automatic device-count scaling or partitioning. Device placement arrives
  annotated (ADR-7).
- **Communication is explicit ops.** `mo.transfer` and the collectives reach
  downstream passes as operations, whatever inserted them. The compiler never
  pattern-matches communication out of other op sequences (C7, ADR-6).
- **Kernels are Mojo** (C3). This does not prohibit runtime primitives in C++.
- **A warm MEF cache hit skips compilation** (QA-S2). Keep the early-return
  path in `FrameworkFrontend.cpp` ahead of graph compilation.
- **Python source traces survive lowering and fusion** (QA-C1), via
  `AttachPythonStackTraces.cpp`. A transform that drops locations breaks the
  user-facing stack trace.
- **Custom ops work in eager and compiled modes** (QA-E1). Changing the
  custom-op interface means preserving both paths.

## Do not expand — exceptions that are not precedents

These exist. Each one looks like house style when you open the file, and
copying it is the most common way an agent damages this architecture.

- **Device gates inside compiler passes.**
  `MODialect/Patterns/Fusion.cpp` checks `deviceInfo.getArch() != "sm_100a"`
  in `MatmulAddPattern` and `MegaFFNNvfp4Pattern`. Passes, analyses, and
  pipeline structure are meant to be device-agnostic, with device-specific
  logic in kernels or the driver (principle 3, QA-P3, ADR-2). Do not add a
  third arch check.
- **Named composite fusion patterns.**
  `MODialect/Transforms/PatternFusion.cpp` recognizes named chains such as
  all-reduce + add + RMS-norm + quantize. Fusion eligibility is supposed to
  be a function of kernel *structure*, not op identity (C8, QA-E3,
  principle 6). New fusion work goes through structural MAP fusion. If you
  are adding a pattern, read the `add-gc-pattern` skill, and justify in the
  PR why it cannot be structural.
- **Builtin-specific lowering.** `Conversion/MOToMAP/MOToMAP.cpp` dispatches
  on op identity (`MO::SliceOp`, `BroadcastToOp`, `ReshapeOp`,
  `TransposeOp`). Custom and builtin ops are supposed to follow the same
  rules for every structural decision (principle 5).
- **Cross-pass analysis preservation.** `SimpleBufferReusePass` calls
  `markAnalysesPreserved<BufferLifetimeAnalysisPass>()` on an analysis its
  erasures partly invalidate, correct only because of what the *next* pass
  queries. Passes are supposed to be independent units with documented
  contracts (principle 7). Do not add another pass that depends on this, and
  do not reorder the passes around it without re-deriving why it is safe.
- **Execution policy baked into generated Mojo.** `MOGGDialect/MOGGOps.cpp`
  emits async task creation and worker-affinity selection
  (`task_id_for_device`, `desired_worker_id`, `$affinity_device_id`). These
  are executable decisions, not hints the runtime may override. The compiler
  is supposed to express what, not how, and leave execution strategy to the
  runtime (principles 1, 2, 9; QA-CR1, QA-CR4). Do not add new scheduling or
  placement decisions to generated code.
- **The quadratic allocation planner.** `DynamicMemoryPlanningPass` builds an
  `n * n` `canShare` matrix over allocations. Compile cost is supposed to
  scale with graph structure, near-linearly in layer count (principle 8,
  QA-S4). Don't add another all-pairs analysis.
- **Device graph capture spans layers.** Staging, partitioning, dependency
  computation, and stub generation run as compiler passes in
  `GraphCompiler.cpp`, while the rest of capture lives in the driver,
  `ExecutableModel`, and the Python pipeline layer. Capture is supposed to be
  a self-contained backend capability (QA-CR3). Do not widen the compiler's
  share of it.

## Don't assume these exist

- **Compiler-derived communication from sharding annotations** (ADR-8,
  QA-D5). Experimental MAX sharding supplies tensor layouts and resharding
  policy above the compiler, but graph authors still insert communication ops
  by hand. There is no sharding algebra, no device mesh in the IR, and no
  per-op propagation rules.
- **An explicit compilation-target manifest** (QA-B1). The target is encoded
  implicitly in the MEF and is not validated at load.
- **Standalone Mojo output** (QA-C3). Dumping generated Mojo works; a
  self-contained program reproducing a whole graph does not exist.
- **Documented fusion coverage** (QA-D4). There is no complete list of what
  fuses, so don't cite one.

## Don't claim what you haven't measured

A large share of the spec is labeled *measurement required*: compile time
(QA-S1, QA-S3, QA-S4), runtime overhead and parallel dispatch (QA-R1, QA-R2,
QA-R3), portability across device types (QA-P1, QA-P2), numerics (QA-B2), and
fusion performance (QA-D1, QA-D3).

A passing test is not the measurement. These requirements ask for profiles,
growth exponents, confidence intervals, or device traces. Don't write "meets
QA-R1" in a PR body on the strength of a green test, and don't flip a label to
*enforced today* without attaching the evidence.

## When a change needs the architecture to move

If a change can only work by relaxing an enforced invariant or by extending
one of the exceptions above, that is an architecture decision, not a detail to
settle in review. Say so in the PR, state which item it affects, and get the
decision made explicitly. The spec records those as ADRs.

## Two rules commonly read stricter than they are

- **A new backend may require targeted compiler changes** (principle 4,
  QA-P2). The failure signal is restructuring a pass or the pipeline for a
  backend, not the existence of a compiler diff. The line between targeted
  and architectural is undefined, so argue it in the PR rather than assuming
  any compiler change disqualifies a bring-up.
- **Compile time is mostly Mojo, not the graph compiler** (principle 8,
  QA-S4). Repeated layers are deduplicated and their kernels cached, so N
  identical layers cost well under N times one layer. An end-to-end compile
  time that does not separate graph compiler passes from kernel generation
  cannot tell you anything about a graph compiler change.
