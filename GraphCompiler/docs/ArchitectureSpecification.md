# Graph Compiler Architecture Specification

## Purpose of this document

This document is intended to guide future graph compiler changes by answering
the question: "Does a given change fit into the existing architecture or require
a change to that architecture?" The answer will help us scope the work necessary
to make that change.

*This* version is the canonical one that should receive updates. The original
version of this document was in
[Notion](https://www.notion.so/modularai/Graph-compiler-architecture-specification-32f1044d37bb8029aef9d15543f7421c?source=copy_link).

## Requirement status

The compiler still has architecture-specific fusion rules and emits runtime
scheduling decisions. Other requirements, such as compile-time budgets, need
measurements before we can say they're met. Use these labels to distinguish
current contracts from work we still need to do.

- Enforced today: the implementation or architectural boundary supports this
  contract today. Preserve it when changing code. This label doesn't certify
  every test or deployment configuration.
- Existing exception (do not expand): current code departs from the stated
  requirement. Keep necessary fixes scoped to that exception; don't use it as
  a precedent for new architecture. Broader changes need an explicit design
  decision.
- Migration target: the full capability or contract isn't established yet.
  Move toward it when the task calls for that work. Don't assume it's already
  available or turn an unrelated change into a migration.
- Measurement required: implementation or tests alone don't establish the
  stated guarantee. Collect the specified measurements, coverage, or review
  evidence before claiming compliance. This label doesn't mean the requirement
  has failed.

Each numbered principle, quality attribute, constraint, and ADR has one label.
The accompanying note explains partial support and any remaining evidence.
For mixed requirements, a concrete implementation exception takes precedence
over missing measurements. An unfinished capability is a migration target.

These labels record implementation status, not permission to weaken the
requirements. Update the label and its evidence when the implementation changes.
The labels reflect a source review, with cited exceptions checked on
October 15, 2026, against revision
[`ceaf9aea9ed8`][review-revision].
Performance targets remain unverified by this source review.

A requirement can also be wrong, rather than merely unmet. Where review
establishes that a stated requirement overshoots what the team enforces, the
requirement itself is amended rather than labeled.

## Table of contents

- [Requirement status](#requirement-status)
- [1. Introduction and goals](#1-introduction-and-goals)
- [2. Guiding principles](#2-guiding-principles)
- [3. Quality attributes](#3-quality-attributes)
  - [3.1 Portability (heterogeneous hardware)](#31-portability-heterogeneous-hardware)
  - [3.2 Extensibility (framework and kernel integration)](#32-extensibility-framework-and-kernel-integration)
  - [3.3 Startup latency](#33-startup-latency)
  - [3.4 Runtime performance and hardware utilization](#34-runtime-performance-and-hardware-utilization)
  - [3.5 Developer productivity](#35-developer-productivity)
  - [3.6 Correctness and debuggability](#36-correctness-and-debuggability)
  - [3.7 Deployability and distribution](#37-deployability-and-distribution)
  - [3.8 Compiler-runtime contract](#38-compiler-runtime-contract)
- [4. Architectural constraints](#4-architectural-constraints)
  - [4.1 Scope and boundary constraints](#41-scope-and-boundary-constraints)
  - [4.2 Technical constraints](#42-technical-constraints)
  - [4.3 Integration constraints](#43-integration-constraints)
- [5. Design decisions](#5-design-decisions)

## 1. Introduction and goals

The graph compiler is a component of the MAX platform that transforms high-level
computation graphs into optimized executable artifacts for heterogeneous
hardware. Its primary role is enabling MAX to reach near-peak hardware
utilization for new models and platforms in a short period of time.

The usage environment is ahead-of-time (AOT) compilation with a large compile
time budget.

**Key goals (in priority order)**:

- **Portability**: produce correct, performant output across different hardware
  backends from the same graph.
- **Productivity**: reduce the need for hand-written kernels by automatically
  fusing composed primitives.
- **Performance**: minimize runtime overhead and maximize hardware utilization
  of compiled execution compared to eager mode.

**Non-goals**:

- JIT compilation for quick experimentation.
- An end-to-end optimizing compiler that produces complete model code and owns
  all decisions.

> [!NOTE]
> The MAX Graph API is currently the sole IR producer for the graph compiler.
> This assumption scopes the portability and deployment requirements: the
> input IR format is controlled, and the graph compiler does not need to accept
> arbitrary IR dialects.

## 2. Guiding principles

These principles are decision rails for the graph compiler team. When a
design choice is not obvious, check the proposal against these principles to
resolve it or make the trade-off explicit.

1. **The compiler expresses what, not how.** The graph API is declarative
   (dependencies, resource requirements, lifetimes), not prescriptive or
   imperative. The graph compiler will refine these requires based on the
   target hardware. But execution strategy is the runtime's job.

   Status: Existing exception (do not expand). The compiler still emits task
   launches, joins, and allocation ordering. See [QA-CR1](#qa-cr1).

2. **Every decision has exactly one owner.** For any decision category
   (memory, ordering, synchronization), one component is authoritative. If
   you need to read both compiler and runtime code to understand a single
   behavior, ownership is split.

   Status: Existing exception (do not expand). Execution policy spans
   generated code and runtime components. See QA-CR4.

3. **Device-specific logic does not enter the compiler.** Passes, analyses,
   and pipeline structure are device-agnostic. All device-specific logic
   lives in kernels or in the runtime/driver layer.

   Status: Existing exception (do not expand). Compiler fusion patterns still
   select NVIDIA `sm_100a` implementations. See [QA-P3](#qa-p3).

4. **A new backend must not require architectural compiler changes.** A new
   backend within an existing device class integrates through kernels,
   driver, and runtime. Targeted graph compiler changes during a bring-up are
   accepted. Restructuring passes or the pipeline to accommodate a backend is
   the failure signal.

   Status: Measurement required. What separates a targeted change from an
   architectural one is not defined, so the bound is unenforced rather than
   met. Check a same-class backend bring-up diff and its test results. Don't
   read the allowance for targeted changes as license for restructuring.

5. **Custom ops and builtin ops follow the same rules.** The compiler does
   not distinguish between custom and builtin ops for any structural
   decision (fusion eligibility, memory planning, optimization).

   Status: Existing exception (do not expand). Shared kernel interfaces
   coexist with builtin-specific lowering and fusion. See [QA-E3](#qa-e3).

6. **Resist special-casing; generalize or push up.** Before adding a
   pattern-specific pass, ask whether the optimization can be expressed as a
   general structural transformation, or whether the pattern can be exposed
   through the graph API or kernel contract for the author to mark
   explicitly.

   Status: Existing exception (do not expand). `PatternFusionPass` still
   recognizes named composite patterns. Use structural fusion for new work;
   see [C8](#c8).

7. **Passes are independent units with documented invariants.** Each pass
   has a defined input and output contract. Cross-pass dependencies through
   implicit shared state or undocumented ordering are forbidden.

   Status: Existing exception (do not expand). `SimpleBufferReusePass`
   preserves a partially invalid analysis based on what the next pass queries.
   See [the preservation assumption][analysis-preservation].

8. **Compilation cost scales with graph structure, not content.** Compile
   time grows no faster than structural complexity. Repeated structures
   (decoder layers) are deduplicated, and the kernels generated for them are
   cached, so N identical layers cost substantially less than N times one
   layer.

   Status: Existing exception (do not expand). Memory planning builds an
   allocation-pair matrix with quadratic space and work. See [the
   planner][memory-planning]. Most measured compilation time is Mojo kernel
   generation, not graph compiler passes, so an unattributed end-to-end
   number tests this principle poorly; see QA-S4.

9. **Do not bind what the runtime should decide later.** If a decision
   depends on information available only at runtime, or if binding it at
   compile time forces unnecessary recompilation, keep it dynamic.

   Status: Existing exception (do not expand). Execution policy and some KV
   cache parameters are fixed in generated code. See [QA-CR1](#qa-cr1) and
   ADR-4.

10. **The graph author states intent; the compiler derives mechanism.** The
    graph API captures intent (sharding, logical computation), not mechanism
    (communication ops, memory layout). The compiler derives mechanism from
    intent.

    Status: Migration target. Experimental MAX sharding provides layouts and
    resharding policy, but the full input-IR contract remains a target. See
    [QA-D5](#qa-d5).

11. **If fusion behavior is not predictable, it is not useful.** Fusion
    patterns are documented and testable. Adding or removing an op in a
    fusable sequence causes smooth performance changes, not cliffs.

    Status: Measurement required. Fusion tests establish transformations, not
    smooth performance changes. See [QA-D3](#qa-d3).

12. **Errors are synchronous, specific, and traceable.** Errors surface at
    compile time with a message that identifies the offending operation,
    explains the cause, and maps back to user source code.

    Status: Measurement required. Source traces and static validation exist.
    Check diagnostic coverage separately from runtime-dependent shape errors;
    see QA-C1 and QA-C2.

## 3. Quality attributes

This section defines testable quality-attribute requirements for the graph
compiler. Each requirement follows the stimulus-response-measure pattern to
ensure it is verifiable. Requirements are derived from stakeholder Q&A sessions.

### 3.1 Portability (heterogeneous hardware)

**QA-P1. Single graph, multiple device types.** The same graph compiles
and executes correctly on any supported device type without source-level
changes.

Status: Measurement required. Verify the same graph against reference outputs
on each supported device type. Source inspection alone doesn't establish
numerical portability.

- *Stimulus*: a graph targeting 1 NVIDIA GPU is recompiled for 1 AMD GPU.
- *Response*: compilation succeeds; output matches reference within numerical
  tolerance.
- *Measure*: functional correctness verified via golden-output tests on each
  supported device type.

> [!NOTE]
> A graph designed for 1 GPU is not expected to work efficiently on 8 GPUs,
> but it is expected to work on 1 NPU.

**QA-P2. No architectural graph compiler changes for same-class backends.**
Adding support for a new backend within an existing device class (for example,
AMD GPU after NVIDIA GPU) restructures no graph compiler pass and does not
change the shape of the pipeline. The bulk of integration work is confined to
the Driver level. Targeted graph compiler changes are expected: bug fixes that
correct latent assumptions the new backend exposes, and bounded additions
within an existing pass.

Status: Measurement required. This needs the backend bring-up diff and full
test results. Compiler architecture gates remain an obstacle; see
[QA-P3](#qa-p3). The boundary between a targeted change and an architectural
one is not defined, so a bring-up diff needs that judgment made case by case.

- *Stimulus*: a new backend within an existing device class is integrated.
- *Response*: the graph compiler passes its full test suite without
  architectural modification.
- *Measure*: the diff for backend bring-up restructures no graph compiler
  pass and does not change the pipeline's shape. Targeted compiler changes,
  if any, are enumerated and justified individually rather than counted
  against the requirement. Validated by the backend's bring-up tracking.

<a id="qa-p3"></a>**QA-P3. Kernel signature portability.** From the graph
compiler's perspective, kernels are portable by signature. The graph compiler
does not depend on kernel implementation details that are device-specific.

Status: Existing exception (do not expand). `MatmulAddPattern` and
`MegaFFNNvfp4Pattern` check `sm_100a` inside the compiler. See [the device
gates][device-gates].

- *Stimulus*: a kernel with a given signature is used in a graph compiled for
  two different device types.
- *Response*: the graph compiler generates correct dispatch for both devices
  using the same kernel signature.
- *Measure*: no device-specific kernel selection logic exists in graph compiler
  passes.

### 3.2 Extensibility (framework and kernel integration)

<a id="qa-e1"></a>**QA-E1. Custom ops in both compiled and eager modes.** Custom
kernel definitions compose with all execution modes (compiled and eager) through
a unified interface.

Status: Enforced today. The shared custom-op interface supports eager
execution and graph staging, with [representative tests][custom-op-tests].
Preserve both paths when changing the interface.

- *Stimulus*: a user registers a custom op using the documented API.
- *Response*: the custom op executes correctly in both eager mode and compiled
  mode.
- *Measure*: integration tests verify custom op execution in both modes for a
  representative set of ops.

**QA-E2. Dynamic shape support.** The graph compiler supports dynamic shapes
for all ops, including custom ops, without requiring shape specialization at
the graph level.

Status: Measurement required. Custom-op tests check reuse across concrete
shapes with one compilation. That evidence doesn't establish the stated
guarantee for every op; see [the reuse test][shape-reuse].

- *Stimulus*: a graph containing ops with symbolic (dynamic) shape dimensions
  is compiled.
- *Response*: the compiled artifact handles varying input shapes at runtime
  without recompilation.
- *Measure*: a compiled artifact executes correctly on inputs with different
  concrete shapes in a single test run.

<a id="qa-e3"></a>**QA-E3. Fusable kernel contract.** A kernel is eligible for
fusion if and only if it satisfies the fusable kernel contract -- specifically,
it uses the proper Mojo types for its input and output tensors. Fusion is
determined entirely by kernel structure (for example, elementwise access pattern
or prologue/epilogue interfaces), not by op identity or any per-op compiler
logic. Adding a new op whose kernel conforms to this contract requires zero
compiler changes; the existing fusion infrastructure handles it automatically.

Status: Existing exception (do not expand). MAP provides structural fusion,
but named patterns and builtin-specific lowering remain. See [pattern
fusion][pattern-fusion] and [MO-to-MAP lowering][mo-to-map].

- *Stimulus*: a kernel using the correct Mojo tensor types is placed adjacent
  to fusion-eligible ops.
- *Response*: the graph compiler fuses the kernel with its neighbors, with no
  op-specific compiler logic.
- *Measure*: integration tests verify that fusion occurs for conforming kernels
  and does not occur for non-conforming kernels. No compiler code is added or
  modified to fuse a new conforming op.

> [!NOTE]
> User-facing documentation for the fusable kernel contract is currently
> incomplete and needs to be created.

<a id="qa-e4"></a>**QA-E4. Extensibility contract stability.** The extensibility
contract (kernel API, MEF surface) evolves under a best-effort compatibility
policy, with a trajectory toward a formal deprecation policy with tooling
support.

Status: Measurement required. Best-effort compatibility is the policy. Verify
prior-version kernels, actionable errors, and release-note coverage before
claiming this requirement is met.

- *Stimulus*: a kernel authored against a previous API version is compiled with
  the current compiler.
- *Response*: the kernel either compiles successfully or produces a clear,
  actionable error pointing to the API change.
- *Measure*: API-breaking changes are documented in release notes with
  migration guidance.

### 3.3 Startup latency

**QA-S1. AOT cold-start compilation time.** Cold-start compilation time for
Llama 3.1 405B-class models does not regress beyond the current baseline (~3
minutes for the compilation portion of ~5 minutes total cold start on a CPU
comparable to those used on common cloud machines hosting multiple GPUs).

Status: Measurement required. Attach cold-start measurements and evidence of
the 30% regression alert for the stated model and hardware class.

- *Stimulus*: a Llama 3.1 405B model is compiled from scratch (no cache) on a
  supported device.
- *Response*: compilation completes within the time budget.
- *Measure*: compilation time is tracked in CI; regressions exceeding
  30% from baseline trigger alerts.

> [!NOTE]
> Orthogonal to compilation cache at the Mammoth level.
> Aimed at scenarios like new version deployment.
> Regressions are tolerable when in favor of runtime performance.

**QA-S2. Warm-start avoids recompilation.** When a valid MEF cache entry exists
for the current compilation target, serving startup skips compilation entirely.

Status: Enforced today. A valid MEF cache hit returns the artifact before
graph compilation. Preserve [the cache-hit path][mef-cache]; cache lookup and
key construction still do work.

- *Stimulus*: a serving container starts with a warm MEF cache.
- *Response*: startup proceeds directly to initialization; no compilation
  passes execute.
- *Measure*: startup time breakdown shows zero time in compilation when cache
  is warm.

**QA-S3. Compilation time does not block developer iteration.** Compilation
time for current models remains below 10 minutes (cold start) on a CPU
comparable to those used on common cloud machines hosting multiple GPUs to
avoid blocking developer iteration when a new system version requires
recompilation.

Status: Measurement required. Verify cold-start compile times and warning
coverage for the standard model set.

- *Stimulus*: a developer deploys a new version of the graph compiler and
  recompiles the standard model set.
- *Response*: all models in the standard set compile within 10 minutes each.
- *Measure*: CI tracks per-model compilation time; exceeding 10 minutes
  triggers a warning.

**QA-S4. Near-linear compile-time scaling with model depth.** Compilation
time for models with repeated layer structure scales near-linearly with
layer count. The target growth exponent is 1.0 (linear); the warning
threshold is 1.2 and the hard ceiling is 1.5.

Status: Measurement required. Measure the growth exponent across layer counts,
and attribute it: most compilation time is Mojo kernel generation (EmitMojo,
JitCompileKernels), which caches across repeated structure, so an exponent
measured end to end says little about the graph compiler's own passes. Report
the graph compiler's exponent separately from the Mojo one, or the number
cannot settle this requirement. The [quadratic allocation
planner][memory-planning] is a known risk, not proof that the end-to-end
ceiling is exceeded.

- *Stimulus*: a decoder-only transformer is compiled at three or more layer
  counts (for example, N, 2N, 4N) with the same per-layer structure and
  device count.
- *Response*: the empirical growth exponent alpha, estimated by least-squares
  regression on log(time) versus log(layers), does not exceed 1.5.
- *Measure*: alpha is computed from compile-time-vs-layer-count measurements
  in the benchmark suite and tracked in CI. Exceeding 1.2 triggers a warning;
  exceeding 1.5 triggers an alert.

> [!NOTE]
> An exponent of 1.0 is linear scaling; 1.5 means that doubling the layer
> count increases compilation time by approximately 2.8x; 2.0 (quadratic)
> would mean 4x. The warning threshold at 1.2 (~2.3x for doubled layers)
> flags early drift before it becomes a blocking problem.
>
> The two dominant cost centers that drive super-linear behavior are kernel
> selection (EmitMojo + JitCompileKernels, ~90% of compilation time) and
> fusion analysis (unconstrained). Kernel caching should make selection
> cost independent of layer count for repeated structures. Fusion is
> capped heuristically.

### 3.4 Runtime performance and hardware utilization

<a id="qa-r1"></a>**QA-R1. Runtime overhead under 1%.** Runtime overhead
introduced by the graph compiler and runtime infrastructure (dispatch,
scheduling, memory management), time of driver operations like loading kernels,
executing kernels, memory allocations and copies excluded, does not exceed 1% of
total execution time on supported hardware.

Status: Measurement required. Verify the overhead fraction with serving
profiles and the exclusions stated below.

- *Stimulus*: a compiled model executes under a representative workload.
- *Response*: profiling shows >=99% of wall-clock time is spent in kernel
  execution.
- *Measure*: profiler data from end-to-end serving benchmarks is analyzed for
  overhead fraction.

> [!NOTE]
> This requirement is in tension with [QA-CR1](#qa-cr1) (execution policy
> independence). Making the compiled artifact execution-model-agnostic
> introduces an abstraction layer that may add dispatch overhead. When the two
> conflict, [QA-R1](#qa-r1) takes priority for production workloads;
> [QA-CR1](#qa-cr1) ensures the abstraction does not architecturally prevent
> achieving the overhead target.

**QA-R2. Compiled execution outperforms eager.** Compiled execution of the
designated model set outperforms eager execution of the same models by a
statistically significant and practically meaningful margin.

Status: Measurement required. Define the model set and useful margin, then
compare eager and compiled execution with confidence intervals.

- *Stimulus*: a model from the designated set runs in both eager and compiled
  modes on the same hardware.
- *Response*: compiled mode achieves higher throughput or lower latency.
- *Measure*: A/B benchmarks with confidence intervals show compiled mode is
  faster.

> [!NOTE]
> Target margin is to be defined once eager mode is measurement-ready. If
> compiled execution is not faster than eager, the graph compiler's value
> proposition is undermined.

**QA-R3. Parallel dispatch of independent operations.** Independent operations
within a compiled graph execute in parallel; each operation dispatches as soon
as its inputs are ready.

Status: Measurement required. Async dispatch exists, but launch machinery
doesn't prove overlap. Verify independent subgraphs with device traces.

- *Stimulus*: a graph containing independent subgraphs is compiled and
  executed.
- *Response*: independent operations overlap in execution.
- *Measure*: trace analysis shows overlapping kernel launches for independent
  ops.

### 3.5 Developer productivity

This section distinguishes two tiers of developer productivity that the graph
compiler provides. The first tier is automatic structural fusion for
compositions whose access patterns the compiler can reason about mechanically
(elementwise chains, prologue/epilogue around structured kernels). The second
tier is custom kernel ergonomics: for complex patterns beyond structural
fusion's reach (attention variants, MoE gating, novel architectures), the
productivity contribution is not "the compiler fuses it for you" but "writing,
integrating, and composing a custom kernel is fast and predictable."

The boundary between the two tiers is determined by kernel structure
([QA-E3](#qa-e3), [C8](#c8)): if a kernel's access pattern conforms to the
fusable contract (elementwise, prologue/epilogue enabled), fusion is automatic.
If it does not, the author writes a custom kernel and the extensibility contract
([QA-E1](#qa-e1), [QA-E4](#qa-e4)) ensures it integrates cleanly.

> [!NOTE]
> Complex multi-op patterns such as attention-like subgraphs do not decompose
> into structurally fusable primitives today. Fusing them generically would
> require either fragile pattern matching (which violates
> [principle 6](#2-guiding-principles): resist special-casing) or an algebraic
> fusion framework that does not exist. Until that changes, the honest
> productivity story for these patterns is custom kernel ergonomics, not
> automatic fusion.

**QA-D1. Structural fusion for composable primitives.** The graph compiler fuses
compositions of primitives whose kernels conform to the fusable kernel contract
([QA-E3](#qa-e3)), eliminating the need for hand-written glue kernels for
structurally simple compositions.

Status: Existing exception (do not expand). Structural MAP fusion exists
alongside op-specific composite fusion. Preserve the structural path; the
speed-of-light target also needs a defined threshold and measurements.

- *Stimulus*: a model author expresses a chain of elementwise ops, or an
  elementwise op adjacent to a matmul with prologue/epilogue fusion enabled,
  as composed primitives instead of a single hand-written kernel.
- *Response*: the graph compiler fuses the chain into one kernel (or fuses
  the elementwise op into the matmul's prologue/epilogue), with no
  op-specific compiler logic.
- *Measure*: fused output achieves performance within a defined percentage
  of theoretical peak (speed of light) for the subgraph. The same fusion
  applies to custom ops whose kernels conform to the fusable contract.

> [!NOTE]
> The target percentage is not yet defined. The performance goal is table
> stakes; the primary success criterion is that the author does not need to
> write a custom kernel for compositions within the fusable contract.

**QA-D2. Custom kernel ergonomics for complex patterns.** For model portions
that exceed the fusable contract (multi-op patterns with non-trivial access
patterns, novel architectures), the graph compiler supports fast integration
of custom kernels that compose cleanly with the rest of the graph.

Status: Measurement required. Custom-kernel integration exists. The one-day
target and the full portability and fusion claims still need representative
validation.

- *Stimulus*: a researcher invents a novel subgraph pattern (for example, a
  new attention variant or gating mechanism) that cannot be expressed as a
  composition of structurally fusable primitives.
- *Response*: the researcher writes a custom kernel, registers it through the
  documented API ([QA-E1](#qa-e1)), and the kernel participates in the graph
  like any builtin op: it is dispatchable on all supported devices (via
  signature portability, [QA-P3](#qa-p3)), its boundaries are eligible for
  structural fusion (if its inputs/outputs conform), and it works in both eager
  and compiled modes.
- *Measure*: time from "kernel code written" to "integrated and running in a
  compiled graph" is under one day for a kernel of moderate complexity.
  No graph compiler changes are required to integrate the kernel.

> [!NOTE]
> The one-day target is aspirational and not yet measured. The key structural
> requirement is that custom kernel integration does not require compiler
> changes ([QA-E3](#qa-e3)) and that custom ops follow the same rules as
> builtins ([principle 5](#2-guiding-principles)).

<a id="qa-d3"></a>**QA-D3. No-surprises contract.** Fusion and optimization
behavior is predictable for documented fusion patterns: no optimization causes
an unexpected performance cliff relative to unfused execution.

Status: Measurement required. Verify performance changes for documented
patterns. Transformation tests alone don't establish performance monotonicity.

- *Stimulus*: a graph author adds or removes an op in a documented fusable
  sequence.
- *Response*: performance degrades or improves smoothly according to the
  hardware resource occupancy (memory bandwidth/compute throughput).
- *Measure*: regression tests for documented fusion patterns verify
  performance monotonicity.

> [!NOTE]
> Strict performance monotonicity for arbitrary compositions is not
> achievable in general (compiler optimizations can cause non-monotonic
> behavior due to register allocation, cache effects, and fusion boundary
> shifts). This contract is scoped to documented fusion patterns.

**QA-D4. Documented fusion coverage and authoring guidance.** The set of op
patterns the compiler can fuse is documented, measurable, and maintained.
The documentation (or a compiler report mode) also serves as guidance for
graph authors on how to structure code for effective fusion and on when to
write a custom kernel instead.

Status: Migration target. The extensibility guide describes tensor interfaces
and fused access. Complete pattern coverage, authoring guidance, and links to
coverage tests remain to be established.

- *Stimulus*: a developer or model author consults the fusion coverage
  documentation to determine whether a given op sequence fuses or whether a
  custom kernel is the appropriate path.
- *Response*: the documentation accurately reflects the current compiler's
  fusion capabilities and provides actionable guidance for structuring
  fusion-friendly graphs, including explicit guidance on the boundary where
  a custom kernel is the better choice.
- *Measure*: fusion coverage tests are linked to the documentation;
  discrepancies are caught in CI. The documentation covers all
  fusion-eligible patterns, or the compiler emits structured metadata about
  fusion decisions.

> [!NOTE]
> The [extensibility guide](ExtensibilityUserGuide.md) describes tensor
> interfaces and fused access. Complete fusion coverage, measurement, and
> authoring guidance remain to be established. Static documentation, compiler
> hints, report mode, and IDE integration are possible forms.

<a id="qa-d5"></a>**QA-D5. Distributed graph authoring with compiler-derived communication.**
A model author expresses a distributed computation using sharding annotations on
tensors. The graph compiler derives the necessary communication operations from
the annotations. When the derived communication is insufficient or suboptimal,
the author can override it by inserting explicit communication ops for
fine-grained control.

Status: Migration target. MAX now has [experimental tensor
layouts][sharding-types] and [resharding policy][resharding-policy]. The
complete annotation, propagation, ambiguity, and override contract described
here is not established by that implementation.

- *Stimulus*: a model author writes a graph with tensors annotated for
  distribution across N devices (for example, `shard(dim=0, axis="tp")`),
  relying on compiler-derived communication for the common case.
- *Response*: the graph compiler derives communication operations from the
  sharding annotations and produces correct output matching the
  single-device reference within numerical tolerance. Author-inserted
  explicit communication ops, when present, take precedence over
  compiler-derived ones.
- *Measure*: the default authored graph contains no explicit communication
  ops and is functionally correct. When explicit overrides are present,
  the compiler respects them and produces correct output.

> [!NOTE]
> See [C7](#c7) for the constraint on explicit communication ops.
>
> Experimental MAX sharding now provides tensor layouts, placement rules, and
> resharding policy above the compiler input. The full contract here remains
> a migration target. Direct graph authors still insert communication ops
> (`mo.transfer`, collective operations). The design direction is derived
> communication by default, with explicit ops for performance tuning. See
> [ADR-8][sharding-adr]
> for rationale and trade-offs.

### 3.6 Correctness and debuggability

**QA-C1. Python stack traces for graph ops.** The graph compiler emits metadata
that enables accurate Python stack traces mapping compiled graph ops back to
their source graph-construction call site.

Status: Enforced today. The compiler materializes Python source locations and
attaches traces to assertions. Preserve [trace propagation][source-traces]
through lowering and fusion.

- *Stimulus*: a compiled graph op produces an error at runtime.
- *Response*: the error includes a Python stack trace pointing to the line in
  the user's code where the op was constructed.
- *Measure*: error output includes source file, line number, and function name
  from the graph construction code.

**QA-C2. Synchronous error surfacing.** Errors from unsupported ops, dtype
mismatches, and shape violations are surfaced synchronously at the SDK boundary
with actionable diagnostics, never deferred to runtime as silent failures.

Status: Measurement required. Static kernel validation and error tests exist.
Verify the stated SDK diagnostics; errors that depend on runtime values cannot
all be diagnosed at compilation.

- *Stimulus*: a user constructs a graph with an unsupported op or invalid
  dtype.
- *Response*: the graph compiler raises an error at compile time with a message
  identifying the offending op and the nature of the error.
- *Measure*: integration tests verify that representative invalid graphs
  produce specific, actionable error messages.

**QA-C3. Standalone Mojo output** (optional, stretch goal). The graph compiler
can optionally emit standalone Mojo source that reproduces the compiled
computation for debugging purposes.

Status: Migration target. Dumping generated Mojo is available. A
self-contained program reproducing the whole graph remains an optional target.

- *Stimulus*: a developer requests standalone Mojo output for a compiled graph.
- *Response*: if supported, the output is a self-contained Mojo program
  reproducing the graph computation.
- *Measure*: the emitted Mojo program compiles and produces matching results.

> [!NOTE]
> End-to-end standalone Mojo output is not currently possible. The Mojo
> toolchain team owns debuggability; there is no explicit handoff contract yet.

### 3.7 Deployability and distribution

**QA-B1. Compilation target match avoids recompilation.** A compiled MEF
artifact encodes its compilation target explicitly (CPU architecture + GPU
architecture + MAX/Mojo version) and executes without recompilation when the
deployment target matches. When loaded on a mismatched target, the mismatch
is detected and reported.

Status: Migration target. MEF caching exists, but it doesn't establish the
explicit target manifest and load-time mismatch contract. Complete that
contract before treating artifacts as validated for deployment.

- *Stimulus*: a MEF artifact compiled for a specific target is deployed on a
  machine matching that target, and separately on a mismatched target.
- *Response*: on the matching target, the artifact loads and executes correctly
  without triggering recompilation. On the mismatched target, the mismatch is
  detected and reported.
- *Measure*: serving tests verify artifact reuse across identical target
  configurations. Runtime logs or errors indicate target mismatch when
  applicable.

> [!NOTE]
> The compilation target is currently encoded implicitly in the MEF file and is
> not documented or error-checked at runtime. The cache key definition (CPU
> arch, GPU arch, MAX/Mojo version) needs to be formalized, and explicit
> encoding with runtime validation is a future goal.

**QA-B2. Numerical behavior verification.** Compiled artifacts produce
numerical results that match reference outputs within defined tolerance,
verified through end-to-end logit verification tests.

Status: Measurement required. Logit-verification infrastructure exists. Check
passing results, model coverage, and tolerances for the designated set.

- *Stimulus*: a compiled model runs inference on a reference input set.
- *Response*: output logits match the reference within tolerance.
- *Measure*: logit verification tests pass in CI for all models in the
  designated set.

### 3.8 Compiler-runtime contract

This section addresses the quality of the boundary between the graph compiler's
output and the runtime's execution model. The compiler and runtime are
acknowledged as a continuum ([C2](#c2)), but the nature of the contract between
them -- what the compiler prescribes vs. what the runtime decides -- affects
portability to new execution models, GPU execution efficiency, and the ability
to support device-native features like graph capture.

<a id="qa-cr1"></a>**QA-CR1. Execution policy independence.** The compiled
artifact expresses computation dependencies and resource requirements without
prescribing a specific runtime execution strategy (CPU dataflow, GPU stream
submission, device graph replay, or future execution models).

Status: Existing exception (do not expand). Generated Mojo contains async task
creation and worker-affinity policy. See [async call
emission][async-emission]; these are executable decisions, not overridable
hints.

- *Stimulus*: a new runtime execution backend is implemented (for example, a
  GPU stream-based backend after the existing CPU dataflow backend).
- *Response*: the new backend executes existing compiled artifacts correctly,
  using its native execution model for dispatch, ordering, and
  synchronization -- without recompilation and without changes to the graph
  compiler.
- *Measure*: the diff for the new backend is confined to the runtime; no graph
  compiler passes or MEF output format changes are required. Validated by
  executing the regression test suite on the new backend with artifacts
  compiled by the unmodified compiler.

> [!NOTE]
> Today the compiler produces MGP, an imperative execution script with
> baked-in async dispatch policy (`isAsync` attribute), chain-based ordering,
> and CPU-side memory lifecycle operations. This prescribes a CPU dataflow
> execution model. A stream-based GPU backend cannot use this artifact without
> either (a) interpreting and discarding the CPU-specific decisions, or (b)
> requiring compiler changes. The quality attribute asks for an artifact format
> where those decisions are either absent or expressed as overridable hints.

**QA-CR2. GPU dispatch path overhead.** For workloads where the majority of
computation is GPU-resident, the runtime dispatch path from compiled artifact
to device kernel launch does not introduce per-kernel CPU overhead beyond a
defined budget.

Status: Measurement required. Measure CPU dispatch time on representative
hardware. The dispatch implementation alone doesn't establish the provisional
5-microsecond budget.

- *Stimulus*: a compiled model with N sequential GPU kernel launches is
  executed.
- *Response*: the per-kernel CPU-side dispatch time (excluding the kernel
  execution itself) does not exceed the target budget.
- *Measure*: CPU-side dispatch time per GPU kernel, measured as wall-clock
  time between successive `cuLaunchKernel` (or equivalent) calls minus kernel
  execution time. Target: under 5 microseconds per kernel on representative
  hardware.

> [!NOTE]
> The current dispatch path for each GPU kernel includes: MEF dataflow
> evaluation, AsyncValue operand readiness check, NUMA-aware WorkQueue worker
> selection, thread dispatch, DeviceContext bind/unbind, Mojo kernel
> invocation, and AsyncValue result marking. This machinery was designed for
> CPU-side scheduling and duplicates work that GPU stream ordering already
> provides. The quality attribute targets the dispatch path after the
> architectural mismatch is addressed, not incremental tuning of the existing
> path.
>
> The 5-microsecond target is provisional and should be calibrated against the
> kernel launch latency of the device API itself (approximately 3-5
> microseconds for `cuLaunchKernel` on current hardware). The goal is that
> runtime overhead is a small constant on top of the device API cost, not a
> multiple of it.

**QA-CR3. Device graph capture as a self-contained backend capability.**
Device graph capture (for example, CUDA graphs) is implementable as a
self-contained capability of a single runtime component, without requiring
coordination across multiple system layers.

Status: Existing exception (do not expand). Capture still spans compiler
passes, model execution, Python orchestration, and the driver. The [compiler
pipeline][compiler-pipeline] stages device graphs and generates recording
stubs.

- *Stimulus*: device graph capture support is added or modified for a
  supported device type.
- *Response*: all changes are confined to a single runtime component (the
  device-specific execution backend). No changes are required in the graph
  compiler, the MEF format, the model execution orchestration layer, or the
  Python pipeline layer.
- *Measure*: the diff for adding or modifying graph capture support touches
  exactly one component. No cross-layer workarounds (for example, suppressing
  CPU-side memory allocation during capture, or filtering out non-capturable
  operations) are present in the implementation.

> [!NOTE]
> Today, graph capture logic is dispersed across three layers:
> `CUDADeviceStream::beginCapture/endCapture` (Driver),
> `ExecutableModel::capture/replay` (MEFModel), and
> `PipelineModel.execute_with_capture` (Python pipelines). Capture is
> difficult because the runtime interleaves CPU-side work (memory allocation
> primitives, chain resolution, async dispatch) with GPU stream operations. A
> clean capture requires a "GPU-stream-pure" execution path, which the current
> architecture does not provide. This quality attribute measures whether the
> architecture supports graph capture as a natural backend mode rather than a
> cross-cutting concern.

**QA-CR4. Single-owner execution policy decisions.** Each category of runtime
execution policy (memory allocation strategy, dispatch ordering,
synchronization mechanism, device context management) has a single
architectural owner -- either the compiler or the runtime -- with no split
ownership or cross-layer negotiation.

Status: Existing exception (do not expand). The compiler chooses task
boundaries and allocation dependencies while runtime components execute them.
Define the policy boundary before claiming single ownership; see
[QA-CR1](#qa-cr1) and QA-CR3.

- *Stimulus*: a developer traces the implementation of a specific execution
  policy decision (for example, "when is buffer X allocated?" or "what
  determines the execution order of kernels K1 and K2?").
- *Response*: the answer is localized to a single component. The decision is
  either fully determined by the compiler (visible in the compiled artifact)
  or fully determined by the runtime (implemented in one runtime component),
  not split across both.
- *Measure*: for each of the defined policy categories (memory allocation
  timing, buffer reuse strategy, execution ordering, async dispatch policy,
  device context management, synchronization), an architectural review
  confirms single ownership. No policy category requires reading both
  compiler pass code and runtime dispatch code to understand the decision.

## 4. Architectural constraints

This section captures constraints on the graph compiler's design that limit
architectural freedom. These constraints are non-negotiable unless explicitly
renegotiated with stakeholders.

### 4.1 Scope and boundary constraints

- **C1. MAX as sole IR producer.** The graph compiler accepts input graph
  exclusively from MAX. No external IR producers (for example, ONNX importers,
  TorchScript converters) are in scope.

  Status: Enforced today. Keep MAX as the supported producer boundary.
  Internal IR tools and tests don't extend the public producer contract.

- <a id="c2"></a>**C2. Graph compiler to GraphRT boundary.** The graph compiler
  produces MEF artifacts consumed by GraphRT. The runtime starts at GraphRT; the
  graph compiler and runtime are viewed as a continuum, but MEF is the contract
  boundary.

  Status: Enforced today. Compilation produces MEF for GraphRT. The boundary
  exists even though execution-policy independence remains an exception under
  [QA-CR1](#qa-cr1).

### 4.2 Technical constraints

- **C3. Kernel language = Mojo.** Graph execution is supplied by a set of
  user-written and generated Mojo kernels.

  Status: Enforced today. Kernel generation and custom-kernel integration use
  Mojo. This doesn't prohibit runtime primitives implemented in C++.

- **C4. Buffer non-aliasing assumption.** The compiler assumes that two
  different `mo.buffer` values do not alias. All optimization passes
  (fusion, memory planning, elision) rely on this invariant.

  Status: Enforced today. This is a compiler precondition, not a claim that
  every caller's buffers are checked for aliasing. Preserve it when
  constructing or rewriting buffer values.

- **C5. Graph encodes device count explicitly.** The input graph encodes the
  number of target devices explicitly. The graph compiler does not perform
  automatic device-count scaling or partitioning.

  Status: Enforced today. Device references and graph structure specify the
  device set. Changing device count requires an appropriate graph.

- **C6. One MEF per graph.** The graph compiler produces one MEF file per
  graph. A model may define multiple graphs (for example, separate prefill and
  decode graphs), each compiled into its own MEF.

  Status: Enforced today. Keep one artifact per top-level graph. Internal
  subgraphs and callables can live inside that artifact.

### 4.3 Integration constraints

- <a id="c7"></a>**C7. Communication primitives as ops.** Communication in the
  compiled IR is expressed as explicit ops (`mo.transfer`, collective
  operations). These ops may be either compiler-derived from sharding
  annotations (see [QA-D5](#qa-d5)) or author-inserted for fine-grained control.
  The graph compiler dispatches them uniformly regardless of origin. The graph
  compiler does not pattern-match communication primitives from other op
  sequences.

  Status: Enforced today. Communication reaches downstream compiler passes as
  explicit operations. Compiler-derived communication under [QA-D5](#qa-d5)
  remains a separate migration target.

  > [!NOTE]
  > Communication remains explicit at the compiler boundary. Experimental MAX
  > sharding can express distribution above that boundary; the full contract in
  > [ADR-8](#adr-8-explicit-sharding-annotations-with-compiler-derived-communication)
  > remains a migration target. Both paths must preserve explicit communication
  > ops for downstream passes.

- <a id="c8"></a>**C8. Fusion is a function of kernel structure.** See
  [QA-E3](#qa-e3) for the testable requirement. Fusion eligibility is determined
  by kernel structure, not by op identity; the graph compiler does not have
  op-identity-based fusion rules.

  Status: Existing exception (do not expand). Named composite patterns still
  exist in [PatternFusionPass][pattern-fusion]. They aren't precedents for
  adding new op-identity fusion rules.

  > [!NOTE]
  > Some ad-hoc kernel fusion patterns remain in the compiler. Migrate them
  > toward the structure-based contract in [QA-E3](#qa-e3); don't expand them
  > without an explicit architecture decision.

## 5. Design decisions

This section records important, expensive, or risky architecture decisions
for the graph compiler in ADR (architecture decision record) format.

### ADR-1: AOT compilation as primary execution mode

Status: Existing exception (do not expand). AOT remains the primary mode, but
eager custom-op execution can use compiled bindings. Preserve the primary-mode
decision without assuming eager execution never invokes compilation; see [the
binding tests][shape-reuse].

- **Context**: the graph compiler must balance compilation cost against runtime
  performance. Users have different latency requirements for interactive
  development vs. production serving.
- **Decision**: the graph compiler operates in AOT mode with a large compile
  time budget. JIT compilation for quick experimentation is out of scope.
  Eager mode uses a separate interpreter path.
- **Alternatives considered**:
  - JIT compilation with tiered optimization: adds complexity and startup
    unpredictability.
  - Unified JIT/AOT compiler: requires meeting tight latency targets for
    interactive use while achieving peak performance for serving.
- **Consequences**:
  - (+) Longer compilation is acceptable if it yields better runtime
    performance.
  - (+) Simplifies the compiler's optimization strategy (can afford expensive
    analyses).
  - (-) Cold-start time is non-trivial (~3 minutes for 405B); requires
    compilation artifacts caching for production.
  - (-) Developer iteration gated by recompilation when the compiler changes.

### ADR-2: Kernel portability at signature level

Status: Existing exception (do not expand). Kernel interfaces coexist with
architecture-specific compiler fusion gates. See [QA-P3](#qa-p3).

- **Context**: the graph compiler needs to handle kernels across multiple
  device types. Full kernel implementation portability is impractical because
  kernels make device-specific decisions internally.
- **Decision**: from the graph compiler's perspective, kernels are portable by
  signature. Internally, kernels perform compile-time and runtime
  device-specific logic, but the graph compiler dispatches based on the
  signature alone.
- **Alternatives considered**:
  - Full implementation portability: would require a single kernel codebase
    for all devices; prohibitively constraining.
  - Per-device kernel registries in the graph compiler: adds device-specific
    logic to the compiler.
- **Consequences**:
  - (+) The graph compiler stays largely device-agnostic; a new backend is
    mostly kernel, driver, and runtime work, with any compiler changes
    targeted rather than structural (principle 4, QA-P2).
  - (+) Kernel authors have full freedom to specialize internally.
  - (-) The graph compiler cannot reason about kernel performance
    characteristics per device.

### ADR-3: Device graph APIs over megakernels for runtime overhead

Status: Enforced today. Device graph capture and replay are implemented. This
preference doesn't establish either the overhead budget or the
single-component boundary; see [QA-R1](#qa-r1) and QA-CR3.

- **Context**: runtime overhead from kernel launch latency can be reduced
  either by writing larger megakernels (fewer launches) or by using device
  graph APIs (for example, CUDA graphs) that amortize launch cost.
- **Decision**: lean toward device graph APIs (for example, CUDA graphs) to
  lower runtime overhead instead of writing larger kernels.
- **Alternatives considered**:
  - Megakernel approach: reduces launches but increases kernel complexity and
    reduces composability.
  - Kernel batching in the runtime: requires runtime-level scheduling
    sophistication.
- **Consequences**:
  - (+) Keeps kernels modular and composable.
  - (+) Leverages vendor-provided infrastructure for launch amortization.
  - (-) Requires integration with vendor-specific graph capture APIs.
  - (-) Not all workloads benefit equally from graph capture.

### ADR-4: KV cache as runtime concern, not compiler concern

Status: Existing exception (do not expand). KV cache dtype is passed as a
custom-op parameter. Enumerate which settings can change without recompilation
instead of assuming every configuration is dynamic; see [KV kernel
parameters][kv-parameters].

- **Context**: KV cache management (allocation, sizing, paging, eviction) is a
  critical component of LLM serving performance. It could be handled by the
  compiler (specializing artifacts per configuration) or by the runtime
  (keeping parameters dynamic).
- **Decision**: the graph compiler does not specialize artifacts by KV cache
  configuration. All user-configurable KV cache parameters are dynamic at
  runtime, handled by the KV cache custom op.
- **Alternatives considered**:
  - Compiler-specialized KV cache artifacts: could enable more aggressive
    memory planning, but forces recompilation on configuration changes.
  - Hybrid approach: specialize for common configurations, fall back to
    dynamic for others.
- **Consequences**:
  - (+) A single compiled artifact serves all KV cache configurations.
  - (+) No recompilation on configuration changes at serving startup.
  - (-) Potentially foregoes optimization opportunities that require
    static KV cache knowledge.

### ADR-5: Best-effort extensibility contract

Status: Enforced today. Best-effort compatibility remains the policy. Formal
deprecation tooling is a future goal; QA-E4 separately requires evidence of
compatibility and migration guidance.

- **Context**: the graph compiler's extensibility interfaces (kernel API, MEF
  format, fusion contract) are used by internal and external kernel authors.
  Strict versioning requires significant infrastructure investment.
- **Decision**: the extensibility contract operates under a best-effort
  compatibility policy. The trajectory is toward a formal deprecation policy
  with tooling support, but this is not in place yet.
- **Alternatives considered**:
  - Strict semantic versioning from day one: high infrastructure cost, slows
    compiler evolution.
  - No compatibility guarantees: unacceptable for external users and partners.
- **Consequences**:
  - (+) Compiler team can evolve interfaces quickly.
  - (+) Pragmatic for the current stage of maturity.
  - (-) External users may encounter breakage without clear migration paths.
  - (-) Must be resolved before broad external adoption.

### ADR-6: Communication primitives as first-class ops

Status: Enforced today. Explicit communication ops remain the downstream
contract. Read the manual-authoring description with ADR-8 and the
experimental MAX sharding work noted in [QA-D5](#qa-d5).

- **Context**: distributed workloads require communication primitives
  (all-reduce, scatter, gather). These could be pattern-matched from
  higher-level graph structure or expressed as explicit ops.
- **Decision**: communication primitives are expressed as explicit ops in the
  input graph (`mo.transfer`, collective operations implemented as builtin or
  custom ops). The graph compiler does not pattern-match for communication.
- **Alternatives considered**:
  - Pattern-matching for communication: would allow the compiler to
    automatically insert communication, but is fragile and hard to debug.
  - Compiler-generated communication plans: requires the compiler to
    understand network topology.
- **Consequences**:
  - (+) Communication decisions are explicit and visible in the graph.
  - (+) Graph authors and serving infrastructure have full control over
    communication strategy.
  - (-) Graph authors must explicitly insert communication ops.
  - (-) The compiler cannot automatically optimize communication placement.

> [!NOTE]
> Industry experience (JAX explicit sharding, PyTorch DTensor, GSPMD/Shardy)
> shows that explicit sharding annotations on tensor types can derive
> communication deterministically while preserving author control. See
> [ADR-8](#adr-8-explicit-sharding-annotations-with-compiler-derived-communication)
> for the planned evolution.

### ADR-7: Graph-level device annotation

Status: Enforced today. Device placement is explicit in the compiler input.
Higher-level APIs can derive that input without giving the compiler automatic
device-count scaling.

- **Context**: the graph compiler needs to know which devices ops target. This
  information could be inferred, defaulted, or explicitly annotated.
- **Decision**: devices are explicitly annotated on the graph by the graph
  author. The graph compiler targets homogeneous device sets. The graph encodes
  the number of devices explicitly; a graph for 1 GPU does not automatically
  generalize to 8 GPUs.
- **Alternatives considered**:
  - Automatic device placement: requires cost models and topology awareness
    in the compiler.
  - Default device with override: simpler, but hides placement decisions.
- **Consequences**:
  - (+) Device placement decisions are explicit and reproducible.
  - (+) Simplifies the compiler by avoiding automatic placement logic.
  - (-) Graph authors must reason about device placement.
  - (-) Separate graphs are needed for different device counts.

### ADR-8: Explicit sharding annotations with compiler-derived communication

Status: Migration target. Experimental MAX sharding supplies part of the
machinery. The input-IR representation, propagation coverage, ambiguity rules,
and lowering ownership still need the contract described here.

- **Context**: distributed model authoring currently requires graph authors
  to manually insert communication ops (`mo.transfer`, all-reduce, etc.) and
  reason about data movement across devices. This does not scale well to complex
  parallelism strategies (expert, pipeline, mixed). The graph API has been
  identified as unsuitable for sharding expressed as `split`+`concat`
  sequences.

  Industry experience with three levels of sharding abstraction -- auto
  (GSPMD/Shardy), explicit ("sharding in types"), and manual (`shard_map`,
  current MAX) -- shows that each has distinct failure modes:

  - **Auto**: the compiler decides placement from hints. Production
    experience shows opacity ("compiler tickling"), regression risk when
    propagation algorithms change (GSPMD-to-Shardy migration caused OOMs
    and performance regressions), and debugging difficulty. Conflicts with
    the stated non-goal of being "an end-to-end optimizing compiler that
    owns all decisions."
  - **Manual** (status quo): the author inserts all communication ops.
    Maximizes control but does not scale; graph authoring effort grows with
    parallelism complexity.
  - **Explicit**: sharding is part of the tensor type. Each op has a
    deterministic sharding propagation rule. Ambiguous cases are
    compile-time errors requiring author annotation. The compiler derives
    communication from the sharding specifications. Preserves author
    control while eliminating manual communication insertion.

- **Decision**: adopt explicit sharding annotations on tensor types in the input
  IR. Sharding specifies how each tensor dimension maps to a device mesh axis
  (shard, replicate). The graph compiler derives communication ops from the
  annotations during a lowering stage before the existing pipeline. Ambiguous
  sharding (where propagation rules cannot determine a unique output sharding)
  is a compile-time error, requiring the author to annotate explicitly. Authors
  can also insert explicit communication ops to override compiler-derived
  communication for fine-grained performance control (see [QA-D5](#qa-d5) and
  [C7](#c7)). After lowering, communication appears as explicit ops in the IR
  regardless of origin, preserving the invariants of downstream passes.
- **Alternatives considered**:
  - Manual communication only (status quo,
    [ADR-6](#adr-6-communication-primitives-as-first-class-ops)): does not scale
    to complex parallelism; graph authoring effort for distributed models is
    high and error-prone.
  - Auto sharding (GSPMD-style): compiler decides placement from sparse
    hints; production experience shows opacity and regression risk.
  - Fully automatic sharding with manual escape hatch (`shard_map`-style):
    mixes abstraction levels; the automatic component introduces the same
    opacity as GSPMD. The adopted approach avoids this by keeping sharding
    explicit in types, with manual overrides only for communication ops.
- **Consequences**:
  - (+) Graph authors express intent (how tensors are distributed) rather
    than mechanism (which communication ops to insert).
  - (+) Sharding is deterministic and queryable at compile time, supporting
    [QA-D3](#qa-d3) (no-surprises contract).
  - (+) AOT compilation eliminates the eager-mode dispatch overhead that
    plagues DTensor in PyTorch.
  - (+) Aligns with industry direction (JAX explicit sharding, MLIR
    mesh/shard dialect).
  - (+) Explicit communication ops as an override escape hatch mitigate
    performance problems without abandoning the sharding abstraction.
  - (-) Requires designing a sharding algebra: at minimum shard, replicate,
    and partial/unreduced placements. Backward-pass correctness is subtle
    (JAX and DTensor experience shows that `sum` on sharded dimensions and
    broadcast between sharded and replicated tensors require careful
    treatment to avoid silent numerical errors).
  - (-) Every op in the MO dialect needs a sharding propagation rule.
    Coverage gaps require a fallback (for example, explicit communication
    for ops without rules). JAX's experience shows that `reshape`, `scatter`,
    and `gather` require special-case logic.
  - (-) Amends [ADR-6](#adr-6-communication-primitives-as-first-class-ops):
    communication ops are still present in the IR after lowering, but are
    compiler-generated rather than author-written.
  - (-) Requires defining a device mesh abstraction in the IR.

> [!NOTE]
> Open design questions:
>
> - The minimal sharding algebra (shard, replicate, partial) and whether
>   unreduced/reduced states are needed for training workloads.
> - How sharding annotations interact with the existing fusion pipeline
>   ([QA-E3](#qa-e3)) and the fusable kernel contract.
> - The fallback mechanism for ops without sharding propagation rules.
> - Whether the sharding lowering stage belongs in the graph compiler or in
>   MAX (as a pre-processing step before IR emission).
> - Sharding annotation may have significant runtime overhead in eager mode.
>
> Requires alignment with MAX on input types.

[device-gates]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp#L4425
[pattern-fusion]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/MODialect/Transforms/PatternFusion.cpp#L49
[analysis-preservation]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp#L101
[memory-planning]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp#L172
[async-emission]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp#L4558
[compiler-pipeline]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/GraphCompiler.cpp#L417
[mef-cache]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L431
[source-traces]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/MGPDialect/Transforms/AttachPythonStackTraces.cpp#L44
[custom-op-tests]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/max/tests/integration/tensor/test_custom_op_def.py#L249
[shape-reuse]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/max/tests/integration/tensor/test_custom_op_def.py#L415
[sharding-types]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/max/python/max/experimental/sharding/types.py
[resharding-policy]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/max/python/max/experimental/sharding/_auto_reshard.py
[kv-parameters]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/max/python/max/nn/kernels.py#L710
[mo-to-map]: https://github.com/modularml/modular/blob/ceaf9aea9ed803119df09686e567a9edf3cb9f05/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp#L744

[review-revision]: https://github.com/modularml/modular/commit/ceaf9aea9ed803119df09686e567a9edf3cb9f05
[sharding-adr]: #adr-8-explicit-sharding-annotations-with-compiler-derived-communication
