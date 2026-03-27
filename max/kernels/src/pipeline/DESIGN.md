# Pipeline scheduling framework: design overview

A reviewer's guide to the compile-time software pipeline scheduler in
`max/kernels/src/pipeline/`. This document is honest about what exists, where
it came from, what works well, and what doesn't.

## What this is

A compile-time software pipeline scheduler for GPU matmul kernels, implemented
entirely in Mojo. It replaces ~800 lines of handwritten, manually-tuned
schedule tables with a framework that derives schedules from a dependency graph
and a hardware cost model.

The framework runs entirely at Mojo compile time via `comptime`. The output is
a flat sequence of `ScheduleEntry` values that the kernel unrolls into
straight-line code. Zero runtime overhead.

Today it drives two production AMD matmul schedules: a single-buffer default
and a double-buffer ping-pong. Both run on MI355X (CDNA3).

## Origin and provenance

This framework grew bottom-up from the AMD ping-pong matmul kernel. The
original kernel had 800+ lines of handwritten schedule tables — manually
ordered operations with manually computed wait counts. Every time the MMA
grid, buffer depth, or data type changed, the schedule tables had to be
rewritten from scratch. FP8 support required a completely separate schedule
because the K-dimension tiling changes the MMA count per block.

The framework extracted the scheduling logic into reusable components. The
design drew on well-known compiler infrastructure:

- **Dependency graph and list scheduling** from LLVM's `MachineScheduler`
  (specifically the `SUnit`/`SDep`/`ScheduleDAG` layering and the principle
  that the graph never schedules itself)
- **Algorithm/schedule separation** from Halide (declare what to compute,
  let the framework derive how)
- **Modulo scheduling theory** from Rau 1994 and GAG96 (ASAP/ALAP priority,
  resource-constrained scheduling, dependency graph structure)

None of this is novel scheduling theory. The contribution is applying these
ideas at the kernel-operation level in a way that runs at Mojo compile time
and produces schedules that match or beat hand-tuned code.

**Important scope note:** The current scheduler optimizes within-iteration
makespan (single iteration time span), not the initiation interval (II)
that modulo scheduling minimizes. Inter-iteration overlap is handled by the
fixed prologue/epilogue structure, not derived from II. The buffer depth is
a configuration parameter (currently always 2), not derived from the
schedule as modulo scheduling theory prescribes. MII computation and C2/C4
validation are not yet implemented. These are future work items — the
framework is currently a pragmatic code generator for AMD ping-pong style
kernels, not a general-purpose modulo scheduler.

## Why operation-level scheduling

The ping-pong matmul has ~24 operations per loop iteration: 8 global loads
(DRAM to LDS), 8 fragment loads (LDS to registers), and 8 MMAs. These
operations are grouped into 8 MMA-centered blocks with double-buffered LDS
stages and warp-group staggering.

This structure is invisible to LLVM's instruction scheduler:

1. **Lost abstraction.** By codegen time, "load tile A stage 0" is a sequence
   of `buffer_load_dwordx4` instructions. The warp-group cooperation pattern
   (WG0 runs one phase ahead of WG1) has no representation in LLVM IR.

2. **Hardware counter granularity.** AMD CDNA tracks async operations via
   `vmcnt` and `lgkmcnt` counters at the operation level, not the instruction
   level. One `buffer_load` that writes 256 bytes to LDS is one `vmcnt` tick.
   Scheduling at this granularity is natural.

3. **Debuggability.** When a wait count is wrong, the symptom is silent data
   corruption or a 10x performance cliff. The kernel author needs to see and
   reason about the schedule at the tile level. An opaque compiler pass
   doesn't help.

## What works well

### The trait-based API is genuinely concise

A complete schedule definition for the 24-op ping-pong kernel is ~150 lines
(`pingpong_schedule.mojo`). The `PipelineSchedule` trait has 2 required
methods:

```mojo
trait PipelineSchedule:
    def config(self) -> PipelineConfig:      # pipeline structure
        ...
    def build_body(self) -> List[OpDesc]:     # what ops exist
        ...
```

Three optional methods have defaults (edge derivation, tuning knobs, kernel
transform hook). The framework derives everything else: dependency edges,
execution ordering, wait counts, prologue, epilogue.

The ping-pong schedule's `build_body()` is three lines:

```mojo
def build_body(self) -> List[OpDesc]:
    var logical = self.declare_ops()              # pure algorithm
    var annotated = annotate_ops(logical, costs)  # attach hw costs
    return double_buffer_reorder(annotated, cfg)  # execution layout
```

This is a real improvement over the 800-line manual tables.

### The CSP scheduler finds minimum-makespan orderings

The exhaustive backtracking scheduler (`optimal_schedule()`) explores all
valid orderings and returns the minimum within-iteration makespan
permutation. For 24 ops it's ~200 leaves — instant at compile time. Early
termination when `makespan == lower_bound` proves no better ordering exists.

Note: "optimal" here means minimum single-iteration makespan, not minimum
initiation interval. The scheduler does not optimize for inter-iteration
overlap — that is handled by the fixed prologue/epilogue structure.

This is brute-force, not clever. It works because n=24 is small. For larger
operation counts it would need pruning or a smarter algorithm. But for the
matmul kernels we care about, exhaustive search is the right tool: it's
simple, correct, and provides a hard guarantee.

### Compile-time verification catches real bugs

`verify_schedule()` runs 6 structural checks via `constrained[]` (Mojo's
`static_assert`). These caught real bugs during development:

- Cross-iteration LDS races where FP8's `num_k_mmas=1` didn't provide enough
  MMA latency between a prefetch write and the next block's fragment read
- Global load redistribution placing prefetch loads in blocks that would
  create stage conflicts under warp staggering
- Missing barrier annotations that would have caused silent data corruption

The checks are:

1. Total op count matches expected
2. Phase partition (prologue + kernel + epilogue = total)
3. Each MMA block terminated by exactly one MMA
4. Stage consistency within blocks
5. Global load distribution within capacity limits
6. Cross-iteration LDS safety under warp stagger

### Performance matches hand-tuned code

BF16 8K×8K×8K: ~1312 TFLOPS (MI355X). FP8 8K×8K×8K: ~2608 TFLOPS. These
match the performance of the original handwritten schedules. The FP8 number
beats AMD's vendor BLAS by ~14%.

## What doesn't work well / known weaknesses

### The layering is LLVM-inspired but not LLVM-quality

The mapping to LLVM components is aspirational. Real LLVM has:

- `HazardRecognizer` that checks legality *during* scheduling (preventing
  illegal states). We only verify *after* (post-hoc). An invalid intermediate
  schedule can be constructed; we catch it at the end.
- `ScheduleDAGMutation` as a real extensibility mechanism with stable APIs.
  Our `EdgeRule` table is a thin pattern-match list, not a proper mutation
  framework.
- Multiple scheduling strategies that share a common interface
  (`MachineSchedStrategy`). Our strategies are hardcoded function calls
  (`greedy_schedule` vs `optimal_schedule`), selected by an enum. Not
  properly pluggable.

### `program_builder.mojo` is still too large (1273 lines)

This file does too many things: block construction, global load
redistribution, wait/drain derivation, all verification checks, schedule
transforms, and the main `build_kernel_program()` entry point. It should be
at least 3 files (construction, derivation, transforms). It's the residue
of the original monolith and hasn't been fully decomposed.

### The config surface area is large and under-documented

`PipelineConfig` has 14 fields. `ScheduleConfig` adds another ~10 tuning
knobs. `TargetProfile` bundles both with a cost model. The valid ranges and
interactions between these knobs are not documented, and some invalid
combinations produce silent wrong results rather than compile errors.
Several checks that should be `constrained[]` (compile-time errors) are still
`debug_assert` (runtime-only in debug builds).

### Only one target exists

The framework is designed to be target-independent (hardware model injected
via `TargetCostModel` + `PipelineConfig`), but only AMD CDNA3 (MI355X) is
implemented. It's unclear whether the abstraction generalizes to NVIDIA
(which uses `cp.async` barriers and `mbarrier` instead of `vmcnt`/`lgkmcnt`)
or to non-matmul kernels (attention, convolution). The abstraction hasn't
been tested against a second target, so it might be shaped too specifically
around CDNA3's memory model.

### No negative tests

There are 28 tests, all positive (valid schedules compile correctly). There
are no tests for invalid inputs: cyclic dependency graphs, infeasible CSP
instances, out-of-range config values, or schedules that should fail
verification. This means the error paths are untested.

### The phase derivation recipes are brittle

`phase_derivation.mojo` (871 lines) uses `PhaseAction`/`PhaseStep` recipes
to generate prologues and epilogues. These recipes are declarative pattern
matches over the op body — emit all global loads, then barrier, then fragment
loads, and so on. The recipe approach is clean in principle but the concrete
recipes are tightly coupled to the ping-pong kernel's structure. A
substantially different kernel (different number of buffer stages, different
MMA grid layout) would likely need new recipes rather than being able to
compose from existing ones.

### The DSL has rough edges

`PipelineBody` (the builder DSL) uses `load()`, `fan[]()`, `grid[]()` methods
that are convenient for the ping-pong case but somewhat ad-hoc. There's no
formal grammar or composition rules. The builder is more of a convenience
wrapper than a proper DSL.

## Architectural principles that do hold

Despite the weaknesses, these LLVM-inspired invariants are genuinely
maintained:

1. **The graph never schedules itself.** `LoopBody` is a pure data structure.
   `greedy_schedule()` and `optimal_schedule()` are external functions that
   consume it and return a permutation. No scheduling logic lives in the
   graph.

2. **Flat, value-typed data structures.** `OpDesc` is 9 `Int` fields. No
   pointers, no heap allocation, no indirection. This is essential for Mojo
   `comptime` evaluation where the interpreter manipulates these values
   directly. It's also simpler than LLVM's `SUnit` with its pointer-based
   predecessor/successor lists.

3. **Single source of truth.** The `PipelineProgram` is built once and all
   phases (prologue, kernel, epilogue, wait counts) are derived from it.
   There is no separate hand-maintained prologue that can drift from the
   kernel.

4. **Algorithm/schedule separation is real.** `declare_ops()` returns 24
   logical ops with no hardware costs. `annotate_ops()` attaches costs from
   the target model. Changing the cost model (different MMA latency, different
   LDS bandwidth) changes the schedule without touching the algorithm
   declaration. This separation was validated when FP8 support required
   different MMA counts but the same algorithm.

## Package structure

```text
pipeline/
├── types.mojo             # OpDesc, DepEdge, ResourceKind, OpRole (644 lines)
├── dependency_graph.mojo  # OpNode, LoopBody (196 lines)
├── schedulers.mojo        # greedy/optimal schedulers (439 lines)
├── pipeline_dsl.mojo      # ScheduleEntry, Pipe builder DSL (250 lines)
├── config.mojo            # PipelineConfig, ScheduleConfig (361 lines)
├── program.mojo           # MMABlockSpec, PipelineProgram (278 lines)
├── program_builder.mojo   # Block construction + verification (1273 lines) ← too big
├── phase_derivation.mojo  # Prologue/epilogue recipes (871 lines)
└── compiler.mojo          # PipelineSchedule trait, compile_schedule() (220 lines)
```

Total: ~4500 lines across 9 files, plus ~300 lines for the ping-pong
schedule consumer (`pingpong_schedule.mojo`).

Import DAG (acyclic, leaves at top):

```text
types
  └─ dependency_graph
       └─ schedulers
       └─ pipeline_dsl
            └─ config
                 └─ program
                      └─ phase_derivation
                      └─ program_builder
                           └─ compiler
```

## Comparison with prior art (honest version)

| Dimension         | LLVM `MachinePipeliner`                    | Halide                   | This framework                                              |
|-------------------|--------------------------------------------|--------------------------|-------------------------------------------------------------|
| Abstraction level | Machine instructions                       | Tensor expressions       | Kernel operations (~24 per iter)                            |
| Maturity          | Decades, many targets                      | Decade+, many backends   | Months, one target                                          |
| Scheduling        | Swing modulo scheduler (minimizes II)      | Auto-scheduler or manual | Brute-force CSP, within-iteration makespan (works for n≤24) |
| Verification      | During scheduling (hazard recognizer)      | Bounds inference         | Post-hoc only (weaker)                                      |
| Extensibility     | `ScheduleDAGMutation`, multiple strategies | `Generator`/`Pipeline`   | Trait + enum (limited)                                      |
| User-facing API   | None (compiler internal)                   | `compute_at`/`store_at`  | `PipelineSchedule` (2 required methods)                     |
| Test coverage     | Extensive (decades of regression tests)    | Extensive                | 28 positive tests, 0 negative                               |

The honest assessment: this framework applies well-known scheduling ideas at a
novel (for us) abstraction level, with a clean compile-time execution model
that exploits Mojo's `comptime` capability. The trait-based API, CSP
scheduler, and compile-time verification are genuinely useful. But it has been
validated on exactly one kernel shape (ping-pong matmul) on one target
(MI355X), and several quality-of-life improvements are needed before it's a
robust, general-purpose tool.

## What a reviewer should focus on

1. **Is the abstraction level right?** 24 operations per iteration, grouped
   into MMA-centered blocks. Too coarse? Too fine? Does this generalize
   beyond matmul?

2. **Is `PipelineSchedule` the right trait boundary?** 2 required + 3
   optional methods. Does this capture the right kernel-vs-framework split?

3. **`PipelineConfig` (14 fields) + `ScheduleConfig` (~10 fields):** Is
   this configuration surface manageable? Should some fields be derived
   rather than specified?

4. **`program_builder.mojo` (1273 lines):** This needs to be split. What's
   the right decomposition?

5. **Post-hoc verification vs online hazard checking:** Should we move to
   checking legality during scheduling (like LLVM's `HazardRecognizer`)
   rather than after?
