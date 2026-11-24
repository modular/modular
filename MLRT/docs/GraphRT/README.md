# GraphRT Runtime Library

The "GraphRT" library provides a low-level execution framework for programs
expressed as asynchronous dataflow graphs. Unlike many other dataflow graph
execution frameworks, GraphRT does not hard code a set of operators or
specialize the types of values passed between operations to "tensor".

Instead, GraphRT aims to be the "MLIR of runtimes" - an openly extensible
compilation target that serves many domains. The GraphRT approach is to allow
clients to define their own set of operators (which it calls "primitives") that
are implemented in C++, and allows them to take and produce arbitrary C++
data values as edges of the dataflow graph.

Being fully MLIR native, GraphRT supports regions which allow natural control
flow constructs to be implemented as primitives, instead of being special cases.
Another advantage of being MLIR native is that it is an ideal compilation target
for domain specific compilers.

## Placement in the ecosystem

GraphRT is very different than a typical machine learning "operator graph"
interpreters. Those frameworks typically have an opinionated set of operators
for math, control flow, and often have an implicit "tensor" type that is passed
between operators. That level of representation is often (but not always)
target/hardware independent. While these abstractions can support "graph level
optimizations", they don't typically expose buffers, accelerator details, and
other minutiae that is required to get high performance from heterogenous
accelerators.

On the other hand, accelerated ML frameworks typically uses "lowering" (in a
compiler sense) to transform the operator graph into a lower-level
representation that is specific to the execution hardware, for example by fusing
kernels into macro kernels, utilizing proprietary accelerator APIs like CUDA,
and exposing buffer allocation and other details. At this level of abstraction,
the representation is typically "target specific" because it knows the set of
hardware being targeted.

This is the level of abstraction that GraphRT exists in: while it can represent
high level operator graphs, it is designed for extensibility so it can support
execution of arbitrary (even heterogenous) accelerated implementations. It
supports arbitrary data on edges of the graph because there is no agreement in
the ML ecosystem about the "best" format for a tensor, because we want to be
able to expose low-level details like buffers, DMAs, and interrupts. GraphRT
is implicitly asynchronous because accelerators are asynchronous, and
multi-threaded CPU implementations are often difficult or impossible to
statically schedule.

Typical accelerator implementations have a bespoke runtime specific to their
accelerator (and often, the organizational chart of the team that produced it)
which makes heterogenous acceleration extremely difficult. In contrast, GraphRT
embraces MLIR ideas of extensibility and dialect composition, allowing it to
support mixing of accelerators (and the drivers for those accelerators) into a
single program.

## GraphRT documentation

Here are some good entry point documents to understand the GraphRT library:

- GraphRT builds directly on the
  [AsyncRT library](../../../AsyncRT/docs/README.md), its
  [Runtime abstraction](../../../AsyncRT/docs/AsyncRTRuntime.md), and uses its
  [`AsyncValue` related types](../../../AsyncRT/docs/AsyncValue.md) pervasively.
- An overview of [Key Concepts in GraphRT](KeyConceptsInGraphRT.md).
- A guide to [defining and implementing GraphRT
  primitives](DefiningPrimitives.md).
- Documentation for the [MEF binary file format](ModularExecutorFormat.md).

## GraphRT code base

Beyond documentation, it can be helpful to browse the code. The major
components of the GraphRT module are:

1. a set of MLIR components for representing GraphRT programs, notably the
   [MEF Dialect](../../include/MLRT/GraphRT/MEFDialect) which defines
   `mef.invokable`.
1. a binary file format named the ["Modular Executor Format"
   (MEF)](ModularExecutorFormat.md) which is designed to be compact and
   efficiently executable by a low-dependency interpreter.
1. the [MEFExecutor library](../../include/MLRT/GraphRT/MEFExecutor/) implements
   the interpreter for the MEF file.
1. translators for converting [MLIR to MEF](../../lib/GraphRT/MLIRToMEF/) (a
   core part of the typical compilation flow), and from
   [MEF to MLIR](../../lib/GraphRT/MEFToMLIR/) (more of a
   debugging/introspection/testing aid).
1. a [set of example primitives](../../lib/GraphRT/Primitives/) implementing
   integer arithmetic, common control flow operations, and for writing tests.
   This library is fully generic and can be used without these, but they are
   nonetheless useful for writing tests, as examples, and even for some
   realistic higher level clients.
1. [tests for all the relevant functionality](../../test/GraphRT/), of course!
