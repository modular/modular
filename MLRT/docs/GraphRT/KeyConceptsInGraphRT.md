# Key Concepts in GraphRT

The main idea of GraphRT is an "executor" which efficiently executes a
user-defined program expressed as a dataflow graph. Like MLIR, the framework
is domain-independent - the focus is making it easy to define and extend
domain-specific abstractions and provide the infrastructure for working in this
domain.

This document describes the fundamental concepts that you may encounter in
GraphRT.

## Non-blocking, asynchronous, and lock-free

As with [the AsyncRT library](../../../AsyncRT/docs/README.md), GraphRT is
designed to be completely non-blocking, and built on the principle of executing
asynchronous operations across an efficient thread pool (which aims to have a
1:1 mapping to hardware threads in the steady state). The implementation of
the MEF executor is fully thread safe, but uses lightweight atomic operations
instead of locks. As with AsyncRT, GraphRT is designed with full error
propagation support.

GraphRT is also compatible with "regions of synchronicity" for specialized
situations that don't benefit from asynchronous designs.

## Primitives: the ingredients of a GraphRT program

A GraphRT program is usually described in MLIR form. For example, here is an
example of a Fibonacci computation:

```mlir
mef.invokable @fib(%arg: i32) -> i32 {
  %one = grt.constant.si32 1
  %cond = grt.lessequal.si32 %arg, %one
  %res = grt.if %cond (%arg: i32) -> i32 {
    mef.output %arg : i32
  } else {
    %min_one = grt.constant.si32 -1
    %n_min_one = grt.add.si32 %arg, %min_one
    %n_min_two = grt.add.si32 %n_min_one, %min_one
    %fib_n_min_one = grt.invoke @fib(%n_min_one) : (i32) -> (i32)
    %fib_n_min_two = grt.invoke @fib(%n_min_two) : (i32) -> (i32)
    %sum = grt.add.si32 %fib_n_min_one, %fib_n_min_two
    mef.output %sum : i32
  }

  mef.output %res : i32
}
```

While the MEF format and MEF dialect has hard coded primitives for defining
invokables and returning from regions (`mef.invokable` and `mef.output`)
respectively, it does not otherwise hard code any other operations that do
actual computation.

This example is using the
["grt" dialect](../../include/MLRT/GraphRT/GRTDialect/), which is just one of
many MLIR dialects that provide GraphRT primitives. In the case of the `grt`
dialect, this includes simple integer and control flow operations like `grt.if`
and `grt.invoke`.

GraphRT benefits from MLIR's ability to compose and intermix dialects together.
In this case, it allows primitives from "grt" to intermix with the runtime APIs
for CPU APIs (for example Intel MKL, ruy, etc), framework specific CPU
interfaces, as well as the CPU side of drivers for diverse accelerators. It is
powerful to be able to drive these from a consistent low-level runtime, and
being "hackable" makes it possible to explore and iterate on new ideas quickly.

This works because GraphRT "doesn't know what it is doing. Each primitive is
typically defined as a C++ function and is registered with GraphRT at load time.
This keeps GraphRT domain independent, and the primitive libraries are what
carry dependencies on various bits of hardware or heavy software libraries, not
the GraphRT implementation itself.

Primitive implementation functions take and return some number of `AsyncValue`
objects, some number of MLIR attributes, and optionally regions and invokable
references. You can optionally use the [MLIR ODS
framework](https://mlir.llvm.org/docs/OpDefinitions/) to define these primitives
in a way understood by MLIR, which gives access to pretty syntax, verification
and makes it easier for compiler to generate the operations. For more details
on defining primitives, please see the [Defining Primitives in
GraphRT](DefiningPrimitives.md) guide.

## Inputs/outputs of primitives can be arbitrary C++ types

Typical machine learning interpreters hard code the input and output types for
operations as some framework-specific "Tensor" type. GraphRT is far more
general than that, allowing values of any arbitrary C++ type to be produced or
consumed by primitives in its interpreter. These types don't even need to be
copyable or movable. This is one of the reasons that GraphRT is multi-framework
capable, and it opens up other opportunities as well.

While the Fibonacci example above use integers for simplicity, you can pass
around tensors from multiple frameworks, optimized runtime representations like
`TensorShape` (much more efficient than a 1D tensor), hardware-specific runtime
objects (like CUDA streams) and can expose buffers and buffer allocation
(important for low-level memory optimizations) without having to switch to
another runtime.

The key thing to know here is that the MEF executor does not type checking
itself: it is fully dynamically typed, taking and passing around `AsyncValue`s
to primitives. Each `AsyncValue` carries around a lightweight C++ type
descriptor (queryable with `->isType<T>()`), which allows the primitives to
be completely dynamically typed if they want. For example, you don't want a
different "invoke" primitive for each signature you want to support.

On the other hand, many primitives do have static types they work with, for
example `grt.add.si32` takes and returns `int32_t` values. We do all type
checking at the MLIR level, by defining primitives in ODS and checking their
type constraints in the verifier. This shifts the checking burden from runtime
to compiler, but it also means that MEF is **not** and will never be a memory
safe "trusted compute" execution format like Java bytecode tried to be.

## Modular Executor Format ("MEF") File Format

To efficiently execute a kernel graph, the runtime does not directly process the
program’s MLIR representation. Instead the
[MLIRToMEF](../../include/MLRT/GraphRT/MLIRToMEF.h) library converts the program
to an in-memory format called Modular Executor Format, or **MEF**. The MEF
format is isomorphic to the MLIR representation, but is `mmap`-able, more
cache-friendly, and requires many fewer dependencies for execution compared to
executing MLIR's directly. This is good for code size and performance.
For more information on the file format, see the [Modular Executor Format (MEF)
File Format](ModularExecutorFormat.md) documentation.

## MEF executor: why not call it an "interpreter"?

The `MEFExecutor` is the library that reads a MEF file and executes invokables
on demand. It is designed to be low-dependency (and thus low code size) and
efficient.

A key design point of the MEF executor is that it executes primitives "strictly"
when their dataflow graph inputs are available, not according to their lexical
order in the invokable. Recall that each primitive takes and returns
asynchronous values and may have unpredictable latency. Because we cannot (in
general) know how long each operation in the program will take, the compiler
cannot produce a reasonable static schedule for an invokable.

Instead, GraphRT uses a dataflow executor that trigger execution of strict
primitives once all their inputs are available. For example, consider
execution of this invokable that computes `(a+a)+(b+b)`:

```mlir
mef.invokable @dataflow_example(%a: i32, %b: i32) -> i32 {
  %a2 = grt.add.si32 %a, %a
  %b2 = grt.add.si32 %b, %b
  %a2b2 = grt.add.si32 %a2, %b2
  mef.output %a2b2 : i32
}
```

In GraphRT, each input and output SSA value is implicitly held in an
`AsyncValue` - this means that the invokable is run with two `AsyncValues`, and
those values may not be "ready". Consider if the first operand (`%a`) is not
ready but the second operand (`%b`) is: in this case, the `grt.add.si32` that
produces `%a2` value cannot be run (its operands are not available yet).
However, the `grt.add.si32` that produces `%b2` value does have all its
operands available, so it is ready to go and it executes - ahead of the first
add.

Executing the primitives in dataflow order is a powerful feature that allows us
to work with accelerators and concurrent CPU algorithms efficiently, and makes
it much more likely that we'll continue discovering work to keep our thread
queues full. However, it is such a fundamental difference from most
interpreters (which generally execute things "top down" with a program counter)
that we remind ourselves of it by calling it an "Executor" instead of an
"Interpreter".

While almost everything is defined as a "strict" primitive, GraphRT does also
support "non-strict primitives" for power-user cases, see below for more
information.

## Chains for side effect ordering

One challenge with dataflow-based execution is that it only works correctly in
the case of side-effect free arithmetic operations. While this is the majority
case in many domains, we also want to support operations that can print, can
read or write mutable buffers, or need other synchronization constructs that we
haven't dreamed of yet. Consider the following pseudo-code program:

```mlir
mef.invokable @dataflow_example(%a: i32, %b: i32) {
  %a2 = grt.add.si32 %a, %a
  %b2 = grt.add.si32 %b, %b
  "print" %a2
  "print" %b2
  mef.output
}
```

With a normal interpreter, this would be no problem: the operations execute in
top-down lexical order, so the first print would be ordered before the second
printout. However, we have a dataflow executor, and as we discussed above, the
`%b2` value may be computed earlier than the `%a2` value, which could cause the
print-outs to come out of order.

The solution to this problem is to explicitly serialize side effects by turning
the implicit dependencies into explicitly ordered dataflow edges. We do this
with a `M::AsyncRT::Chain` type, which is threaded through the operations. This
is a legal MEF dialect program:

```mlir
mef.invokable @dataflow_example(%a: i32, %b: i32, %ch0: !mef.chain) -> !mef.chain {
  %a2 = grt.add.si32 %a, %a
  %b2 = grt.add.si32 %b, %b
  %ch1 = grt.print.si32 %a2, %ch0
  %ch2 = grt.print.si32 %b2, %ch1
  mef.output %ch2 : !mef.chain
}
```

In this invokable there is no dependence between the two add operations, so they
can run in any order (or, indeed, in parallel) but the prints have a data
dependence between them, so the first runs before the second one.

If you look at the [definition of the
`M::AsyncRT::Chain`](../../../AsyncRT/include/AsyncRT/Support/Chain.h) struct,
you'll see that it is empty. What is happening here though is that the
`AsyncValue` containing the chain is only marked as "ready" when the side
effect for the print is completed. This composes naturally with GraphRT's
notion of strict and non-strict primitives.

## Implicit error propagation

In addition to being a "[future or promise of a
value](https://en.wikipedia.org/wiki/Futures_and_promises)", an `AsyncValue` may
also be fulfilled with an `Error` value. This is important because many
asynchronous operations can fail, or depend transitively on other operations
that fail (for example a matrix multiplication may produce a shape error,
invalidating any computation that uses its results). One challenge with this is
that it is very easy to forget to write code "defensively" that checks for all
the possible error conditions.

The GraphRT executor directly supports error propagation for strict primitives:
if any input operand to a primitive is resolved to an error, the executor will
implicitly skip invoking the primitive and propagate the error to the result
value(s) of the primitive. This ensures that errors are correctly propagated,
and moves the burden of error propagation from primitives out to the executor
itself.

Cancelation is also supported by the executor: if the `Runtime` is canceled, the
executor will stop dispatching of new primitives when their operands become
available: instead it will implicitly propagate the cancel value to their
results, quickly quashing a computation.

## Non-strict primitives

"Strict" execution of primitives when their dataflow inputs become available is
the vastly most common behavior we want for asynchronous programs, but there are
other more advanced use cases (like control flow operators) that want to kick
off an operation before all the inputs are available: these are called
"non-strict" primitives.

Instances of "non-strict" primitives are handled differently than "strict"
primitives by the MEF executor: instead of triggering the primitive when *all*
operands becomes available, it triggers the primitive when *any* operands are
available.

As a concrete example, consider the following invocation:

```mlir
mef.invokable @dataflow_example(%a: i32, %b: i32) -> i32 {
  %a2 = grt.add.si32 %a, %a
  %b2 = "some long latency operation"(%b) : (i32) -> i32
  %a2b2 = grt.invoke @other_invokable(%a2, %b2) : (i32, i32) -> i32
  mef.output %a2b2 : i32
}
```

As we discussed earlier, it is possible that the `%a` and `%b` inputs are
available at different times, and it is entirely common for an "add" operation
to take longer to execute than "some long latency operation". In higher level
machine learning graphs, some operators take a lot less time to execute than
others - for example some might be accelerated and some might not.

On the other side of the call, the code embodied by `other_invokable` may
include arbitrarily complicated logic, and some of that logic may only depend
on one or the other argument to it. We'd really like to kick off that logic as
soon as we can, to make sure we keep the CPU thread pool and accelerators as
loaded up as we can.

For this reason, we don't want `grt.invoke` to be a strict primitive that waits
for both arguments to become available before dispatching the async call to
`@other_invokable`. Instead, we want to start the execution as soon as *either*
argument is available. As such, we mark `grt.invoke` as a non-strict primitive.

Non-strict primitives are generally useful for control flow operations (for
example `grt.if` can start execution of its body as soon as the condition
resolves (it doesn't need all inputs to be available) and `grt.repeat` can
start execution when the iteration count is known. It is also useful for some
primitives that do not use all their operands, for example, consider a [ternary
operator](https://en.wikipedia.org/wiki/Ternary_operation) like the `cond ? a :
b`: there is no reason to block on availability of `a` or `b`: as soon as the
condition is known, the `AsyncValue` for the right side can be forwarded on as
the result.

A primitive may opt-into non-strict behavior by adopting the `NonStrictOp` trait
in its MLIR ODS description. Doing so means it needs to be ready to handle
argument `AsyncValue`s that are not ready yet, so it requires more careful logic
when implementing the primitive. Please see the [Defining Primitives in
GraphRT](DefiningPrimitives.md) guide for more information.

## Lowering to MEF

Finally, MEF and the MEF dialect in MLIR is designed to be a low-level
execution-oriented format. While it is possible to do some optimizations at
this level, it is expected that machine learning frameworks will "lower" to this
level of representation using MLIR's fantastic support for doing this. As such,
the machine learning graph will pass through many levels of abstraction,
exposing lots of opportunities for optimization.
