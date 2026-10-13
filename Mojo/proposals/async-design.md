# Designing async in Mojo

**Status**: Draft.

Date: Oct 1, 2026

We're starting design work on async programming in Mojo. This post explains
what we want async to do, what we have already decided, and what is still open.

Tracking issue:
[modular/modular#7211](https://github.com/modular/modular/issues/7211)

## Where things stand today

Mojo already has an unfinished async implementation. `async def` compiles to
a stackless coroutine: the compiler turns the function into a state machine
whose frame holds the state that lives across each `await`. The standard
library has private `Coroutine` and `RaisingCoroutine` types and a minimal
internal task runtime. None of this is stable or exported for general use. The
work is to design the user-facing model and then build the compiler, library,
and runtime support behind it.

## Goals of async programming

Async programming lets you express a program sequentially but execute it
asynchronously. The intent of the feature is to reduce the complexity
necessary to hide latency. For example, if an operation is blocked on a busy
resource outside the current thread, such as an accelerator, a socket, or a
disk, the thread can do other work until the result is ready.

Without language support for async, the programmer must express such
operations as structs, explicitly defining the state machine and writing the
scheduler that drives it to completion (or cancellation). That approach costs
you:

- Direct-style control flow. Loops and branches that depend on async results
  have to be split into "build the graph" and "run the graph" phases.
- Static types. Dependencies become runtime IDs and dictionary lookups, so
  shape and type mismatches show up at runtime instead of compile time.
- Precise lifetimes. Intermediate values sit in runtime collections, so the
  compiler can't destroy them at their last use.
- Composition. Every async operation needs a wrapper struct, and helper
  functions have to expose scheduler internals instead of taking and returning
  values.

With async as a language feature, the goal is twofold.

First, we want to remove the readability and maintenance overhead that comes
with explicit asynchronous programming (listed above). Instead, asynchronous
code should resemble synchronous code, something like this:

```mojo
async def mlp(
    x: Tensor, weights: List[Tensor], ctx: DeviceContext
) raises -> Tensor:
    var h = x
    for w in weights:
        h = await ctx.matmul(h, w)
    return h
```

The compiler generates the task state, lifetimes, and suspension points.

Second, we want to expose the primitives users need for fine-grained control
over their coroutines: where they execute, where they are allocated, and how
they are scheduled.

## Two scheduling problems we want to serve

We have found it useful to sort async workloads by how they are scheduled.
Mojo's design should work well for both:

1. **Resource-directed (pull-based).** A DAG of tasks runs on a fixed set of
   resources, for example matmuls on two GPUs. The scheduler polls resources
   and dependencies and starts each task when its inputs are done and its
   resource is free. This is the common case for ML and HPC work on the host.
2. **Event-driven (push-based).** An open-ended stream of independent tasks,
   such as client requests in a server, waits on external events like socket
   readiness or timers. An event loop wakes the task waiting on each event and
   runs it on a thread pool. This is the common case for networking and
   services.

## Features that achieve these goals

- **`async def` and `await`** for writing async code in direct style, with
  full type checking and Mojo's normal ownership and lifetime rules.
- **A task abstraction that supports both scheduling models.** A task must be
  pollable for pull-based schedulers and wakeable for push-based ones.
- **A thread-safety model**, similar to Rust's `Send`/`Sync` or Swift's
  `Sendable`, so the compiler can check which values are allowed to cross
  threads inside tasks.
- **Pluggable executors.** The standard library provides a default runtime,
  but users can write their own schedulers, such as GPU-stream-aware or
  single-threaded ones, against a stable interface. We will release a
  dedicated forum for this design after the language feature stabilizes.
- **Error handling and cancellation** that fit the rest of Mojo (`raises`).
- **Low overhead**, so async is usable in performance-critical code, and
  ideally on constrained targets.

## Ways to implement these features

### Decisions we have committed to

- **Stackless coroutines.** Each coroutine is a compiler-generated state
  machine with a frame, not a separate call stack. The frames are small and
  their size is predictable, and they run on the normal thread stack, so they
  work cleanly with C and C++ ABIs. They are also the only practical option on
  SIMT and embedded targets, where giving every task its own stack is
  expensive or impossible. The cost is function coloring: an `async` function
  can only be awaited from another `async` context.
- **The compiler generates coroutines; the library schedules them.** The
  language provides the state-machine transformation and a small set of
  primitives. Executors, task types, and synchronization types belong in the
  library.
- **Coroutine becomes a trait.** If you want to hand-roll your own coroutine,
  you can, and it will be compatible with the Mojo standard library runtime.
- **Async without `malloc`.** Async should work on targets with no heap
  allocator.
- **Fallible coroutines.** Any coroutine can fail, and whoever awaits or polls
  it can retrieve the error. This works today, but the interface is not yet
  stable.
- **Cancellable.** All coroutines should be safe to cancel (they should clean
  up after themselves) but must be done so explicitly.
- **Cold start by default.** Today coroutines are cold-started by default, but
  a compiler optimization can transform cold starts into hot starts.

### Open questions

These are the questions we most want feedback on:

- **Async on the GPU.** Should `async def` be usable inside GPU kernels, or
  only on the host to coordinate device work?
- **Type erasure.** Storing different futures in one collection needs some
  form of existential or boxing, like `Box<dyn Future>` in Rust. How much
  does this depend on existentials landing in Mojo?
- **Function coloring.** Given stackless coroutines, what can we do to reduce
  the cost of coloring? For example, a blocking `wait()` bridge, or APIs that
  are generic over sync and async.

## How other languages do it

| Language   | Coroutine model            | Colored         | Start                           | Runtime                                                       |
|------------|----------------------------|-----------------|---------------------------------|---------------------------------------------------------------|
| Rust       | Stackless (state machine)  | Yes             | Cold (lazy futures)             | Not built in; poll + waker, third-party executors (Tokio)     |
| C++20      | Stackless                  | Yes             | Configurable via `promise_type` | Not built in; library-defined                                 |
| Swift      | Stackless async frames     | Yes             | Cold, tasks start on creation   | Built-in cooperative pool, structured concurrency, `Sendable` |
| Kotlin     | Stackless (CPS transform)  | Yes (`suspend`) | Configurable                    | Library (kotlinx.coroutines), structured concurrency          |
| Python     | Stackless                  | Yes             | Cold coroutines, hot tasks      | `asyncio` event loop in stdlib                                |
| JavaScript | Stackless                  | Yes             | Hot (eager promises)            | Single-threaded event loop                                    |
| Go         | Stackful (growable stacks) | No              | Hot (`go f()`)                  | Built-in M:N scheduler                                        |

Most systems languages chose stackless coroutines with colored functions and
left the executor to libraries. Go avoids coloring by giving every goroutine
its own growable stack. That is a trade-off we don't think works for GPUs and
embedded targets.

## Plan of execution

1. **Build it as a prototype first.** Commit a small working executor, built
   on today's coroutine implementation, to an experimental directory. This
   gives us something concrete to discuss and benchmark, and it shows where
   the current compiler support falls short.
2. **Move pieces into the standard library over time.** As compiler support
   improves and the open questions above get answered, move the prototype's
   dependencies (task types, executor interface, synchronization primitives)
   from the experimental directory into the standard library, one reviewed
   piece at a time.

We'll post design proposals for each open question in this thread as they
become ready. If you have use cases, especially ones that push on GPU,
no-allocator, or embedded requirements, please share them here.

## A final note on what exists today

Earlier, we said we want users to be able to write the following:

```mojo
async def mlp(
    x: Tensor, weights: List[Tensor], ctx: DeviceContext
) raises -> Tensor:
    var h = x
    for w in weights:
        h = await ctx.matmul(h, w)
    return h
```

However, we didn't say how a user calls `mlp`, or what resumes the coroutine
the compiler generates.

The compiler turns `mlp` into a coroutine. A library `Task` type connects that
coroutine to Mojo's async runtime, which resumes it when the awaited operation
completes. The library makes this connection by accessing the coroutine's
resume function through an embedded `co` operation
(``__mlir_op.`co.resume` ``) and binding it to the waiter callback in AsyncRT.
Any coroutine-related function or state that the compiler generates is
accessible from Mojo source code through these embedded operations. Today, the
coroutine frame exposes:

- **Result and error slots** (`co.set_byref_error_result`): the caller
  allocates memory for the result, and optionally for an error, then stores
  pointers to that memory in the coroutine frame. The coroutine writes its
  return value or raised error through those pointers when it completes.
- **Direct results** (`co.set_results` and `co.get_results`): the SSA values
  the async function returns directly, stored in the coroutine object. The
  library uses `co.get_results` to read whether a completed coroutine raised
  an error.
- **Resume function** (`co.resume`): a function pointer that resumes the
  suspended coroutine. This is what the runtime calls when the awaited
  operation completes.
- **Completion callback** (`co.get_callback_ptr`): a pointer to a closure
  that runs when the coroutine exits. The closure is two pointers wide (a
  function pointer and one data field). AsyncRT sets it to mark the task's
  completion token as available, and `TaskGroup` sets it to count down its
  outstanding tasks.
- **Lifetime** (`co.destroy`): frees any memory associated with the coroutine
  handle. Every handle returned from calling an async function must be
  destroyed exactly once.

Today, calling tasks looks something like this:

```mojo
def test_runtime_task() raises:
    print("== test_runtime_task")

    async def test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async def test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await create_task(test_asyncrt_add[1](a)) + await create_task(
            test_asyncrt_add[2](b)
        )

    var task = create_task(test_asyncrt_add_two_of_them(10, 20))
    print(task.wait())
```

Keep in mind that this shows what exists today, not our final design.
