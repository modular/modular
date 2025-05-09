## Abstract

In this document, I propose some changes to how stackless coroutines are
modelled in Mojo. In particular, I propose that we:

- Replace `async` with a `yields T` syntax, that allows us to abstract over
  whether a function/trait/library contains suspension points. A function that
  is not a coroutine (i.e. does not suspend) is equivalent to a function
  declared as `yields Never`. This feature is extremely powerful. For example,
  it allows us to write higher-order functions that don't care whether their
  function argument is a coroutine. Also, it allows us to abstract over whether
  an I/O library uses non-blocking I/O.
- Remove the requirement that the `await` keyword be used to wait for a
  coroutine. This turns out to be a necessary consequence of abstracting over
  suspension.

In this model, whether a function is a coroutine is relegated to an **effect**
appearing at the end of a function's signature, very similar to `raises`. Like
`raises`, the effect is viral by default, so we still have "colored functions".
Despite this, the design that I present resolves the vast majority of the
complaints that people have about function coloring. In particular, it prevents
the need to duplicate functions/traits/libraries into sync versions and async
versions.

All of this is achieved without performance tradeoffs. In fact, readers
interested in maximizing the performance of Mojo's coroutines may be interested
in reading the section towards the end of this document wherein I propose that
Mojo coroutines be non-allocating. This not a novel idea—Rust coroutines work
this way—but it may nevertheless be interesting to Modular, since coroutines in
today's Mojo are heap-allocated by default.

This proposal was inspired by discussions with @owenhilyard and @lattner. Prior
art is discussed in Appendix B.

## Motivation

Mojo requires a concurrency model that maximizes performance. We need to support
the concurrent execution of millions of tasks that are waiting on I/O or other
shared resources, while minimizing memory consumption and compute. The canonical
use case is a high-throughput server.

To achieve this level of performance, Mojo's concurrency model needs to be built
upon stackless coroutines. For various reasons, the performance of this model is
unbeatable. For starters, the memory requirements of a stackless coroutine can
be calculated at compile time, which allows the coroutine's execution state to
be stored in a small, fixed-size memory block. Furthermore, Rust has
demonstrated—in contrast to C++—that coroutines don't need to allocate
additional memory during their execution. Altogether, this means that stackless
coroutines can be invoked, suspended, and resumed with very low overhead, and
memory consumption is minimized. This is exactly what Mojo needs.

Stackless coroutines are normally exposed to programmers through the `async` and
`await` keywords. Declaring a function to be `async` declares it to be a
coroutine, and `await` "waits for" a coroutine to finish. Fundamentally, a
coroutine can only be awaited by another coroutine, unless you're willing to
block an OS thread while you wait. This is what gives rise to the widely known
"function coloring" problem, wherein the `async` and `await` keywords are viral
throughout codebases.

There are several distinct issues that come up in discussions about function
coloring. The first issue is that it can be annoying to write `async` and
`await` all over a codebase, especially in the case where you need to add I/O to
a function that was previously "pure".

It's worth assessing the `async` and `await` keywords separately. For example,
Kotlin has shown that writing `await` at every call site is
[unnecessary](https://kotlinlang.org/docs/composing-suspending-functions.html),
so it's fair to criticize the burden of needing to write `await` every time you
invoke a coroutine. It's syntactically noisy, it disrupts function chaining, and
it bloats the size of diffs whenever a function is changed from sync to async,
or vice versa. The main utility of `await` is to make suspension points stand
out in the source code. This is analogous to how Swift uses the `try` keyword to
make exception-driven control flow stand out.

For the most part, the `async` keyword (or Kotlin's `suspend`) subsumes the
utility of `await`. Like `await`, it allows programmers and compilers to
identify whether a function call can suspend, by observing its signature.
Combined with lifetime analysis, this enables reasoning about what data will
needs to be retained while a coroutine is suspended. In turn, this allows the
programmer to reason about the memory consumption of suspended coroutines, and
how to reduce it. The `async` keyword also allows the programmer to check
whether they are holding any locks across a suspension point, which can
sometimes be a major performance bottleneck. Finally, if Mojo is to offer
allocation-free coroutines (like Rust), the compiler needs to prevent a
coroutine from calling itself recursively, except in cases where the programmer
has explicitly allocated memory for each recursive call. For the compiler to
check this, it needs to know whether a recursive function is `async`.

Another major issue that arises in "two-color" languages is the inability to
abstract over whether a function is `async`. The simplest examples are the
standard `map` and `fold` functions from functional programming. `map` is a
higher-order function that takes a data structure and a function pointer, and
applies it to every element of the data structure. If `map` were a method on
Mojo's `List` type, its signature would be as follows:
`fn map[f: fn (Int) -> Int](self) -> Self:`

The problem is that we haven't allowed `f` to be a coroutine. One way to resolve
this would be to add a second overload:
`async fn map[f: async fn (Int) -> Int](self) -> Self:`

Unfortunately, this requires the implementation of `map` to be duplicated, which
would be a maintenance nightmare. Ideally, we want the ability to define `map`
in such a way that it becomes a coroutine whenever it is passed a coroutine as a
parameter. A similar problem occurs with traits: many languages that have
traits/protocols offer sync and async versions of key traits. For example, Rust
offers both `Iterator` and `AsyncIterator`. And at a larger scale: many
languages duplicate **entire libraries**, to offer a sync version and async
version. Altogether, it's clear that Mojo desperately needs a way to abstract
over the "asyncness" of functions, traits, and libraries. **The purpose of this
document is to present a means of abstracting over "asyncness" using Mojo's
parameter system.**

In summary:

- To maximize the performance of Mojo's concurrency model, coroutines must be
  explicitly declared as `async`, or something equivalent to that.
- To avoid a situation where functions/traits/libraries are split into sync and
  async versions, Mojo needs a means of abstracting over "asyncness".

## My proposal

### Introduce the syntax `fn foo() yields T`

I propose we introduce a keyword `yields`, which works very similarly to
`raises`. A function that suspends must be annotated as `yields`, as in "yields
control". For example:

`fn read_file(path: String) yields -> String:`

This syntax would replace the use of `async fn` for declaring Mojo-native
coroutines, but we can keep `async fn` around for Python integration. The
primary reason for introducing a new syntax is that we want `yields` to be able
to take a type:

`fn read_file(path: String) yields Blah -> String:`

This syntax has several uses. Firstly, it allows us to declare the types of the
values yielded by **generators**:

`fn prime_numbers() yields Int64:`

Generators and coroutines can make use of the same compiler transform, where a
seemingly sequential function definition is transformed into a state machine
that `yields` periodically. That's why we're discussing generators here.

In Python, the type that a generator yields can be declared by writing the
generator-function's return type as `-> Generator[int, None, None]`. However,
this also exposes the concrete type of the generator, and forces the generator's
state to be heap-allocated. For performance reasons, this is not ideal for Mojo.
In comparison, the syntax `yields Int64` reveals nothing about the generator's
implementation—it only describes an aspect of its interface.

The focus of this proposal is the use of coroutines for asynchronous I/O, so I
will say no more about generators. For the I/O use case, the purpose of allowing
`yields` to take a type is that we can interpret `yields Never`—where `Never` is
an uninstantiatable type—as meaning that **a function is NOT a coroutine.** For
example, the following signature describes a function that reads a file without
yielding control of the OS thread:

`fn read_file(path: String) yields Never -> String:`

This is equivalent to omitting `yields` entirely:

`fn read_file(path: String) -> String:`

With this syntax, we have the ability to write functions that are **generic**
over whether an argument is a coroutine:

`fn higher_order[T: AnyType](f: fn() yields T) yields T:`

This is a really neat trick. Chris
[proposed](https://github.com/modular/mojo/pull/3946#issuecomment-2601176112)
doing this for `raises`; I've merely adapted the trick to `yields`. We can even
write functions that are generic over `yields` and `raises` simultaneously:

```plaintext
fn higher_order[
    T1: AnyType, T2: AnyType
](f: fn() yields T1 raises T2) yields T1 raises T2:
```

### Use the `if`-operator to _programmatically_ configure whether a function is a coroutine

The coolest thing about this design is that it allows us to use Mojo's parameter
system to **programmatically control** whether a function is a coroutine:

`fn read_file(path: String) yields Blah if config['async'] else Never -> String:`

Here, I'm using Mojo's ternary `if`-operator. `config['async']` is a
compile-time Boolean. For the moment, let's not worry about where this value
comes from, or what its name is. Owen
[proposed](https://github.com/modular/mojo/pull/3946) passing the value around
as an implicit parameter, but maybe it could be set using compiler flags, or
maybe Mojo could allow modules to be provided with arbitrary parameters when
they are imported. The choice that we make here is orthogonal to the rest of the
design that I'm presenting, so we can shelve this concern for the moment.

The syntax `yields Blah if config['async'] else Never` is verbose, so let's
introduce some sugar:

1. The first argument of the conditional (`Blah`) can be omitted, in which case
   it defaults to whatever the default value for `yields` is. (Perhaps `None`.)
2. The third argument of the conditional (`Never`) can be omitted, in which case
   it defaults to `Never`.

If we drop both of these arguments from the signature of `read_file`, it
becomes:

`fn read_file(path: String) yields if config['async'] -> String:`

This is highly readable, in my opinion! This signature communicates very plainly
that `read_file` will yield (be a coroutine) only if `config['async']` has been
set. This is peak simplicity: I can't imagine "programmable async" being any
simpler than this.

Using Mojo's "associated aliases" feature, we can even define traits in such a
way that their implementers can choose whether the trait methods are coroutines
or not:

```plaintext
trait Iterator[T: AnyType]:
    alias yields_: Bool
    fn __next__(mut self) raises StopIteration, yields if yields_ -> T: ...

fn get_integer[IterType: Iterator[Int]](mut iterator: IterType) yields if IterType.yields_ -> Int:
    try:
        return next(iterator)     # Later on, I'll explain why we're not using 'await' here.
    except:
        return 0
```

In Python, Rust, Swift, and all other languages with `async`/`await` syntax, the
standard library traits—including `Iterator`—are duplicated into "sync" and
"async" versions. More broadly, many different APIs have undergone "mitosis",
ultimately splitting the languages' libraries into sync and async versions. This
issue has generated a lot of negative sentiment in languages with
`async`/`await`. As shown above, the `yields if <cond>` syntax allows us to
avoid this problem. **This would be an enormous win for Mojo.**

As a final note: the `yields if <cond>` syntax allows Mojo's I/O libraries to be
generic over whether the I/O is blocking or non-blocking. There are a few
reasons why somebody may want blocking I/O:

1. For "gradual disclosure of complexity", when teaching people programming.
2. To avoid the nuisance of declaring the `yields` effect when writing simple
   scripts, such as a script that reads a file, does some calculations, and then
   writes to a file.
3. To achieve better performance when writing a program where the number of
   concurrent tasks is always less than the number of cores. (Such programs
   don't benefit from yielding control of a core.)
4. To define an I/O library that is usable from other languages—such as
   C—without those languages needing to know about coroutines or executors.

In summary: Being able to configure whether a function is a coroutine or not is
**very** useful, and can be achieved by combining the `yields T` effect with the
ternary `if`-operator.

### Let the `if`-operator be used for `raises`

We should allow the `if`-operator to be used for `raises`, in addition to
`yields`. This would have a lot of utility. For example, it would provide a
mechanism for configuring whether functions that allocate memory should raise an
exception on allocation failure:
`fn append(mut self: List[T], owned value: T) raises if config['allocation exceptions']:`

Another use case for `raises if <cond>` is to configure whether a long-running
task should respond to a cancellation request from its executor:
`fn long_running_task() yields, raises Cancelled if config['cancellable']:`

This feature is **_desperately_** needed in languages with `async`/`await`. As a
concrete example, in the Swift community there have been
[debates](https://forums.swift.org/t/asyncsequences-and-cooperative-task-cancellation/62657)
about whether "asynchronous sequences" should respond to cancellation requests,
and if so, whether Swift needs a parallel universe of "asynchronous sequences
that don't cancel themselves". All of this complexity can be avoided by allowing
the end user to configure whether they want a function to raise cancellation
exceptions. This is made trivial by the `raises if <cond>` syntax.

### Drop the requirement that coroutines be `await`ed

Let's think some more about `read_file`:

`fn read_file(path: String) yields if config['async'] -> String:`

As mentioned earlier, the `yields if <cond>` syntax allows people to use this
function both as a coroutine (with non-blocking I/O), and as a blocking
function, depending on their needs. Let's suppose that `read_file` is used by
another function `foo` which also wants to abstract over its coroutine status:

```plaintext
fn foo(path: String) yields if config['async']:
    data = await read_file(path)
    print(data)
```

Here, we're awaiting the result of `read_file`, and then printing it. In Mojo,
the expression `await read_file(path)` desugars to
`read_file(path).__await__()`. In other words, `await` does not have any magical
properties—it's just a function call.

Unfortunately, this design is incompatible with the semantics of
`yields if config['async']`. When `config['async']` is false, the definitions of
`read_file` and `foo` reduce to:

```plaintext
fn read_file(path: String) -> String:
    ...
fn foo(path: String):
    data = await read_file(path)
    print(data)
```

`read_file` becomes a function that returns a `String`. In turn, this means that
`await read_file(path)` is awaiting a string. This is nonsense! A string can't
be awaited—it doesn't have an `__await__` method. And in fact, if
`String.__await__` was defined, the situation would be even worse: the
synchronous and asynchronous versions of `foo` would be awaiting different
things!

None of this is acceptable. To make the `yield if <cond>` feature behave
properly, I can only see two options:

1. Introduce some compiler magic to **pretend you didn't write `await`** when
   `<cond>` is set to false.
2. Change the semantics of coroutines, so that `read_file(path)` awaits the
   coroutine, rather than returning an instance of `Coroutine`. In other words,
   we eliminate the need to write `await`.

If we chose the second option, our program would be written as follows:

```plaintext
fn read_file(path: String) yields if config['async'] -> String:
    ...
fn foo(path: String) yields if config['async']:
    var data: String = read_file(path)           # no 'await'
    print(data)
```

**Note:** We would still use the `await` keyword for other purposes. I'm not
saying we should remove the keyword from Mojo.

In my opinion, this model is the right choice, for many different reasons:

1. This model ensures that the `await` keyword isn't treated specially by the
   compiler—it's just sugar for `__await__()`.
2. This model makes function chaining more ergonomic: we can write
   `read_file(path).split('\n')` instead of
   `(await read_file(path)).split(' \n')`.
3. This model makes **refactoring** a program more ergonomic: when you add the
   `yields` effect to a function, you no longer need to go and add the `await`
   keyword to every call site. This reduces the syntactic burden of function
   coloring. (That said, you might need to add the `yields` effect to the
   function's callers, if they don't already have it.)
4. This model eliminates the need for the following Python features:
   `async for`, `async with`, `__aiter__`, `__anext__`, `__aenter__`, and
   `__aexit__`. These features **ONLY** exist so that the compiler can
   [insert some calls to 'await'](https://snarky.ca/unravelling-the-async-with-statement/)
   into the desugaring of `for` and `with` statements. We'll still need these
   features for backwards compatibility with Python coroutines, but we wouldn't
   need to teach all of this stuff to people writing pure Mojo. In the long
   term, this would make Mojo a simpler, more approachable language.
5. If we were to stick with `await read_file(...)`, programmers would need to do
   mental gymnastics to determine the expression's meaning in the case where
   `read_file` has been specialized to a synchronous function. Remember: after
   specialization, the signature of `read_file` becomes
   `fn read_file(path: String) -> String`, so the meaning of
   `await read_file(...)` becomes truly baffling, especially if the IDE
   _displays_ the type of `read_file` as being `fn(String) -> String` when you
   hover over it!
6. In contrast, if we drop the requirement to use `await`, a coroutine can be
   explained as simply "a function that has a `yields` effect". It would have
   all of the affordances of an ordinary function, including the property that
   `f()` evaluates the function. This is a simpler model for learners—especially
   people who are being taught about asynchronous I/O for the first time. We can
   explain that functions that yield propagate a `yields` effect, analogous to
   how functions that raise propagate the `raises` effect. **With this model, we
   don't need to expose learners to a `Coroutine` object**, and we don't need to
   explain why `read_file(path)` doesn't _actually_ read the file. Instead,
   function calls continue to work the same, regardless of whether a function
   contains `yields` in its signature. In short: this model is great for
   learnability.

Coincidentally, this model is similar to
[Kotlin's](https://kotlinlang.org/docs/composing-suspending-functions.html).
Kotlin has the syntax `suspend fun`, which is the equivalent of Mojo's
`async fn`, and yet it doesn't require `await` to be used at call sites.

Some people might feel uncomfortable with removing `await`, on the thesis that
it helps document where all of the suspension points are in a function body.
Personally, I don't think this is a big deal, for the following reasons:

1. In IDEs, we can highlight the identifiers of coroutines differently, e.g.
   give them a distinct color. This would make it easy to tell which function
   calls can suspend, and which can't.
2. The `yields` effect can be understood as being similar to `raises`, in the
   sense that the effect is visible in function signatures, but not at the call
   site. In practice, the invisibility of exceptions in Mojo isn't a big deal.
   And in Python, programmers make do with far less information, because there
   isn't an explicit `raises` effect. Finally, Swift tried making exceptions
   explicit at call sites via the `try` keyword ("marked propagation"), and
   people have mixed feelings about that. In summary: if Mojo doesn't need
   "marked exception points", then I doubt it needs "marked suspension points".
   (Especially if we have the aforementioned IDE highlighting.)
3. In languages where programs are mostly single-threaded (such as JavaScript
   and Python), `await` points are used to reason about the moments at which
   "concurrency happens". This doesn't apply to Mojo. Mojo programs are
   typically (highly) multi-threaded, so concurrency happens _everywhere_. Said
   another way: in Mojo, `await` points are not synchronization points. They are
   just points at which the current thread—one of many—might switch to another
   task.
4. In languages with green threads, such as Go and Java, suspension points are
   not explicit. I haven't seen Go or Java users complain about this. To the
   contrary, they normally talk about the lack of function coloring (i.e.
   explicit suspension) as the primary _advantage_ of green threads.

Given all of the above, it doesn't seem like we'd be losing much by having
suspension points be a transparent effect, like exceptions. A dedicated IDE
color for functions that `yield` would likely be sufficient. That's already a
step beyond what Go offers!

In summary: To be generic over whether a function `yields`, it seems that we
need to eliminate `await` at call sites. This may take some getting used to, but
beyond that, it's not a big deal.

### Coroutines should expose an associated `Task` type

As previously mentioned, the plan is for `read_file(path)` to denote a `String`,
rather than a `Coroutine`/`Task`. But we still need the latter type (let's call
it `Task`) for modelling the execution state of a coroutine. An instance of
`Task` models a unit of work that can be carried out by an executor, potentially
in parallel with other work. Given that `read_file(path)` returns a `String`,
we'll need to find a different syntax for constructing a `Task`.

If Mojo is to offer non-allocating coroutines—as Rust does—we will actually need
a distinct `Task` type for each distinct coroutine. Therefore, it makes sense
for `Task` to be an **associated type** of a coroutine. For example, the task
type associated with the `read_file` coroutine would be `read_file.Task`. Once
we have this, the evaluation of `read_file(path)` can be packaged up into a
`Task` by passing the arguments to the `Task` constructor, instead of passing
them to the coroutine: `var task: read_file.Task = read_file.Task(path)`

`read_file.Task` would be value-semantic and its state would be stored inline.
This means the state of `task` would be stored in the enclosing function's stack
frame. This avoids the need to allocate memory. Several such tasks can be
constructed, and then `await`ed simultaneously. The executor can then run the
tasks concurrently, while the parent task is paused. When they are finished, the
parent task can be resumed. No memory needs to be allocated at any point in this
process, except when constructing the `String` that `read_file` returns.
Alternatively, tasks can be "detached" from their parent task by submitting them
directly to an executor, e.g. `executor.submit(task^)`. This would return a
`Future` that can be `await`ed.

**Note:** In Python, a `Task` object is reference-semantic, and points to a task
that has already been submitted for execution. In the above design, that's
equivalent to implicitly passing every task to `executor.submit` as soon as it
is created. This is not the right default for Mojo, for two reasons:

1. `executor.submit` needs to allocate memory to store the task. This has
   runtime overhead.
2. If the child task executes concurrently with the parent task, and one of the
   tasks is able to mutate `path`, this constitutes a data race. In comparison,
   if the parent task must **yield** before the child task is executed, the data
   race is eliminated.

All that said, the exact design for how tasks are created and managed is outside
of the scope of this proposal. The purpose of this section was merely to show
that although `read_file(path)` immediately executes the coroutine and return a
`String`, it's still easy to create a first-class "unit of work" that can
execute concurrently with other work, and that this can be done without
superfluous memory allocations.

## Summary

I have proposed that we replace `async` with a `yields T` syntax, that allows us
to abstract over whether a function/trait/library contains suspension points. A
function that is not a coroutine (i.e. does not suspend) is equivalent to a
function declared as `yields Never`. For this design to work well, we need to
remove the requirement that the `await` keyword be used to wait for a coroutine.
This isn't a big deal, because the information that `await` communicates is
still visible in type signatures, and we can display it in IDEs. As a bonus,
eliminating `await` at call sites reduces the burden of turning a function into
a coroutine, and vice versa.

The goal of this proposal is to make the experience of using Mojo coroutines as
painless as possible, so that programmers can focus on getting their jobs done.
I believe I have made good progress in that direction. In this proposal, a
coroutine is signified by the `yields` effect, which is no more burdensome than
`raises`. Both of these effects are programmable, meaning Mojo programmers don't
need to duplicate their APIs into versions that propagate `yield`/`raise`
effects, and versions that don't.

## Unresolved questions

This proposal is just a high-level design—it doesn't attempt to resolve every
question concerning Mojo coroutines. In my mind, the biggest unanswered question
is where the compile-time Boolean variables that enable/disable `yields` and
`raises` originate from. For example, I claimed the signature of `read_file`
would be written
`fn read_file(path: String) yields if config['async'] -> String:`, but I didn't
explain where `config` comes from.

I can think of a few different options:

- Make it a function parameter:
  `fn read_file[config_async: Bool](...) yields if config_async`
- Introduce the concept of **module parameters** to Mojo, so that the function
  can be configured when it is imported:
  `from io_library[async_io=True] import read_file`
- Have a few compile-time variables built into the compiler, that can be set
  using compiler flags.
- For _methods_ that yield or raise, we can make use of the parameters and
  `alias`es defined by the enclosing struct.

Which of these alternatives is best would be situational. Struct parameters and
associated aliases are a great alternative, where applicable. Function
parameters have the benefit of already existing in Mojo today, but they are
verbose and too complicated for beginners—nobody wants to repeatedly write
`read_file[async_io=True](...)` when reading files. Personally, I think it's
worth investigating adding module parameters to Mojo. This feature wouldn't
require much effort to implement, because module parameters are more or less the
same as struct parameters. The only question is whether the feature is
sufficiently useful to be worth adding. Maybe OCaml veterans can answer that
question. (OCaml has had module parameters for decades.)

## Appendix A: Why it's okay to diverge from async/await syntax

There's one elephant in the room that I haven't addressed: given that I've
proposed replacing `async` with `yields`, and eliminating `await`, the model
that I've presented doesn't _look_ like "async/await". Hopefully, it's clear
that I haven't proposed any syntactic changes just out of personal preference.
Each change was driven by the need to solve a particular problem. There's no way
to achieve the benefits that I've demonstrated while sticking with the exact
same syntax as Python.

Regardless, there are other reasons why diverging from Python's syntax might be
a good idea:

- Mojo coroutines will have semantic differences from Python coroutines, for the
  sake of performance. If we were to use the same `async`/`await` syntax as
  Python, programmers might make incorrect assumptions about how Mojo coroutines
  work, and how to use them effectively. Changing the syntax makes clear that
  the semantics are different.
- Last I heard, Mojo aims to support all of Python's existing features. This
  implies that `async def f()` should declare a _Python_ coroutine, not a Mojo
  coroutine. In particular, it implies that `f()` return an instance of Python's
  `coroutine` class. If it were to return anything else—e.g. a Mojo `Coroutine`
  struct—then it would be much more difficult to port Python code to Mojo, given
  that Mojo coroutines will have differences from Python coroutines. In short:
  it seems sensible to reserve the `async def` syntax, and therefore `async fn`,
  for Python coroutines. By happenstance, my proposal does this.

Taking a few steps back: Mojo aims to be the best language for writing
high-performance applications, especially AI applications. It will still be
around in 20 years, and if it succeeds in its goals, the next generation of
programmers will be taught Mojo early on in their education. This suggests that
there is value in implementing the best possible design for coroutines, rather
than focusing on syntactic similarly with existing languages. Given this, I
think it's worth making a clean break from the cliché async/await syntax, and
doing what's best for Mojo. Hopefully I've justified why these changes are
necessary to make Mojo as great as it can be.

## Appendix B: Prior art

My proposal was heavily inspired by
[this proposal](https://github.com/modular/mojo/pull/3946) from @owenhilyard,
and
[this post](https://github.com/modular/mojo/pull/3946#issuecomment-2601176112)
from @lattner. Owen and Chris noticed the issue with the `await` keyword
appearing in async-generic code, which was what inspired my proposal to remove
it.

My proposal also has a few resemblances to Rust's
[keyword generics initiative](https://github.com/rust-lang/keyword-generics-initiative),
which is still in progress. As part of that initiative, there have been
explorations into syntaxes such as `?async` and `async<A>` for declaring that a
function is "maybe async", where `A` is a new kind of "generic parameter".
Beyond that, there isn't much overlap with the design presented in this
proposal. (Or if there _is_ overlap, it's not intentional.)

People sometimes claim that Zig offers stackless coroutines without function
coloring. They usually point to
[this blog post](https://kristoff.it/blog/zig-colorblind-async-await/).
Unfortunately, the design presented in that blog post was just an experiment,
and it was removed from Zig years ago. I am told it had multiple unresolved
flaws. The designers of Zig plan to re-implement a similar design in a future
version of Zig, but nothing has been finalized yet. Therefore, it doesn't seem
appropriate for me to compare Zig's design(s) to the design presented in this
proposal.
