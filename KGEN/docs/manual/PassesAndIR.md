# Passes and Intermediate Representations

The best way to start understanding a compiler is to understand the various IR
stages, the differences between them, and which code makes those transformations
happen.

The Mojo compiler transforms Mojo code into various intermediate
representations, such as LIT, KGEN, LLVM, and many others.

This doc covers the basics of what our IR looks like, and how the passes
transform from Mojo to all of those.

## MLIR

The Mojo compiler is built on MLIR. To see it, put this snippet into a
`main.mojo` file:

```mojo
fn foo(arg: Int):
  pass

fn main():
  foo(5)
```

and run this command to run the parser/type-checker:

`br //KGEN/tools/kgen-translate -- -import-mojo main.mojo`

The output contains this for the `main` function:

```mlir
lit.fn @”main()”() -> !kgen.none attributes {sourceName = “main”, specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{5}>
  %1 = lit.call @main::@”foo(::Int)”(%0) : !lit.generator<(“arg”: !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

This is MLIR. One could think of it as a more extensible and customizable form
of LLVM IR.

Useful background on MLIR:

- [MLIR Documentation Overview](https://mlir.llvm.org/docs/)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin)

## Mojo Dialects

Above, we saw some MLIR that **contained multiple “dialects” at once.**

Here it is again:

```mlir
lit.fn @”main()”() -> !kgen.none attributes {sourceName = “main”, specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{5}>
  %1 = lit.call @main::@”foo(::Int)”(%0) : !lit.generator<(“arg”: !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Here we see instructions from two dialects (`lit` and `kgen`) working together.

Some things from the `lit` dialect:

- `lit.call` - A call operation.
- `!lit.ref` - A reference type.
- `!lit.trait` - A trait.

Some things from the `kgen` dialect:

- `!kgen.none` - The "none" type.
- `!kgen.param.constant` - Makes a compile-time value.

(We'll cover these, the rest of the `lit`/`kgen` things, and things from other
dialects further below.)

Mojo has several dialects: `kgen`, `lit`, `kgen`, `pop`, and `hlcf`. KGEN IR
also uses the upstream `index`, and `llvm` dialects. The `lit` dialect should
more properly be named `mojo` perhaps but currently reflects how “lit” Mojo is
🔥.

`lit` is a high-level dialect for building kernel libraries. It is lowered to
`kgen` before elaboration. The `kgen` dialect is the canonical dialect for
describing parametric IR. The dialect defines the parameter system and the
types, attributes, and operations for interacting with parameters.

`hlcf` and `index` are non-parameterized, target-independent dialects that exist
in “KGEN IR” pre-elaboration and post-elaboration. `llvm` is a target-dependent
dialect that can exist at all levels of KGEN IR. However, it locks the
particular kernel to the LLVM target.

`pop` (which stands for “parametric operations”) are parameterized,
target-independent dialects used to build parametric kernels.

In summary:

- `lit` exist pre-elaboration. They are lowered to `kgen` and `pop` before
  elaboration.
- `kgen` contains all the instructions that must be understood and monomorphized
  by the elaborator.
- `pop` exists pre and post elaboration. Operations in the dialect become
  non-parametric post-elaboration. They are lowered to `llvm` when executing
  kernels.
- `index` and `hlcf` exist pre and post elaboration. They are lowered to `llvm`
  when executing kernels.
- `llvm` can exist at all levels of KGEN IR to describe target-specific
  operations, but then the kernel can only target LLVM.

## Passes

The Mojo compiler has a lot of stages. Some of the big ones are:

- Parsing, which does lexing, parsing, and type-checking.
- Elaborating, which instantiates generics, for example `fn add[x: Int](...)`
  into `fn add[3](...)`, `fn add[42](...)`, `fn add[1337](...)` etc.
- Lowering to LLVM.

...but there are a lot more.

You can learn about all of them in Weiwei's excellent
[Mojo Compilation Model](https://www.notion.so/modularai/Mojo-Compilation-Model-Now-and-Future-6028a58015034f38b037e520ee2e2d78)
doc.

You can see all the passes that run for a particular program by running
`kgen --mlir-print-ir-before-all -elaborate main.mojo 2>&1 | grep 'IR Dump Before'`.
For example when run on a simple `fn main(): pass` it mentions these passes
coming after the parser:

- DebugInfoStrip
- LowerSemanticCF
- VerifyParameters
- CheckLifetimes
- AnnotateKernels
- VerifyKernels
- LowerLIT
- MOGGPreElabPipeline
- RemoveUnusedParams
- EliminateDeadSymbols
- SROA
- Mem2Reg
- Canonicalizer
- InlineParametric
- SCCP
- ApplyInliner
- OutlineClosures
- CSE
- LiftAndFoldApply
- ElaborateGenerators
- EliminateDuplicateFunctions
- ResolveCompilerPromises
- LowerArgConventions
- LowerCallingConventions
- EnsureNoParameters
- AutomaticInline
- RaiseForLoops
- LoopUnrolling
- ArgPromotion
- SimplifyCF
- LowerLoops
- LowerClosures
- LowerAsyncFunctions
- DeadArgumentElimination

...and many of these passes are run multiple times.

The `mojo` command will run the entire pipeline from beginning to end, but you
can use `kgen` to run specific passes, and `kgen-translate` to run only the
parser. For more details on those, and other commands, see
[Mojo Dev Tools](https://www.notion.so/modularai/Mojo-Dev-Tools-027879ef5e4d480ea6f8f73b1cbc2ad3).

## MLIR Guide

- `%name` — A run-time value; an MLIR SSA value; the result of an MLIR
  operation.
- `@name` — A
  [SymbolRefAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)
- `!name` — An MLIR type.
- `#expr` or `{expr}` — Compile-time data.
- Anything else is an MLIR **operation.**

All these are explained in the next sections.

For now, forget about compile-time data (`#expr`), and pretend they only come
into play with generics are involved. Let's start with a program that just uses
run-time values, operations, and types.

### Operations, run-time values, and types

For example, this program:

```mojo
fn main():
    var x = 42
```

...when run through the parser, gives this MLIR:

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %x = lit.var.decl "x" var : !lit.ref<!Int, mut *"x`">
  %0 = kgen.param.constant: !Int = <{42}>
  lit.ref.store %0, %x : <!Int, mut *"x`">
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Here's what each of those lines means.

` %x = lit.var.decl "x" var : !lit.ref<!Int, mut *"x``"> `

This is declaring a **run-time value** (or in other words an **MLIR SSA value**)
named `%x`.

It will contain the result of `lit.var.decl "x" var` which is an **MLIR
operation**. Every MLIR operation is followed by **operands**, like the `"x"`
and `var` here.

After that and the `:`, we specify the operations resulting **MLIR type**,
`!lit.ref<!Int, mut \*"x``">```.

Note that the type directly describes the operation's result specifically.

` %x = ( lit.var.decl "x" var : !lit.ref<!Int, mut *"x``"> ) `

In our MLIR, the `%x =` is always the lowest precedence.

Now the next line:

`%0 = kgen.param.constant: !Int = <{42}>`

The `%0 =` is always the lowest precedence, so we first look at the
`kgen.param.constant: !Int = <{42}>` part.

That `=` is not an assignment like the first `=`. One should interpret this like
a (hypothetical) `kgen.param.constant <{42}> : !Int`.

Since there's no `%`/`@`/`!`/`#`/`{` symbol in front of `kgen.param.constant`,
it's an MLIR operation.

That operation's operand is `<{42}>`, which is how we write constants (and
parameter expressions in general, but we'll get there later).

Now the next line:

` lit.ref.store %0, %x : <!Int, mut *"x``"> `

This follows the same rules; The `lit.ref.store` operation takes operands `%0`
and `%x`, and the operation's result type is ` <!Int, mut *"x``"> `. Let's
explore that type a little more.

For lit.ref.store specifically, the `<..., ...>` is actually shorthand for
`!lit.ref<..., ...>`. So that line is more like:

` lit.ref.store %0, %x : !lit.ref<!Int, mut *"x``"> `

As you can see, there's a lot of context-dependent sugar in our MLIR. If you
don't know what something means, ask in slack (and then add the answer to this
guide!). Or, if you're feeling brave, you can try and trace the printing logic
(usually in a `_.td` and its corresponding `_.cpp` file).

### Symbols

Anything with a `@` in front is an **MLIR symbol ref.**

For example, this program:

```mojo
fn my_func(x: Int):
    pass

fn main():
    my_func(42)
```

parses `main` to this MLIR:

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{42}>
  %1 = lit.call @mymain::@"my_func(::Int)"(%0) : !lit.generator<("x": !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Let's talk about this line:

`%1 = lit.call @mymain::@"my_func(::Int)"(%0) : !lit.generator<("x": !Int) -> !kgen.none>`

The `@mymain::@"my_func(::Int)"` is an **MLIR symbol ref**. It refers to
something defined somewhere else.

In the above `lit.call` line, the type after the `:` doesn't describe the
operation's type, it describes the type of the symbol ref. In other words,
that's `my_func`'s type, not the `lit.call`'s result type.

### Compile-time Data

Anything with a `#` in front of it (`#Thing`), or surrounded with curly braces
like `{Thing}` is **compile-time data**, often referred to as an "attribute",
"value", "constant", or "parameter". The terminology is confusing, so for now,
just call it "compile-time data".

Let's see some compile-time data. This program:

```mojo
fn my_func[N: Int](x: Int):
    pass

fn main():
    my_func[73](42)
```

...parses `main` to this MLIR:

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{42}>
  %1 = lit.call @mymain::@"my_func[::Int](::Int)"<:!Int {73}>(%0) : !lit.generator<("x": !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Notice the `lit.call` line's new part: `<:!Int {73}>`. That `{73}` is making
some compile-time data (`{73}`) of type `!Int`.

We've also seen this before; `%0 = kgen.param.constant: !Int = <{42}>` had a
`<{42}>` which was a compile-time data operand to the `kgen.param.constant` op,
though that one didn't have the type (`:!Int`) in front.

All compile-time data has a type. We'll talk about that more further below.

### Compile-time Data Terminology: Parameters, Attributes, Constants, Values

Every stage of the compiler, up to the elaborator, deals with Mojo's
compile-time metaprogramming, and handling data at compile-time. We call that
compile-time data "**parameters**".

In Mojo, a "parameter" is not an argument. "Parameter" means compile-time data.

More specifically, “parameter” means one of three things. For example, in this
snippet:

```mojo
struct Foo[T: Stringable]:
    var field: T

fn main():
  var f = Foo[Int]()
```

- A "parameter declaration" (or "param decl" or "input param"), is like the
  `T: Stringable` in that first line.
- A "parameter reference" (or "param ref"), is like the mention of `T` in
  `var field: T`. It refers to a param decl.
- A "parameter value" (or "param value"), is the `Int`.

All of them in the same sentence: The param value `Int` is fed into `Foo`'s
param decl `T: Stringable` and makes its way to the param ref `T` in `field: T`.

When people say “in parameter-space” or "in the parameter domain", that means
“at compile time”.

There can be subtle differences between the various terms:

- "Attribute" refers to MLIR attributes. There are typed attributes and untyped
  attributes. All parameters are typed attributes, and most (but not all) typed
  attributes are parameters.
- "Value" is often short for "parameter value", but in rare cases it can mean a
  run-time value.
- "Constant" is equivalent to "parameter value", but probably refers to
  hard-coded parameter values.

### Does Compile-time Data Have Types?

(Spoiler alert: in Mojo and LIT, yes, but in KGEN, no.)

To better frame the question, let's start by observing some things about C++. In
C++, compile-time data _kind of_ has types.

- The `N` in `template<int N> ...` has type `Int`.
- The `T` in `template<typename T>` kind of has a type, `typename`.

This is not how Mojo works. However, **that is how kgen works.** kgen's `type`
is basically C++'s `typename`.

However, Mojo (and the LIT dialect which comes right after Mojo), compile-time
data does have types.

They're a bit more strict in that way, to enable better type-checking (akin to
Java generics) compared to C++'s "duck-typed" templates.

In Mojo, we need to specify the rough shape of `T`, by specifying a trait.

`template<int N, typename T> class Vec { ...` in C++ would therefore be
equivalent to

`struct Vec[N: Int, T: Copyable]: ...` in Mojo.

In that `Vec`, we can say two things:

- `N`'s type is `Int`.
- `T`'s type is `Copyable`.

"Type" is a relative term. `N`'s type is `Int`, and Int's type is something
else, and that has a type, and so on. Everything has a type.
