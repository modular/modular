---
title: Mojo🔥 programming manual
---

Mojo🔥 is a first-class programming language that utilizes a number of
next-generation compiler technologies. The compiler itself has integrated
caching, is multithreaded, is built to be distributed across a cluster,
incorporates autotuning features and powerful compile time metaprogramming, and
is built on top of MLIR. Beside using MLIR as an implementation detail, Mojo
exposes the full power of MLIR to advanced library developers that want to work
with exotic hardware.

Mojo is designed to become a superset of Python over time, complementing its
dynamic features with state of the art systems programming features and lifting
the vast Python library ecosystem. This allows it to address the traditional use
cases of C, C++, Rust etc, but also CUDA and other accelerator systems. By
bringing together the best of dynamic languages and systems languages, we hope
to provide a **unified** programming model that works across levels of
abstraction, is friendly for novice programmers to learn and use, and scales
across many use cases from accelerators through to application programming and
scripting.

This document is intended to be an introduction to the Mojo programming
language, fit for consumption by Mojo programmers. It assumes knowledge of
Python and systems programming concepts, but does not expect the reader to be a
compiler nerd.

## High level motivation for a new language

Mojo started with the goal of bringing an innovative programming model to
accelerators and other heterogeneous systems that are pervasive in machine
learning. This drove the need for powerful compile-time metaprogramming, the
integration of adaptive compilation techniques, caching throughout the
compilation flow, and other things that are not supported by existing
languages.

While accelerators are important, one of the most prevalent and sometimes
overlooked “accelerators” is the host CPU. CPUs today are getting lots of
tensor-core-like accelerator blocks and other dedicated AI acceleration units,
but they also importantly serve as the “fall back” to support operations more
specialized accelerators don’t - things like data loading, pre- and
post-processing, and integrations with foreign systems. Applied AI systems need
to address all these use cases and it is clear that we couldn’t lift AI with a
limited “accelerator language” that only worked with specific accelerated use
cases.

While innovating in compiler internals and while support for current and
emerging accelerators is critical to our mission, we didn’t see a
need to innovate in *syntax* or *community*. We decided to embrace the Python
ecosystem because it is so widely used, it is loved by the AI ecosystem, and
because it is really nice! For more information on the challenges with Python,
how Mojo compares to other members of the Python family, and other details, see
the "Detailed Motivation and Related Work" section later in this document.

## Mojo as a member of the Python family

The Mojo language has lofty goals - we want full compatibility with the Python
ecosystem, we would like predictable low-level performance and low-level
control, and we need the ability to deploy subsets of code to accelerators. We
also don’t want
ecosystem fragmentation - we hope that people find our work to be useful over
time, and don’t want something like the Python 2 => Python 3 migration to
happen again. These are no small goals!

Fortunately, while Mojo is a brand new code base, we aren’t really starting
from scratch conceptually. Embracing Python massively simplifies our design
efforts, because most of the syntax is already specified. We can instead focus
our efforts on building the compilation model and designing specific systems
programming features. We also benefit from tremendous work on other languages
(e.g. Clang, Rust, Swift, Julia, Zig, Nim, etc), and leverage the massive MLIR
compiler ecosystem.  We also benefit from experience with the Swift programming
language, which migrated most of a massive Objective-C community over to a
new language.

After discussion, we decided that the right *long-term goal* for Mojo is to
provide a **superset of Python** (i.e. be compatible with existing programs)
and to embrace the CPython immediately for long-tail ecosystem enablement.
To a Python programmer, we expect and hope that Mojo will be immediately
familiar, while also
providing new tools for developing systems-level code that enable you to do
things that Python falls back to C and C++ for. We aren’t trying to convince
the world that “static is good” or “dynamic is good” - our belief is that both
are good when used for the right applications, and that the language should
enable the programmer to make the call.

### How compatible is Mojo with Python really?

Mojo has many core features of Python including async/await, error handling,
variadics, etc, but… it is still very early and missing many features - so
today it isn’t very compatible. Mojo doesn’t even support classes yet! That
said, we have experience with other similar projects that give some insights on
how this will go assuming that we stay properly focused.

We have experience with two major but different compatibility journeys: the
**Clang** compiler is a C, C++ and Objective-C (and CUDA, OpenCL, …) that is
part of LLVM. A major goal of Clang was to be a “compatible replacement” for
GCC, MSVC and other existing compilers. It is hard to make a direct comparison,
but the complexity of the Clang problem appears to be an order of magnitude
bigger than implementing a compatible replacement for Python.  The journey there
gives good confidence we can do this right for the Python community.

Another example is the [Swift programming language](https://www.swift.org/),
which embraced the Objective-C runtime and language ecosystem and progressively
shifted millions of programmers (and huge amounts of code) incrementally over
to a completely different programming language. With Swift, we learned
lessons about how to be “run-time compatible” and cooperate with a
legacy runtime. In the case of Python and Mojo, we expect Mojo to cooperate
directly with the CPython runtime and have similar support for integrating with
CPython classes and objects without having to compile the code itself. This
will allow us to talk to a massive ecosystem of existing code, but provide a
progressive migration approach where incremental work put in for migration will
yield incremental benefit.

Overall, we believe that the north star of compatibility, continued vigilance
on design, and incremental progress towards full compatibility will get us to
where we need to be in time.

### Intentional Differences From Python

While compatibility and migratability are key to success, we also want Mojo to
be a first class language on its own, and cannot be hobbled by not being able
to introduce new keywords or add a few grammar productions. As such, our
approach to compatibility is two fold:

1. We plan to utilize CPython to run all existing Python3 code “out of the box”
without modification and use its runtime, unmodified, for full compatibility
with the entire ecosystem. Running code this way will get no benefit from Mojo,
but the sheer existence and availability of this ecosystem will rapidly
accelerate the bring-up of Mojo, and leverage the fact that Python is really
great for high level programming already.

2. We will provide a mechanical migrator that provides very good compatibility
for people who want to move Python code to Mojo. For example, Mojo provides a
backtick feature that allows use of any keyword as an identifier, providing a
trivial mechanical migration path for code that uses those keywords as
identifiers or keyword arguments. Code that migrates to Mojo can then utilize
the advanced systems programming features.

Together, this allows Mojo to integrate well in a mostly-CPython world, but
allows Mojo programmers to be able to progressively move code (a module or file
at a time) to Mojo. This approach was used and proved by the Objective-C to
Swift migration that Apple performed. Swift code is able to subclass and
utilize Objective-C classes, and programmers were able to adopt Swift
incrementally in their applications. Swift also supports building APIs that are
useful for Objective-C programmers, and we expect Mojo to be a great way to
implement APIs for CPython as well.

It will take some time to build Mojo and the migration support, but we feel
confident that this will allow us to focus our energies and avoid distractions.
We also think the relationship with CPython can build from both directions -
wouldn't it be cool if the CPython team eventually reimplemented the
interpreter in Mojo instead of C? 🔥

## Using the Mojo Compiler

The simplest way to run a mojo program is to type “mojo hello.🔥”, for
example:

```mojo
    $ cat hello.🔥
    def main():
       print("hello world")
       for x in range(9, 0, -3):
           print(x)
    $ mojo hello.🔥
    hello world
    9
    6
    3
    $
```

On the other hand, we realize that not everyone is so bold as to be ready for
emoji file extensions: the Mojo toolchain also fully supports the `.mojo`
suffix as well.

If you are interested in diving into the more of the internal implementation
details of Mojo, it can be instructive to look at types in the standard
library, example code in notebooks, blogs and other sample code.

## Basic Systems Programming Extensions

Given our goal of compatibility and Python’s strength with high-level
applications and dynamic APIs, we don’t have to spend much time talking about
how those portions of the language works. On the other hand, Python’s support
for systems programming is mostly delegated to C, and we want to provide a
single system that is great in that world. As such, this section breaks down
each major component and feature and describes how to use them with examples.

### `'let'` and ‘`var'` declarations

Inside a `'def'` in Mojo, you may assign to a name and it implicitly creates a
function scope variable just like in Python. This provides a very dynamic and
low ceremony way to write code, but is a challenge for two reasons: 1) systems
programmers often want to declare that a value is immutable, and 2) may want to
get an error if they mistype a variable name in an assignment.

To support this, Mojo supports ‘let’ and ‘var’ declarations which introduce a
new scoped runtime value: ‘`let`’ is immutable and ‘`var`’ is mutable. These
values use lexical scoping and support name shadowing:

```mojo
    def your_function(a, b):
      let c = a
      c = b  # error: c is immutable

     if c != b:
       var c = b
       stuff()
```

‘let’ and ‘var’ declarations support type specifiers as well as patterns, and
late initialization:

```mojo
    def your_function():
      let x: Int8 = 42
      let y: Int64 = 17

      var (a, b) = (x, y)

      var z: Int8
      if x != 0:
        z = 1
      else:
        z = foo()
      use(z)
```

Note that ‘var’ and ‘let’ are completely opt-in when in ‘def’ declarations: you
may also use implicitly declared values as normal in Python, and they get
function scope.

### `'struct'` Types

Mojo builds on MLIR and LLVM which provide a modern compiler code generation
stack that powers many systems programming languages. This allows us to expose
low-level control over data layout, indirection-free access to fields, as well
as low-level tricks for bit-swizzling and other niche tricks. One extremely
important feature of modern systems programming languages is the ability to
build high level and safe abstractions on top of these low-level shenanigans,
with zero abstraction penalties. In Mojo, this is provided by a ``struct``
declaration.

‘`struct`’ declarations in Mojo are similar in many ways to classes: they
support methods, fields, operator overloading, decorators for meta programming,
etc. On the other hand, where classes are extremely dynamic with
dynamic dispatch, dynamic method swizzling, and dynamically bound instance
properties, structs are static, bound at compile time, and are stored inlined
into their container instead of being implicitly indirect and reference
counted. This approach has precedent in other languages, e.g. Swift, C# and
others.

Here’s a simple definition of a struct:

```mojo
    struct MyPair:
      var first: Int
      var second: Int

      def __init__(self&, first: Int, second: Int):
          self.first = first
          self.second = second

      def __lt__(self, rhs: MyPair) -> Bool:
          return self.first < rhs.first or
                (self.first == rhs.first and
                 self.second < rhs.second)
```

As you can see, Mojo structs are very similar to classes, the biggest
difference is that all instance properties _must_ be explicitly declared with a
‘var’ or ‘let’ declaration. This allows the Mojo compiler to layout and access
the value precisely in memory without indirection or other overhead. Struct
fields are bound statically: they aren’t looked up with a dictionary
indirection. As such, you cannot ‘`del`’ a method or reassign it at runtime.
This enables the Mojo compiler to perform guaranteed static dispatch, use
guaranteed static access to fields, and inline `MyPair` into the stack frame or
enclosing type that uses it without indirection or other overheads.

One common thing about Mojo and Python is that they both focus on enabling
expressive API design, and push complexity from the language itself into the
package ecosystem. Structs in Mojo supercharge this capability by providing
zero-cost abstraction capabilities typically seen in languages like C++, Swift,
Rust, and Zig, which compose beautifully with the operator overloading and
other features that Python has supported for years. These capabilities compose
to allow **all** the “standard types” (like `Int`, `Bool`, `String` and even
`Tuple`) to be implemented as structs in the standard library instead of being
built into the language/compiler.

You might be wondering what the “`&`” means on the `self` argument: this
indicates that the value is mutable, please see the “By-Reference” arguments
section below.

### Strong type checking

Another feature of structs is that a `struct` definition defines a
compile-time-bound name, and references to that name in a type context are
treated as a strong specification for the value being defined. For example,
consider the following code:

```mojo
    def pairTest() -> Bool:
      let p = MyPair(1, 2)
      return p < 4 # gives a compile time error
```

If you attempt to run this code, you’ll get a compile time error telling you
that “4” cannot be converted to ``MyPair``, which is what the RHS of `__lt__`
requires. This is a familiar experience when working with systems programming
languages, but is not how Python works. Python has a syntactically identical
feature for “MyPy” type annotations, but they are not enforced by the compiler:
instead they are hints that inform static analysis. By tying types to specific
declarations, we are able to handle both the classical type annotation hints as
well as the strong type specifications without breaking compatibility.

Beyond type checking, strong types are also very important for code generation.
Because we know that these types are correct, we can specialize on the types,
pass values in registers, and generally be as efficient as C for argument
passing and other low level details. This also is the foundation of the safety
and predictability guarantees that Mojo provides to systems programmers.

#### A note on “Int” vs “int”

You might note that Mojo uses a capital-`Int` type defined in its standard
library, which differs in case from the lower-case-`int` that is used by MyPy.
This is intentional, and a good thing. The Mojo standard library “Int” is
defined to be a fixed width integer sized to match the CPU register (like
`ssize_t` in C). In contrast, the Python “int” type is a boxed object that
supports arbitrary precision arithmetic, and has a somewhat broader API - e.g.
support for object identity tests.

Mojo does this for two reasons:

1. We need to expose a simple and predictable programming model to systems
programmers and give them full control over the hardware. We cannot afford to
rely on fancy compiler analysis to “devirtualize” common cases.

2. Mojo needs to grow into a full superset of Python to avoid breaking the
ecosystem. We cannot do that by changing the behavior of core types like the
integers that everyone uses. By using different names, we can support both
types in the same system.

As an additional minor point, `Int` is just a struct built into the standard
library type, so it is nice that it can follow the naming conventions of user
defined types.

### Overloaded functions & methods

While strong type checking is good for predictability, control, and safety, it
forces you to keep the type checker happy. This can be a challenge when you
want to define expressive APIs that “just work” because some methods should
accept many different static types, and shouldn’t require the user of the API
to remember different names for all the different use cases. Python handles
this by accepting arbitrary inputs and uses dynamic dispatch to resolve what
to do on the fly - this will work in Mojo, but may not give you the
predictability, control, or performance you might be seeking.

To solve this problem, Mojo offers full support for “overloaded methods”. This
is a common feature seen in many programming languages (including C++, Java,
Swift, etc) where you can define the same function name with multiple different
signatures. When resolving a function call, Mojo will try each candidate and
use the one that works (if only one works), pick the closest match (if it can
determine a close match) or report the call as being ambiguous if it can’t
figure out which one to pick. In the latter case, you can resolve the ambiguity
by adding an explicit cast on the call site. Let’s look at an example:

```mojo
struct Array[T: AnyType]:
    fn __getitem__(self, idx: Int) -> T: ...
    fn __getitem__(self, idx: Range) -> ArraySlice: ...
```

Mojo doesn’t support overloading solely on result type, and doesn’t use result
type or contextual type information for type inference, keeping things simple,
fast, and predictable.  Mojo will never produce an "expression too complex"
error, because its type-checker is simple and fast by definition.

### `'fn'` Definitions

The extensions above are the cornerstone that provides low-level programming
and provide abstraction capabilities, but many systems programmers prefer more
control and predictability than what ‘def’ in Mojo provides. To recap, ‘def’ is
defined by necessity to be very dynamic, flexible and generally compatible with
Python: arguments are mutable, local variables are implicitly declared on first
use, and scoping isn’t enforced. This is great for high level programming and
scripting, but is not always great for systems programming. To complement this,
Mojo provides an ‘`fn`’ declaration which is like a “strict mode” for ‘`def'`.

> Alternative: instead of using a new keyword like fn, we could instead add a
modifier or decorator like '@strict def'. However, we need to take new keywords
anyway and there is little cost to doing so. Also, in practice in systems
programming domains, 'fn' is used all the time so it probably makes sense to
make it first class.

‘`fn`’ and ‘`def`’ are always interchangeable from an interface level: there is
nothing a ‘def’ can provide that a ‘`fn`’ cannot (or vice versa). The
difference is that a ‘`fn`’ is more limited and controlled on the _inside_ of
its body (alternatively: pedantic and strict). Specifically, ‘`fn`’s have a
number of limitations compared to ‘`def`’s:

1. Argument values default to being immutable in the body of the function (like
a ‘let’), instead of mutable (like a ‘var’). This catches accidental mutations,
and permits the use of non-copyable types as arguments.

2. Argument values require a type specification (except for `self` in a
method), catching accidental omission of type specifications. Similarly, a
missing return type specifier is interpreted as returning `None` instead of an
unknown return type. Note that both can be explicitly declared to return
“`object`”, which allows one to opt-in to the behavior of a `def` if desired.

3. Implicit declaration of local variables is disabled, so all locals must be
declared. This catches name typos and dovetails with the scoping provided by
‘let’ and ‘var’.

4. Both support raising exceptions, but this must be explicitly declared on a
‘fn’ with the `raises` keyword.

Programming patterns will vary widely across teams, and this level of
strictness will not be for everyone. We expect that folks who are used to C++
and already use MyPy-style type annotations in Python to prefer the use of
‘fn’s, but higher level programmers and ML researchers to continue to use
‘def’. Mojo allows you to freely intermix ‘def’ and ‘fn’ declarations, e.g.
implementing some methods with one and others with the other, and allows each
team or programmer to decide what is best for their use-case.

### The`__copyinit__` and `__moveinit__` Special Methods

Mojo supports full “value semantics” as seen in languages like C++, and more
advanced support than languages like Swift and Rust because it supports
non-movable types like Atomic. This is accessed by implementing special methods
like `__init__` and `__del__` on structs, which give control over the lifetime
of the logical value maintained by that struct. For example, consider a dynamic
string type that needs to allocate memory for the string data when constructed
and destroy it when the value is destroyed:

```mojo
    struct MyString:
        var data: Pointer[Int8]

        # StringRef has a data + length field
        def __init__(self&, input: StringRef):
            let data = Pointer[Int8].alloc(input.length+1)
            data.memcpy(first.data, input.length)
            data[input.length] = 0
            self.data = Pointer[Int8](data)

        def __del__(owned self):
            self.data.free()
```

This MyString type is implemented using low level functions to show a
simple example of how this works - a more realistic implementation would use
short string optimizations, etc.  However, if you go ahead and try this out, you
might be surprised:

```mojo
    fn useStrings():
        var a: MyString = "hello"
        print(a)   # Should print "hello"
        var b = a  # ERROR: MyString doesn't implement __copyinit__

        a = "Goodbye"
        print(b)   # Should print "hello"
        print(a)   # Should print "Goodbye"
```

The compiler isn’t allowing us to make a copy of our string: MyString contains
an instance of Pointer (which is equivalent to a low-level C pointer), and Mojo
can’t know “what the pointer means” or “how to copy it” - this is one reason
why application level programmers should use higher level types like arrays and
slices! More generally, some types (like atomic numbers) cannot be copied or
moved around at all, because their address provides an **identity** just like a
class instance does.

In this case, we do want our string to be copyable around, to enable this, we
implement the `__copyinit__` special method, which is conventionally
implemented like this:

```mojo
    struct MyString:
        ...
        def __copyinit__(self&, existing: Self):
            self.data = Pointer(strdup(self.data.address))
```

With this implementation, our code above works correctly and the “b = a” copy
produces a logically distinct instance of the string with its own lifetime and
data. The copy is made with the C strdup`()` function as instructed by the
lines of code above.  Mojo also supports the `__moveinit__` method which allows
both Rust-style moves (which take a value when a lifetime ends) and C++-style
moves (where the contents of a value is removed but the destructor still runs),
and allows defining custom move logic.  Please see the "Value Lifecycle"
document for more information.

Mojo provides full control over the lifetime of a value, including the ability
to make types copyable, move-only, and not-movable. This is more control than
languages like Swift and Rust, which require values to at least be movable. If
you are curious how `existing` can be passed into the `__copyinit__` method
without itself creating a copy, check out the section on "Borrowed" argument
convention below.

## Parameterization: Compile time meta-programming

One of Python’s most amazing features is its extensible runtime
meta-programming features. This has enabled a wide range of libraries and
provides a flexible and extensible programming model that Python programmers
everywhere benefit from. Unfortunately, these features also come at a cost:
because they are evaluated at runtime, they directly impact run-time efficiency
of the underlying code. Because they are not known to the IDE, it is difficult
for IDE features like code completion to understand them and use them to
improve the developer experience.

Outside the Python ecosystem, static meta-programming is also an important part
of development, enabling the development of new programming paradigms and
advanced libraries. There are many examples of prior art in this space, with
different tradeoffs, for example:

1. Preprocessors (e.g. C preprocessor, Lex/YACC, etc) are perhaps the heaviest
handed. They are fully general, but the worst in terms of developer experience
and tools integration.

2. Some languages (like Lisp and Rust) support (sometimes “hygienic”) macro
expansion features, enabling syntactic extension and boilerplate reduction with
somewhat better tooling integration.

3. Some older languages like C++ have very large and complex metaprogramming
languages (templates) that are a dual to the _runtime_ language. These are
notably difficult to learn and have poor compile times and error messages.

4. Some languages (like Swift) build many features into the core language in a
first class way to provide good ergonomics for common cases at the expense of
generality.

5. Some newer languages like Zig integrate a language interpreter into the
compilation flow, and allow the interpreter to reflect over the AST as it is
compiled. This allows many of the same features as a macro system with better
extensibility and generality.

For Modular’s work in AI, high performance machine learning kernels and
accelerators, we need high abstraction capabilities provided by advanced
metaprogramming systems. We needed high level zero-cost abstractions,
expressive libraries, and large scale integration of multiple variants of
algorithms. We want library developers to be able to extend the system, just
like they do in Python, providing an extensible developer platform.

That said, we are not willing to sacrifice developer experience (including
compile times and error messages) nor are we interested in building a parallel
language ecosystem that is difficult to teach. We can learn from these previous
systems, but also have new technologies to build on top of, including MLIR and
fine-grained language-integrated caching technologies.

As such, Mojo supports a full compile-time metaprogramming functionality built
into the compiler as a separate stage of compilation - after parsing, semantic
analysis, and IR generation, but before lowering to target-specific code. It
uses the same host language for runtime programs as it does for metaprograms,
and leverages MLIR to represent and evaluate these programs in a predictable
way.

Let’s take a look at some simple examples.

> Note on naming: after going around in circles on naming, we converged on
calling these things "parameters". Python programmers use the words
"arguments" and "parameters" fairly interchangeably as near-synonyms for
"things that get passed into functions". We have currently decided to reclaim
the word "parameter", "parameter expression" to mean compile time value, but
use "argument" and "expression" to refer to runtime values.  This allows us to
align around words like "parameterized" and "parametric".

### Defining parameterized types and functions

Mojo structs and functions may each be parameterized, but an example can help
motivate why we care. Let’s look at a
“[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)” type,
which represents a low-level vector register in hardware that holds multiple
instances of a scalar data-type. Hardware accelerators these days are getting
exotic datatypes, and it isn’t uncommon to work with CPUs that have 512-bit or
longer SIMD vectors. There is a lot of diversity in hardware (including many
brands like SSE, AVX-512, NEON, SVE, RVV, etc) but many operations are common
and used by numerics and ML kernel developers - this type exposes them to Mojo
programmers.

Here is a (cut down) version of the SIMD API in the Mojo standard library:

```mojo
    struct SIMD[type: DType, size: Int]:
        var value: … # Some low-level MLIR stuff here

        # Create a new SIMD from a number of scalars
        fn __init__(self&, *elems: SIMD[type, 1]):  ...

        # Fill a SIMD with a duplicated scalar value.
        @staticmethod
        fn splat(x: SIMD[type, 1]) -> SIMD[type, size]: ...

        # Cast the elements of the SIMD to a different elt type.
        fn cast[target: DType](self) -> SIMD[target, size]: ...

        # Many standard operators are supported.
        fn __add__(self, rhs: Self) -> Self: ...
```

Parameters in Mojo are declared in square brackets using an extended version of
the [PEP695 syntax](https://peps.python.org/pep-0695/). They are named and have
types like normal values in a Mojo program, but they are evaluated at compile
time instead of runtime by the target program. The runtime program may use the
value of parameters - because the parameters are resolved at compile time
before they are needed by the runtime program - but the compile time parameter
expressions may not use runtime values.

In the case of the `SIMD` excerpt above, there are three declared parameters:
the SIMD struct is parameterized by a ‘`type`’ parameter and a ‘`size`’
parameter. The ‘`cast`’ method is further parameterized with a ‘`target`’
parameter. Because SIMD is a parameterized type, the type of a ‘self’ argument
carries the parameters - the full type name is “`SIMD[type, size]`”. While it
is always valid to write this out (as shown in the return type of `splat`),
this can be verbose: we recommend using the `Self` type (from
[PEP673](https://peps.python.org/pep-0673/)) like the `__add__` example does.

### Using parameterized types and functions

For this type, ‘size’ specifies the number of elements in a SIMD vector and
the type specifies the element type - for example, you might use a “4xFloat” to
represent a small floating point vector, or a “32xbfloat16’s” on an AVX-512
system with the ‘bfloat16’ machine learning type:

```mojo
    fn funWithSIMD():
        # Make a vector of 4 floats.
        let smallVec = SIMD[DType.f32, 4](1.0, 2.0, 3.0, 4.0)

        # Make a big vector containing 1.0 in bfloat16 format.
        let bigVec = SIMD[DType.bf16, 32].splat(1.0)

        # Do some math and convert the elements to float32.
        let biggerVec = (bigVec+bigVec).cast[DType.f32]()

        # You can write types out explicitly if you want of course.
        let biggerVec2 : SIMD[DType.f32, 32] = biggerVec
```

Note that the “cast” method needs an additional parameter to indicate what type
to cast to: that is handled by parameterizing the call to “cast”. The example
above shows the use of concrete types, but the major power of parameters comes
from the ability to define parametric algorithms and types, e.g. it is quite
easy to define parametric algorithms, e.g. ones that are length- and
DType-agnostic:

```mojo
    fn rsqrt[width: Int, dt: DType](x: SIMD[dt, width]) -> SIMD[dt, width]:
       return 1 / sqrt(x)
```

The Mojo compiler is fairly smart about type inference with parameters. Note
that this function is able to call the parametric “`sqrt(x)`” function without
specifying the parameters, the compiler infers its parameters as if you wrote
“`sqrt[width,type](x)`” explicitly. Also note that “`rsqrt`” chose to define
its first parameter named “width” but the SIMD type names it “`size`” without
challenge.

### Parameter expressions are just Mojo code

All parameters and parameter expressions are typed using the same type system
as the runtime program: ‘Int’ and ‘DType’ are implemented in the Mojo standard
library as structs. Parameters are quite powerful, supporting the use of
expressions with operators, function calls etc at compile time, just like a
runtime program. This enables the use of many ‘dependent type’ features, for
example, you might want to define a helper function to concatenate two SIMD
vectors:

```mojo
    fn concat[ty: DType, len1: Int, len2: Int](
        lhs: SIMD[ty, len1], rhs: SIMD[ty, len2]) -> SIMD[ty, len1+len2]:
          ...

    fn use_vectors(a: SIMD[DType.f32, 4], b: SIMD[DType.f16, 8]):
        let x = concat(a, a)  # Length = 8
        let y = concat(b, b)  # Length = 16
```

Note how the result length is the sum of the input vector lengths, and you can
express that with a simple + operation.  For a more complex example, take a look
at the `SIMD.shuffle` method in the standard library: it takes two input SIMD
values, a vector shuffle mask as a list, and returns a SIMD that matches the
length of the shuffle mask.

### Powerful Compile-time Programming

While simple expressions are useful, sometimes you want to write imperative
compile-time logic with control flow. For example, the “isclose” function in
Math.mojo uses exact equality for integers but “close” comparison for floating
point. You can even do compile time recursion, e.g. here is an example “tree
reduction” algorithm that sums all elements of a vector recursively into a
scalar:

```mojo
    struct SIMD[type: DType, size: Int]:
        ...
        fn reduce_add(self) -> SIMD[type, 1]:
            @parameter
            if size == 1:
                return self[0]
            elif size == 2:
                return self[0] + self[1]

            # Extract the top/bottom halves, add them, sum the elements.
            let lhs = self.slice[size // 2](0)
            let rhs = self.slice[size // 2](size // 2)
            return (lhs + rhs).reduce_add()
```

This makes use of the “`@parameter if`” feature, which is an if statement that
runs at compile time. It requires that its condition be a valid parameter
expression, and ensures that only the live branch of the if is compiled into
the program.

### Mojo Types are just Parameter Expressions

While we’ve shown how you can use parameter expressions within types, in both
Python and Mojo, type annotations can themselves be arbitrary expressions.
Types in Mojo have a special metatype type, allowing type-parametric algorithms
and functions to be defined, for example one can define an algorithm like the
C++ `std::vector` class like this:

```mojo
    struct DynamicVector[type: AnyType]:
          ...
        fn reserve(self&, new_capacity: Int): ...
        fn push_back(self&, value: type): ...
        fn pop_back(self&): ...
        fn __getitem__(self, i: Int) -> type: ...
        fn __setitem__(self&, i: Int, value: type): ...

    fn use_vector():
        var v = DynamicVector[Int]()
        v.push_back(17)
        v.push_back(42)
        v[0] = 123
        print(v[1])      # Prints 42
        print(v[0])      # Prints 123
```

Notice that the ‘type’ parameter is being used as the formal type for the
‘value’ arguments and the return type of the `__getitem__` function. Parameters
allow the `DynamicVector` type to provide different APIs based on the different
use-cases. There are many other cases that benefit from more advanced use
cases. For example, the parallel processing library defines the
`parallelForEachN` algorithm, which executes a closure N times in parallel,
feeding in a value from the context. That value can be of any type:

```mojo
    fn parallelize[
        arg_type: AnyType,
        func: fn(Int, arg_type) -> None,
    ](rt: Runtime, num_work_items: Int, arg: arg_type):
        # Not actually parallel: see Functional.mojo for real impl.
        for i in range(num_work_items):
           func(i, arg)
```

This is possible because the ‘func’ parameter is allowed to refer to the
earlier ‘arg_type’ parameter, and that refines its type in turn.

Another example where this is important is with variadic generics, where an
algorithm or data structure may need to be defined over a list of heterogenous
types:

```mojo
    struct Tuple[*ElementTys: AnyType]:
       var _storage : *ElementTys
```

> Note: we don't have enough metatype helpers in place yet, but we should be
able to write something like this in the future, though overloading is still a
better way to handle this:

```mojo
struct Array[T: AnyType]:

    fn __getitem__[IndexType: AnyType](self, idx: IndexType)
       -> (ArraySlice[T] if issubclass(IndexType, Range) else T):
       ...

```

### `alias`: Named Parameter Expressions

It is very common to want to *name* compile time values. Whereas ‘var’ defines a
runtime value, and ‘let’ defines a runtime constant, we need a way to define a
compile time temporary value. For this, Mojo uses an ‘`alias`’ declaration. For
example, the DType struct implements a simple enum using aliases for the
enumerators like this (the actual internal implementation details vary a bit):

```mojo
    struct DType:
       var value : Int8
       alias invalid = DType(0)
       alias bool = DType(1)
       alias si8 = DType(2)
       alias ui8 = DType(3)
       alias si16 = DType(4)
       alias ui16 = DType(5)
       ...
       alias f32 = DType(15)
```

This allows clients to use ‘DType.f32’ as a parameter expression (which also
works as a runtime value of course) naturally. Note that this is invoking the
runtime constructor for DType at compile time.

Types are another common use for alias: because types are just compile time
expressions, it is very handy to be able to do things like this:

```mojo
    alias F32 = SIMD[DType.f32, 1]
    alias UI8 = SIMD[DType.ui8, 1]

    var x : F32   # F32 works like a "typedef"
```

Like ‘var’ and ‘let’, aliases obey scope and you can use local aliases within
functions as you’d expect.

### Autotuning / Adaptive compilation

Mojo parameter expressions allow you to write portable parametric algorithms
like you can do in other languages, but when writing high performance code you
still have to pick concrete values to use for the parameters. For example when
writing high performance numeric algorithms, you might want to use memory
tiling to accelerate the algorithm, but the dimensions to use depend highly on
the available hardware features, the sizes of the cache, what gets fused into
the kernel, and many other fiddly details.

Even vector length can be difficult to manage, because the vector length of a
typical machine depends on the datatype, and some datatypes like bfloat16 don’t
have full support on all implementations. Mojo helps by providing an autotune
function in the standard library. For example if you want to write a
vector-length-agnostic algorithm to a buffer of data, you might write it like
this:

```mojo
    def exp_buffer[dt: DType](data: ArraySlice[dt]):
        # Pick vector length for this dtype and hardware
        alias vector_len = autotune(4, 1, 8, 16, 32)

        # Use it as the vectorization length
        vectorize[exp[dt, vector_len]](data)
```

When compiling instantiations of this code Mojo forks compilation of this
algorithm and decides which value to use by measuring what works best in
practice for the target hardware. It evaluates the different values of the
`vector_len` expression and picks the fastest one according to a user-defined
performance evaluator. Because it measures and evaluates each option
individually, it might pick a different vector length for F32 than for SI8, for
example. This simple feature is pretty powerful - going beyond simple integer
constants - because functions and types are also parameter expressions.

Autotuning is an inherently exponential technique that benefits from internal
implementation details of the Mojo compiler stack (particularly MLIR,
integrated caching, and distribution of compilation). This is also a power-user
feature and needs continued development and iteration over time.

TODO: Write up evaluators when their design settles down a little bit.

## Argument Passing Control and Memory Ownership

In both Python and Mojo, much of the language revolves around function calls: a
lot of the (apparently) built-in functionality is implemented in the standard
library with “dunder” methods. Mojo takes this a step further than Python, by
putting the most basic things (like integers and the object type itself) into
the standard library.

### Why argument conventions are important

In Python all fundamental values are references to objects - a Python
programmer typically thinks about the programming model as everything being
reference semantic. However, at the CPython or machine level, we can see that
the references themselves are actually passed _by-copy_, by copying a pointer
and adjusting reference counts.

This approach provides a comfortable programming model (though which is
occasionally surprising due to reference sharing) but it requires all values to
be heap allocated. Mojo classes (TODO: will) follow the same reference-semantic
implementation approach as Python, but this isn’t practical for simpler types
like integers in a systems programming context. In these scenarios, you want
these values to live on the stack or even in hardware registers. As such, Mojo
structs are always inlined into their container, whether that be as the field
of another type or into the stack frame of the containing function.

This raises an interesting question: how do you implement methods that need to
mutate ‘self’ of a structure type, e.g. “`__iadd__"`? How does “`let"` work and
how does it prevent mutation? How are the lifetimes of these values controlled
to keep Mojo a memory safe language?

The answer is that the Mojo compiler uses dataflow analysis and type
annotations to provide full control over value copies, aliasing of references,
and mutation control. The features provided are similar in many ways to what
the Rust language provides, but they work somewhat differently in order to make
Mojo easier to learn and integrate better into the Python ecosystem without
requiring a massive annotation burden.

### By-Reference Arguments

Let’s start with the simple case: passing mutable references to values vs
passing immutable references. As we already know, arguments that are passed to
fn’s are immutable by default:

```mojo
struct Int:
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: Int) -> Int: ...

    # ... but this cannot work for __iadd__
    fn __iadd__(self, rhs: Int):
       self = self + rhs  # ERROR: cannot assign to self!
```

The problem here is that `__iadd__` needs to mutate the internal state of the
integer. The solution in Mojo is to declare that the argument is passed “by
reference” by using the & marker on the argument name (`self` in this case):

```mojo
struct Int:
    # ...
    fn __iadd__(self&, rhs: Int):
       self = self + rhs    # OK
```

Because this argument is passed by-reference, the ‘self’ argument is mutable in
the callee, and any changes are visible in the caller - even if the caller has
a non-trivial computation to access it, like an array subscript:

```mojo
fn show_mutation():
    var x = 42
    x += 1
    print(x)    # prints 43 of course

    var a = InlinedFixedVector[16, Int](...)
    a[4] = 7
    a[4] += 1    # Mutate an element within the InlinedFixedVector
    print(a[4])  # Prints 8

    let y = x
    y += 1       # ERROR: Cannot mutate 'let' value
```

Mojo implements the in-place mutation of the InlinedFixedVector element by
emitting a call to `__getitem__` into a temporary buffer, followed by a store
with `__setitem__` after the call. Mutation of the ‘let’ value fails because it
isn’t possible to form a mutable reference to an immutable value. Similarly,
the compiler rejects attempts to use a subscript with a by-ref argument if it
implements `__getitem__` but not `__setitem__`.

There is nothing special about ‘self’ in Mojo, and you can have multiple
different by-ref arguments. For example, you can define and use a swap function
like this:

```mojo
fn swap(lhs&: Int, rhs&: Int):
  let tmp = lhs
  lhs = rhs
  rhs = tmp

fn show_swap():
    var x = 42
    var y = 12
    swap(x, y)
    print(x)  # Prints 12
    print(y)  # Prints 42
```

A very important aspect of this system is that it all composes correctly.

> Alternative: instead of using the `&` sigil, we could call this an `inout`
argument.  Such a spelling would align better with other argument convention
keywords, and is more correct given how Mojo's computed LValues work.

### “Borrowed” Argument Convention

Now that we know how by-reference argument passing works, you may wonder how
by-value argument passing works and how that interacts with the `__copyinit__`
method which implements copy constructors. In Mojo, the default convention for
passing arguments to functions is to pass with the “borrowed” argument
convention. You can spell this out explicitly if you’d like:

```mojo
fn useSomethingBig(borrowed a: SomethingBig, b: SomethingBig):
  """'a' and 'b' are passed the same, because 'borrowed' is the default."""
  a.print_id()
  b.print_id()
```

This default applies to all arguments uniformly, including the `self` argument
of methods. The borrowed convention passes an _immutable reference_ to the
value from the caller’s context, instead of copying the value. This is much
more efficient when passing large values, or when passing expensive values like
a reference counted pointer (which is the default for Python/Mojo classes),
because the copy constructor and destructor don’t have to be invoked when
passing the argument. Here is a more elaborate example building on the code
above:

```mojo
# A type that is so expensive to copy around we don't even have a
# __copyinit__ method.
struct SomethingBig:
    var id_number: Int
    var huge: InlinedArray[Int, 100000]
    fn __init__(self&): …

    # self is passed by-reference for mutation as described above.
    fn set_id(self&, number: Int):
        self.id_number = number

    # Arguments like self are passed as borrowed by default.
    fn print_id(self):  # Same as: fn print_id(borrowed self):
        print(self.id_number)

fn try_something_big():
  # Big thing sits on the stack: after we construct it it cannot be
  # moved or copied.
  let big = SomethingBig()
  # We still want to do useful things with it though!
  big.print_id()
  # Do other things with it.
  useSomethingBig(big, big)
```

Because the default argument convention is borrowed, we get very simple and
logical code which does the right thing by default: for example, we don’t want
to copy or move all of SomethingBig just to invoke the “`print_id`” method, or
when calling `useSomethingBig`.

The borrowed convention is similar and has precedent to other languages. For
example, the borrowed argument convention is similar in some ways to passing an
argument by “`const&`” in C++. This avoids a copy of the value, and disables
mutability in the callee. The borrowed convention differs from “`const&`” in
C++ in two important ways though:

1. The Mojo compiler implements a borrow checker (similar to Rust) that
prevents code from dynamically forming mutable references to a value when there
are immutable references outstanding, and prevents having multiple mutable
references to the same value. You are allowed to have multiple borrows (as the
call to “`useSomethingBig`” does above) but cannot pass something by mutable
reference and borrow at the same time. (TODO: Not currently enabled).

2. Small values like “`Int`”, “`Float`”, and “`SIMD`” are passed directly in
machine registers instead of through an extra indirection (this is because they
are declared with the “`@register_passable`” decorator, see below). This is a
[significant performance
enhancement](https://www.forrestthewoods.com/blog/should-small-rust-structs-be-passed-by-copy-or-by-borrow/)
when compared to languages like C++ and Rust, and moves this optimization from
every call site to being declarative on a type.

Rust is another important language and the Mojo and Rust borrow checkers
enforce the same exclusivity invariants. The major difference between Rust and
Mojo is that no sigil is required on the caller side to pass by borrow, Mojo is
more efficient when passing small values, and Rust defaults to moving values by
default instead of passing them around by borrow. These policy and syntax
decisions allows Mojo to provide an arguably easier to use programming model.

### “Owned” Argument Convention and postfix `^` operator

The final argument convention that Mojo supports is the `owned` argument
convention.  This convention is used for functions that want to take exclusive
ownership over a value, and it is often used with the postfix `^` operator.

For example, consider working with a move-only type like a unique pointer, while
the borrow convention makes it easy to work with the unique pointer without
ceremony, at some point you may want to transfer ownership to some other
function.  This is what the `^` operator does:

```mojo
fn usePointer():
  let ptr = SomeUniquePtr(...)
  use(ptr)        # Perfectly fine to pass to borrowing function.
  use(ptr)
  take_ptr(ptr^)  # pass ownership of the `ptr` value to another function.

  use(ptr) # ERROR: ptr is no longer valid here!
```

For movable types, the `^` operator ends the lifetime of a value binding and
transfers the value to something else (in this case, the `take_ptr` function).
To support this, you can define functions as taking owned arguments, e.g. you
define `take_ptr` like so:

```mojo
fn take_ptr(owned p: SomeUniquePtr):
  use(p)
```

Because it is declared `owned`, the `take_ptr` function knows it has unique
access to the value.  This is very important for things like unique pointers,
can be useful to avoid copies, and is a generalization for other cases as well.

For example, you will notably see the `owned` convention on destructors and on
consuming move initializers, e.g., our `MyString` type from earlier my be
defined as:

```mojo
 struct MyString:
        var data: Pointer[Int8]

        # StringRef has a data + length field
        def __init__(self&, input: StringRef): ...
        def __copyinit__(self&, existing: Self): ...

        def __moveinit__(self&, owned existing: Self):
          self.data = existing.data

        def __del__(owned self):
            self.data.free()
```

This is because you need to own a value to destroy it or to steal its parts!

### `@register_passable` Struct Decorator

As described above, the default fundamental model for working with values is
that they live in memory so they have identity, which means they are passed
indirectly to and from functions (equivalently, they are passed ‘by reference’
at the machine level). This is great for types that cannot be moved, and is a
good safe default for large objects or things with expensive copy operations.
However, it is really inefficient for tiny things like a single integer or
floating point number!

To solve this, Mojo allows structs to opt-in to being passed in a register
instead of passing through memory with the `@register_passable` decorator.
You’ll see this decorator on types like `Int` in the standard library:

```mojo
@register_passable("trivial")
struct Int:
   var value: __mlir_type.`!pop.scalar<index>`

   fn __init__(value: __mlir_type.`!pop.scalar<index>`) -> Self:
       return Self {value: value}
   ...
```

The basic `@register_passable` decorator does not change the fundamental
behavior of a type: it still needs to have a `__copyinit__` method to be
copyable, may still have a `__init__` and `__del__` methods, etc. The major
effect of this decorator is on
internal implementation details: `@register_passable` types are typically
passed in machine registers (subject to the details of the underlying
architecture of course).

There are only a few observable effects of this decorator to the typical Mojo
programmer:

1. `@register_passable` types are not being able to hold instances of types
that are not themselves `@register_passable`.

2. instances of `@register_passable` types do not have predictable identity,
and so the ‘self’ pointer is not stable/predictable (e.g. in hash tables).

3. `@register_passable`arguments and result are exposed to C and C++ directly,
instead of being passed by-pointer.

4. The `__init__` and `__copyinit__` methods of this type are implicitly static
(like `__new__` in Python) and returns its result by-value instead of taking
`self&`.

We expect that this decorator will be used pervasively on core standard library
types, but is safe to ignore for general application level code.

The `Int` example above actually uses the "trivial" variant of this decorator.
It changes the passing convention as described above but also disallows copy
and move constructors and destructors (synthesizing them all trivially).

> TODO: Trivial needs to be decoupled to its own decorator since it applies to
memory types as well.

### How ‘`def`’ argument passing works

Argument passing in `def` functions is sugar for argument passing in `fn`:

1. If there is no explicit type annotation, the compiler defaults to type
`Object`.

2. Arguments with explicit markers (e.g. by reference or owned) obey their
   marker.

3. Arguments without an argument convention are passed by implicit copy into a
mutable var with the same name as the argument. Implicit copy requires that the
type have a `__copyinit__` method.

These functions are equivalent (other than keyword argument label to callers):

```mojo
def example(a&: Int, b: Int, c):
    ...

fn example(a&: Int, b_in: Int, c_in: Object):
    var b = b_in
    var c = c_in
    ...
```

As you can see, the ‘a’ argument with an explicit type and convention is
treated exactly as before. The typed ‘b’ argument maintains its type, but gets
a mutable shadow copy so the callee can modify the value inside the body of the
def. The ‘c’ argument gets an implicit Object type, and is mutable in the body.
These copies typically add no overhead, because small types like Object
references are cheap to copy. The expensive part is the reference count
adjustment, which is eliminated by a move optimization.

## Lifetimes

TODO: Explain how returning references work, tied into lifetimes which dovetail
with parameters.  This is not enabled yet.

## Type Traits

This is a feature very much like Rust traits or Swift protocols or Haskell type
classes. Note, this is not implemented yet.

## Advanced/Obscure Mojo Features

This section describes power-user features that are important for building the
bottom-est level of the standard library. This level of the stack is inhabited
by narrow features that require experience with compiler internals to
understand and utilize effectively.

TOWRITE: Each builtin decorator should be mentioned.  Eventually decorators
should appear in the API docs.

> TODO: We need to decide how to namespace these, should these go into a 'mojo'
package or something?

### `@always_inline` decorator

`@always_inline("nodebug")`: same thing but without debug information so you
don’t step into the + method on Int.

### Magic operators

C++ code has a number of magic operators that intersect with value lifecycle, things like "placement new", "placement delete" and "operator=" that reassign over an existing value.  Mojo is a safe language when you use all its language features and compose on top of safe constructs, but of any stack is a world of C-style pointers and rampant unsafety.  Mojo is a pragmatic language, and since we are interested in both interoperating with C/C++ and in implementing safe constructs like String directly in Mojo itself, we need a way to express unsafe things.

The Mojo standard library `Pointer[element_type]` type is implemented with an
underlying `!pop.pointer<element_type>` type in MLIR, and we desire a way to
implement these C++-equivalent unsafe constructs in Mojo.  Eventually these will
migrate to all being methods on the Pointer type, but until then, some need to
be exposed as builtin operators.

TODO: document all of these:

```
__get_address_as_lvalue(x)
__get_address_as_uninit_lvalue(x)
__get_lvalue_as_address(x):  use Pointer.address_of instead
__get_address_as_owned_value(x)
```

### Direct Access to MLIR

TOWRITE: Mojo is zero cost abstractions piled up, turtles all the way down to
MLIR.

How to use `__mlir_type`, `__mlir_op`, `__mlir_type` with some simple examples.

## Detailed Motivation and Related Work

Mojo started with the goal of bringing an innovative programming model to
accelerators and other heterogeneous systems that are pervasive in machine
learning. That said, one of the most important and prevalent “accelerators” is
actually the host CPU. These CPUs are getting lots of tensor-core-like
accelerator blocks and other dedicated AI acceleration units, but they also
importantly serve as the “fall back” to support operations the accelerators
don’t. This includes tasks like data loading, pre- and post-processing, and
integrations with foreign systems written (e.g.) in C++.

As such, it became clear that we couldn’t build a limited accelerator language
that targets a narrow subset of the problem (e.g. just work for tensors). We
needed to support the full gamut of general purpose programming. At the same
time, we didn’t see a need to innovate in syntax or community, and so we
decided to embrace and complete the Python ecosystem.

### Why Python?

Python is the dominant force in both the field ML and also countless other
fields. It is easy to learn, known by important cohorts of programmers (e.g.
data scientists), has an amazing community, has tons of valuable packages, and
has a wide variety of good tooling. Python supports development of beautiful
and expressive APIs through its dynamic programming features.

Arguably, machine learning is what really propelled Python to being such a
dominant programming language. This happened when frameworks like TensorFlow
and PyTorch embraced it as a frontend to their high-performance runtimes
implemented in C++. This was a good decision, because it opened the door for a
wide range of data scientists and researchers to work in the AI space with high
productivity, and relative ease of entry.

For Modular today, Python is a non-negotiable part of our API surface stack -
this is dictated by our customers. Given that everything else in our stack is
negotiable, it stands to reason that we should start from a “Python First”
approach.

More subjectively, we feel that Python is a beautiful language - designed with
simple and composable abstractions, eschews needless punctuation that is
redundant-in-practice with indentation, and built with powerful (dynamic)
metaprogramming features that are a runway to extend to what we need for
Modular. We hope that those in the Python ecosystem see our new direction as
taking Python ahead to the next level - completing it - instead of trying to
compete with it.

### What’s wrong with Python?

Python has well known problems - most obviously, poor low-level performance and
cpython implementation decisions like the GIL. While there are many active
projects underway to improve these obvious challenges, the issues brought by
Python go deeper and particularly impact the AI field. Instead of talking about
those technical limitations, we’ll talk about the implications of these
limitations here in 2023.

Note that everywhere we refer to Python in this section is referring to the
cpython implementation. We'll talk about other implementations in a bit.

#### The Two-World Problem

For a variety of reasons, Python isn’t suitable for systems programming.
Fortunately, Python has amazing strengths as a glue layer, and low-level
bindings to C and C++ allow building libraries in C, C++ and many other
languages with better performance characteristics. This is what has enabled
things like numpy, TensorFlow and PyTorch and a vast number of other libraries
in the ecosystem.

Unfortunately, while this approach is an effective way to building high
performance Python libraries, its approach comes with a cost: building these
hybrid libraries is very complicated, requiring low-level understanding of the
internals of cpython, requires knowledge of C/C++/… programming (undermining
one of the original goals of using Python in the first place), makes it
difficult to evolve large frameworks, and (in the case of ML) pushes the world
towards “graph based” programming models which have worse fundamental usability
than “eager mode” systems. TensorFlow was an exemplar of this, but much of the
effort in PyTorch 2 is focused around discovering graphs to enable more
aggressive compilation methods.

Beyond the fundamental nature of the two-world problem in terms of system
complexity, it makes everything else in the ecosystem more complicated.
Debuggers generally can’t step across Python and C code, and those that can
aren’t widely accepted. Package ecosystems need to deal with C/C++ code etc
instead of a single world. Projects like PyTorch with significant C++
investments are intentionally trying to move more of their codebase to Python
because they know it gains usability.

#### The Three-World and N-World Problem

The two-world problem is commonly felt across the Python ecosystem, but things
are even worse for developers of machine learning frameworks. AI is pervasively
accelerated, and those accelerators use bespoke programming languages like
CUDA. While CUDA is a relative of C++, it has its own special problems and
limitations, and does not have consistent tools like debuggers or profilers. It
is also effectively vendor locked to a single hardware maker!

The AI world has an incredible amount of innovation on the hardware front, and
as a consequence, complexity is spiraling out of control. There are now many
attempts to build limited programming systems for accelerators (OpenCL, Sycl,
OneAPI, …) as well as use aspects of Python syntax to make accelerator
programming look like Python (OpenAI’s Triton and many others). This complexity
explosion is continuing to increase and none of these systems solve the
fundamental fragmentation in tools and ecosystem that is hurting the industry
so badly.

#### Mobile and Server Deployment

Another challenge for the Python ecosystem is one of deployment. There are many
facets to this, including folks who want to carefully control dependencies,
some folks prefer to be able to deploy hermetically compiled “a.out” files, and
multithreading and performance are also very important. These are areas where
we would like to see the Python ecosystem take steps forward.

### Attempts to “fix” Python

There are many many approaches to fix Python, including recent work to speed up
Python and replace the GIL, languages that look like Python but are subsets of
it, and embedded DSLs that integrate with Python but that are not first class
languages. While we cannot do an exhaustive list of all the efforts, we can
talk about some of the challenges in these areas, and why they aren’t suitable
for Modular’s use.

#### Improving CPython and JIT compiling Python

Recently, significant energy has been put into improving CPython performance
and other implementation issues, and this is showing huge results for the
community. This work is fantastic because it incrementally improves the current
CPython implementation. Python 3.11 has delivered improvements of 10-60% faster
than Python 3.10 through internal improvements, and [Python
3.12](https://github.com/faster-cpython/ideas/wiki/Python-3.12-Goals) aims to
go further with a trace optimizer. Many other projects are attempting to tame
the GIL, and projects like PyPy (among many others) have used JIT compilation
and tracing approaches to speed up Python.

These are great efforts, but are not helpful in getting a unified language onto
an accelerator. Many accelerators these days support very limited dynamic
features, and often do so with terrible performance. Furthermore, systems
programmers don’t just seek “performance” they also typically want a lot of
“**predictability and control**” over how a computation happens.

While we are a fan of these approaches, and feel they are valuable and exciting
to the community, they unfortunately do not satisfy our needs. We are looking
to eliminate the need to use C or C++ within Python libraries, we seek the
highest performance possible, and we cannot accept dynamic features at all in
some cases, so these approaches don’t help.

#### Python Subsets and other Python-like Languages

There are many attempts to build a “deployable” Python, one example is
TorchScript from the PyTorch project. These are useful in that they often
provide low-dependence deployment solutions, reduce dynamic features, and
sometimes have high performance. Because they use the base Python syntax, they
can be easier to learn than a novel language.

On the other hand, these languages have not seen wide adoption - because they
are a subset, they generally don’t interoperate with the Python ecosystem, do
not have fantastic tooling (e.g. debuggers), and often change out inconvenient
behavior in Python (e.g. infinite precision integers) unilaterally, which
breaks compatibility and fragments the ecosystem.

The challenges with these approaches is that they attempt to solve a weak point
of Python, but aren’t as good at Python’s strong points. At best, these can
provide a new alternative to C and C++, but without solving the dynamic use
cases of Python, they cannot solve the “two world problem”. This approach
drives fragmentation, and incompatibility makes _migration_ difficult to
impossible - recall how challenging the **Python 2** to **Python 3** migration
was.

#### Embedded DSLs in Python

Another common approach is to build an embedded DSL in Python, typically
installed with a Python decorator. There are many examples of this, e.g. the
`@tf.function` decorator in TensorFlow, the `@triton.jit` in OpenAI’s Triton
programming model, etc. A major benefit of these systems is that they maintain
compatibility with all of the Python ecosystem tooling, and integrate natively
into Python logic, allowing an embedded mini language to co-exist with the
strengths of Python for dynamic use cases.

Unfortunately, the embedded mini-languages provided by these systems often have
surprising limitations, don’t integrate well with debuggers and other workflow
tooling, and do not support the level of native language integration that we
seek for a language that unifies heterogeneous compute and is the primary way
to write large scale kernels and systems. We hope to move the usability of the
overall system forward by simplifying things and making it more consistent.
Embedded DSLs are an expedient way to get demos up and running, but we are
willing to put in the additional effort and work to provide better usability
and predictability for our use-case.
