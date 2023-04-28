---
title: Mojo🔥 programming manual
---

Mojo is a first-class programming language that utilizes a number of
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

## Using the Mojo Compiler

The simplest way to run a mojo program is to type "mojo hello.🔥", for
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

Given our goal of compatibility and Python's strength with high-level
applications and dynamic APIs, we don't have to spend much time talking about
how those portions of the language works. On the other hand, Python's support
for systems programming is mostly delegated to C, and we want to provide a
single system that is great in that world. As such, this section breaks down
each major component and feature and describes how to use them with examples.

### `let` and `var` declarations

Inside a `def` in Mojo, you may assign a value to a name and it implicitly
creates a function scope variable just like in Python. This provides a very
dynamic and low-ceremony way to write code, but it is a challenge for two
reasons:

1) Systems programmers often want to declare that a value is immutable.
2) They may want to get an error if they mistype a variable name in an
assignment.

To support this, Mojo provides scoped runtime value declarations: `let` is
immutable and `var` is mutable. These values use lexical scoping and support
name shadowing:

```mojo
def your_function(a, b):
    let c = a
    c = b  # error: c is immutable

    if c != b:
        var c = b
        stuff()
```

`let` and `var` declarations support type specifiers as well as patterns, and
late initialization:

```mojo
def your_function():
    let x: Int8 = 42
    let y: Int64 = 17

    let z: Int8
    if x != 0:
        z = 1
    else:
        z = foo()
    use(z)
```

Note that `let` and `var` are completely opt-in when in `def` declarations. You
can still use implicitly declared values as with Python, and they get
function scope as usual.

### `struct` Types

Mojo builds on MLIR and LLVM which provide a modern compiler code generation
stack that powers many systems programming languages. This allows us to expose
low-level control over data layout, indirection-free access to fields, as well
as low-level tricks for bit-swizzling and other niche tricks. One extremely
important feature of modern systems programming languages is the ability to
build high-level and safe abstractions on top of these low-level shenanigans,
with zero abstraction penalties. In Mojo, this is provided by the `struct`
type.

A `struct` in Mojo is similar in many ways to a Python `class`: they both support
methods, fields, operator overloading, decorators for meta programming, etc. On
the other hand, where classes are extremely dynamic with dynamic dispatch,
dynamic method swizzling, and dynamically bound instance properties, structs
are static, bound at compile time, and are stored inlined into their container
instead of being implicitly indirect and reference-counted. This approach has
precedent in other languages such as Swift, C# and others.

Here's a simple definition of a struct:

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

The biggest difference compared to a Python `class` is that all instance properties in
a `struct` **must** be explicitly declared with a `var` or `let` declaration.
This allows the Mojo compiler to layout and access property values
precisely in memory without indirection or other overhead.

Struct fields are bound statically: they aren't looked up in a dictionary.
As such, you cannot `del` a method or reassign it at runtime. This
enables the Mojo compiler to perform guaranteed static dispatch, use guaranteed
static access to fields, and inline a struct into the stack frame or enclosing
type that uses it without indirection or other overheads.

Both Python and Mojo focus on enabling
expressive API design, and push complexity from the language itself into the
package ecosystem. Structs in Mojo supercharge this capability by providing
zero-cost abstraction capabilities typically seen in languages like C++, Swift,
Rust, and Zig, which compose beautifully with the operator overloading and
other features that Python has supported for years. These capabilities compose
to allow *all* the "standard types" (like `Int`, `Bool`, `String` and even
`Tuple`) to be implemented as structs in the standard library instead of being
built into the language/compiler.

:::{.callout-note}

If you're wondering what the `&` means on the `self` argument: this
indicates that the value is mutable, which is explained below in
[By-reference arguments](#by-reference-arguments).

:::

#### `Int` vs `int`

You might note that Mojo uses a capital `Int` type defined in its standard
library, which differs in case from the lower-case `int` that is used by
[MyPy](https://mypy.readthedocs.io/). This is intentional, and a good thing.
The Mojo standard library `Int` is defined to be a fixed width integer sized to
match the CPU register (like `ssize_t` in C). In contrast, the Python `int`
type is a boxed object that supports arbitrary precision arithmetic, and has a
somewhat broader API - e.g. support for object identity tests.

Mojo does this for two reasons:

1. We need to expose a simple and predictable programming model to systems
programmers and give them full control over the hardware. We cannot afford to
rely on fancy compiler analysis to "devirtualize" common cases.

2. Mojo needs to grow into a full superset of Python to avoid breaking the
ecosystem. We cannot do that by changing the behavior of core types like the
integers that everyone uses. By using different names, we can support both
types in the same system.

As an additional minor point, `Int` is just a struct built into the standard
library type, so it is nice that it can follow the naming conventions of user
defined types.


### Strong type checking

Although you can still use dynamic types just like in Python, Mojo also allows
you to use strong type checking in your program. This should be familiar to
any systems programmer as it provides predictability, control, and safety for
your code.

One of the primary ways to employ strong type checking is with Mojo's `struct`
type. A `struct` definition in Mojo defines a compile-time-bound name, and
references to that name in a type context are treated as a strong specification
for the value being defined. For example, consider the following code that uses
the `MyPair` struct shown above:

```mojo
def pairTest() -> Bool:
    let p = MyPair(1, 2)
    return p < 4 # gives a compile time error
```

When you run this code, you'll get a compile time error telling you that "4"
cannot be converted to `MyPair`, which is what the RHS of `MyPair.__lt__`
requires.

This is a familiar experience when working with systems programming languages,
but it's not how Python works. Python has a syntactically identical feature for
[MyPy](https://mypy.readthedocs.io/) type annotations, but they are not
enforced by the compiler: instead they are hints that inform static analysis.
By tying types to specific declarations, Mojo is able to handle both the
classical type annotation hints as well as the strong type specifications
without breaking compatibility.

Beyond type checking, strong types are also very important for code generation.
Because we know that these types are correct, we can specialize on the types,
pass values in registers, and generally be as efficient as C for argument
passing and other low level details. This also is the foundation of the safety
and predictability guarantees that Mojo provides to systems programmers.


### Overloaded functions & methods

Also like Python, you can define functions in Mojo without specifying
argument data types and let Mojo infer them. This is nice when you want
expressive APIs that just work by accepting arbitrary inputs and let dynamic
dispatch decide how to handle the data. However, when you want to ensure
type safety as discussed above, Mojo also offers full support for overloaded
functions and methods.

Essentially, this allows you to define multiple functions with the same name
but with different arguments. This is a common feature seen in many languages
such as C++, Java, and Swift.

When resolving a function call, Mojo tries each candidate and use the one that
works (if only one works), or it picks the closest match (if it can determine a
close match), or it reports that the call as ambiguous if it can't figure
out which one to pick. In the latter case, you can resolve the ambiguity by
adding an explicit cast on the call site. Let's look at an example:

```mojo
struct Array[T: AnyType]:
    fn __getitem__(self, idx: Int) -> T: ...
    fn __getitem__(self, idx: Range) -> ArraySlice: ...
```

You can overload methods in structs and classes, and overload module-level
functions.

Mojo doesn't support overloading solely on result type, and doesn't use result
type or contextual type information for type inference, keeping things simple,
fast, and predictable.  Mojo will never produce an "expression too complex"
error, because its type-checker is simple and fast by definition.

Again, if you leave your argument names without type definitions, then the
function behaves just like Python with dynamic types. As soon as you define a
single argument type, Mojo will look for overload candidates and resolve
function calls as described above.

### `fn` Definitions

The extensions above are the cornerstone that provides low-level programming
and provide abstraction capabilities, but many systems programmers prefer more
control and predictability than what 'def' in Mojo provides. To recap, 'def' is
defined by necessity to be very dynamic, flexible and generally compatible with
Python: arguments are mutable, local variables are implicitly declared on first
use, and scoping isn't enforced. This is great for high level programming and
scripting, but is not always great for systems programming. To complement this,
Mojo provides an `fn` declaration which is like a "strict mode" for `def`.

> Alternative: instead of using a new keyword like fn, we could instead add a
modifier or decorator like '@strict def'. However, we need to take new keywords
anyway and there is little cost to doing so. Also, in practice in systems
programming domains, 'fn' is used all the time so it probably makes sense to
make it first class.

`fn` and `def` are always interchangeable from an interface level: there is
nothing a 'def' can provide that a `fn` cannot (or vice versa). The
difference is that a `fn` is more limited and controlled on the *inside* of
its body (alternatively: pedantic and strict). Specifically, `fn`s have a
number of limitations compared to `def`s:

1. Argument values default to being immutable in the body of the function (like
a `let`), instead of mutable (like a `var`). This catches accidental mutations,
and permits the use of non-copyable types as arguments.

2. Argument values require a type specification (except for `self` in a
method), catching accidental omission of type specifications. Similarly, a
missing return type specifier is interpreted as returning `None` instead of an
unknown return type. Note that both can be explicitly declared to return
"`object`", which allows one to opt-in to the behavior of a `def` if desired.

3. Implicit declaration of local variables is disabled, so all locals must be
declared. This catches name typos and dovetails with the scoping provided by
`let` and `var`.

4. Both support raising exceptions, but this must be explicitly declared on a
`fn` with the `raises` keyword.

Programming patterns will vary widely across teams, and this level of
strictness will not be for everyone. We expect that folks who are used to C++
and already use MyPy-style type annotations in Python to prefer the use of
`fn`s, but higher level programmers and ML researchers to continue to use
`def`. Mojo allows you to freely intermix `def` and `fn` declarations, e.g.
implementing some methods with one and others with the other, and allows each
team or programmer to decide what is best for their use-case.

### The `__copyinit__` and `__moveinit__` Special Methods

Mojo supports full "value semantics" as seen in languages like C++ and Swift,
and it makes defining simple aggregates of fields very easy with its `@value`
decorator (described in more detail below).

For advanced use cases, Mojo allows you to define custom constructors (using
Python's existing `__init__` special method), custom destructors (using the
existing `__del__` special method) and custom copy and move constructors using
the new `__copyinit__` and `__moveinit__` special methods.

These low-level customization hooks can be useful when doing low level systems
programming, e.g. with manual memory management.  For example, consider a
dynamic string type that needs to allocate memory for
the string data when constructed and destroy it when the value is destroyed:

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

This `MyString` type is implemented using low level functions to show a
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

The compiler isn't allowing us to make a copy of our string: `MyString` contains
an instance of `Pointer` (which is equivalent to a low-level C pointer), and
Mojo can't know "what the pointer means" or "how to copy it" - this is one reason
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

With this implementation, our code above works correctly and the "b = a" copy
produces a logically distinct instance of the string with its own lifetime and
data. The copy is made with the C strdup`()` function as instructed by the
lines of code above.  Mojo also supports the `__moveinit__` method which allows
both Rust-style moves (which take a value when a lifetime ends) and C++-style
moves (where the contents of a value is removed but the destructor still runs),
and allows defining custom move logic.  Please see the "Value Lifecycle"
section below for more information.

Mojo provides full control over the lifetime of a value, including the ability
to make types copyable, move-only, and not-movable. This is more control than
languages like Swift and Rust, which require values to at least be movable. If
you are curious how `existing` can be passed into the `__copyinit__` method
without itself creating a copy, check out the section on "Borrowed" argument
convention below.

## Parameterization: Compile time meta-programming

One of Python's most amazing features is its extensible runtime
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

2. Some languages (like Lisp and Rust) support (sometimes "hygienic") macro
expansion features, enabling syntactic extension and boilerplate reduction with
somewhat better tooling integration.

3. Some older languages like C++ have very large and complex metaprogramming
languages (templates) that are a dual to the *runtime* language. These are
notably difficult to learn and have poor compile times and error messages.

4. Some languages (like Swift) build many features into the core language in a
first class way to provide good ergonomics for common cases at the expense of
generality.

5. Some newer languages like Zig integrate a language interpreter into the
compilation flow, and allow the interpreter to reflect over the AST as it is
compiled. This allows many of the same features as a macro system with better
extensibility and generality.

For Modular's work in AI, high performance machine learning kernels and
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

Let's take a look at some simple examples.

> Note on naming: after going around in circles on naming, we converged on
calling these things "parameters". Python programmers use the words
"arguments" and "parameters" fairly interchangeably as near-synonyms for
"things that get passed into functions". We have currently decided to reclaim
the word "parameter", "parameter expression" to mean compile time value, but
use "argument" and "expression" to refer to runtime values.  This allows us to
align around words like "parameterized" and "parametric".

### Defining parameterized types and functions

Mojo structs and functions may each be parameterized, but an example can help
motivate why we care. Let's look at a
"[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)" type,
which represents a low-level vector register in hardware that holds multiple
instances of a scalar data-type. Hardware accelerators these days are getting
exotic datatypes, and it isn't uncommon to work with CPUs that have 512-bit or
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
the SIMD struct is parameterized by a `type` parameter and a `size`
parameter. The `cast` method is further parameterized with a `target`
parameter. Because SIMD is a parameterized type, the type of a 'self' argument
carries the parameters - the full type name is "`SIMD[type, size]`". While it
is always valid to write this out (as shown in the return type of `splat`),
this can be verbose: we recommend using the `Self` type (from
[PEP673](https://peps.python.org/pep-0673/)) like the `__add__` example does.

### Using parameterized types and functions

For this type, 'size' specifies the number of elements in a SIMD vector and
the type specifies the element type - for example, you might use a "4xFloat" to
represent a small floating point vector, or a "32xbfloat16's" on an AVX-512
system with the 'bfloat16' machine learning type:

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

Note that the "cast" method needs an additional parameter to indicate what type
to cast to: that is handled by parameterizing the call to "cast". The example
above shows the use of concrete types, but the major power of parameters comes
from the ability to define parametric algorithms and types, e.g. it is quite
easy to define parametric algorithms, e.g. ones that are length- and
DType-agnostic:

```mojo
fn rsqrt[width: Int, dt: DType](x: SIMD[dt, width]) -> SIMD[dt, width]:
    return 1 / sqrt(x)
```

The Mojo compiler is fairly smart about type inference with parameters. Note
that this function is able to call the parametric `sqrt(x)` function without
specifying the parameters, the compiler infers its parameters as if you wrote
`sqrt[width,type](x)` explicitly. Also note that `rsqrt` chose to define
its first parameter named "width" but the SIMD type names it `size` without
challenge.

### Parameter expressions are just Mojo code

All parameters and parameter expressions are typed using the same type system
as the runtime program: 'Int' and 'DType' are implemented in the Mojo standard
library as structs. Parameters are quite powerful, supporting the use of
expressions with operators, function calls etc at compile time, just like a
runtime program. This enables the use of many 'dependent type' features, for
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
compile-time logic with control flow. For example, the "isclose" function in
Math.mojo uses exact equality for integers but "close" comparison for floating
point. You can even do compile time recursion, e.g. here is an example "tree
reduction" algorithm that sums all elements of a vector recursively into a
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

This makes use of the `@parameter if` feature, which is an if statement that
runs at compile time. It requires that its condition be a valid parameter
expression, and ensures that only the live branch of the if is compiled into
the program.

### Mojo Types are just Parameter Expressions

While we've shown how you can use parameter expressions within types, in both
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

Notice that the 'type' parameter is being used as the formal type for the
'value' arguments and the return type of the `__getitem__` function. Parameters
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

This is possible because the 'func' parameter is allowed to refer to the
earlier 'arg_type' parameter, and that refines its type in turn.

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

It is very common to want to *name* compile time values. Whereas `var` defines a
runtime value, and `let` defines a runtime constant, we need a way to define a
compile time temporary value. For this, Mojo uses an `alias` declaration. For
example, the `DType` struct implements a simple enum using aliases for the
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

This allows clients to use `DType.f32` as a parameter expression (which also
works as a runtime value of course) naturally. Note that this is invoking the
runtime constructor for DType at compile time.

Types are another common use for alias: because types are just compile time
expressions, it is very handy to be able to do things like this:

```mojo
alias F32 = SIMD[DType.f32, 1]
alias UI8 = SIMD[DType.ui8, 1]

var x : F32   # F32 works like a "typedef"
```

Like `var` and `let`, aliases obey scope and you can use local aliases within
functions as you'd expect.

### Autotuning / Adaptive compilation

Mojo parameter expressions allow you to write portable parametric algorithms
like you can do in other languages, but when writing high performance code you
still have to pick concrete values to use for the parameters. For example when
writing high performance numeric algorithms, you might want to use memory
tiling to accelerate the algorithm, but the dimensions to use depend highly on
the available hardware features, the sizes of the cache, what gets fused into
the kernel, and many other fiddly details.

Even vector length can be difficult to manage, because the vector length of a
typical machine depends on the datatype, and some datatypes like `bfloat16`
don't have full support on all implementations. Mojo helps by providing an
`autotune` function in the standard library. For example if you want to write a
vector-length-agnostic algorithm to a buffer of data, you might write it like
this:

```mojo
from Autotune import autotune

def exp_buffer_impl[dt: DType](data: ArraySlice[dt]):
    # Pick vector length for this dtype and hardware
    alias vector_len = autotune(1, 4, 8, 16, 32)

    # Use it as the vectorization length
    vectorize[exp[dt, vector_len]](data)
```

When compiling instantiations of this code, Mojo forks compilation of this
algorithm and decides which value to use by measuring what works best in
practice for the target hardware. It evaluates the different values of the
`vector_len` expression and picks the fastest one according to a user-defined
performance evaluator. Because it measures and evaluates each option
individually, it might pick a different vector length for F32 than for SI8, for
example. This simple feature is pretty powerful - going beyond simple integer
constants - because functions and types are also parameter expressions.

Users can instrument the search of `exp_buffer_impl` by providing a performance
evaluator and using the `search` standard library function. `search` takes an
evaluator and a forked function and returns the fastest implementation selected
by the evaluator as a parameter result.

```mojo
from Autotune import search

fn exp_buffer[dt: DType](data: ArraySlice[dt]):
    # Forward declare the result parameter.
    alias best_impl: fn(ArraySlice[dt]) -> None

    # Perform search!
    search[
      fn(ArraySlice[dt]) -> None,
      exp_buffer_impl[dt],
      exp_evaluator[dt] -> best_impl
    ]()

    # Call the selected implementation
    best_impl(data)
```

In this example, we provided `exp_evaluator` to the search function as the
performance evaluator. Performance evaluators are invoked with a list of
candidate functions and should return the index of the best one. Mojo's
standard library provides a `Benchmark` module that you can use to time
functions.

```mojo
from Benchmark import Benchmark

fn exp_evaluator[dt: DType](
    fns: Pointer[fn(ArraySlice[dt]) -> None],
    num: Int
):
    var best_idx = -1
    var best_time = -1
    for i in range(num):
        candidate = fns[i]
        let buf = Buffer[dt]()

        # Benchmark this candidate.
        fn setup():
            buf.fill_random()
        fn wrapper():
            candidate(buf)
        let cur_time = Benchmark(2).run[wrapper, setup]()

        # Track the index of the fastest candidate.
        if best_idx < 0:
            best_idx = i
            best_time = cur_time
        elif best_time > cur_time:
            best_idx = f_idx
            best_time = cur_time

    # Return the fastest implementation.
    return best_idx
```

Autotuning is an inherently exponential technique that benefits from internal
implementation details of the Mojo compiler stack (particularly MLIR,
integrated caching, and distribution of compilation). This is also a power-user
feature and needs continued development and iteration over time.

## Argument Passing Control and Memory Ownership

In both Python and Mojo, much of the language revolves around function calls: a
lot of the (apparently) built-in functionality is implemented in the standard
library with "dunder" methods. Mojo takes this a step further than Python, by
putting the most basic things (like integers and the object type itself) into
the standard library.

### Why argument conventions are important

In Python all fundamental values are references to objects - a Python
programmer typically thinks about the programming model as everything being
reference semantic. However, at the CPython or machine level, we can see that
the references themselves are actually passed *by-copy*, by copying a pointer
and adjusting reference counts.

This approach provides a comfortable programming model (though which is
occasionally surprising due to reference sharing) but it requires all values to
be heap allocated. Mojo classes (TODO: will) follow the same reference-semantic
implementation approach as Python, but this isn't practical for simpler types
like integers in a systems programming context. In these scenarios, you want
these values to live on the stack or even in hardware registers. As such, Mojo
structs are always inlined into their container, whether that be as the field
of another type or into the stack frame of the containing function.

This raises an interesting question: how do you implement methods that need to
mutate 'self' of a structure type, e.g. "`__iadd__"`? How does "`let"` work and
how does it prevent mutation? How are the lifetimes of these values controlled
to keep Mojo a memory safe language?

The answer is that the Mojo compiler uses dataflow analysis and type
annotations to provide full control over value copies, aliasing of references,
and mutation control. The features provided are similar in many ways to what
the Rust language provides, but they work somewhat differently in order to make
Mojo easier to learn and integrate better into the Python ecosystem without
requiring a massive annotation burden.

### By-Reference Arguments

Let's start with the simple case: passing mutable references to values vs
passing immutable references. As we already know, arguments that are passed to
`fn`'s are immutable by default:

```mojo
struct Int:
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: Int) -> Int: ...

    # ... but this cannot work for __iadd__
    fn __iadd__(self, rhs: Int):
        self = self + rhs  # ERROR: cannot assign to self!
```

The problem here is that `__iadd__` needs to mutate the internal state of the
integer. The solution in Mojo is to declare that the argument is passed "by
reference" by using the `&` marker on the argument name (`self` in this case):

```mojo
struct Int:
    # ...
    fn __iadd__(self&, rhs: Int):
        self = self + rhs    # OK
```

Because this argument is passed by-reference, the 'self' argument is mutable in
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
with `__setitem__` after the call. Mutation of the `let` value fails because it
isn't possible to form a mutable reference to an immutable value. Similarly,
the compiler rejects attempts to use a subscript with a by-ref argument if it
implements `__getitem__` but not `__setitem__`.

There is nothing special about 'self' in Mojo, and you can have multiple
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

### "Borrowed" Argument Convention

Now that we know how by-reference argument passing works, you may wonder how
by-value argument passing works and how that interacts with the `__copyinit__`
method which implements copy constructors. In Mojo, the default convention for
passing arguments to functions is to pass with the "borrowed" argument
convention. You can spell this out explicitly if you'd like:

```mojo
fn use_something_big(borrowed a: SomethingBig, b: SomethingBig):
    """'a' and 'b' are passed the same, because 'borrowed' is the default."""
    a.print_id()
    b.print_id()
```

This default applies to all arguments uniformly, including the `self` argument
of methods. The borrowed convention passes an *immutable reference* to the
value from the caller's context, instead of copying the value. This is much
more efficient when passing large values, or when passing expensive values like
a reference counted pointer (which is the default for Python/Mojo classes),
because the copy constructor and destructor don't have to be invoked when
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
    use_something_big(big, big)
```

Because the default argument convention is borrowed, we get very simple and
logical code which does the right thing by default: for example, we don't want
to copy or move all of SomethingBig just to invoke the "`print_id`" method, or
when calling `use_something_big`.

The borrowed convention is similar and has precedent to other languages. For
example, the borrowed argument convention is similar in some ways to passing an
argument by "`const&`" in C++. This avoids a copy of the value, and disables
mutability in the callee. The borrowed convention differs from "`const&`" in
C++ in two important ways though:

1. The Mojo compiler implements a borrow checker (similar to Rust) that
prevents code from dynamically forming mutable references to a value when there
are immutable references outstanding, and prevents having multiple mutable
references to the same value. You are allowed to have multiple borrows (as the
call to "`use_something_big`" does above) but cannot pass something by mutable
reference and borrow at the same time. (TODO: Not currently enabled).

2. Small values like "`Int`", "`Float`", and "`SIMD`" are passed directly in
machine registers instead of through an extra indirection (this is because they
are declared with the "`@register_passable`" decorator, see below). This is a
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

### "Owned" Argument Convention and postfix `^` operator

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
indirectly to and from functions (equivalently, they are passed 'by reference'
at the machine level). This is great for types that cannot be moved, and is a
good safe default for large objects or things with expensive copy operations.
However, it is really inefficient for tiny things like a single integer or
floating point number!

To solve this, Mojo allows structs to opt-in to being passed in a register
instead of passing through memory with the `@register_passable` decorator.
You'll see this decorator on types like `Int` in the standard library:

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
and so the 'self' pointer is not stable/predictable (e.g. in hash tables).

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

### How `def` argument passing works

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

As you can see, the 'a' argument with an explicit type and convention is
treated exactly as before. The typed 'b' argument maintains its type, but gets
a mutable shadow copy so the callee can modify the value inside the body of the
def. The 'c' argument gets an implicit Object type, and is mutable in the body.
These copies typically add no overhead, because small types like Object
references are cheap to copy. The expensive part is the reference count
adjustment, which is eliminated by a move optimization.

## "Value Lifecycle": Birth, life and death of a value

Now that we have an understanding of the different ingredients that can go into
building functions and the types system, we can look at how to put together
together to model important types that you may want to express in Mojo.

Many existing languages express design points with different tradeoffs: C++, for
example, is very powerful but often accused of
"getting the defaults wrong" which leads to bugs and mis-features.  Swift is
easy to work with, but has a less predictable model that copies values a lot and
is dependent on an "ARC optimizer" for performance. Rust started with strong
value ownership goals to satisfy its borrow checker, but relies on values being
movable, which makes it challenging to express custom move constructors and
can put a lot of stress on `memcpy` performance. In Python, everything is a
reference to a class, so it has never really faced these issues.

For Mojo, we benefit from learning from these existing systems, and aim to
provide a model that is very powerful while still easy to learn and understand.
We also don't want to require "best effort" and difficult-to-predict
optimization passes built into a "sufficiently smart" compiler.

To explore these issues, we look at different value classifications and the
relevant Mojo features that go into expressing them, and build from the
bottom-up.  We use C++ as the primary comparison point in examples because it is
widely known but occasionally reference other languages if they provide a better
comparison point.

### Types that cannot be instantiated

The most bare-bones type in Mojo is one that doesn't allow you to create
instances of it: these types have no initializer at all, and if they have a
destructor, it will never be invoked (because there cannot be instances to
destroy):

```mojo
struct NoInstances:
    var state: Int  # Pretty useless

    alias my_int = Int

    @staticmethod
    fn print_hello():
        print("hello world")
```

Mojo types do not get default constructors, move constructors, memberwise
initializers or anything else by default, so it is impossible to create an
instance of this `NoInstances` type.  In order to get them you need to define
an `__init__` method or use a decorator that synthesizes an initializer.  As
shown, these types can be useful as "namespaces", because you can refer to
static members like `NoInstances.my_int` or `NoInstances.print_hello()` even
though you cannot instantiate an instance of the type.

### Non-movable and non-copyable types

If we take a step up the ladder of sophistication, you’ll get to types that can
be instantiated, but once so they are pinned to an address in memory and cannot
be implicitly moved or copied.  This can be useful to implement types like
atomic operations (e.g. `std::atomic` in C++) or other types where the memory
address of the value is its identity and critical to its purpose:

```mojo
struct Atomic:
    var state: Int

    fn __init__(self&, state: Int = 0):
        self.state = state

    fn __iadd__(self&, rhs: Int):
        #...atomic magic...

    fn get_value(self) -> Int:
        return atomic_load_int(self.state)
```

This class defines an initializer but no copy or move constructors, so once it
is initialized it can never be moved or copied.  This is safe and useful because
Mojo's ownership system is fully "address correct" - when this is initialized
onto the stack or in the field of some other type, it never needs to move.

Note that Mojo’s approach just controls the builtin operations like `a = b`
copies and the `x^` consume operator.  One useful pattern that can be used for
types like this is to add an explicit `copy()` method (a non-"dunder" method)
which can be useful to explicitly make copies of an instance when it is known
safe to the programmer.

### Unique "move-only" types

If we take one more step up the ladder of capabilities, we will encounter types
that are "unique" - there are many examples of this in C++, e.g. types like
`std::unique_ptr`, or even a `FileDescriptor` type that owns an underlying POSIX
file descriptor.  These types are pervasive in languages like Rust, where
copying is discouraged, but "move" is free. In Mojo, you can declare these by
implementing the `__moveinit__` method with a consuming existing like this:

```mojo
# This is a simple wrapper around POSIX-style fcntl.h functions.
struct FileDescriptor:
    var fd: Int

    # This is the new.
    fn __moveinit__(self&, consuming existing: Self):
        self.fd = existing.fd

    # This takes ownership of a POSIX file descriptor.
    fn __init__(self&, fd: Int):
        self.fd = fd

    fn __init__(self&, path: String):
        # Error handling omitted, call the open(2) syscall.
        self = FileDescriptor(open(path, ...))

    fn __del__(owned self):
        close(self.fd)   # pseudo code, call close(2)

    fn dup(self) -> Self:
        # Invoke the dup(2) system call.
        return Self(dup(self.fd))
    fn read(...): ...
    fn write(...): ...
```

The new concept is that we added a "consuming move constructor" which is named
`__moveinit__`.  The consuming move initializer takes ownership of an existing
`FileDescriptor`, and moves its internal implementation details over to a new
instance.  This is because instances of `FileDescriptor` may exist at different
locations, and they can be logically moved around - stealing the body of one
value and moving it another.

Here is an egregious example that will invoke this multiple times:

```mojo
fn egregious_moves(owned fd1: FileDescriptor):
    # fd1 and fd2 have different addresses in memory, but the
    # consume operator moves unique ownership from fd1 to fd2.
    let fd2 = fd1^

    # Do it again, a use of fd2 after this point will produce an error.
    let fd3 = fd2^

    # We can do this all day...
    let fd4 = fd3^
    fd4.read(...)
    # fd4.__del__() runs here
```

Note how ownership of the value is transferred between various values that own
it, using the postfix-`^` ‘consume’ operator to destroy a previous binding.  If
you are familiar with C++, the simple way to think about the consume operator is
like `std::move`, but in this case, we can see that it is able to move things
without resetting them to a state that can be destroyed: in C++, if your move
operator failed to change the old value’s `fd` instance, it would get closed
twice.

Mojo tracks the liveness of values and allows you to define custom move
constructors.  This is rarely needed, but extremely powerful when it is.  For
example, some types like the
<code>[llvm::SmallVector type](https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h)</code>
use the "inline storage" optimization technique, and they may want to be
implemented with an "inner pointer" into their instance.  This is a well known
trick to reduce pressure on the malloc memory allocator, but it means that a
"move" operation needs custom logic to update the pointer when that happens.

With Mojo, this is as simple as implementing a custom `__moveinit__` method.
This is something that is also easy to implement in C++ (though, with
boilerplate in the cases where you don’t need custom logic) but is difficult to
implement in other popular memory safe languages.

One additional note is that while the Mojo compiler provides good predictability
and control, it is also very sophisticated.  It reserves the rights to eliminate
temporaries and the corresponding copy/move operations.  If this is
inappropriate for your type, you should use explicit methods like `copy()`
instead of the dunder methods.

### Types that support a "Stealing Move"

One challenge with memory safe languages is that they need to provide a
predictable programming model around what the compiler is able to track, and
static analysis in a compiler is inherently limited.  For example, while it is
possible for a compiler to understand the that the two array accesses in the
first example below are to different array elements, it is (in general)
impossible to reason about the second example:

```c++
std::pair<T, T> getValues1(MutableArray<T> &array) {
    return { std::move(array[0]), std::move(array[1]) };
}
std::pair<T, T> getValues2(MutableArray<T> &array, size_t i, size_t j) {
    return { std::move(array[i]), std::move(array[j]) };
}
```

The problem here is that there is simply no way (looking at just the function
body above) to know or prove that the dynamic values of `i` and `j` are not the
same.  While it is possible to maintain dynamic state to track whether
individual elements of the array are live, this often would cause significant
runtime expense (even when move/consumes are not used), which is something that
Mojo and other systems programming languages are not keen to do.  There are a
variety of ways to deal with this, including some pretty complicated solutions
that aren’t always easy to learn.

Mojo takes a pragmatic approach to let Mojo programmers get their job done
without having to work around its type system. As seen above, it doesn’t force
types to be copyable, movable or even constructable, but it does want types to
express their full contract and it wants to enable fluent design patterns that
programmers expect from languages like C++.  The (well known) observation here
is that many objects have contents that can be "stolen" without needing to
disable their destructor, either because they have a "null state" (like an
optional type or nullable pointer) or because they have a null value that is
efficient to create and a no-op to destroy (e.g. `std::vector` can have a null
pointer for its data).

To support these use-cases, the consume operator supports arbitrary LValues, and
when applied to one, it invokes the "stealing move constructor".  This
constructor must set up the new value to be in a live state, and can mutate the
old value, but needs to put it into a state where its destructor will still
work.  For example, if we want to put our `FileDescriptor` into a vector and
move out of it, we might choose to extend it to know that `-1` is a sentinel
that means that it is "null".  We can implement this like so:

```mojo
# This is a simple wrapper around POSIX-style fcntl.h functions.
struct FileDescriptor:
    var fd: Int

    # This is the new key capability.
    fn __moveinit__(self&, existing&: Self):
        self.fd = existing.fd
        existing.fd = -1  # neutralize 'existing'.

    fn __moveinit__(self&, consuming existing: Self): # as above
    fn __init__(self&, fd: Int): # as above
    fn __init__(self&, path: String): # as above

    fn __del__(owning self):
        if self.fd != -1:
            close(self.fd)   # pseudo code, call close(2)
```

Notice how the "stealing move" constructor takes the file descriptor from an
existing value, and mutates that value so that its destructor won’t do anything.
This technique has tradeoffs, and is therefore not the best for every type.  We
can see that it adds one (inexpensive) branch to the destructor, because it has
to check for the sentinel case.  It is also generally considered bad form to
make types like this nullable, because a more general feature like an
`Optional[T]` type is a better way to handle this.

That said, we plan to implement `Optional[T]` in Mojo itself, and `Optional`
needs this functionality.  We also believe that the library authors understand
their domain problem better than language designers do, and generally prefer to
give library authors full power over that domain.  As such you can choose (but
don’t have to) to make your types participate in this behavior in an opt-in way.

### Copyable Types

The next step up from moveable types are copyable types.  Copyable types are
also very common - programmers generally expect things like strings, and arrays
to be copyable, and every Python Object reference is copyable - by copying the
pointer and adjusting the reference count.

There are many ways to implement copyable types.  One can implement reference
semantic types like Python or Java, where you propagate shared pointers around,
one can use immutable data structures that are easily shareable because they are
never mutated once created, and one can implement deep value semantics through
lazy copy-on-write like Swift does.  Each of these approaches has different
tradeoffs, and Mojo takes the opinion that while we want a few common sets of
collection types, that we can also support a wide range of specialized ones that
focus on particular use cases.

In Mojo, you can do this by implementing the `__copyinit__` method.  Here is an
example of that using a simple `String` in pseudo code:

```mojo
struct MyString:
    var data: Pointer[Int8]

    # StringRef is a pointer + length and works with StringLiteral.
    def __init__(self&, input: StringRef):
        self.data = ...

    # Copy the string by deep copying the underlying malloc'd data.
    def __copyinit__(self&, existing: Self):
        self.data = strdup(existing.data)

    # This isn't required, but optimizes unneeded copies.
    def __moveinit__(self&, owned existing: Self):
        self.data = existing.data

    def __del__(owned self):
        free(self.data.address)

    def __add__(self, rhs: MyString) -> MyString: ...
```

This simple type is a pointer to a "null terminated" string data allocated with
malloc, using old-school C APIs for clarity.  It implements the `__copyinit__`
which maintains the invariant that each instance of MyString owns their
underlying pointer and frees it on destruction.  This implementation builds on
tricks we’ve seen above, and implements a `__moveinit__` constructor, which
allows it to completely eliminate temporary copies in some common cases.  You
can see this behavior in this code sequence:

```mojo
fn test_my_string():
    var s1 = MyString("hello ")

    var s2 = s1    # s2.__copyinit__(s1) runs here

    print(s1)

    var s3 = s1^   # s3.__moveinit__(s1) runs here

    print(s2)
    # s2.__del__() runs here
    print(s3)
    # s3.__del__() runs here
```

In this case you can see both why a copy constructor is needed: without one, the
duplication of the `s1` value into `s2` would be an error - because you
cannot have two live instances of the same non-copyable type.  The move
constructor is optional, but helps the assignment into `s3`: without it, the
compiler would invoke the copy constructor from s1, then destroy the old `s1`
instance.  This is logically correct, but introduces extra runtime overhead.

Mojo destroys values eagerly, which allows it to use frequently transform
copy+destroy pairs into a move operation, which can lead to much better
performance than C++ without requiring the need for pervasive micro-management
of `std::move`.

### Trivial Types

The most flexible types are ones that are just "bags of bits".  These types are
"trivial" because they can be copied, moved, and destroyed without invoking
custom code.  Types like these are arguably the most common basic type that
surrounds us: things like integers and floating point values are all trivial.
From a language perspective, Mojo doesn’t need special support for these, it
would be perfectly fine for type authors to implement these things as no-ops,
and allow the inliner to just make them go away.

There are two reasons that approach would be suboptimal: one is that we don’t
want the boilerplate of having to define a bunch of methods on trivial types,
and second, we don’t want the compile time overhead of generating and pushing
around a bunch of function calls, only to have them inline away to nothing.
Furthermore, there is an orthogonal concern, which is that many of these types
are trivial in another way: they are tiny, and should be passed around in the
registers of a CPU, not indirectly in memory.

As such, Mojo provides a struct decorator that solves all of these problems.
You can implement a type with the `@register_passable("trivial")` decorator,
and this tells Mojo that the type should be copyable and movable, but that it
has no user-defined logic for doing this.  It also tells Mojo to prefer to pass
the value in CPU registers, which can lead to efficiency benefits.

TODO: This decorator is due for a reconsideration.  Lack of custom logic
copy/move/destroy logic, and "passability in a register" are orthogonal concerns
and should be split.  This former logic should be subsumed into a more general
`@value("trivial")` decorator which is orthogonal from `@register_passable`.

### `@value` decorator

Mojo's approach (described above) provides simple and predictable hooks that
give you the ability to express exotic low-level things like `Atomic` correctly.
This is great for control and for a simple programming model, but most structs
we all write are simple aggregations of other types, and we don't want to have
to write a lot of boilerplate for them!  To solve this, Mojo provides a `@value`
decorator for structs that synthesizes the boilerplate for you.

The `@value` decorator takes a look at the fields of your type, and generates
members that are missing.  Consider a simple struct like this, for example:

```mojo
@value
struct MyPet:
    var name: String
    var age: Int
```

Mojo will notice that you do not have a memberwise initializer, a move
constructor or a copy constructor and will synthesize these for you as if you
had written:

```mojo
fn __init__(self&, owned name: String, age: Int):
    self.name = name^
    self.age = age

fn __copyinit__(self&, existing: Self):
    self.name = existing.name
    self.age = existing.age

fn __moveinit__(self&, owned existing: Self):
    self.name = existing.name^
    self.age = existing.age
```

If your type contains any move-only fields, it cannot (and therefore will not)
generate a copy constructor for you of course.  Mojo only synthesizes these for
you when they don't exist, so it is ok to override its behavior by defining your
own version of these.  For example, it is fairly common to want to define a
custom copy constructor but use the default memberwise and move constructor.

There is no way to suppress generation of specific methods or customize
generation at this time, but we can add arguments to the `@value` generator to
do this if there is demand.

Note that the `@value` decorator only works on types whose members are copyable
and/or movable.  If you have something like `Atomic` in your struct, then it
probably isn't a value type, and you don't want these members anyway.

## Behavior of Destructors

Any struct in Mojo can have a destructor, which is automatically run when the
values lifetime ends, for example, a simple string might look like this (in
pseudo code):

```mojo
struct MyString:
    var data: Pointer[Int8]

    def __init__(self&, input: StringRef): ...
    def __add__(self, rhs: MyString) -> MyString: ...
    def __del__(owned self):
        free(self.data.address)
```

The Mojo compiler automatically invokes the destructor when the value is dead,
and provides strong guarantees about when the destructor is run.  Mojo uses
compiler static analysis to reason about your code and decide when to insert
calls to the destructor.  For example:

```mojo
fn use_strings():
    var a = MyString("hello a")
    var b = MyString("hello b")
    print(a)
    # a.__del__() runs here


    print(b)
    # b.__del__() runs here

    a = MyString("temporary a")
    # a.__del__() runs here

    other_stuff()

    a = MyString("final a")
    print(a)
    # a.__del__() runs here
```

In the code above you’ll see that the `a` and `b` values are created early on,
and each initialization of a value is matched with a call to a destructor.
Notice also where the calls are happening: in the `b` variable for example, Mojo
keeps the value live across the (unrelated) print of the `a` variable until the
print of the `b` variable, and destroys it immediately after that call.  The `a`
value is destroyed immediately after its first print, and immediately after
reassigning it a new (unused) temporary value, and after its final print.

Mojo destroys values using an **"As Soon As Possible"** (ASAP) policy, behaving
like
a hyper-active garbage collector that is run after every call - and when we say
every call, we mean it!  Code that uses internal expressions (like `a+b+c+d`)
will destroy the intermediate expressions eagerly when they are not needed -
destruction is not deferred to the end of the statement like in C++. Mojo
fully understands control flow, including loops, ifs, and try/except of course.

Now, this may be surprising to a C++ programmer: this invalidates the use of the
[RAII pattern](https://en.cppreference.com/w/cpp/language/raii) that C++
programmers use widely.  So, why does Mojo destroy things so eagerly, instead of
using C++-style scoped destruction?  Well I’m glad you asked, there are many
good reasons!

The Mojo design has a number of strong advantages over the C++ model:

1. Recall that Python doesn’t really have scopes beyond the whole function, and
   Mojo needs to provide a workable model that behaves correctly in the presence
   of Python-style ‘def’s.
2. Because Python doesn’t provide strong guarantees on object destruction, it
   doesn’t encourage the RAII pattern.  To solve for the RAII pattern, Mojo (and
   Python) provides a <code>[with
   statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement)</code> that provides scoped access to resources,
   which is more deliberate and more syntactically clear than RAII.
3. The Mojo approach eliminates the need for types to implement re-assignment
   operators, like `operator=(const T&)` and `operator=(T&&)` in C++, making it
   easier to define types and eliminating a concept.
4. Mojo does not allow mutable references to overlap with other mutable
   references or with immutable borrows.  One major way that it provides a
   predictable programming model is by making sure that references to objects
   die as soon as possible, avoiding confusing situations where the compiler
   thinks a value could still be alive and interfere with another value, but
   that isn’t clear to the user.
5. Destroying values at last use composes nicely with "move" optimization,
   which transforms a "copy+del" pair into a "move" of a value, a generalization
   of C++ move optimizations like NRVO.
6. Destroying values at the end of scope in C++ is problematic for some common
   patterns like tail recursion, because the destructor calls happen after the
   tail call. This can be a significant performance and memory problem for
   certain functional programming patterns.

The Mojo approach is more similar to how Rust and Swift work, because they both
have strong value ownership tracking and provide memory safety.  One difference
is that their implementation requires the use of a [dynamic "drop
flag"](https://doc.rust-lang.org/nomicon/drop-flags.html) - they maintain hidden
shadow variables to keep track of the state of your values to provide safety.
These are often optimized away, but the Mojo approach eliminates this overhead
entirely, making the generated code faster and avoiding ambiguity.

### Field Sensitive Lifetime Management

In addition to Mojo’s lifetime analysis being fully control flow aware, it is
also fully field sensitive (each field of a structure is tracked independently).
It separately keeps track of whether a "whole object" is initialized with an
initializer or destroyed with a whole object destructor.  For example, consider
this code:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString
    fn __init__(self&): ...
    fn __del__(owned self): ...

fn use_two_strings():
    var ts = TwoStrings()
    # ts.str1.__del__() runs here

    other_stuff()

    ts.str1 = MyString("hello a")     # Overwrite ts.str1
    print(ts.str1)
    # ts.__del__() runs here
```

Note that the `ts.str1` field is immediately destroyed after being set up,
because Mojo knows that it will be overwritten down below.  You can also see
this when using the consume operator, for example:

```mojo
fn consume_and_use_two_strings():
    var ts = TwoStrings()
    consume(ts.str1^)

    # ts is partially initialized here!
    other_stuff()

    ts.str1 = MyString()  # All together now
    use(ts)               # This is ok
    # ts.__del__() runs here
```

Notice that the code consumes one of the fields: for the duration of
`other_stuff()` the `str1` field is completely uninitialized.  Fortunately for
the code above, `str1` is reinitialized before it is used by the `use` function
- and if it weren’t, Mojo will reject the code with an uninitialized field
error.

Mojo's rule on this is powerful and intentionally straightforward: fields can be
temporarily consumed, but the "whole object" must be constructed with the
aggregate type’s initializer, and destroyed with the aggregate destructor.  This
means that it isn’t possible to create an object by initializing its fields, nor
is it possible to tear down an object by destroying its fields:

```mojo
fn consume_and_use_two_strings():
    var ts = TwoStrings()
    consume(ts.str1^)
    consume(ts.str2^)
    # Error: cannot run the 'ts' destructor without initialized fields.

    var ts2 : TwoStrings
    ts2.str1 = MyString()  # All together now
    ts2.str2 = MyString()  # All together now
    use(ts2) # Error: 'ts2' isn't fully initialized
```

While we could allow patterns like this to happen, we reject this because "a
value is more than a sum of its parts".  Consider a `FileDescriptor` that
contains an POSIX file descriptor as an integer value for example - there is a
big difference between destroying the integer (a noop!) and destroying the
`FileDescriptor` (it might call the `close()` system call).  Because of this, we
require all full value initialization to go through initializers and be
destroyed with their full value destructor.

For what it's worth, Mojo does internally have an equivalent of the Rust
"[mem::forget](https://doc.rust-lang.org/std/mem/fn.forget.html)" function which
explicitly disables a destructor, and has a corresponding internal feature for
"blessing" an object, but they aren’t exposed for user consumption at this
point.

### Field lifetimes in `__init__`

The behavior of an `__init__` method works almost like any other method - there
is a small bit of magic though, in that it knows that the fields of an object
are uninitialized, but it believes the full object is initialized.  This means
that you can use ‘self’ as a whole object as soon as all the fields are
initialized:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString

    fn __init__(self&, cond: Bool, other: MyString):
        self.str1 = MyString()
        if cond:
            self.str2 = other
            use(self)  # Safe to use immediately!
            # self.str2.__del__(): destroyed because overwritten below.

        self.str2 = self.str1
        use(self)  # Safe to use immediately!
```

Similarly, it is completely safe for initializers in Mojo to completely
overwrite `self`, e.g. by delegating to other initializers:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString

    fn __init__(self&): ...
    fn __init__(self&, cond: Bool, other: MyString):
        self = TwoStrings()  # basic
        self.str1 = MyString("fancy")
```

### Field lifetimes of `owned` arguments in `__del__ `and `__moveinit__`

A final bit of magic exists for the ‘owned’ arguments of a destructor and move
initializer.  To recap, these methods are defined like this:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString
    fn __init__(...)

    fn __moveinit__(self&, owned existing: Self): ...
    fn __del__(owned self): ...
```

These methods face an interesting but obscure problem: both of these methods are
in charge of dismantling the `owned existing`/`self` value, either in destroying
sub-elements that have to do with them, or using them to implement deletion
logic for their own type.  The move constructor wants to create a new `self`
instance by stealing parts from an existing instance.  As such, they both want
to consume and transform elements of the ‘owned’ value, and definitely don’t
want the owned values destructor to run!  The most egregious example of this is
the `__del__` method, which would turn into an infinite loop.

To solve this problem, Mojo handles these two methods specially, by assuming
that their whole values are destroyed upon reaching any return from the method.
This means that the whole object may be used before the field values are
consumed, for example, this works as you expect:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString
    fn __init__(...)
    fn __moveinit__(self&, owned existing: Self): ...

    fn __del__(owned self):
        log(self)       # Self is still whole
        # self.str2.__del__(): Mojo destroys str2 since it isn't used

        consume(^str1)
        # Everything has now been consumed, no destructor is run on self.
```

You should not generally have to think about this, but if you have logic with
inner pointers into members, you may need to keep them alive for some logic
within the destructor or move initializer itself.  You can do this by assigning
to the discard pattern:

```mojo
fn __del__(owned self):
    log(self) # Self is still whole

    consume(^str1)
    _ = self.str2
    # self.str2.__del__(): Mojo destroys str2 after its last use.
```

In this case, if "consume" implicitly refers to some value in `str2` somehow,

this will ensure that str2 isn’t destroyed until the last use when it is
accessed by the `_` pattern.

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
don't step into the + method on Int.

### `@parameter` decorator

The `@parameter` decorator can be placed on nested functions that capture
runtime values to create "parametric" capturing closures. This is an unsafe
feature in Mojo, because we do not currently model the lifetimes of
capture-by-reference. A particular aspect of this feature is that it allows
closures that captures runtime values to be passed as parameter values.

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
