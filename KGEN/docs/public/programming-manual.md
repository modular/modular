---
title: Mojo🔥 programming manual
description: A comprehensive tour of Mojo language features with code examples.
website:
  open-graph:
    image: /static/images/mojo-social-card.png
  twitter-card:
    image: /static/images/mojo-social-card.png
---

Mojo is a programming language that is as easy to use as Python but with the
performance of C++ and Rust. Furthermore, Mojo provides the ability to leverage
the entire Python library ecosystem.

Mojo achieves this feat by utilizing next-generation compiler technologies with
integrated caching, multithreading, and cloud distribution technologies.
Furthermore, Mojo's autotuning and compile-time metaprogramming features allow
you to write code that is portable to even the most exotic hardware.

More importantly, **Mojo allows you to leverage the entire Python ecosystem**
so you can continue to use tools you are familiar with. Mojo is designed to
become a **superset** of Python over time by preserving Python's dynamic
features while adding new primitives for [systems
programming](https://en.wikipedia.org/wiki/Systems_programming). These new
system programming primitives will allow Mojo developers to build
high-performance libraries that currently require C, C++, Rust, CUDA, and other
accelerator systems. By bringing together the best of dynamic languages and
systems languages, we hope to provide a **unified** programming model that
works across levels of abstraction, is friendly for novice programmers, and
scales across many use cases from accelerators through to application
programming and scripting.

This document is an introduction to the Mojo programming language, not a
complete language guide. It assumes knowledge of Python and systems programming
concepts. At the moment, Mojo is still a work in progress and the documentation
is targeted to developers with systems programming experience. As the language
grows and becomes more broadly available, we intend for it to be friendly and
accessible to everyone, including beginner programmers. It's just not there
today.

## Using the Mojo compiler

You can run a Mojo program from a terminal just like you can with Python. So if
you have a file named `hello.mojo` (or `hello.🔥`—yes, the file extension can
be an emoji!), just type `mojo hello.mojo`:

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

Again, you can use either the emoji or the `.mojo` suffix.

If you are interested in diving into the internal implementation
details of Mojo, it can be instructive to look at types in the standard
library, example code in notebooks, blogs and other sample code.

## Basic systems programming extensions

Given our goal of compatibility and Python's strength with high-level
applications and dynamic APIs, we don't have to spend much time explaining
how those portions of the language work. On the other hand, Python's support
for systems programming is mainly delegated to C, and we want to provide a
single system that is great in that world. As such, this section breaks down
each major component and feature and describes how to use them with examples.

### `let` and `var` declarations

Inside a `def` in Mojo, you may assign a value to a name and it implicitly
creates a function scope variable just like in Python. This provides a very
dynamic and low-ceremony way to write code, but it is a challenge for two
reasons:

1) Systems programmers often want to declare that a value is immutable for
    type-safety and performance.
2) They may want to get an error if they mistype a variable name in an
    assignment.

To support this, Mojo provides scoped runtime value declarations: `let` is
immutable, and `var` is mutable. These values use lexical scoping and support
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
    let x: SI8 = 42
    let y: SI64 = 17

    let z: SI8
    if x != 0:
        z = 1
    else:
        z = foo()
    use(z)
```

Note that `let` and `var` are completely opt-in when in `def` declarations. You
can still use implicitly declared values as with Python, and they get
function scope as usual.

### `struct` types

Mojo is based on MLIR and LLVM, which offer a cutting-edge compiler and code
generation system used in many programming languages. This lets us have better
control over data organization, direct access to data fields, and other ways to
improve performance. An important feature of modern systems programming
languages is the ability to build high-level and safe abstractions on top of
these complex, low-level operations without any performance loss. In Mojo, this
is provided by the `struct` type.

A `struct` in Mojo is similar to a Python `class`: they both support methods,
fields, operator overloading, decorators for metaprogramming, etc. Their
differences are as follows:

- Python classes are dynamic: they allow for dynamic dispatch, monkey-patching
(or "swizzling"), and dynamically binding instance properties at runtime.

- Mojo structs are static: they are bound at compile-time (you cannot add
methods at runtime). Structs allow you to trade flexibility for performance
while being safe and easy to use.

Here's a simple definition of a struct:

```mojo
@value
struct MyPair:
    var first: Int
    var second: Int
    def __lt__(self, rhs: MyPair) -> Bool:
        return self.first < rhs.first or
              (self.first == rhs.first and
               self.second < rhs.second)
```

Syntactically, the biggest difference compared to a Python `class` is that all
instance properties in a `struct` **must** be explicitly declared with a `var`
or `let` declaration.

In Mojo, the structure and contents of a "struct" are set in advance and can't
be changed while the program is running. Unlike in Python, where you can add,
remove, or change attributes of an object on the fly, Mojo doesn't allow that
for structs. This means you can't use `del` to remove a method or change its
value in the middle of running the program.

However, the static nature of `struct` has some great benefits! It helps Mojo
run your code faster. The program knows exactly where to find the struct's
information and how to use it without any extra steps or delays.

Mojo's structs also work really well with features you might already know from
Python, like operator overloading (which lets you change how math symbols like
`+` and `-` work with your own data). Furthermore, *all* the "standard types"
(like `Int`, `Bool`, `String` and even `Tuple`) are made using structs. This
means they're part of the standard set of tools you can use, rather than being
hardwired into the language itself. This gives you more flexibility and control
when writing your code.

:::{.callout-note}

If you're wondering what the `inout` means on the `self` argument: this
indicates that the argument is mutable and changes made inside the function are
visible to the caller. For details, see below about
[inout arguments](#inout-arguments).

:::

#### `Int` vs `int`

In Mojo, you might notice that we use `Int` (with a capital "I"), which is
different from Python's `int` (with a lowercase "i"). This difference is on
purpose, and it's actually a good thing!

In Python, the `int` type can handle really big numbers and has some extra
features, like checking if two numbers are the same object. But this comes with
some extra baggage that can slow things down. Mojo's `Int` is different. It's
designed to be simple, fast, and tuned for your computer's hardware to handle
quickly.

We made this choice for two main reasons:

1. We want to give programmers who need to work closely with computer hardware
(systems programmers) a transparent and reliable way to interact with hardware.
We don't want to rely on fancy tricks (like JIT compilers) to make things
faster.

2. We want Mojo to work well with Python without causing any issues. By using a
different name (Int instead of int), we can keep both types in Mojo without
changing how Python's int works.

As a bonus, `Int` follows the same naming style as other custom data types you
might create in Mojo. Additionally, `Int` is a `struct` that's included in
Mojo's standard set of tools.

### Strong type checking

Even though you can still use flexible types like in Python, Mojo lets you use
strict type checking. Type-checking can make your code more predictable,
manageable, and secure.

One of the primary ways to employ strong type checking is with Mojo's `struct`
type. A `struct` definition in Mojo defines a compile-time-bound name, and
references to that name in a type context are treated as a strong specification
for the value being defined. For example, consider the following code that uses
the `MyPair` struct shown above:

```mojo
def pairTest() -> Bool:
    let p = MyPair(1, 2)
    return p < 4 # gives a compile-time error
```

When you run this code, you'll get a compile-time error telling you that "4"
cannot be converted to `MyPair`, which is what the RHS of `MyPair.__lt__`
requires.

This is a familiar experience when working with systems programming languages,
but it's not how Python works. Python has syntactically identical features for
[MyPy](https://mypy.readthedocs.io/) type annotations, but they are not
enforced by the compiler: instead, they are hints that inform static analysis.
By tying types to specific declarations, Mojo can handle both the classical
type annotation hints and strong type specifications without breaking
compatibility.

Type checking isn't the only use-case for strong types. Since we know the types
are accurate, we can optimize the code based on those types, pass values in
registers, and be as efficient as C for argument passing and other low-level
details. This is the foundation of the safety and predictability guarantees
Mojo provides to systems programmers.

### Overloaded functions and methods

Like Python, you can define functions in Mojo without specifying argument data
types and Mojo will handle them dynamically. This is nice when you want
expressive APIs that just work by accepting arbitrary inputs and let dynamic
dispatch decide how to handle the data. However, when you want to ensure type
safety, as discussed above, Mojo also offers full support for overloaded
functions and methods.

This allows you to define multiple functions with the same name but with
different arguments. This is a common feature seen in many languages, such as
C++, Java, and Swift.

When resolving a function call, Mojo tries each candidate and uses the one that
works (if only one works), or it picks the closest match (if it can determine a
close match), or it reports that the call is ambiguous if it can't figure
out which one to pick. In the latter case, you can resolve the ambiguity by
adding an explicit cast on the call site. Let's look at an example:

```mojo
struct Array[T: AnyType]:
    fn __getitem__(self, idx: Int) -> T: ...
    fn __getitem__(self, idx: Range) -> ArraySlice: ...
```

You can overload methods in structs and classes and overload module-level
functions.

Mojo doesn't support overloading solely on result type, and doesn't use result
type or contextual type information for type inference, keeping things simple,
fast, and predictable.  Mojo will never produce an "expression too complex"
error, because its type-checker is simple and fast by definition.

Again, if you leave your argument names without type definitions, then the
function behaves just like Python with dynamic types. As soon as you define a
single argument type, Mojo will look for overload candidates and resolve
function calls as described above.

Functions and methods can be overloaded on both argument and parameter
signatures. See [Overloading on parameters](#overloading-on-parameters) for more
information and the description on overload resolution rules.

### `fn` definitions

The extensions above are the cornerstone that provides low-level programming
and provide abstraction capabilities, but many systems programmers prefer more
control and predictability than what `def` in Mojo provides. To recap, `def` is
defined by necessity to be very dynamic, flexible and generally compatible with
Python: arguments are mutable, local variables are implicitly declared on first
use, and scoping isn't enforced. This is great for high level programming and
scripting, but is not always great for systems programming. To complement this,
Mojo provides an `fn` declaration which is like a "strict mode" for `def`.

> Alternative: instead of using a new keyword like `fn`, we could instead add a
modifier or decorator like `@strict def`. However, we need to take new keywords
anyway and there is little cost to doing so. Also, in practice in systems
programming domains, `fn` is used all the time so it probably makes sense to
make it first class.

As far as a caller is concerned, `fn` and `def` are interchangeable: there is
nothing a `def` can provide that a `fn` cannot (and vice versa). The
difference is that a `fn` is more limited and controlled on the *inside* of
its body (alternatively: pedantic and strict). Specifically, `fn`s have a
number of limitations compared to `def` functions:

1. Argument values default to being immutable in the body of the function (like
a `let`), instead of mutable (like a `var`). This catches accidental mutations,
and permits the use of non-copyable types as arguments.

2. Argument values require a type specification (except for `self` in a
method), catching accidental omission of type specifications. Similarly, a
missing return type specifier is interpreted as returning `None` instead of an
unknown return type. Note that both can be explicitly declared to return
`object`, which allows one to opt-in to the behavior of a `def` if desired.

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

For more about argument behavior in Mojo functions, see the section below about
[Argument passing control and memory
ownership](#argument-passing-control-and-memory-ownership).

### The `__copyinit__` and `__moveinit__` special methods {#copy-and-move}

Mojo supports full "value semantics" as seen in languages like C++ and Swift,
and it makes defining simple aggregates of fields very easy with its `@value`
decorator (described in more detail below).

For advanced use cases, Mojo allows you to define custom constructors (using
Python's existing `__init__` special method), custom destructors (using the
existing `__del__` special method) and custom copy and move constructors using
the new `__copyinit__` and `__moveinit__` special methods.

These low-level customization hooks can be useful when doing low level systems
programming, e.g. with manual memory management.  For example, consider a
dynamic string type that needs to allocate memory for the string data when
constructed and destroy it when the value is destroyed:

```mojo
struct MyString:
    var data: Pointer[UI8]

    # StringRef has a data + length field
    def __init__(inout self, input: StringRef):
        let data = Pointer[UI8].alloc(input.length+1)
        data.memcpy(input.data, input.length)
        data[input.length] = 0
        self.data = Pointer[UI8](data)

    def __del__(owned self):
        self.data.free()
```

This `MyString` type is implemented using low-level functions to show a
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

The Mojo compiler doesn't allow us to copy the string from `a` to `b` because
it doesn't know how to. `MyString` contains an instance of `Pointer` (which is
equivalent to a low-level C pointer) and Mojo doesn't know what kind of data it
points to or how to copy it. More generally, some types (like atomic numbers)
cannot be copied or moved around because their address provides an **identity**
just like a class instance does.

In this case, we do want our string to be copyable. To enable this, we
implement the `__copyinit__` special method, which is conventionally
implemented like this:

```mojo
struct MyString:
    ...
    def __copyinit__(inout self, existing: Self):
        self.data = Pointer(strdup(existing.data.address))
```

With this implementation, our code above works correctly, and the `b = a` copy
produces a logically distinct instance of the string with its own lifetime and
data. The copy is made with the C-style `strdup()` function as instructed by the
lines of code above.  Mojo also supports the `__moveinit__` method, which allows
both Rust-style moves (which take a value when a lifetime ends) and C++-style
moves (where the contents of a value are removed, but the destructor still runs)
and allows defining custom move logic. For more discussion about value
lifetimes, see the [Value lifecycle](#value-lifecycle) section below.

Mojo provides full control over the lifetime of a value, including the ability
to make types copyable, move-only, and not-movable. This is more control than
languages like Swift and Rust offer, which require values to at least be
movable. If you are curious how `existing` can be passed into the
`__copyinit__` method without itself creating a copy, check out the section on
[Borrowed arguments](#borrowed-arguments) below.

## Argument passing control and memory ownership

In both Python and Mojo, much of the language revolves around function calls: a
lot of the (apparently) built-in behaviors are implemented in the standard
library with "dunder" (double-underscore) methods. Inside these
magic functions is where a lot of memory ownership is determined through
argument passing.

Let's review some details about how Python and Mojo pass arguments:

- All values passed into a *Python* `def` function use reference semantics. This
means the function can modify mutable objects passed into it and those changes
are visible outside the function. However, the behavior is sometimes surprising
for the uninitiated, because you can change the object that an argument points
to and that change is not visible outside the function.

- All values passed into a *Mojo* `def` function use value semantics by default.
Compared to Python, this is an important difference: A Mojo `def` function
receives a copy of all arguments—it can modify arguments inside the function,
but the changes are **not** visible outside the function.

- All values passed into a Mojo [`fn` function](#fn-definitions) are immutable
references by default. This means the function can read the original object (it
is *not* a copy), but it cannot modify the object at all.

This convention for immutable argument passing in a Mojo `fn` is called
"borrowing." In the following sections, we'll explain how you can change the
argument passing behavior in Mojo, for both `def` and `fn` functions.

### Why argument conventions are important

In Python, all fundamental values are references to objects—as described above,
a Python function can modify the original object. Thus, Python developers are
used to thinking about everything as reference semantic. However, at the
CPython or machine level, you can see that the references themselves are
actually passed *by-copy*—Python copies a pointer and adjusts reference counts.

This Python approach provides a comfortable programming model for most people,
but it requires all values to be heap-allocated (and results are occasionally
surprising results due to reference sharing). Mojo classes (TODO: will) follow
the same reference-semantic approach for most objects, but this isn't practical
for simple types like integers in a systems programming context. In these
scenarios, we want the values to live on the stack or even in hardware
registers. As such, Mojo structs are always inlined into their container,
whether that be as the field of another type or into the stack frame of the
containing function.

This raises some interesting questions: How do you implement methods that need
to mutate `self` of a structure type, such as `__iadd__`? How does `let` work,
and how does it prevent mutation? How are the lifetimes of these values
controlled to keep Mojo a memory-safe language?

The answer is that the Mojo compiler uses dataflow analysis and type
annotations to provide full control over value copies, aliasing of references,
and mutation control. These features are similar in many ways to features in
the Rust language, but they work somewhat differently in order to make
Mojo easier to learn, and they integrate better into the Python ecosystem
without requiring a massive annotation burden.

In the following sections, you'll learn about how you can control memory
ownership for objects passed into Mojo `fn` functions.

### Immutable arguments (`borrowed`) {#borrowed-arguments}

A borrowed object is an **immutable reference** to an object that a function
receives, instead of receiving a copy of the object. So the
callee function has full read-and-execute access to the object, but it cannot
modify it (the caller still has exclusive "ownership" of the object).

Although this is the default behavior for `fn` arguments, you can explicitly
define it with the `borrowed` keyword if you'd like (you can also apply
`borrowed` to `def` arguments):

```mojo
fn use_something_big(borrowed a: SomethingBig, b: SomethingBig):
    """'a' and 'b' are passed the same, because 'borrowed' is the default."""
    a.print_id()
    b.print_id()
```

This default applies to all arguments uniformly, including the `self` argument
of methods. This is much more efficient when passing large values or when
passing expensive values like a reference-counted pointer (which is the default
for Python/Mojo classes), because the copy constructor and destructor don't
have to be invoked when passing the argument. Here is a more elaborate example
building on the code above:

```mojo
# A type that is so expensive to copy around we don't even have a
# __copyinit__ method.
struct SomethingBig:
    var id_number: Int
    var huge: InlinedArray[Int, 100000]
    fn __init__(inout self): …

    # self is passed inout for mutation as described above.
    fn set_id(inout self, number: Int):
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

Because the default argument convention for `fn` functions is `borrowed`, Mojo
has simple and logical code that does the right thing by default. For example,
we don't want to copy or move all of `SomethingBig` just to invoke the
`print_id()` method, or when calling `use_something_big()`.

This borrowed argument convention is similar in some ways to passing an
argument by `const&` in C++, which avoids a copy of the value and disables
mutability in the callee. However, the borrowed convention differs from
`const&` in C++ in two important ways:

1. The Mojo compiler implements a borrow checker (similar to Rust) that
prevents code from dynamically forming mutable references to a value when there
are immutable references outstanding, and it prevents multiple mutable
references to the same value. You are allowed to have multiple borrows (as the
call to `use_something_big` does above) but you cannot pass something by mutable
reference and borrow at the same time. (TODO: Not currently enabled).

2. Small values like `Int`, `Float`, and `SIMD` are passed directly in machine
registers instead of through an extra indirection (this is because they are
declared with the [`@register_passable`
decorator](#register_passable-struct-decorator)). This is a [significant
performance
enhancement](https://www.forrestthewoods.com/blog/should-small-rust-structs-be-passed-by-copy-or-by-borrow/)
when compared to languages like C++ and Rust, and moves this optimization from
every call site to being declarative on a type.

Similar to Rust, Mojo's borrow checker enforces the exclusivity of invariants.
The major difference between Rust and Mojo is that Mojo does not require a
sigil on the caller side to pass by borrow. Also, Mojo is more efficient when
passing small values, and Rust defaults to moving values instead of passing
them around by borrow. These policy and syntax decisions allow Mojo to provide
an easier-to-use programming model.

### Mutable arguments (`inout`) {#inout-arguments}

On the other hand, if you define an `fn` function and want an argument to be
mutable (so that changes to the argument *inside* the function are visible
*outside* the function), you must declare the argument as mutable with the
`inout` keyword.

Consider the following example, in which the `__iadd__` function tries
to modify `self`:

```mojo
struct Int:
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: Int) -> Int: ...

    # ... but this cannot work for __iadd__
    fn __iadd__(self, rhs: Int):
        self = self + rhs  # ERROR: cannot assign to self!
```

The problem here is that `self` is immutable because this is a Mojo `fn`
function, so it can't change the internal state of the argument. The solution
is to declare that the argument is mutable by adding the `inout` keyword
on the `self` argument name:

```mojo
struct Int:
    # ...
    fn __iadd__(inout self, rhs: Int):
        self = self + rhs    # OK
```

:::{.callout-note}

**Tip:** When you see `inout`, it means that any changes made to the argument
**in**side the function are visible **out**side the function.

:::

Now the `self` argument is mutable in the function and any changes are visible
in the caller—even if the caller has a non-trivial computation to access it,
like an array subscript:

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

Mojo implements the in-place mutation of the above `InlinedFixedVector` element
by emitting a call to `__getitem__` into a temporary buffer, followed by a
store with `__setitem__` after the call. Mutation of the `let` value fails
because it isn't possible to form a mutable reference to an immutable value.
Similarly, the compiler rejects attempts to use a subscript with an `inout`
argument if it implements `__getitem__` but not `__setitem__`.

Of course, you can declare multiple `inout` arguments. For example, you can
define and use a swap function like this:

```mojo
fn swap(inout lhs: Int, inout rhs: Int):
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

:::{.callout-note}

Notice that we don't call this argument passing "by reference." Although the
`inout` convention is conceptually the same, we don't call it by-reference
passing because the implementation may actually pass values using pointers.

:::

### Transfer arguments (`owned` and `^`) {#owned-arguments}

The final argument convention that Mojo supports is the `owned` argument
convention. This convention is used for functions that want to take exclusive
ownership over a value, and it is often used with the postfixed `^` operator.

For example, imagine you're working with a move-only type like a unique
pointer. While the borrow convention makes it easy to work with the unique
pointer without ceremony, at some point you might want to transfer ownership to
some other function. This is what the `^` "transfer" operator does:

```mojo
fn usePointer():
    let ptr = SomeUniquePtr(...)
    use(ptr)        # Perfectly fine to pass to borrowing function.
    take_ptr(ptr^)  # Pass ownership of the `ptr` value to another function.

    use(ptr) # ERROR: ptr is no longer valid here!
```

For movable types, the `^` operator ends the lifetime of a value binding and
transfers the value ownership to something else (in this case, the `take_ptr()`
function). To support this, you can define functions as taking `owned`
arguments. For example, you define `take_ptr()` like so:

```mojo
fn take_ptr(owned p: SomeUniquePtr):
    use(p)
```

Because it is declared `owned`, the `take_ptr()` function knows it has unique
access to the value.  This is very important for things like unique pointers,
and it's useful when you want to avoid copies.

For example, you will notably see the `owned` convention on destructors and on
consuming move constructor. For example, our `MyString` type from earlier can
be defined as follows:

```mojo
struct MyString:
    var data: Pointer[UI8]

    # StringRef has a data + length field
    def __init__(inout self, input: StringRef): ...
    def __copyinit__(inout self, existing: Self): ...

    def __moveinit__(inout self, owned existing: Self):
        self.data = existing.data

    def __del__(owned self):
        self.data.free()
```

Specifying `owned` in the `__del__` function is important because you must
own a value to destroy it.

### Comparing `def` and `fn` argument passing

Mojo's `def` function is essentially just sugaring for the `fn` function:

- A `def` argument without an explicit type annotation defaults to `Object`.

- A `def` argument without a convention keyword (such as `inout` or `owned`) is
passed by implicit copy into a mutable var with the same name as the argument.
(This requires that the type have a `__copyinit__` method.)

For example, these two functions have the same behavior:

```mojo
def example(inout a: Int, b: Int, c):
    # b and c use value semantics so they're mutable in the function
    ...

fn example(inout a: Int, b_in: Int, c_in: Object):
    # b_in and c_in are immutable references, so we make mutable shadow copies
    var b = b_in
    var c = c_in
    ...
```

The shadow copies typically add no overhead, because references for small types
like `Object` are cheap to copy. The expensive part is adjusting the reference
count, but that's eliminated by a move optimization.

## Python integration

It's easy to use Python modules you know and love in Mojo. You can import
any Python module into your Mojo program and create Python types from Mojo
types.

### Importing Python modules

To import a Python module in Mojo, just call `Python.import_module()` with the
module name:

```mojo
from PythonInterface import Python

# This is equivalent to Python's `import numpy as np`
let np = Python.import_module("numpy")

# Now use numpy as if writing in Python
a = np.array([1, 2, 3])
print(a)
```

Yes, this imports Python NumPy, and you can import *any other Python module*.

Currently, you cannot import individual members (such as a single Python class
or function)—you must import the whole Python module and then access members
through the module name.

#### Importing local Python modules

If you have some local Python code you want to use in Mojo, just add
the directory to the Python path and then import the module.

For example, suppose you have a Python file like this:

```{.python filename="mypython.py"}
import numpy as np

def my_algorithm(a, b):
    array_a = np.random.rand(a, a)
    return array_a + b
```

Here's how you can import it and use it in Mojo:

```{.mojo filename="mojo-code.mojo"}
from PythonInterface import Python

Python.add_to_path("path/to/module")
let mypython = Python.import_module("mypython")

let c = mypython.my_algorithm(2, 3)
print(c)
```

There's no need to worry about memory management when using Python in Mojo.
Everything just works because Mojo was designed for Python from the beginning.

### Mojo types in Python

Mojo primitive types implicitly convert into Python objects.
Today we support lists, tuples, integers, floats, booleans, and strings.

For example, given this Python function that prints Python types:

```{.python filename="mypython2.py"}
def type_printer(my_list, my_tuple, my_int, my_string, my_float):
    print(type(my_list))
    print(type(my_tuple))
    print(type(my_int))
    print(type(my_string))
    print(type(my_float))
```

You can import it and pass it Mojo types, no problem:

```{.mojo filename="mojo-code.mojo"}
from PythonInterface import Python

Python.add_to_path("/path/to/module")
let mypython2 = Python.import_module("mypython2")
mypython2.type_printer([0, 3], (False, True), 4, "orange", 3.4)
```

It will output the types after implicit conversion to Python types:

```python
<class 'list'>
<class 'tuple'>
<class 'int'>
<class 'str'>
<class 'float'>
```

Mojo doesn't have a standard Dictionary yet, so it is not yet possible
to create a Python dictionary from a Mojo dictionary. You can work with
Python dictionaries in Mojo though! To create a Python dictionary, use the
`dict` method:

```mojo
from PythonInterface import Python
from PythonObject import PythonObject
from IO import print
from Range import range

def main() -> None:
    let dictionary = Python.dict()
    dictionary["fruit"] = "apple"
    dictionary["starch"] = "potato"
    let keys: PythonObject = ["fruit", "starch", "protein"]
    let N: Int = keys.__len__().to_index()
    print(N, "items")
    for i in range(N):
        if Python.is_type(dictionary.get(keys[i]), Python.none()):
            print(keys[i], "is not in dictionary")
        else:
            print(keys[i], "is included")
```

The output:

```text
3 items
fruit is included
starch is included
protein is not in dictionary
```

## Parameterization: compile-time metaprogramming

One of Python's most amazing features is its extensible runtime
metaprogramming features. This has enabled a wide range of libraries and
provides a flexible and extensible programming model that Python programmers
everywhere benefit from. Unfortunately, these features also come at a cost:
because they are evaluated at runtime, they directly impact run-time efficiency
of the underlying code. Because they are not known to the IDE, it is difficult
for IDE features like code completion to understand them and use them to
improve the developer experience.

Outside the Python ecosystem, static metaprogramming is also an important part
of development, enabling the development of new programming paradigms and
advanced libraries. There are many examples of prior art in this space, with
different tradeoffs, for example:

1. Preprocessors (e.g. C preprocessor, Lex/YACC, etc) are perhaps the heaviest
handed. They are fully general but the worst in terms of developer experience
and tools integration.

2. Some languages (like Lisp and Rust) support (sometimes "hygienic") macro
expansion features, enabling syntactic extension and boilerplate reduction with
somewhat better tooling integration.

3. Some older languages like C++ have very large and complex metaprogramming
languages (templates) that are a dual to the *runtime* language. These are
notably difficult to learn and have poor compile times and error messages.

4. Some languages (like Swift) build many features into the core language in a
first-class way to provide good ergonomics for common cases at the expense of
generality.

5. Some newer languages like Zig integrate a language interpreter into the
compilation flow, and allow the interpreter to reflect over the AST as it is
compiled. This allows many of the same features as a macro system with better
extensibility and generality.

For Modular's work in AI, high-performance machine learning kernels, and
accelerators, we need high abstraction capabilities provided by advanced
metaprogramming systems. We needed high-level zero-cost abstractions,
expressive libraries, and large-scale integration of multiple variants of
algorithms. We want library developers to be able to extend the system, just
like they do in Python, providing an extensible developer platform.

That said, we are not willing to sacrifice developer experience (including
compile times and error messages) nor are we interested in building a parallel
language ecosystem that is difficult to teach. We can learn from these previous
systems but also have new technologies to build on top of, including MLIR and
fine-grained language-integrated caching technologies.

As such, Mojo supports compile-time metaprogramming built
into the compiler as a separate stage of compilation—after parsing, semantic
analysis, and IR generation, but before lowering to target-specific code. It
uses the same host language for runtime programs as it does for metaprograms,
and leverages MLIR to represent and evaluate these programs predictably.

Let's take a look at some simple examples.

:::{.callout-note}

**About "parameters":** Python developers use the words "arguments" and
"parameters" fairly interchangeably for "things that are passed into
functions." We decided to reclaim "parameter" and "parameter expression" to
represent a compile-time value in Mojo, and continue to use "argument" and
"expression" to refer to runtime values. This allows us to align around words
like "parameterized" and "parametric" for compile-time metaprogramming.

:::

### Defining parameterized types and functions

You can parameterize structs and functions by specifying parameter names and
types in square brackets (using an extended version of the [PEP695
syntax](https://peps.python.org/pep-0695/)). Unlike argument values, parameter
values are known at compile-time, which enables an additional level of
abstraction and code reuse, plus compiler optimizations such as
[autotuning](#autotuning-adaptive-compilation).

For instance, let's look at a
[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) type,
which represents a low-level vector register in hardware that holds multiple
instances of a scalar data-type. Hardware accelerators are constantly
introducing new vector data types, and even CPUs may have 512-bit or longer SIMD
vectors. In order to access the SIMD instructions on these processors, the data
must be shaped into the proper SIMD width (data type) and length (vector size).

However, it's not feasible to define all the different SIMD variations with
Mojo's built-in types. So, Mojo's `SIMD` type (defined as a struct) exposes the
common SIMD operations in its methods, and makes the SIMD data type and size
values parametric. This allows you to directly map your data to the SIMD vectors
on any hardware. Here is a (cut down) version of Mojo's `SIMD` type:

```mojo
struct SIMD[type: DType, size: Int]:
    var value: … # Some low-level MLIR stuff here

    # Create a new SIMD from a number of scalars
    fn __init__(inout self, *elems: SIMD[type, 1]):  ...

    # Fill a SIMD with a duplicated scalar value.
    @staticmethod
    fn splat(x: SIMD[type, 1]) -> SIMD[type, size]: ...

    # Cast the elements of the SIMD to a different elt type.
    fn cast[target: DType](self) -> SIMD[target, size]: ...

    # Many standard operators are supported.
    fn __add__(self, rhs: Self) -> Self: ...
```

Defining each SIMD variant with parameters is great for code reuse because the
`SIMD` type can express all the different vector variants statically, instead of
requiring the language to pre-define every variant.

Because `SIMD` is a parameterized type, the `self` argument in its functions
carries those parameters—the full type name is `SIMD[type, size]`. Although
it's valid to write this out (as shown in the return type of `splat()`), this
can be verbose, so we recommend using the `Self` type (from
[PEP673](https://peps.python.org/pep-0673/)) like the `__add__` example does.

### Overloading on parameters

Functions and methods can be overloaded on their parameter signatures. The
overload resolution logic filters for candidates according to the following
rules, in order of precedence:

1) Candidates with the minimal number of implicit conversions (in both arguments
and parameters).
2) Candidates without variadic arguments.
3) Candidates without variadic parameters.
4) Candidates with the shortest parameter signature.

If there is more than one candidate after applying these rules, the overload
resolution fails. For example:

```mojo
@register_passable("trivial")
struct MyInt:
    """A type that is implicitly convertible to `Int`."""
    var value: Int

    @always_inline("nodebug")
    fn __init__(_a: Int) -> Self:
        return Self {value: _a}

fn foo[x: MyInt, a: Int](): pass

fn foo[x: MyInt, y: MyInt](): pass

fn bar[a: Int](b: Int): pass

fn bar[a: Int](*b: Int): pass

fn bar[*a: Int](b: Int): pass

fn waldo[a: Int](b: Int): pass

fn waldo[a: Int, T: AnyType](y: T): pass

fn parameterOverloads[a: Int, b: Int, x: MyInt]():
    # `foo[x: MyInt, a: Int]()` is called because it requires no implicit
    # conversions, whereas `foo[x: MyInt, y: MyInt]()` requires one.
    foo[x, a]()

    # `bar[x: Int](y: Int)` is called because it does not have variadic
    # arguments or parameters.
    bar[a](b)

    # `waldo[x: Int](y: Int)` is called because it has a shorter parameter
    # signature than `waldo[x: Int, T: AnyType](y: T)`.
    waldo[a](b)
```

### Using parameterized types and functions

You can instantiate parametric types and functions by passing values to the
parameters in square brackets. For example, for the `SIMD` type above, `type`
specifies the data type and `size` specifies the length of the SIMD vector (it
must be a power of 2):

```mojo
fn funWithSIMD():
    # Make a vector of 4 floats.
    let small_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    # Make a big vector containing 1.0 in float16 format.
    let big_vec = SIMD[DType.float16, 32].splat(1.0)

    # Do some math and convert the elements to float32.
    let bigger_vec = (big_vec+big_vec).cast[DType.float32]()

    # You can write types out explicitly if you want of course.
    let bigger_vec2 : SIMD[DType.float32, 32] = bigger_vec
```

Note that the `cast()` method also needs a parameter to specify the type you
want from the cast (the method definition above expects a `target` parametric
value). Thus, just like how the `SIMD` struct is a generic type definition, the
`cast()` method is a generic method definition that gets instantiated at
compile-time instead of runtime, based on the parameter value.

The `funWithSIMD()` function above shows the use of concrete types (that is, it
instantiates `SIMD` using known type values), but the major power of parameters
comes from the ability to define parametric algorithms and types (code that
uses the parameter values). For example, here's how to define a parametric
algorithm with `SIMD` that is type- and width-agnostic:

```mojo
fn rsqrt[dt: DType, width: Int](x: SIMD[dt, width]) -> SIMD[dt, width]:
    return 1 / sqrt(x)
```

Notice that the `x` argument is actually a `SIMD` type based on the function
parameters. The runtime program can use the value of parameters, because the
parameters are resolved at compile-time before they are needed by the runtime
program (but compile-time parameter expressions cannot use runtime values).

The Mojo compiler is also smart about type inference with parameters. Note
that the above function is able to call the parametric
[`sqrt()`](/mojo/MojoStdlib/Math.html#sqrt) function
without specifying the parameters—the compiler infers its parameters as if you
wrote `sqrt[type, simd_width](x)` explicitly. Also note that `rsqrt()` chose to
define its first parameter named `width` even though the `SIMD` type names it
`size`, and there is no problem.

### Parameter expressions are just Mojo code

A parameter expression is any code expression (such as `a+b`) that occurs where
a parameter is expected. Parameter expressions support operators and function
calls, just like runtime code, and all parameter types use the same type
system as the runtime program (such as `Int` and `DType`).

Because parameter expressions use the same grammar and types as runtime
Mojo code, you can use many "dependent type" features. For example, you might
want to define a helper function to concatenate two SIMD vectors:

```mojo
fn concat[ty: DType, len1: Int, len2: Int](
    lhs: SIMD[ty, len1], rhs: SIMD[ty, len2]) -> SIMD[ty, len1+len2]:
      ...

fn use_vectors(a: SIMD[DType.float32, 4], b: SIMD[DType.float16, 8]):
    let x = concat(a, a)  # Length = 8
    let y = concat(b, b)  # Length = 16
```

Note how the resulting length is the sum of the input vector lengths, and you
can express that with a simple `+` operation. For a more complex example, take
a look at the [`SIMD.shuffle()`](/mojo/MojoStdlib/SIMD.html#shuffle) method in
the standard library: it takes two input SIMD values, a vector shuffle mask as
a list, and returns a SIMD that matches the length of the shuffle mask.

### Powerful compile-time programming

While simple expressions are useful, sometimes you want to write imperative
compile-time logic with control flow. For example, the `isclose()` function in
the Mojo `Math` module uses exact equality for integers but "close" comparison
for floating-point. You can even do compile-time recursion. For instance, here
is an example "tree reduction" algorithm that sums all elements of a vector
recursively into a scalar:

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

This makes use of the `@parameter if` feature, which is an `if` statement that
runs at compile-time. It requires that its condition be a valid parameter
expression, and ensures that only the live branch of the `if` statement is
compiled into the program.

### Mojo types are just parameter expressions

While we've shown how you can use parameter expressions within types,
type annotations can themselves be arbitrary expressions (just like in Python).
Types in Mojo have a special metatype type, allowing type-parametric algorithms
and functions to be defined. For example, you can define an algorithm like the
C++ `std::vector` class like this:

```mojo
struct DynamicVector[type: AnyType]:
    ...
    fn reserve(inout self, new_capacity: Int): ...
    fn push_back(inout self, value: type): ...
    fn pop_back(inout self): ...
    fn __getitem__(self, i: Int) -> type: ...
    fn __setitem__(inout self, i: Int, value: type): ...

fn use_vector():
    var v = DynamicVector[Int]()
    v.push_back(17)
    v.push_back(42)
    v[0] = 123
    print(v[1])      # Prints 42
    print(v[0])      # Prints 123
```

Notice that the `type` parameter is used as the formal type for the
`value` arguments and the return type of the `__getitem__` function. Parameters
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

This is possible because the `func` parameter is allowed to refer to the
earlier `arg_type` parameter, and that refines its type in turn.

Another example where this is important is with variadic generics, where an
algorithm or data structure may need to be defined over a list of heterogeneous
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

### `alias`: named parameter expressions

It is very common to want to *name* compile-time values. Whereas `var` defines a
runtime value, and `let` defines a runtime constant, we need a way to define a
compile-time temporary value. For this, Mojo uses an `alias` declaration. For
example, the `DType` struct implements a simple enum using aliases for the
enumerators like this (the actual internal implementation details vary a bit):

```mojo
struct DType:
    var value : UI8
    alias invalid = DType(0)
    alias bool = DType(1)
    alias int8 = DType(2)
    alias uint8 = DType(3)
    alias int16 = DType(4)
    alias int16 = DType(5)
    ...
    alias float32 = DType(15)
```

This allows clients to use `DType.float32` as a parameter expression (which also
works as a runtime value) naturally. Note that this is invoking the
runtime constructor for `DType` at compile-time.

Types are another common use for alias: because types are compile-time
expressions, it is handy to be able to do things like this:

```mojo
alias Float32 = SIMD[DType.float32, 1]
alias UInt8 = SIMD[DType.uint8, 1]

var x : Float32   # Float32 works like a "typedef"
```

Like `var` and `let`, aliases obey scope, and you can use local aliases within
functions as you'd expect.

By the way, both `None` and `AnyType` are defined as [type
aliases](/mojo/MojoBuiltin/TypeAliases.html).

### Autotuning / Adaptive compilation

Mojo parameter expressions allow you to write portable parametric algorithms
like you can do in other languages, but when writing high-performance code you
still have to pick concrete values to use for the parameters. For example, when
writing high-performance numeric algorithms, you might want to use memory
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
individually, it might pick a different vector length for Float32 than for
Int8, for example. This simple feature is pretty powerful—going beyond simple
integer constants—because functions and types are also parameter expressions.

You can instrument the search of `exp_buffer_impl()` by providing a performance
evaluator and using the [`search()`](/mojo/MojoStdlib/Autotune.html#search)
standard library function. `search()` takes an evaluator and a forked function
and returns the fastest implementation selected by the evaluator as a parameter
result.

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

Autotuning has an exponential runtime. It benefits from internal implementation
details of the Mojo compiler stack (particularly MLIR, integrated caching, and
distribution of compilation). This is a power-user feature and needs continued
development and iteration over time.

## "Value Lifecycle": Birth, life and death of a value {#value-lifecycle}

At this point, you should understand the core semantics and features for Mojo
functions and types, so we can now discuss how they fit together to express
new types in Mojo.

Many existing languages express design points with different tradeoffs: C++, for
example, is very powerful but often accused of
"getting the defaults wrong" which leads to bugs and mis-features.  Swift is
easy to work with, but has a less predictable model that copies values a lot and
is dependent on an "ARC optimizer" for performance. Rust started with strong
value ownership goals to satisfy its borrow checker, but relies on values being
movable, which makes it challenging to express custom move constructors and
can put a lot of stress on `memcpy` performance. In Python, everything is a
reference to a class, so it never really faces issues with types.

For Mojo, we've learned from these existing systems, and we aim to
provide a model that's very powerful while still easy to learn and understand.
We also don't want to require "best effort" and difficult-to-predict
optimization passes built into a "sufficiently smart" compiler.

To explore these issues, we look at different value classifications and the
relevant Mojo features that go into expressing them, and build from the
bottom-up. We use C++ as the primary comparison point in examples because it is
widely known, but we occasionally reference other languages if they provide a
better comparison point.

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
instance of this `NoInstances` type.  In order to get them, you need to define
an `__init__` method or use a decorator that synthesizes an initializer.  As
shown, these types can be useful as "namespaces" because you can refer to
static members like `NoInstances.my_int` or `NoInstances.print_hello()` even
though you cannot instantiate an instance of the type.

### Non-movable and non-copyable types

If we take a step up the ladder of sophistication, we’ll get to types that can
be instantiated, but once they are pinned to an address in memory, they cannot
be implicitly moved or copied.  This can be useful to implement types like
atomic operations (such as `std::atomic` in C++) or other types where the memory
address of the value is its identity and is critical to its purpose:

```mojo
struct Atomic:
    var state: Int

    fn __init__(inout self, state: Int = 0):
        self.state = state

    fn __iadd__(inout self, rhs: Int):
        #...atomic magic...

    fn get_value(self) -> Int:
        return atomic_load_int(self.state)
```

This class defines an initializer but no copy or move constructors, so once it
is initialized it can never be moved or copied.  This is safe and useful because
Mojo's ownership system is fully "address correct" - when this is initialized
onto the stack or in the field of some other type, it never needs to move.

Note that Mojo’s approach controls only the built-in move operations, such as
`a = b` copies and the [`^` transfer operator](#owned-arguments). One
useful pattern you can use for your own types (like `Atomic` above) is to add
an explicit `copy()` method (a non-"dunder" method). This can be useful to make
explicit copies of an instance when it is known safe to the programmer.

### Unique "move-only" types

If we take one more step up the ladder of capabilities, we will encounter types
that are "unique" - there are many examples of this in C++, such as types like
`std::unique_ptr` or even a `FileDescriptor` type that owns an underlying POSIX
file descriptor. These types are pervasive in languages like Rust, where
copying is discouraged, but "move" is free. In Mojo, you can implement these
kinds of moves by defining the `__moveinit__` method to take ownership of a
unique type. For example:

```mojo
# This is a simple wrapper around POSIX-style fcntl.h functions.
struct FileDescriptor:
    var fd: Int

    # This is how we move our unique type.
    fn __moveinit__(inout self, owned existing: Self):
        self.fd = existing.fd

    # This takes ownership of a POSIX file descriptor.
    fn __init__(inout self, fd: Int):
        self.fd = fd

    fn __init__(inout self, path: String):
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

The consuming move constructor (`__moveinit__`) takes ownership of an existing
`FileDescriptor`, and moves its internal implementation details over to a new
instance.  This is because instances of `FileDescriptor` may exist at different
locations, and they can be logically moved around—stealing the body of one
value and moving it into another.

Here is an egregious example that will invoke `__moveinit__` multiple times:

```mojo
fn egregious_moves(owned fd1: FileDescriptor):
    # fd1 and fd2 have different addresses in memory, but the
    # transfer operator moves unique ownership from fd1 to fd2.
    let fd2 = fd1^

    # Do it again, a use of fd2 after this point will produce an error.
    let fd3 = fd2^

    # We can do this all day...
    let fd4 = fd3^
    fd4.read(...)
    # fd4.__del__() runs here
```

Note how ownership of the value is transferred between various values that own
it, using the postfix-`^` "transfer" operator, which destroys a previous
binding and transfer ownership to a new constant. If you are familiar with C++,
the simple way to think about the transfer operator is like `std::move`, but in
this case, we can see that it is able to move things without resetting them to
a state that can be destroyed: in C++, if your move operator failed to change
the old value’s `fd` instance, it would get closed twice.

Mojo tracks the liveness of values and allows you to define custom move
constructors. This is rarely needed, but extremely powerful when it is. For
example, some types like the [`llvm::SmallVector
type`](https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h)
use the "inline storage" optimization technique, and they may want to be
implemented with an "inner pointer" into their instance. This is a well-known
trick to reduce pressure on the malloc memory allocator, but it means that a
"move" operation needs custom logic to update the pointer when that happens.

With Mojo, this is as simple as implementing a custom `__moveinit__` method.
This is something that is also easy to implement in C++ (though, with
boilerplate in the cases where you don’t need custom logic) but is difficult to
implement in other popular memory-safe languages.

One additional note is that while the Mojo compiler provides good predictability
and control, it is also very sophisticated.  It reserves the right to eliminate
temporaries and the corresponding copy/move operations.  If this is
inappropriate for your type, you should use explicit methods like `copy()`
instead of the dunder methods.

### Types that support a "stealing move"

One challenge with memory-safe languages is that they need to provide a
predictable programming model around what the compiler is able to track, and
static analysis in a compiler is inherently limited.  For example, while it is
possible for a compiler to understand that the two array accesses in the
first example below are to different array elements, it is (in general)
impossible to reason about the second example (this is C++ code):

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
individual elements of the array are live, this often causes significant
runtime expense (even when move/transfers are not used), which is something that
Mojo and other systems programming languages are not keen to do.  There are a
variety of ways to deal with this, including some pretty complicated solutions
that aren’t always easy to learn.

Mojo takes a pragmatic approach to let Mojo programmers get their job done
without having to work around its type system. As seen above, it doesn’t force
types to be copyable, movable, or even constructable, but it does want types to
express their full contract, and it wants to enable fluent design patterns that
programmers expect from languages like C++.  The (well known) observation here
is that many objects have contents that can be "stolen" without needing to
disable their destructor, either because they have a "null state" (like an
optional type or nullable pointer) or because they have a null value that is
efficient to create and a no-op to destroy (e.g. `std::vector` can have a null
pointer for its data).

To support these use-cases, the [`^` transfer operator](#owned-arguments)
supports arbitrary LValues, and when applied to one, it invokes the "stealing
move constructor." This constructor must set up the new value to be in a live
state, and it can mutate the old value, but it must put the old value into a
state where its destructor still works. For example, if we want to put our
`FileDescriptor` into a vector and move out of it, we might choose to extend it
to know that `-1` is a sentinel which means that it is "null". We can implement
this like so:

```mojo
# This is a simple wrapper around POSIX-style fcntl.h functions.
struct FileDescriptor:
    var fd: Int

    # This is the new key capability.
    fn __moveinit__(inout self, inout existing: Self):
        self.fd = existing.fd
        existing.fd = -1  # neutralize 'existing'.

    fn __moveinit__(inout self, owned existing: Self): # as above
    fn __init__(inout self, fd: Int): # as above
    fn __init__(inout self, path: String): # as above

    fn __del__(owned self):
        if self.fd != -1:
            close(self.fd)   # pseudo code, call close(2)
```

Notice how the "stealing move" constructor takes the file descriptor from an
existing value and mutates that value so that its destructor won’t do anything.
This technique has tradeoffs and is not the best for every type. We can see
that it adds one (inexpensive) branch to the destructor because it has to check
for the sentinel case. It is also generally considered bad form to make types
like this nullable because a more general feature like an `Optional[T]` type is
a better way to handle this.

Furthermore, we plan to implement `Optional[T]` in Mojo itself, and `Optional`
needs this functionality.  We also believe that the library authors understand
their domain problem better than language designers do, and generally prefer to
give library authors full power over that domain.  As such you can choose (but
don’t have to) to make your types participate in this behavior in an opt-in way.

### Copyable types

The next step up from movable types are copyable types.  Copyable types are
also very common - programmers generally expect things like strings and arrays
to be copyable, and every Python Object reference is copyable - by copying the
pointer and adjusting the reference count.

There are many ways to implement copyable types.  One can implement reference
semantic types like Python or Java, where you propagate shared pointers around,
one can use immutable data structures that are easily shareable because they are
never mutated once created, and one can implement deep value semantics through
lazy copy-on-write as Swift does.  Each of these approaches has different
tradeoffs, and Mojo takes the opinion that while we want a few common sets of
collection types, we can also support a wide range of specialized ones that
focus on particular use cases.

In Mojo, you can do this by implementing the `__copyinit__` method.  Here is an
example of that using a simple `String` in pseudo-code:

```mojo
struct MyString:
    var data: Pointer[UI8]

    # StringRef is a pointer + length and works with StringLiteral.
    def __init__(inout self, input: StringRef):
        self.data = ...

    # Copy the string by deep copying the underlying malloc'd data.
    def __copyinit__(inout self, existing: Self):
        self.data = strdup(existing.data)

    # This isn't required, but optimizes unneeded copies.
    def __moveinit__(inout self, owned existing: Self):
        self.data = existing.data

    def __del__(owned self):
        free(self.data.address)

    def __add__(self, rhs: MyString) -> MyString: ...
```

This simple type is a pointer to a "null-terminated" string data allocated with
malloc, using old-school C APIs for clarity. It implements the `__copyinit__`,
which maintains the invariant that each instance of `MyString` owns its
underlying pointer and frees it upon destruction. This implementation builds on
tricks we’ve seen above, and implements a `__moveinit__` constructor, which
allows it to completely eliminate temporary copies in some common cases. You
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

In this case, you can see both why a copy constructor is needed: without one,
the duplication of the `s1` value into `s2` would be an error - because you
cannot have two live instances of the same non-copyable type. The move
constructor is optional but helps the assignment into `s3`: without it, the
compiler would invoke the copy constructor from s1, then destroy the old `s1`
instance. This is logically correct but introduces extra runtime overhead.

Mojo destroys values eagerly, which allows it to transform
copy+destroy pairs into single move operations, which can lead to much better
performance than C++ without requiring the need for pervasive micromanagement
of `std::move`.

### Trivial types

The most flexible types are ones that are just "bags of bits".  These types are
"trivial" because they can be copied, moved, and destroyed without invoking
custom code.  Types like these are arguably the most common basic type that
surrounds us: things like integers and floating point values are all trivial.
From a language perspective, Mojo doesn’t need special support for these, it
would be perfectly fine for type authors to implement these things as no-ops,
and allow the inliner to just make them go away.

There are two reasons that approach would be suboptimal: one is that we don’t
want the boilerplate of having to define a bunch of methods on trivial types,
and second, we don’t want the compile-time overhead of generating and pushing
around a bunch of function calls, only to have them inline away to nothing.
Furthermore, there is an orthogonal concern, which is that many of these types
are trivial in another way: they are tiny, and should be passed around in the
registers of a CPU, not indirectly in memory.

As such, Mojo provides a struct decorator that solves all of these problems.
You can implement a type with the `@register_passable("trivial")` decorator,
and this tells Mojo that the type should be copyable and movable but that it
has no user-defined logic for doing this.  It also tells Mojo to prefer to pass
the value in CPU registers, which can lead to efficiency benefits.

TODO: This decorator is due for reconsideration.  Lack of custom logic
copy/move/destroy logic and "passability in a register" are orthogonal concerns
and should be split.  This former logic should be subsumed into a more general
`@value("trivial")` decorator, which is orthogonal from `@register_passable`.

### `@value` decorator

Mojo's approach (described above) provides simple and predictable hooks that
give you the ability to express exotic low-level things like `Atomic` correctly.
This is great for control and for a simple programming model, but most structs
we all write are simple aggregations of other types, and we don't want to have
to write a lot of boilerplate for them! To solve this, Mojo provides a `@value`
decorator for structs that synthesizes the boilerplate for you. `@value` can be
thought of as an extension of Python's `@dataclass` handling the new
`__moveinit__` and `__copyinit__` Mojo methods.

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
fn __init__(inout self, owned name: String, age: Int):
    self.name = name^
    self.age = age

fn __copyinit__(inout self, existing: Self):
    self.name = existing.name
    self.age = existing.age

fn __moveinit__(inout self, owned existing: Self):
    self.name = existing.name^
    self.age = existing.age
```

Mojo synthesizes each of these only when it doesn't exist, so it's okay to
override its behavior by defining your own version for just one or more. For
example, it is fairly common to want a custom copy constructor but use the
default member-wise and move constructor.

:::{.callout-note}

**Note:** If your type contains any [move-only](#unique-move-only-types)
fields, Mojo will not generate a copy constructor because it cannot copy those
fields.

:::

There is no way to suppress the generation of specific methods or customize
generation at this time, but we can add arguments to the `@value` generator to
do this if there is demand.

Note that the `@value` decorator only works on types whose members are copyable
and/or movable.  If you have something like `Atomic` in your struct, then it
probably isn't a value type, and you don't want these members anyway.

## Behavior of destructors

Any struct in Mojo can have a destructor (a `__del__()` method), which is
automatically run when the value's lifetime ends (typically the point at which
the value is last used). For example, a simple string might look like this (in
pseudo code):

```mojo
@value
struct MyString:
    var data: Pointer[UInt8]

    def __init__(inout self, input: StringRef): ...
    def __add__(self, rhs: MyString) -> MyString: ...
    def __del__(owned self):
        free(self.data.address)
```

Mojo destroys values like `MyString` (it calls the `__del__()` destructor)
using an **"As Soon As Possible"** (ASAP) policy that runs after every call.
Mojo does *not* wait until the end of the code block to destroy unused values.
Even in an expression like `a+b+c+d`, Mojo destroys the intermediate
expressions eagerly, as soon as they are no longer needed—it does not wait
until the end of the statement.

The Mojo compiler automatically invokes the destructor when a value is dead
and provides strong guarantees about when the destructor is run. Mojo uses
static compiler analysis to reason about your code and decide when to insert
calls to the destructor. For example:

```mojo
fn use_strings():
    var a = MyString("hello a")
    var b = MyString("hello b")
    print(a)
    # a.__del__() runs here for "hello a"


    print(b)
    # b.__del__() runs here

    a = MyString("temporary a")
    # a.__del__() runs here because "temporary a" is never used

    other_stuff()

    a = MyString("final a")
    print(a)
    # a.__del__() runs again here for "final a"
```

In the code above, you’ll see that the `a` and `b` values are created early on,
and each initialization of a value is matched with a call to a destructor.
Notice that `a` is destroyed multiple times—once for each time it receives a
new value.

Now, this might be surprising to a C++ programmer, because it's different from
the [RAII pattern](https://en.cppreference.com/w/cpp/language/raii) in which
C++ destroys values at the end of a scope. Mojo also follows the principle
that values acquire resources in a constructor and release resources in a
destructor, but eager destruction in Mojo has a number of strong advantages
over scope-based destruction in C++:

- The Mojo approach eliminates the need for types to implement re-assignment
  operators, like `operator=(const T&)` and `operator=(T&&)` in C++, making it
  easier to define types and eliminating a concept.

- Mojo does not allow mutable references to overlap with other mutable
  references or with immutable borrows. One major way that it provides a
  predictable programming model is by making sure that references to objects die
  as soon as possible, avoiding confusing situations where the compiler thinks a
  value could still be alive and interfere with another value, but that isn’t
  clear to the user.

- Destroying values at last-use composes nicely with "move" optimization, which
  transforms a "copy+del" pair into a "move" operation, a generalization of
  C++ move optimizations like NRVO (named return value optimization).

- Destroying values at end-of-scope in C++ is problematic for some common
  patterns like tail recursion because the destructor calls happen after the
  tail call. This can be a significant performance and memory problem for
  certain functional programming patterns.

Importantly, Mojo's eager destruction also works well within Python-style `def`
functions to provide destruction guarantees (without a garbage collector) at a
fine-grain level—recall that Python doesn’t really provide scopes beyond a
function, so C++-style destruction in Mojo would be a lot less useful.

:::{.callout-note}

**Note:** Mojo also supports the Python-style [`with`
statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement),
which provides more deliberately-scoped access to resources.

:::

The Mojo approach is more similar to how Rust and Swift work, because they both
have strong value ownership tracking and provide memory safety.  One difference
is that their implementations require the use of a [dynamic "drop
flag"](https://doc.rust-lang.org/nomicon/drop-flags.html)—they maintain hidden
shadow variables to keep track of the state of your values to provide safety.
These are often optimized away, but the Mojo approach eliminates this overhead
entirely, making the generated code faster and avoiding ambiguity.

### Field-sensitive lifetime management

In addition to Mojo’s lifetime analysis being fully control-flow aware, it is
also fully field-sensitive (each field of a structure is tracked
independently). That is, Mojo separately keeps track of whether a "whole
object" is fully or only partially initialized/destroyed.

For example, building upon the `MyString` struct defined above,
consider this code:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString
    fn __init__(inout self): ...
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
this when using the [transfer operator](#owned-arguments), for example:

```mojo
fn consume_and_use_two_strings():
    var ts = TwoStrings()
    consume(ts.str1^)
    # ts.str1.__moveinit__() runs here

    # ts is now only partially initialized here!
    other_stuff()

    ts.str1 = MyString()  # All together now
    use(ts)               # This is ok
    # ts.__del__() runs here
```

Notice that the code transfers ownership of the `str1` field: for the duration
of `other_stuff()`, the `str1` field is completely uninitialized because
ownership was transferred to `consume()`. Then
`str1` is reinitialized before it is used by the `use()` function (if it
weren’t, Mojo would reject the code with an uninitialized field error).

Mojo's rule on this is powerful and intentionally straight-forward: fields can
be temporarily transferred, but the "whole object" must be constructed with the
aggregate type’s initializer and destroyed with the aggregate destructor. This
means that it isn’t possible to create an object by initializing only its
fields, nor is it possible to tear down an object by destroying only its
fields. For example, this code does not compile:

```mojo
fn consume_and_use_two_strings():
    var ts = TwoStrings() # ts is initialized
    consume(ts.str1^)
    consume(ts.str2^) # Both members are transferred/destroyed
    # Error: cannot run the 'ts' destructor without initialized fields

    var ts2 : TwoStrings # ts2 type is declared but not initialized
    ts2.str1 = MyString()
    ts2.str2 = MyString()  # Both the member are initalized
    use(ts2) # Error: 'ts2' isn't fully initialized
```

While we could allow patterns like this to happen, we reject this because a
value is more than a sum of its parts.  Consider a `FileDescriptor` that
contains a POSIX file descriptor as an integer value: there is a
big difference between destroying the integer (a no-op) and destroying the
`FileDescriptor` (it might call the `close()` system call).  Because of this, we
require all full-value initialization to go through initializers and be
destroyed with their full-value destructor.

For what it's worth, Mojo does internally have an equivalent of the Rust
[`mem::forget`](https://doc.rust-lang.org/std/mem/fn.forget.html) function,
which explicitly disables a destructor and has a corresponding internal feature
for "blessing" an object, but they aren’t exposed for user consumption at this
point.

### Field lifetimes in `__init__`

The behavior of an `__init__` method works almost like any other method - there
is a small bit of magic: it knows that the fields of an object
are uninitialized, but it believes the full object is initialized.  This means
that you can use `self` as a whole object as soon as all the fields are
initialized:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString

    fn __init__(inout self, cond: Bool, other: MyString):
        self.str1 = MyString()
        if cond:
            self.str2 = other
            use(self)  # Safe to use immediately!
            # self.str2.__del__(): destroyed because overwritten below.

        self.str2 = self.str1
        use(self)  # Safe to use immediately!
```

Similarly, it is safe for initializers in Mojo to completely
overwrite `self`, such as by delegating to other initializers:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString

    fn __init__(inout self): ...
    fn __init__(inout self, cond: Bool, other: MyString):
        self = TwoStrings()  # basic
        self.str1 = MyString("fancy")
```

### Field lifetimes of `owned` arguments in `__del__` and `__moveinit__`

A final bit of magic exists for the `owned` arguments of a move
initializer and a destructor.  To recap, these methods are defined like this:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString
    fn __init__(...)

    fn __moveinit__(inout self, owned existing: Self): ...
    fn __del__(owned self): ...
```

These methods face an interesting but obscure problem: both methods
are in charge of dismantling the `owned` `existing`/`self` value. That is,
`__moveinit__()` destroys sub-elements of `existing` in order to transfer
ownership to a new instance, while `__del__()` implements the
deletion logic for its `self`. As such, they both
want to own and transform elements of the `owned` value, and they definitely
don’t want the `owned` value's destructor to run (in the case of the `__del__()`
method, that would turn into an infinite loop).

To solve this problem, Mojo handles these two methods specially by assuming
that their whole values are destroyed upon reaching any return from the method.
This means that the whole object may be used before the field values are
transferred. For example, this works as you expect:

```mojo
struct TwoStrings:
    var str1: MyString
    var str2: MyString
    fn __init__(...)
    fn __moveinit__(inout self, owned existing: Self): ...

    fn __del__(owned self):
        log(self)       # Self is still whole
        # self.str2.__del__(): Mojo destroys str2 since it isn't used

        consume(self.str1^)
        # Everything has now been transferred, no destructor is run on self.
```

You should not generally have to think about this, but if you have logic with
inner pointers into members, you may need to keep them alive for some logic
within the destructor or move initializer itself.  You can do this by assigning
to the `_` "discard" pattern:

```mojo
fn __del__(owned self):
    log(self) # Self is still whole

    consume(self.str1^)
    _ = self.str2
    # self.str2.__del__(): Mojo destroys str2 after its last use.
```

In this case, if `consume()` implicitly refers to some value in `str2` somehow,
this will ensure that `str2` isn’t destroyed until the last use when it is
accessed by the `_` discard pattern.

## Lifetimes

TODO: Explain how returning references work, tied into lifetimes which dovetail
with parameters.  This is not enabled yet.

## Type traits

This is a feature very much like Rust traits or Swift protocols or Haskell type
classes. Note, this is not implemented yet.

## Advanced/Obscure Mojo features

This section describes power-user features that are important for building the
bottom-est level of the standard library. This level of the stack is inhabited
by narrow features that require experience with compiler internals to
understand and utilize effectively.

### `@register_passable` struct decorator

The default model for working with values is
they live in memory, so they have an identity, which means they are passed
indirectly to and from functions (equivalently, they are passed "by reference"
at the machine level). This is great for types that cannot be moved, and is a
safe default for large objects or things with expensive copy operations.
However, it is inefficient for tiny things like a single integer or
floating point number.

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
architecture).

There are only a few observable effects of this decorator to the typical Mojo
programmer:

1. `@register_passable` types are not able to hold instances of types
that are not themselves `@register_passable`.

2. Instances of `@register_passable` types do not have predictable identity,
and so the `self` pointer is not stable/predictable (e.g. in hash tables).

3. `@register_passable` arguments and result are exposed to C and C++ directly,
instead of being passed by-pointer.

4. The `__init__` and `__copyinit__` methods of this type are implicitly static
(like `__new__` in Python) and returns its result by-value instead of taking
`inout self`.

We expect that this decorator will be used pervasively on core standard library
types, but is safe to ignore for general application level code.

The `Int` example above actually uses the "trivial" variant of this decorator.
It changes the passing convention as described above but also disallows copy
and move constructors and destructors (synthesizing them all trivially).

> TODO: Trivial needs to be decoupled to its own decorator since it applies to
memory types as well.

<!--
TOWRITE: Each builtin decorator should be mentioned. Eventually, decorators
should appear in the API docs.

> TODO: We need to decide how to namespace these, should these go into a 'mojo'
package or something?
-->

### `@always_inline` decorator

`@always_inline("nodebug")`: same thing but without debug information so you
don't step into the + method on Int.

### `@parameter` decorator

The `@parameter` decorator can be placed on nested functions that capture
runtime values to create "parametric" capturing closures. This is an unsafe
feature in Mojo, because we do not currently model the lifetimes of
capture-by-reference. A particular aspect of this feature is that it allows
closures that capture runtime values to be passed as parameter values.

### Magic operators

C++ code has a number of magic operators that intersect with value lifecycle,
things like "placement new", "placement delete" and "operator=" that reassign
over an existing value. Mojo is a safe language when you use all its language
features and compose on top of safe constructs, but of any stack is a world of
C-style pointers and rampant unsafety. Mojo is a pragmatic language, and since
we are interested in both interoperating with C/C++ and in implementing safe
constructs like String directly in Mojo itself, we need a way to express unsafe
things.

The Mojo standard library `Pointer[element_type]` type is implemented with an
underlying `!pop.pointer<element_type>` type in MLIR, and we desire a way to
implement these C++-equivalent unsafe constructs in Mojo. Eventually, these
will migrate to all being methods on the Pointer type, but until then, some
need to be exposed as built-in operators.

<!--
TODO: document all of these:

```mojo
__get_address_as_lvalue(x)
__get_address_as_uninit_lvalue(x)
__get_lvalue_as_address(x):  use Pointer.address_of instead
__get_address_as_owned_value(x)
```
-->

### Direct access to MLIR

Mojo provides full access to the MLIR dialects and ecosystem. Please take a
look at the [Low level IR in Mojo](/mojo/notebooks/BoolMLIR.html) to learn how
to use the `__mlir_type`, `__mlir_op`, and `__mlir_type` constructs. All of the
built-in and standard library APIs are implemented by just calling the
underlying MLIR constructs, and in doing so, Mojo effectively serves as syntax
sugar on top of MLIR.
