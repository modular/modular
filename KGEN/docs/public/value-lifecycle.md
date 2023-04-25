# Mojo’s “Value Lifecycle”: Birth, life and death of a value

One of the subtle aspects of a programming language is its approach to defining
types, what capabilities those types can expose, and how
general/flexible/ergonomic it is to use.  This document is a detailed
exploration of the capabilities and features of the Mojo language, and how it is
designed and how the relevant features work.  These features are key underlying
components of its ownership and memory safety model.

Mojo isn’t the first language to try to pin this down - there are many languages
to learn from including C++, Rust, Swift and many more.  Each of these has
different tradeoffs: C++, for example, is very powerful but often accused of
“getting the defaults wrong” which leads to bugs and mis-features.  Swift is
easy to work with, but has a less predictable model that copies values a lot and
is dependent on an “ARC optimizer” for performance. Rust started with strong
value ownership goals to satisfy its borrow checker, but relies on values being
movable, and makes it challenging to express custom move constructors. In
Python, everything is a reference to a class, so it has never really faced these
issues.

We aimed to learn from these and other languages to provide a model that is very
powerful while still easy to learn and understand, and without requiring “best
effort” and difficult-to-predict optimization passes.  We use C++ as the primary
comparison point in examples because it is widely known but occasionally
reference other languages if they provide a better comparison point.

## Defining types with various capabilities

To explore these issues, we look at different value classifications and the
relevant Mojo features that go into expressing them, and build from the
bottom-up.  This document is not meant to be a tutorial on how to define structs
in Mojo, nor is it meant to define best practices - instead it is a detailed
exploration of the features.

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
instance of this “`NoInstances`” type.  In order to get them you need to define
an `__init__` method or use a decorator that synthesizes an initializer.  As
shown, these types can be useful as “namespaces”, because you can refer to
static members like “`NoInstances.my_int`” or “`NoInstances.print_hello()`” even
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
Mojo's ownership system is fully “address correct” - when this is initialized
onto the stack or in the field of some other type, it never needs to move.

Note that Mojo’s approach just controls the builtin operations like `a = b`
copies and the `x^` consume operator.  One useful pattern that can be used for
types like this is to add an explicit `copy()` method (a non-“dunder” method)
which can be useful to explicitly make copies of an instance when it is known
safe to the programmer.

### Unique “move-only” types

If we take one more step up the ladder of capabilities, we will encounter types
that are “unique” - there are many examples of this in C++, e.g. types like
`std::unique_ptr`, or even a `FileDescriptor` type that owns an underlying POSIX
file descriptor.  These types are pervasive in languages like Rust, where
copying is discouraged, but “move” is free. In Mojo, you can declare these by
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


The new concept is that we added a “consuming move constructor” which is named
“`__moveinit__`”.  The consuming move initializer takes ownership of an existing
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

   # We can do this all day…
   let fd4 = fd3^
   fd4.read(...)
   # fd4.__del__() runs here
```

Note how ownership of the value is transferred between various values that own
it, using the postfix-`^` ‘consume’ operator to destroy a previous binding.  If
you are familiar with C++, the simple way to think about the consume operator is
like “`std::move`”, but in this case, we can see that it is able to move things
without resetting them to a state that can be destroyed: in C++, if your move
operator failed to change the old value’s “fd” instance, it would get closed
twice.

Mojo tracks the liveness of values and allows you to define custom move
constructors.  This is rarely needed, but extremely powerful when it is.  For
example, some types like the
<code>[llvm::SmallVector type](https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h)</code>
use the “inline storage” optimization technique, and they may want to be
implemented with an “inner pointer” into their instance.  This is a well known
trick to reduce pressure on the malloc memory allocator, but it means that a
“move” operation needs custom logic to update the pointer when that happens.

With Mojo, this is as simple as implementing a custom `__moveinit__` method.
This is something that is also easy to implement in C++ (though, with
boilerplate in the cases where you don’t need custom logic) but is difficult to
implement in other popular memory safe languages.

One additional note is that while the Mojo compiler provides good predictability
and control, it is also very sophisticated.  It reserves the rights to eliminate
temporaries and the corresponding copy/move operations.  If this is
inappropriate for your type, you should use explicit methods like `copy()`
instead of the dunder methods.

### Types that support a “Stealing Move”

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
is that many objects have contents that can be “stolen” without needing to
disable their destructor, either because they have a “null state” (like an
optional type or nullable pointer) or because they have a null value that is
efficient to create and a no-op to destroy (e.g. `std::vector` can have a null
pointer for its data).

To support these use-cases, the consume operator supports arbitrary LValues, and
when applied to one, it invokes the “stealing move constructor”.  This
constructor must set up the new value to be in a live state, and can mutate the
old value, but needs to put it into a state where its destructor will still
work.  For example, if we want to put our `FileDescriptor` into a vector and
move out of it, we might choose to extend it to know that “-1” is a sentinel
that means that it is “null”.  We can implement this like so:

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

Notice how the “stealing move” constructor takes the file descriptor from an
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
          self.data = …

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

This simple type is a pointer to a “null terminated” string data allocated with
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
duplication of the “`s1`” value into “`s2`” would be an error - because you
cannot have two live instances of the same non-copyable type.  The move
constructor is optional, but helps the assignment into “`s3`”: without it, the
compiler would invoke the copy constructor from s1, then destroy the old "`s1`"
instance.  This is logically correct, but introduces extra runtime overhead.

Mojo destroys values eagerly, which allows it to use frequently transform
copy+destroy pairs into a move operation, which can lead to much better
performance than C++ without requiring the need for pervasive micro-management
of `std::move`.

### Trivial Types

The most flexible types are ones that are just “bags of bits”.  These types are
“trivial” because they can be copied, moved, and destroyed without invoking
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
You can implement a type with the “`@register_passable("trivial")`” decorator,
and this tells Mojo that the type should be copyable and movable, but that it
has no user-defined logic for doing this.  It also tells Mojo to prefer to pass
the value in CPU registers, which can lead to efficiency benefits.

TODO: This decorator is due for a reconsideration.  Lack of custom logic
copy/move/destroy logic, and “passability in a register” are orthogonal concerns
and should be split.  This former logic should be subsumed into a more general
“`@value("trivial")`” decorator which is orthogonal from “`@register_passable`”.

### Boilerplate eliminating decorators

TODO: Describe the `@value` /  `@mojo.value` struct decorator.  This synthesizes
copy/move/memberwise constructors based on the availability of stored
properties, it takes string arguments that customize its behavior.

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

Mojo destroys values using an "as soon as possible" (ASAP) policy, behaving like
a hyper-active garbage collector that is run after every call - and when we say
every call, we mean it!  Code that uses internal expressions (like `a+b+c+d`)
will destroy the intermediate expressions eagerly when they are not needed -
destruction is not deferred to the end of the statement like in C++. Mojo also
fully understands control flow, including loops, ifs, and try/except.

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
5. Destroying values at last use composes nicely with “move” optimization,
   which transforms a “copy+del” pair into a “move” of a value, a generalization
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

## Field Sensitive Lifetime Management

In addition to Mojo’s lifetime analysis being fully control flow aware, it is
also fully field sensitive (each field of a structure is tracked independently).
It separately keeps track of whether a “whole object” is initialized with an
initializer or destroyed with a whole object destructor.  For example, consider
this code:

```mojo
    struct TwoStrings:
      var str1: MyString
      var str2: MyString
      fn __init__(self&): …
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

While we could allow patterns like this to happen, we reject this because a
value is more than a sum of its parts.  Consider a `FileDescriptor` that
contains an POSIX file descriptor as an integer value for example - there is a
big difference between destroying the integer (a noop!) and destroying the
`FileDescriptor` (it might call the `close()` system call).  Because of this, we
require all full value initialization to go through initializers and be
destroyed with their full value destructor.

For what it's worth, Mojo does internally have an equivalent of the Rust
"[mem::forget](https://doc.rust-lang.org/std/mem/fn.forget.html)” function which
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
In this case, if “consume” implicitly refers to some value in “str2” somehow,

this will ensure that str2 isn’t destroyed until the last use when it is
accessed by the `_` pattern.
