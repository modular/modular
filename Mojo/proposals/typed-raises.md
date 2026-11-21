# Typed raises in Mojo

Status: Implemented

## Introduction

Mojo supports typed error handling, allowing functions to declare the type of
error they may raise. Unlike conventional exception systems, which often rely on
a common exception class hierarchy and stack unwinding, Mojo treats errors as
ordinary values that are returned through an alternate function result.

This design provides efficient, statically typed error handling while preserving
the flexibility needed for systems programming, generic libraries, and GPU
programming.

For example:

```mojo
@fieldwise_init
struct ParseError:
    var offset: Int
    var message: String

def parse(input: String) raises ParseError -> Int:
    if input == "":
        raise ParseError(0, "Empty input")
    return 42

def example():
    try:
        print(parse(""))
    except err:
        # err is statically known to be a ParseError.
        print(err.offset, ": ", err.message)
```

The error is an ordinary Mojo struct. The compiler knows its type statically,
allowing the handler to access its fields without dynamic type checks or
downcasts.

This document explains the design and semantics of typed raises, including its
interaction with function types, generic programming, implicit conversions,
no-return functions, and error handling constructs.

## Motivation and design goals

Mojo's original error-handling system used the built-in `Error` type for every
throwing function. While simple and convenient, this model imposed several
limitations.

First, the `Error` type is fundamentally a dynamically descriptive error: it
carries a message rather than providing a statically known structure. This is
useful for reporting problems to users, but less useful when callers need to
understand and recover from particular error conditions.

Second, some Mojo environments have strict constraints on allocation and runtime
facilities. In particular, GPU kernels should be able to report failures using
small, statically known types without requiring strings and heap allocation.

Third, Mojo's generic programming model benefits from treating errors as
ordinary types. A collection algorithm should be able to propagate the error
type of a callback without introducing special-case language features or erasing
error information.

The overall design of Mojo error handling has several goals:

1. **Efficient error handling.** Raising an error should be as efficient as
   returning an alternate result, without requiring stack unwinding or dynamic
   exception machinery.

2. **Strong static typing.** Functions should be able to explicitly communicate
   which error type they produce, allowing callers to handle structured errors
   without casts.

3. **No special error hierarchy.** Any Mojo type should be usable as an error
   type, including register-passable and memory-only types.

4. **Ergonomics for dynamic error handling** Existing functions using bare
   `raises` should continue working, with `Error` remaining the convenient
   default.

5. **Generic composition.** Error types should participate in parametric
   functions, function values, and implicit conversions using the normal Mojo
   type system.

6. **Zero-overhead non-raising functions.** Generic functions instantiated with
   an error type that cannot occur should compile into ordinary non-raising
   functions.

The overall philosophy is to make error handling a natural extension of Mojo's
existing value, type, and function systems, rather than introducing a separate
exception mechanism.

## Basic typed raises

A function may declare its error type by placing a type after the `raises`
keyword:

```mojo
def parse() raises ParseError -> Int:
    ...
```

A function using bare `raises` implicitly uses the built-in `Error` type:

```mojo
def parse() raises -> Int:
    ...
```

This is equivalent to:

```mojo
def parse() raises Error -> Int:
    ...
```

Functions without a `raises` clause are non-raising, and are equivalent to
`raises Never` (discussed below).

A raising function has a single statically-known error type. Its implementation
may raise values of that type or values that can be implicitly converted to it.

```mojo
def example() raises ParseError:
    raise ParseError(0, "Bad input")
```

The error type is checked statically. Attempting to raise a value that cannot be
converted to the declared error type is a compile-time error.

A custom error type does not need to inherit from `Error` or implement a special
exception interface. However, implementing `Writable` is useful for diagnostics,
because it allows errors to be rendered as human-readable messages, and all
`Writable` types implicitly convert to `Error`.

### Error handling

Mojo uses `try` and `except` to handle errors, following Python:

```mojo
try:
    parse()
except err:
    handle(err)
```

The type of `err` is inferred from the raising operations within the `try`
block.

For example, when `parse()` raises `ParseError`, the handler receives a
`ParseError`. No additional type annotation or dynamic type test is necessary.

Mojo supports a single `except` clause, together with optional `else` and
`finally` clauses. Typed raises do not change their basic control-flow
semantics.

## Unifying error types in a try block

A `try` block may contain multiple operations that raise different error types:

```mojo
def parse_header() raises HeaderError:
    ...

def parse_body() raises BodyError:
    ...

try:
    parse_header()
    parse_body()
except err:
    handle(err)
```

The compiler must determine a single type for `err`, even though either
operation may fail.

Mojo infers the error type of a `try` block by considering the error types of
all potentially raising operations within its body.

If the raised types have a common type, Mojo selects that type, applying
implicit conversions where necessary. Otherwise, it falls back to the built-in
`Error` type when the raised types can all be converted to it. If no suitable
common type exists, the compiler emits an error.

The current design instead considers all raised types together, rather than
selecting the first error type. For example:

```mojo
def first() raises Int:
    ...

def second() raises Float64:
    ...

try:
    first()
    second()
except err:    # err is Float64.
    print(err)
```

The compiler searches for a suitable common type - in this case `Int` implicitly
converts to `Float64`, so the compiler picks it. If there is no common type,
the compiler tries implicitly converting all the types to `Error` and uses that
if it works.

The purpose of this inference is to preserve useful static type information when
possible, while allowing compatible error-producing operations to compose
naturally.

## Error propagation and implicit conversions

Typed raises interact with Mojo's normal implicit conversion system.

Consider:

```mojo
def inner() raises SourceError:
    ...

def outer() raises DestinationError:
    inner()
```

This is permitted when `SourceError` can be implicitly converted to
`DestinationError`. If such a conversion does not exist, the call is rejected.

The conversion is conceptually equivalent to:

```mojo
def outer() raises DestinationError:
    try:
        inner()
    except err:
        raise DestinationError(err)
```

Importantly, this does not require both errors to have the same representation.
A callee may return a `SourceError`, while the caller's function signature
expects a `DestinationError`. The compiler handles this by constructing a
temporary error value and converting it when propagating the failure.

This also allows custom error types to interoperate with the built-in `Error`
type when appropriate.

Implicit conversion is especially important for incremental adoption: a library
can introduce precise error types without requiring every caller to adopt them
simultaneously.

## Error types are part of function types

A function's error type is part of its type, just like its argument and normal
result types.

For example:

```mojo
def (Int) -> String

def (Int) raises ParseError -> String
```

These are distinct function types. The first cannot raise; the second may raise
`ParseError`. This distinction matters for function pointers, closures,
higher-order functions, and generic algorithms.

Mojo supports implicit conversions between compatible function types. In
particular, a non-raising function may be used where a raising function is
expected, because it satisfies the contract that it may raise an error without
ever exercising that possibility.

Similarly, conversion between raising function types can be supported when their
error types are compatible under the normal error conversion rules.

This makes error handling compatible with higher-order programming without
introducing dynamic type erasure.

## The `Never` type

An important part of the design is the `Never` type. `Never` represents a type
that can never be constructed. A function returning `Never` cannot return
normally:

```mojo
def abort() -> Never:
    ...
```

The same idea applies to error types. A function declared as:

```mojo
def example() raises Never -> Int:
    return 42
```

can never raise an error, because it is impossible to construct a value of type
`Never`, as such it is semantically equivalent to `def example() -> Int:`.
The absence of `raises` can therefore be understood as syntactic sugar for
`raises Never`.

This observation is more powerful than it initially appears.
Rather than modeling throwing and non-throwing functions as fundamentally
separate kinds of functions, Mojo represents both using the same underlying
abstraction, with the error type indicating whether failure is possible.

This gives generic code a natural way to abstract over the error effect.

### Generic functions that propagate error types

Consider an algorithm that invokes a callback:

```mojo
def apply[ErrorType: AnyType](
    callback: def() raises ErrorType -> Int
) raises ErrorType -> Int:
    return callback()
```

The error type is a parameter of the function.
If the callback raises `ParseError`, the algorithm also raises `ParseError`.

If the callback does not raise, Mojo can infer `ErrorType` as `Never`, and the
instantiated algorithm is also non-raising.

```mojo
def simple() -> Int:
    return 42

def parse_value() raises ParseError -> Int:
    ...

var a = apply(simple)

try:
    var b = apply(parse_value)
except err:
    # err is ParseError.
    ...
```

The first call does not require a `try` block, because its instantiated error
type is `Never`. The second propagates the callback's `ParseError`.

This eliminates the need to write separate throwing and non-throwing overloads
for many generic algorithms.

Languages such as Swift have introduced dedicated features like `rethrows` to
express similar patterns. Mojo instead models the effect directly through the
normal type system, which generalizes naturally to typed errors.

### ABI implications

The equivalence of `raises Never` and non-raising functions must be reflected in
their generated calling conventions.

A conventional raising function conceptually produces either a normal result or
an error, together with an indication of which result is valid.

However, a function raising `Never` cannot produce an error.
The compiler therefore eliminates the impossible error result and its
discriminator, giving the instantiated function the same calling convention as a
normal non-raising function.

This is important for performance as well as semantics: abstraction over the
error type should not introduce runtime overhead when instantiated with `Never`.

## Context managers

Typed raises also integrate with Mojo's `with` statement.
A context manager needs to perform cleanup when the body exits, including when
it raises an error. The error may have an arbitrary custom type.

For example, a context manager can define a parameterized exit handler:

```mojo
def __exit__[ErrType: AnyType](
    self, err: ErrType
) -> Bool:
    ...
```

This allows the context manager to receive the actual error type from its body
rather than requiring every error to be converted to the built-in `Error` type.

The `with` construct is conceptually built on the same error-handling mechanisms
as `try`, `except`, and `finally`.
Typed raises therefore extend to resource management without introducing a
separate exception representation or requiring a common error class hierarchy.

## Implementation model

Mojo's error-handling implementation uses alternate return values rather than
traditional exception unwinding.

Conceptually, a function:

```mojo
def parse() raises ParseError -> Int:
    ...
```

produces one of two possible results:

- An `Int` when execution succeeds.
- A `ParseError` when execution fails.

A discriminator indicates which result is valid.
At the compiler IR level, the normal result and error result can be passed
through separate result locations, with a boolean return value indicating the
outcome.

The control-flow lowering machinery then handles propagation, conversion, and
transfer to the appropriate `except` block.

For a `try` block, the compiler constructs a contextual error destination whose
type is inferred from the raising operations in the block.

When a call produces a different but convertible error type, the compiler can
use a temporary error destination followed by conversion and propagation.

The representation supports both register-passable and memory-only error types,
allowing the compiler to select efficient calling conventions based on the
actual types involved.
After generic specialization, calls raising `Never` require neither an error
result nor an error discriminator.

This approach reuses Mojo's existing type, lifetime, ownership, and lowering
systems rather than requiring an independent runtime exception mechanism.

## Alternatives considered

### Require all functions to raise Error

This is simple but loses the benefits of precise error types, structured
recovery, and allocation-free error representations.

It also forces specialized code into a common representation that may be
unsuitable for GPU kernels and low-level programming.

### Require a common error hierarchy

Many languages require exceptions to inherit from a common base class or
implement a common interface.

Mojo deliberately avoids this requirement. Error values should behave like
ordinary types, and the language should not impose dynamic dispatch,
inheritance, or a mandatory runtime representation.

### Require explicit error conversion at every call

Mojo could require callers to catch and explicitly convert any error that
differs from the enclosing function's declared error type.

Although predictable, this would create unnecessary boilerplate and make precise
error types difficult to adopt across large codebases.

Reusing ordinary implicit conversions provides a more compositional design.

### Introduce a dedicated rethrows feature

Mojo could introduce a special effect-polymorphism construct similar to Swift's
`rethrows`.

Instead, `Never` provides a more general solution: throwing and non-throwing
functions can be modeled using the same parameterized function type.

This avoids introducing a special feature restricted to one kind of higher-order
function.

## Summary

Mojo's typed raises design makes errors ordinary, statically typed values that
can be propagated through an alternate function result.

This provides structured error handling without requiring exception hierarchies,
stack unwinding, or dynamic runtime machinery.

The design integrates with implicit conversions, generic functions, function
types, and context managers.

The `Never` type makes non-raising functions a special case of typed raising
functions, allowing higher-order algorithms to be parameterized over error
behavior without sacrificing efficient calling conventions.

Finally, error-type unification within `try` blocks allows compatible
heterogeneous errors to compose while retaining compile-time type checking.

Together, these features make typed error handling a natural part of Mojo's
programming model, rather than an independent exception subsystem.
