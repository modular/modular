# Stdlib Insider Docs

Owen Hilyard, Created November 17, 2024

**Status**: Accepted

## Motivation

For most languages, people who work on the standard library have the ability
to look inside of the compiler to clarify questions they have about the
semantics of particular operations or compiler builtins. For Mojo, that is
not the case for many people working on the standard library. As a result,
the exact semantics of some important compiler builtins, such as
`lit.ownership.mark_initialized`, are not fully known to a large number of
people working on the standard library. These language-internal dialects are,
like Mojo itself, subject to enhancements, breaking changes, and even complete
removal. This presents a problem since the stdlib is correctness-critical code,
and when people who don't understand the API contract of a construct use it
in correctness-critical code, issues are bound to happen.

## Proposal

In order to help address this, I propose the creation of a "stdlib insider"
document, which contains information on MLIR operations/types, WIP features and
other parts of the language which are either intended to only be used in the
standard library/MAX drivers or language features which are still subject to
change and evolution. This can be substantially less polished than the Mojo
manual, including "X is like Y in C++ but ...", links to academic
papers, links to LLVM docs, pseudocode, and other ways one might explain a
concept to a colleague. This document, likely maintained as a markdown file in
/docs, is intended to be internally facing to core Mojo developers. This means
that a large "everything in here is subject to being totally rewritten in a
bugfix release or security update, do not use outside of the standard library
or MAX" warning, which will hopefully dissuade people from using what is
documented there in ways that will get them stuck on old Mojo versions.

For MLIR operations, I'd like the following information documented
for each operation used in the standard library; Operation name,
arguments (potentially as Mojo function syntax), a description of the
operation, pre-conditions, post-conditions, and clear hazards. Clear hazards
would ways in which the operation can cause UB (ex: is what happens with a
null pointer well defined?), conditions under which the operation will force
the program to abort or other cases which may be surprising behavior.

For MLIR types, information about what a type is intended to do, what
parameters the type has (and their types), the size of the instantiated type
(for alignment), and any non-trivialities in the type (is it ok to copy/move
it?). MLIR attributes should have similar information.

For features, a short description of what the feature is intended for, and
then syntax examples that show the capabilities of the feature. Ideally, some
differentiation between the design and the current implementation should be
present, slowly moving parts of the documentation over as they are available
on nightly. Documenting known sharp edges or limitations is also helpful, for
instance if trait objects could only represent a single trait (ex: no Movable
\+ Copyable + Formattable trait) or if some part of the implementation has a
high time or space complexity (ex: O(N^2) compile time overhead in the number
of traits in a trait object).

## Current State

At present, the majority of MLIR operations are things which I think are
reasonable to explain with a link to LLVM or C++ docs. For example, `pop.max` is
mostly self-explanatory, so unless there are extra semantics I don't think it
really needs more of an explanation than "see C++ std::max".

What I consider the important things to document:

### `pop.external_call`

Lots of people want to call into OpenSSL to get basic HTTPS working, and that
needs to be done correctly. There's also a lot of ABI issues around this,
for instance whether Mojo structs are C layout (for now) or whether we need
a mechanism to force that behavior.

### `co.*`

This area is WIP, but some community discussion around the direction would be
helpful. Information about the API of each of the types would also be nice
since it looks like we would need to use MLIR to implement future combinators
like Rust's `FuturesUnordered`. Documentation around synchronization
requirements is also important for correctness as people move towards async io.

### `#kgen.param.expr<target_get_field`/`#kgen.param.expr<target_has_feature`

Filling out the full target introspection capabilities that are present inside
of the compiler will be fairly labor intensive, but also very useful. If we can
get better documentation on this, the community can help fill out the soup of
x86 CPU among other things. If there is a way to query the data layout
information, that would save us a lot of effort in parts of std that interact
with libc.

### `lit.ownership.mark_initialized`

This operation acts like a "virtual store" of valid data to the memory location
of its argument. This is conventionally used in places where unsafe code is
doing construction of a value a field at a time, but needs to "bless" the whole
object as being constructed.

### `lit.ref.from_pointer`

Documenting the soundness preconditions would help anyone writing smart
pointers or certain types of collections to use this correctly.

### `pop.call_llvm_intrinsic`/`sys.intrinsics.llvm_intrinsic`

There's a lot of places where this is used, but how to use it is not really
documented. In particular specifying legal Mojo type <-> LLVM type conversions,
Documenting limitations is also helpful, for instance, can is `@call` ok to
use? Can I directly write phi nodes? I had a lot of difficulty interacting
with anything that returns a `!llvm.struct` or `!llvm.vec`.

### `pop.inline_asm`

A lot of CPU functionality doesn't have LLVM intrinsics and we'll need to use
assembly (CPU time stamp counters, poking MSRs for runtime capability
discovery, etc). I personally ran into difficulties doing multiple return
(ex: x86 `cpuid` returns results in 4 registers). Information on the asm
dialect, how to create clobbers and scratch registers, and rules for control
flow (ex: can I implement computed gotos, jump into the middle of a function or
return from the current function?).

### `!lit.origin.set`

This is used in Coroutines (presumably to store capture origins), but it looks
like it may be useful for storing collections of references.

### Nested-origin read-only decorators

`@__nested_origins_read_only` and
`@__unsafe_nested_origins_read_only` promise that a function does not mutate
origins nested inside its arguments or captured values. For example, a
`Pointer[Int, origin]` passed by value contains `origin`; the pointer value is
the argument, while its pointee is reached through a nested origin.

The promise is part of the function type. At a call, the compiler treats these
nested origins as read-only for exclusivity checking and does not invalidate
interior references merely because the call receives a value that contains
their origin. The function can still read through such an origin:

```mojo
@__nested_origins_read_only
def load[origin: Origin](ptr: Pointer[Int, origin]) -> Int:
    return ptr[]
```

The checked decorator verifies the function body, including transitive effects
from functions it calls. It rejects mutation through the pointer:

```mojo
@__nested_origins_read_only
def invalid_store[origin: MutOrigin](ptr: Pointer[Int, origin]):
    ptr[] = 0  # Error: mutable access to a nested origin.

@__nested_origins_read_only
def also_invalid[origin: MutOrigin](ptr: Pointer[Int, origin]):
    mutate_through_pointer(ptr)  # Error if the callee may mutate `origin`.
```

The decorator does not make the entire function read-only. Direct mutation
allowed by an argument convention remains allowed. This includes a `mut`
argument, a `ref` whose origin is statically mutable, and result storage.
Owned arguments and local variables can also be mutated because they are local
to the call:

```mojo
@__nested_origins_read_only
def reset(mut value: Int):
    value = 0  # Allowed: this is direct access through `mut`.

@__nested_origins_read_only
def update_local(var value: Int) -> Int:
    value += 1  # Allowed: `value` is owned by this call.
    return value
```

Closure captures follow the same distinction. A capture declared `mut` grants
direct mutable access to the captured value. Origins inside that value remain
read-only:

```mojo
def mutate_capture():
    var value = 0

    @__nested_origins_read_only
    def inner() {mut value}:
        value = 1  # Allowed: direct access through a `mut` capture.

    inner()


def capture_pointer[origin: MutOrigin](ptr: Pointer[Int, origin]):
    @__nested_origins_read_only
    def inner() {ptr}:
        ptr[] = 1  # Error: mutation through a nested captured origin.

    inner()
```

`@__unsafe_nested_origins_read_only` makes the same call-site promise but does
not verify the body. It exists for compiler builtins and implementation
patterns that the current effect analysis cannot prove. Use the checked
decorator whenever verification succeeds.

> [!WARNING]
> The unsafe decorator is a contract, not an escape from origin checking.
> Breaking the promise can let the caller retain aliases or interior
> references across a mutation that should have invalidated them.

For example, the following function can compile, but its annotation is
incorrect and can make callers unsound:

```mojo
@__unsafe_nested_origins_read_only
def invalid_promise[origin: MutOrigin](ptr: Pointer[Int, origin]):
    ptr[] = 0  # The compiler trusts the false read-only promise.
```

Neither decorator means that the function is pure, that it performs no writes,
or that every value reachable by unsafe or untracked means becomes immutable.
They refine only the modeled accesses to nested origins and parametric `ref`
arguments. They do not narrow direct `mut` access, direct mutable captures,
result slots, owned arguments, or local storage.

### The backing allocator(s) for `pop.aligned_alloc` and `pop.global_alloc`

Time spent in GDB has led me to believe this is tcmalloc, but knowing for
sure means we have information like the minimum alignment the allocator will
provide (storing information in the lower bits), what kind of caching it does,
information about how it handles memory overcommit being off (db servers), and
what kind of instrumentation we might have access to.
