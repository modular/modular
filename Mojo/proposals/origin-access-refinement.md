# Refining function access sets in Mojo

Author: Chris Lattner
Status: Proposal.
Date: November 15, 2026.

This proposal introduces an optional `accesses` clause on Mojo functions to
refine the accesses a function may perform through its arguments and origins. It
builds on the existing origin model described in
[Origin design](origin-design.md).

## Introduction

Mojo uses argument conventions and origins to describe the relationships between
values, references, and their lifetimes. These also establish **default**
assumptions about what memory a function may access and how. For example, a
function accepting a `mut` argument is assumed to be able to mutate that entire
argument, even if its implementation only reads one field. Similarly, a value
such as `Span` may carry an origin identifying memory that the function never
actually dereferences, but Mojo takes a conservative assumption that it will get
accessed.

Those conservative assumptions are safe, but can cause unnecessary exclusivity
conflicts and invalidation of interior origins. This document proposes an
optional `accesses` clause that replaces selected default accesses with an
explicit contract:

```mojo
def update_x(mut p: Point)
  accesses(mut p.x):
    p.x += 1
```

The caller knows that this function accesses only `p.x`, even though the
argument convention would otherwise permit mutation of all of `p`. The compiler
verifies that the implementation of the function obeys the contract in
`CheckLifetimes`. Functions without an `accesses` clause continue to use their
default access sets.

The refinement affects call-site argument exclusivity as well as body
verification. A refined contract can make two accesses in
the same call disjoint:

```mojo
def update_fields(mut lhs: Point, mut rhs: Point)
    accesses(mut lhs.x, mut rhs.y):
    lhs.x += 1
    rhs.y += 1

# This is not an exclusivity violation due to the refined access clause.
update_fields(point, point)
```

Without the refinement, passing `point` to both `mut` arguments produces
overlapping mutable accesses. The contract instead gives the call one access to
`point.x` and one to `point.y`, which the origin model can prove disjoint.

Accesses are **upper bounds** on the operations a callee may perform, not a
claim that it necessarily performs them. They do not describe aliasing,
reference provenance, or the lifetime of a returned reference; those remain the
responsibility of origins.

## Motivation

This proposal is the next logical step towards improving the power of function
access sets in Mojo, and provides a generalized model that can eliminate the
`@__unsafe_nested_origins_read_only` decorator that we use as an undocumented
stop-gap in some places. Here we show a few other motivating use-cases for this
feature.

### Comparing spans without accessing their elements

Consider a `Span` equality operation. The span carries a pointer, size, and an
origin describing the underlying elements. Span equality compares the pointer
and size without reading or writing any element:

```mojo
# Simplified Span declaration.
struct Span[mut: Bool, Element: AnyType, origin: Origin[mut]]:
    var data: Pointer[Element, origin]
    var size: Int

    def __eq__(self, other: Self) -> Bool
      accesses(none self.origin):  # doesn't actually access the pointee data.
        return self.data == other.data and self.size == other.size
```

Both spans may refer to the same mutable memory. With a conservative contract,
the call may appear to make conflicting accesses to that memory merely because
the spans carry mutable origins. But the equality operation never dereferences
either origin. The `none` refinements tell the caller that the underlying memory
is untouched, avoiding a false exclusivity conflict without requiring the
compiler to prove the origins distinct.

The metadata of `self` and `other` remains readable. The `none` clauses apply to
the specified element-storage origins, not to the span objects themselves.

Today, this method relies on `@__unsafe_nested_origins_read_only` to avoid the
conservative treatment of its nested origins. The checked `none` specifiers
replace that workaround and let us remove the unsafe attribute.

### Avoiding unnecessary interior-origin invalidation

Mutable accesses matter beyond exclusivity: mutation of a value may invalidate
interior origins derived from it. This is necessary to prevent references into a
value from surviving structural changes that could invalidate them. However, a
function whose signature conservatively suggests mutation may invalidate
interior references even when it cannot mutate the relevant state.

For example:

```mojo
def update_x(mut p: Point)
  accesses(mut p.x):
    p.x += 1
```

The refined contract does not permit mutation of `p.y`, so callers should not
have to treat interior origins associated with unrelated parts of `p` as
invalidated by this call. The same distinction is especially important for
collections: operations on metadata or non-mutating comparisons should not
appear to invalidate references into their storage. The origin model still
determines which interior origins depend on which storage; the access contract
determines which of those origins a call might invalidate.

### Making conditional mutation through parametric `ref` explicit

A `ref` argument with parametric mutability can bind to either a mutable or an
immutable origin. Mojo already rejects an unconditional mutation through `ref`
arguments with parametric mutability because the origin could be immutable.
However, functions could technically use `comptime if` on the reference
mutability and mutate the reference only when actually mutable. This is
something that should be expressible, but isn’t the right default to assume.

Because this is technically possible, today, the caller gets no benefit from
that restriction. If the supplied origin is mutable, the call is conservatively
treated as a possible mutation even when the implementation only propagates the
reference, which is common for parametric accessors that return a reference.

We propose `imm` as the default access for a `ref` whose mutability is
parametric. This lets a function propagate the reference without writing through
it. The rare function that specializes on the origin's mutability and mutates in
the mutable branch can opt in with `accesses(mut value)`:

```mojo
def maybe_update(ref value: T)
  accesses(mut value):
    comptime if origin_of(value).mut:
        value.update()
```

At the call site, the mutable instantiation permits reads and writes, and the
immutable instantiation permits reads. The body must still establish that the
origin is mutable before writing through it. The spelling of the compile-time
mutability test is illustrative.

### Expressing an argument that is not accessed

A function sometimes carries an argument or reference solely to propagate its
origin:

```mojo
def return_ref(mut x: Point) -> ref[x] Point
  accesses(none x):
    return x
```

The function neither reads nor modifies the referent. Returning the reference is
allowed: the `accesses` contract concerns accesses *through* the reference,
while `ref[x]` describes the returned reference's origin and lifetime. This
distinction is central to the proposal.

### Eventual support for global variables

Mojo currently does not support global variables, but we will want to add
support for them in the future. In that future, we will extend `accesses` to
allow functions to opt-in to being able to access specific global variables,
which gives us the correct default: functions do not access globals unless they
opt into it explicitly. Global variables are not part of this proposal, so this
is limited to [Future directions](#future-directions).

## Detailed design

### The `accesses` clause

An `accesses` clause is an optional part of a function signature. It contains a
comma-separated list of access specifiers, each consisting of an access
convention and an origin expression:

```mojo
def example(mut p: Point, mut q: Point)
  accesses(mut p.x, imm p.y, none q):
    ...
```

The three access conventions are:

| Specifier | Permitted accesses through its origin                  |
|-----------|--------------------------------------------------------|
| `none x`  | No reads or writes; references may still be propagated |
| `imm x`   | Reads, but no mutation                                 |
| `mut x`   | Reads and mutation, subject to the origin's mutability |

In particular, **`mut` implies `imm`**. These are permissions, not operations
that the implementation is required to perform.

The clause follows the optional result type, or the argument list and other
function effects when the result is omitted. It can appear inline, but the
formatter should normally place it on the following line with two additional
spaces of indentation. Long lists may be formatted across several lines:

```mojo
def transform(mut p: Point, mut q: Point) -> Int
  accesses(
      mut p.x,
      imm p.y,
      none q,
  ):
    ...
```

This syntax intentionally reuses Mojo's convention vocabulary and avoids
`x: permission` syntax, which would resemble a type annotation. The clause does
not alter the ABI or the argument's declared passing convention.

### Default access sets and replacement

Without `accesses`, each argument receives its default access set according to
its argument convention and the origins reachable through it. Explicit
specifiers **replace**, rather than add to, the defaults for the affected
argument or origin root. Any argument not mentioned retains its default access
set.

```mojo
def example(mut p: Point, mut q: Point)
  accesses(mut p.x):
    # p: only the explicitly specified access to p.x.
    # q: unchanged default access of a mut argument.
    ...
```

In particular, refining `p.x` does not leave an implicit `mut p` permission in
place. Under this rule, reading `p.y` is also forbidden unless an `imm p.y` (or
broader non-overlapping permission) is specified:

```mojo
def update_from_y(mut p: Point)
  accesses(mut p.x, imm p.y):
    p.x += p.y
```

A `ref` with parametric mutability defaults to `imm` for its referent but can be
explicitly relaxed to `none` or upgraded to `mut` if the callee needs to. The
relevant defaults are:

- A read-only borrow permits immutable access.
- A `mut` argument permits mutable (and immutable) access.
- A `ref` with fixed mutability permits the access implied by that mutability.
- A `ref` with parametric mutability permits immutable access by default under
  this proposal.

Other conventions retain their established access rules. The implementation
must derive defaults and explicit refinements through one shared mechanism so
body checking and call checking cannot disagree.

### Origin expressions, not just argument specifications

The operand of an access specifier uses the same origin-expression resolution
rules as the origin inside `ref[x]`: a named origin parameter directly denotes
that origin, while an expression denoting a value implicitly resolves using
`origin_of`.

```mojo
accesses(imm x)             # Origin of an argument.
accesses(mut x.field)       # Origin of a subobject.
accesses(none origin_a)     # Explicit origin parameter.
accesses(none self.origin)  # Origin parameter projected through a value.
```

Both `self.origin` and an explicit name such as `origin_a` can therefore refer
to the same kind of compiler-level origin parameter. This is not a separate
access-path language.

An access operand is an unevaluated origin expression. Resolving it does not
execute runtime code, call getters, or load the referenced storage. The compiler
accepts only expressions that the existing origin-expression resolver can map
to an origin or a statically known subobject projection. It rejects calls,
computed indexing, and other expressions that do not resolve to such a path.

Accesses to an object's own fields are distinct from accesses to memory
described by origins carried inside the object. For example, `none self.origin`
can prevent element dereferences while allowing a `Span` method to read its
pointer and size metadata.

### Access-path replacement

The compiler applies an `accesses` clause in the following order:

1. Construct the default access set for every argument and reachable origin.
2. Group explicit specifiers by the argument or origin root that they refine.
3. Remove the default access for each refined root.
4. Add the explicit, non-overlapping accesses for that root.
5. Leave defaults for unmentioned roots unchanged.

An access to a subobject does not implicitly grant access to siblings. For
example, `accesses(mut p.x)` permits the implementation to read and mutate
`p.x`, but it does not permit reading `p.y`. Identifying `p.x` is an unevaluated
projection and does not itself require runtime access to all of `p`.

The compiler normalizes paths through the origin model before comparing them.
If it cannot prove that two explicit paths are disjoint, it rejects the
refinement rather than assuming that they do not overlap. The programmer can
use a broader access path when the desired distinction cannot be represented.

### Multiple and overlapping specifiers

Non-overlapping specifiers combine to form the replacement access set:

```mojo
accesses(mut p.x, imm p.y)
```

**Overlapping specifications are an error**, even if their access conventions
agree. For example, these are invalid:

```mojo
accesses(imm p, mut p.x)    # Error: overlapping origins.
accesses(none p, none p.x)  # Error: overlapping origins.
accesses(imm p.x, mut p.x)  # Error: duplicate/overlapping origin.
```

The compiler checks overlap after origin resolution, not merely by comparing
source spellings. This prevents alternate spellings for the same origin from
evading the rule. Distinct syntactic fields are only non-overlapping when the
origin model establishes that they represent disjoint storage. Otherwise, the
compiler rejects the refinement.

### Enforcement and effects of calls

The declared access set is a contract between callers and the implementation.
Direct reads and mutations, and accesses performed transitively by called
functions, must be contained in the declared set.

```mojo
def only_read(mut p: Point)
  accesses(imm p):
    print(p.x)  # OK.
    p.x = 1     # Error: exceeds the declared access contract.
```

Call checking substitutes actual argument origins into the callee's access
contract and checks the resulting accesses using the existing exclusivity and
invalidation logic. It uses the refined contract when deciding which accesses
the call acquires; it must not first acquire the argument convention's broader
default access. An unknown or unrefined callee retains conservative default
accesses.

Body checking uses the declared contract of every direct or indirect callee. A
function cannot claim `none x` and then invoke another function that reads or
writes through `x`. Recursive functions do not require access inference or a
fixed-point computation because each declared contract is an upper bound and
each body is checked independently against it. Indirect calls use the contract
carried by the function type. An external function whose body is unavailable is
trusted to satisfy its declared contract in the same way that it is trusted to
satisfy the rest of its signature.

`CheckLifetimes` also uses the access set to evaluate ASAP destruction and
similar operations.

Reference propagation is not a read or write through the referent. Consequently,
`return_ref` above is legal, while dereferencing `x`, accessing its fields, or
calling a function that reads through `x` would violate `none x`. Normal origin
and lifetime rules independently govern whether propagated references are valid.

This rule also permits storing a reference in an origin-tracked aggregate or
capturing it in a closure without dereferencing it. Such propagation must still
preserve the origin and satisfy normal lifetime rules. Comparing pointee values,
loading through the reference, or passing it to a callee with `imm` or `mut`
access is an access and therefore violates `none`.

### Function types and function convertibility

Access contracts are part of a function's externally visible semantics and must
be represented in function types, including function values, callbacks, and
trait requirements. Otherwise indirect calls could silently lose the guarantee
that makes these refinements useful.

Function types use the same clause position as function declarations: after the
optional result type:

```mojo
comptime ReadPoint = def(mut p: Point) accesses(imm p)
comptime MutPoint = def(mut p: Point) accesses(mut p)
comptime ReadAndReturnPoint = def(mut p: Point) -> Point accesses(imm p)
```

An access expression refers to a named function-type parameter just as a
declaration clause refers to a named function parameter. A function type that
refines an argument must therefore name that argument. Return types are optional
for function types; omitting `-> T` means the function returns `None`. Function
types without an `accesses` clause retain the default access set derived from
their argument conventions.

A function with a **narrower** access set can be used where a function with a
**wider** access set is expected, provided the remaining function-type
requirements are satisfied. The opposite conversion is unsafe. Conceptually, if
`A(f)` is the set of accesses permitted by function `f`, then substituting `f`
for a function type `F` requires `A(f) ⊆ A(F)`.

A function with `ReadPoint`'s contract is convertible to `MutPoint`: callers
prepared for possible mutation can safely call a function that only reads. A
function with `MutPoint`'s contract is **not** convertible to `ReadPoint`,
because callers of the latter rely on the non-mutating guarantee. Similarly, a
function promising `none` is usable wherever `imm` or `mut` is permitted,
subject to the same origin/type constraints.

Field-level contracts follow the same set-inclusion rule after normalizing their
origins into corresponding function-type parameters. This requires an
origin-aware comparison rather than textual comparison of parameter names. For
example, `mut p.x` is no more permissive than `mut p`, but `mut p.x` and
`mut p.y` are generally incomparable. The representation of bound origin
expressions in IR is an implementation detail.

These contracts also affect overload resolution and implicit conversions insofar
as the language compares function types. A conversion must not silently discard
a narrower *expected* contract; any adaptation of a callable must preserve its
promised upper bound. Existing calling conventions, result types, and
origin/lifetime constraints remain independently enforced.

### Implementation in the compiler

The parser and type checker parse, preserve, resolve the declared access
specifiers, reject invalid overlaps, and store the information in
`FnMetaOriginDataAttr`, which is part of function types. The parser does not
need to enforce every memory access in the body of the function during ordinary
type checking. Enforcement belongs primarily in `CheckLifetimes` in its initial
“uninit pass,” which already processes every access.

`accesses` can be a contextual keyword. The lexer can continue to produce an
identifier, and the function-signature parser recognizes it only after the
optional result type, or after the argument list and other effects when the
result is omitted. It remains available as an identifier in all other contexts.

The following implementation sequence is non-normative:

1. Record the optional clause on function declarations and function types,
   including origin expressions and conventions.
2. Resolve each expression using the existing origin-expression machinery and
   diagnose overlapping specifications.
3. Construct the default access set and apply replacements for the affected
   origin roots; leave unmentioned roots at their defaults.
4. During `CheckLifetimes`, verify direct and transitive accesses against the
   resulting contract and use the callee's refined set at call sites.
5. Apply the existing lifetime and interior-origin invalidation logic to the
   actual permitted accesses rather than assuming the full default mutation
   footprint.

`CheckLifetimes` diagnostics should identify the disallowed access and the
declaration restricting it.

## Compatibility and staging

Ordinary function refinements are opt-in. The proposed `imm` default for `ref`
arguments with parametric mutability is technically source breaking, but will be
very low impact in practice. Existing functions that conditionally mutate
through it will need to adopt `accesses(mut x)`. We should inventory affected
standard-library and user patterns and decide whether to stage the new default
separately from the initial access-clause implementation.

We can validate this is working well by replacing all uses of
`@__unsafe_nested_origins_read_only`. This swaps out our temporary stop-gap with
a more general solution that is also memory safe.

## Future directions

### Global variables and other state

Mojo does not yet support general global variables. When introduced, access to
global state should require explicit opt-in rather than being available
implicitly to all functions. The same `accesses` vocabulary could eventually
describe `imm` and `mut` access to globals, but global-state syntax, name
resolution, and default capabilities are
**not part of this proposal's detailed design**.

### Access inference and effect polymorphism

The explicit contracts provide a starting point for inferred access summaries,
polymorphic access sets, and callback APIs that preserve or parameterize over
the effects of supplied functions. These are not required for the initial
implementation.

### Broader effects

Access refinement resembles a small, concrete effect system for memory. Other
effects—such as I/O, allocation, exceptions, or synchronization—could eventually
be described separately or integrated into a broader system. This proposal
deliberately confines itself to origin accesses, but we could imagine and
explore explicitly declared symbolic effects in the future if concrete use-cases
arise.

## Summary

`accesses` lets a function narrow its default access contract using origin
expressions and the `none`, `imm`, and `mut` conventions. The contracts are
checked by `CheckLifetimes`, propagate through calls and function types, and
improve both exclusivity checking and interior-origin invalidation. They add
precision to Mojo's existing origin model without changing what origins mean or
requiring a general-purpose effect system.
