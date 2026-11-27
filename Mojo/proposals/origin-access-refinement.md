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

### Treating parametric `ref` as read-only

A `ref` argument with parametric mutability can bind to either a mutable or an
immutable origin. Historically, Mojo modeled a call as potentially mutating the
referent whenever the supplied origin was mutable. This was conservative but
unnecessarily restrictive: the function body could not mutate through a
parametric `ref`, even when a `comptime if` proved that a particular
specialization received a mutable origin.

Mojo now reflects that existing body restriction in the call contract by
defining a parametric `ref` argument as read-only. The caller can therefore
preserve live references derived from the borrowed value itself. This guarantee
does not apply recursively: mutable origins embedded in the value remain
mutable, and the callee may mutate the storage they describe. A `ref` whose
origin is statically known to be mutable remains a mutable reference and uses
the corresponding mutable argument convention.

This change establishes `imm` as the default access for a parametric `ref`
before the general `accesses` syntax is available. Access clauses initially
only narrow an argument's default access. They cannot upgrade a parametric
`ref` from `imm` to `mut`, even when a specialization supplies a mutable
origin. This keeps the initial feature aligned with the access permitted in
function bodies. A future proposal can add widening if a concrete use case
justifies the additional semantics.

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

A `ref` with parametric mutability now defaults to `imm` for its referent. The
proposed access clauses allow that default to be narrowed to `none`, but not
upgraded to `mut`. More generally, an explicit access set must be a subset of
the default access set it replaces. The relevant defaults are:

- A read-only borrow permits immutable access.
- A `mut` argument permits mutable (and immutable) access.
- A `ref` with fixed mutability permits the access implied by that mutability.
- A `ref` with parametric mutability permits immutable access.

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
`FnOriginMetadataAttr`, which is part of function types. The parser does not
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

Ordinary function refinements are opt-in. Mojo has already adopted the `imm`
default for `ref` arguments with parametric mutability as an incremental step
toward this design. This does not remove a capability from function bodies:
mutation through a parametric `ref` was already rejected, including when a
compile-time condition proved that the instantiated origin was mutable. The
change makes call-site analysis match that existing restriction, avoiding
unnecessary invalidation.

The change is memory safe because the callee and caller enforce the same
contract. Body checking rejects mutation through a parametric `ref`, including
mutation guarded by a compile-time mutability test. At a call, the immutable
access does not invalidate references derived from the borrowed value itself.
Mutable origins embedded in the value retain their declared access, so calls
can still invalidate references into the storage they describe. A reference
with statically known mutable origin uses the mutable argument convention
instead, so direct mutation also continues to invalidate dependent references.

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

### Forwarding access sets through generic wrappers

Generic forwarding functions need a dependent access contract. For example,
the access set of `len(value)` should be the access set of the `__len__()`
implementation selected for `T`, after substituting `value` for that method's
`self` argument:

```mojo
trait Sized:
    def __len__(self) -> Int:
        ...

def len[T: Sized](value: T) -> Int:
    return value.__len__()
```

No single fixed contract on `len()` is correct for every conformance. A
container might compute its length entirely from inline metadata and use
`accesses(none self.origin)`. Another type might compute its length by scanning
storage and require `accesses(imm self.origin)`. Giving `len()` the conservative
default loses the precision of the selected method, while declaring the
narrower contract is unsound for conformances that access storage.

This is not ordinary access inference. The result remains symbolic while
`len()` is type checked and becomes concrete only after selecting a conformance
for `T`. The compiler must preserve that dependency in generic function types,
substitute the selected method's origins at an instantiation or call site, and
then apply the normal access-set inclusion checks. This means that generic code
will have to be conservative, but users can use a concrete code of
`len(someSpan)` without thinking the origin within the span is accessed.

Several designs are worth investigating:

- **Explicit access projection**: Allow a forwarding function to project the
  contract of a resolved call. An illustrative spelling is
  `accesses(forward value.__len__())`. The expression would not execute; it
  would resolve the method, obtain its access set, substitute `value` for
  `self`, and expose that result as the contract of `len()`. This is the most
  direct and predictable model for public wrappers and separate compilation.
- **Access-set parameters or associated effects**: Let a trait requirement
  carry a symbolic access-set member that each conformance supplies. Both
  `__len__()` and `len()` could refer to that member. This makes the dependency
  explicit in the trait witness and may generalize well to higher-order APIs,
  but it introduces a new kind of compile-time value and more surface syntax.

Explicit projection is the best initial direction because it keeps exported
contracts reviewable and does not require general effect inference. The
underlying IR should nevertheless represent projected access sets symbolically
so that later inference can produce the same representation instead of creating
a second mechanism.

### Trait requirements, generic storage, and downcasts

Generic collections expose another important interaction between access
contracts and traits. Consider appending a mutable pointer to a list. The
pointer passed as `value` can carry the same mutable origin as a pointer already
stored in `self`:

```mojo
var element = 0
var values = List[MutPointer[Int, origin_of(element)]]()
values.append(Pointer(to=element))
```

The conservative model sees the origin of `element` through both `self` and
`value`. Because both arguments otherwise permit mutation, it reports an
exclusivity conflict. However, `append()` only changes the list representation,
moves the pointer value into storage, and perhaps moves existing elements while
growing the allocation. None of those operations need to mutate the pointee.
This is not a special property of `Pointer`; it should follow from the
operations that `append()` is allowed to perform on an arbitrary `T`.

The relevant operation is the move constructor required by `Movable`.
`Movable` should require move construction not to mutate origins nested in the
value. The existing checked decorator illustrates the initial form of this
requirement:

```mojo
trait Movable:
    @__nested_origins_read_only
    def __init__(out self, *, deinit move: Self):
        ...
```

The move constructor can still modify the immediate storage of `move` and
initialize `self`. It can move a pointer, span, or other reference-carrying
field without mutating the memory described by that field. It may read nested
origins if its declared contract permits immutable access. A move constructor
that dereferences a stored mutable pointer and modifies its pointee does not
satisfy this `Movable` requirement, even if the concrete type could otherwise
declare a wider contract on an unrelated method.

This is a requirement on the trait witness, not an inferred fact about common
conforming types. Every explicit conformance must be checked against the
requirement using the same function-conversion rule as any other trait method:
an implementation can provide a narrower access set, but not a wider one.
Synthesized move constructors satisfy the requirement compositionally when the
move operations selected for their fields satisfy it.

This guarantee lets `List.append()` make the same checked promise:

```mojo
@__nested_origins_read_only
def append[T: AnyType](mut self: List[T], var value: T)
  where conforms_to(T, Movable):
    ...
```

The body can mutate `self`'s length, capacity, allocation, and element slots.
It can also invoke the `Movable` witness for `T`. The witness contract ensures
that moving `value` or relocating existing elements does not mutate origins
nested in those values. The same reasoning applies to operations such as
`Deque.append()` and `LinkedList.append()`.

The guarantee is operation-specific. Storing a `T` does not permanently hide
its nested origins, and a collection is not generally read-only with respect to
its elements. An indexing operation can project an element and expose its
origins. A method constrained by another trait can invoke that trait's
requirements. Copy construction and destruction need their own contracts;
they do not inherit the `Movable` guarantee merely because they operate on the
same type.

Generic types are also not intrinsically opaque. A generic method can test for
an additional conformance and downcast `T` to that trait:

```mojo
trait MayMutateNestedOrigins:
    def update(mut self):
        ...

@__nested_origins_read_only
def update_if_supported[T: Movable](mut value: T):
    comptime if conforms_to(T, MayMutateNestedOrigins):
        rebind[downcast[T, MayMutateNestedOrigins]](value).update()
```

This function must be rejected for the instantiations that enter the branch.
The downcast makes the requirements of `MayMutateNestedOrigins` available, but
it does not erase their access effects. Because `update()` has the conservative
default contract, it can mutate origins nested in `value`, which exceeds the
enclosing read-only promise.

More generally, checking a generic body must use the capabilities of the
current static view:

- An operation selected through a declared bound uses the contract of that
  trait requirement.
- A trait downcast adds the contracts of the target trait. Calls through the
  downcast are checked against those contracts.
- A rebind or compile-time branch that reveals an exact concrete type uses the
  selected concrete operation's contract.
- The enclosing function's contract remains an upper bound after substituting
  the concrete type and resolving compile-time control flow.

Ideally, the compiler diagnoses an incompatible downcast while checking the
generic definition whenever the target trait's requirement is already known to
be too broad. At minimum, it must reject every specialization that makes the
incompatible path reachable. Separate compilation favors definition-time
checking because callers should not discover that a public generic function
violates its own signature only for a particular conforming type.

The general `accesses` syntax also needs a way to quantify over origins nested
in an otherwise opaque generic value. A generic type can contain any number of
origin-carrying fields, so `append()` cannot name those origins individually.
The eventual spelling should distinguish access to the immediate value storage
from access through every origin reachable inside that value. Conceptually,
`append()` needs mutable access to the list and value slots but only immutable
access through nested origins. The exact quantified-origin syntax is left for a
follow-up proposal; the checked `@__nested_origins_read_only` decorator can
serve as the implementation bridge.

This direction can be implemented incrementally:

1. Put the checked nested-origin contract on the `Movable` move requirement and
   verify it on explicit and synthesized conformances.
2. Propagate trait-requirement contracts through witness calls, conditional
   conformance checks, and downcasts. Diagnose a call whose target requirement
   exceeds the enclosing contract.
3. Convert generic collection mutators from the unsafe decorator to the checked
   decorator once all operations they invoke have checked requirements.
4. Add an `accesses` form that quantifies over nested origins and use it to
   replace the checked decorator with ordinary function-type semantics.
5. Add symbolic definition-time checking for generic branches whose access
   sets depend on conformance selection, building on access projection for
   generic wrappers.

These steps deliberately avoid a `Pointer` or collection special case. The
precision comes from checked contracts on trait requirements and from retaining
those contracts when generic code narrows a type.

### Scoped unsafe access widening

Checking an access contract creates a practical need for an explicit escape
hatch. Low-level code sometimes knows a fact that the origin model cannot
express, such as two spans with the same origin covering disjoint ranges, or a
mutation preserving every outstanding interior reference. Such a function
might promise read-only nested origins and perform one carefully audited
mutation that the checker would otherwise reject.

This cannot be an ordinary helper function. Transitive access checking would
correctly observe the helper's mutation and reject the caller. The escape hatch
must therefore have compiler-recognized semantics, either directly or through a
capability represented in IR.

An unsafe escape does not change the function's externally visible contract.
The caller continues to rely on that contract for exclusivity and invalidation.
The programmer is responsible for proving that the hidden access does not break
those assumptions. If the mutation is part of the function's ordinary
observable behavior and callers must account for it, the function should
declare a wider access set instead of using an escape hatch.

Several designs are worth considering:

- **An unchecked value or origin cast**: Existing unsafe origin casts can erase
  the connection between a local view and the function's tracked input origins.
  For example, code that has separately established that three spans do not
  overlap can reconstruct local views with an untracked mutable origin:

  ```mojo
  var unsafe_input = input.unsafe_origin_cast[MutUntrackedOrigin]()
  var unsafe_output = output.unsafe_origin_cast[MutUntrackedOrigin]()
  ```

  Normal operations on these local views no longer appear to access the
  enclosing function's tracked origins. This works with the existing origin
  model, makes the unsafe boundary explicit, and requires no closure or new
  control-flow syntax. It is therefore a practical implementation bridge.

  The cast also erases useful lifetime and exclusivity information. Its result
  can be returned, stored, or passed elsewhere, accidentally laundering the
  unchecked authority. Code should create the untracked view immediately
  before the operation that needs it, keep it local, and retain the original
  tracked value for all other work. A dedicated cast could enforce that its
  result is lexical and non-escaping, making this approach substantially safer
  than an ordinary first-class `MutUntrackedOrigin` value.

- **A scoped closure intrinsic**: A compiler-known function could accept an
  explicit access list and a non-escaping closure:

  ```mojo
  __unsafe_with_accesses(
      mut span1.origin,
      mut span2.origin,
      mut result.origin,
      body=merge_body,
  )
  ```

  `CheckLifetimes` would verify the closure against the widened local set while
  continuing to verify the rest of the enclosing function normally. The
  intrinsic must guarantee that the closure does not escape and that lowering
  introduces no runtime call or allocation. This is implementable without new
  surface syntax and makes each exception easy to find, but closure capture
  inference and diagnostics could make simple low-level code cumbersome.

The recommended short-term bridge is a small, explicit untracked-origin cast at
each existing low-level use site. The preferred language design remains
`unsafe accesses`, potentially prototyped first as a compiler-known scoped
closure intrinsic. Both scoped forms should require an explicit list of widened
origins and access modes; a blanket "disable access checking" operation would
hide unrelated mistakes in the same scope. Longer term, disjoint-range proofs,
pointer-stability guarantees, and other safe abstractions should replace common
uses of every escape hatch.

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
