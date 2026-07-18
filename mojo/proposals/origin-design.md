# Mojo origin and lifetime design

**Status**: Draft

## Introduction

This document explains how Mojo tracks value lifetimes and references inside
the compiler. It complements the user-facing guide on
[lifetimes, origins, and
references](https://mojolang.org/docs/manual/values/lifetimes/),
which describes what programmers need to know about `ref` arguments,
`origin_of()`, and origin parameters. Here we map those language concepts to
their implementation: the LIT dialect IR, the origin attribute algebra, and the
`CheckLifetimes` pass that verifies correctness and inserts destructors.

Origins are symbolic, compile-time values. They do not name storage addresses;
they name the variables (and derived sub-objects) whose lifetimes must be
extended while a reference is live. Most of this machinery is invisible in
source code, but it becomes important when reading compiler IR, debugging
lifetime errors, or extending types like `Pointer`, `Span`, and parametric
`ref` APIs.

## Why origins?

Mojo combines value semantics with references: most arguments are passed by
immutable borrow, `mut` arguments get exclusive mutable access, and local
variables are destroyed as soon as their last use (ASAP destruction). Any
non-owning pointer or `ref` must therefore answer two compile-time questions:

1. **Which value owns the storage?** So the compiler knows whose lifetime to
   extend while the reference is live.
2. **Is mutation allowed through this reference?** So exclusivity and
   parametric APIs (`ref self` methods, `Pointer`, `Span`, etc.) can be checked
   consistently.

An **origin** is the symbolic name the compiler assigns to answer those
questions. Origins are not addresses or run-time handles; they identify
variables (and derived sub-objects like `self.names`) for static dataflow
analysis. The `CheckLifetimes` pass consumes origin information on `!lit.ref`
types to enforce four core invariants:

- **Use before destroy** — a reference cannot outlive the value it names.
- **Lifetime extension** — referenced values stay alive until the last reference
  is gone (including union origins when a `ref` return could point at either
  of two arguments).
- **Argument exclusivity** — mutable references cannot alias other live mutable
  accesses to the same origin for the duration of a call.
- **ASAP destruction** — destructors are inserted at the earliest safe point
  once a value is no longer referenced.

Most high-level code never mention origins explicitly, but the standard library
and other lower-level code needs to deal with them whenever references are used.
Within the compiler though, every `ref` argument, `ref` return, and
origin-parameterized type carries an origin through LIT IR until lifetimes are
verified and lowered away. For the programmer-facing model and examples, see the
[lifetimes manual](https://mojolang.org/docs/manual/values/lifetimes/).

## Compiler pipeline overview

Origin information flows through three main stages: parser emission, lifetime
checking, and LIT lowering. The LIT dialect (`lit.*` ops and `!lit.ref` types)
is the IR representation; it exists only between parsing and `LowerLIT`.

```text
Mojo source
    ▼
Parser (MojoParser) ── emits LIT IR with origins; checks call-site exclusivity
    ▼
LowerSemanticCF     ── lowers semantic control flow for analysis
    ▼
CheckLifetimes      ── check for uninit variable usage, destructor insertion
    ▼
LowerLIT            ── lit → kgen; strips all origin types/attrs
    ▼
Elaboration, code generation, etc...
```

### Mojo Parser

The parser is responsible for most origin construction. When it emits a
function signature, local variable, `ref` argument, or origin-parameterized
type, it attaches the corresponding origin attribute to each `!lit.ref` and
records function-level metadata in `#lit.fn_metadata` (implicit origin
declarations, closure capture sets, etc.). Surface syntax like `origin_of(x)`,
`ref[self]`, named origin parameters, and origin unions is lowered directly
into the attribute algebra defined in `LITAttrs.td` (described below).

The parser also performs **argument exclusivity checking** at each call site.
As arguments are emitted, `ExclusivityChecker` (in `UncheckedCallEmission.cpp`)
tracks which origins are accessed and whether each access is a read or write.
Conflicting mutable aliasing between arguments (or between an argument and a
captured origin) is diagnosed here, before `CheckLifetimes` runs. Read/read
aliasing is permitted; read/write and write/write conflicts on the same origin
are rejected.

### CheckLifetimes

After `LowerSemanticCF` prepares control flow for dataflow analysis,
`CheckLifetimes` checks for use of uninitialized values, and inserts implicit
destructor calls (as well as diagnosing unused stores, performs copy->move
optimization etc). It walks each function's LIT IR, tracking which
values are live, which origins they carry, and how each operation uses or
defines references. It inserts destructor calls at ASAP destruction
points and rejects code that uses uninitialized values (e.g. due to transfers).
Errors here are the classic lifetime failures:
use-after-destroy, returning a reference that outlives its owner, and similar
dataflow violations that cannot be caught from a single call expression alone.

Exclusivity at call sites is already handled by the parser; `CheckLifetimes`
focuses on intraprocedural and cross-operation lifetime rules.

### LowerLIT

Once lifetimes are verified, they are no longer needed. `LowerLIT` converts the
LIT dialect into KGEN to strip them out. References become ordinary pointers:
`!lit.ref<T, origin>` lowers to `!kgen.pointer<T>`. All origin attributes and
origin parameter types are replaced with empty structs — they are compile-time
only and have no run-time representation. After this pass,
origins no longer appear in IR; remaining compilation operates on KGEN types
and the usual elaboration pipeline. This also means that origins are invisible
at elaboration time - there is no conditional logic that can depend on them,
and therefore no code specialization due to them.

Note: The "mut" bool in an origin is tracked separately and can be specialized
on if parametric.

## LIT type representation

The LIT dialect represents origins and references with four main types:

### `!lit.origin<mut>`

An origin *parameter* has *type* `!lit.origin<mut>`, where `mut` is a
compile-time boolean attribute (constant `imm`/`mut`, or parametric). This
type appears in function and struct parameter lists wherever Mojo source names
an `Origin[mut=…]` parameter.

Because mutability is part of the origin *type*, parametric APIs need **two**
implicit or explicit parameters when mutability is inferred at the call site: a
`Bool` for whether the access is mutable, and an `!lit.origin<mut=that_bool>`
for the origin itself. The standard library wraps this pair as
`Origin[mut=is_mutable]`; in IR they remain separate parameters linked by the
type constraint on `!lit.origin<mut>`.

### `!lit.ref<element, origin>`

A reference is a runtime memory location plus an origin attribute (plus
address-space index etc). The element type is the pointee; the origin
attribute (names a parameter of `!lit.origin` type) indicating which value's
lifetime governs the reference and whether mutation is allowed.

Read-only borrows (`a: T`), `mut` arguments, and explicit `ref` arguments all
eventually use `!lit.ref` in LIT IR. The difference is whether mutability is
fixed (`imm`/`mut` on the origin) or parametric (extracted from a `Bool`
parameter, as in the example below).

### `!lit.origin.set` and `!lit.ref.pack`

`!lit.origin.set` is a singleton type for origin-set parameters on parametric
closures (`OriginSet` in source). `!lit.ref.pack` represents a heterogeneous
pack of references that share one origin — used for variadic `ref` packs and
lowered to a struct of `!lit.ref` values with a common lifetime.

### Simple example showing references

A `ref` argument with no origin specifier gets an **inferred** origin: the
parser adds two implicit parameters to the function and ties the argument's
`!lit.ref` to them (`Signatures.cpp`, `processRefOriginSpecifier`).

Mojo source:

```mojo
# More typically written as: def example(ref a: Int):
def example[a_mut: Bool, a_origin: Origin[mut=a_mut](ref a: Int):
    pass
```

Rough LIT IR (names simplified; the parser mangles parameter names):

```mlir
// Note: This is glossing over the struct wrappers (Bool vs !kgen.scalar<bool>)
// for clarity.
lit.fn @"example"<
  a_mut: !kgen.scalar<bool>,
  a_origin: !lit.origin<mut=a_mut>
>(
  %a: !lit.ref<!Int, mut=a_is_mut a_origin> ref
) {
  ...
```

At each call site, the compiler binds `a_mut` and `a_origin` from the
actual argument (mutable vs immutable, origin of the passed value). That is how
a single `ref a: Int` signature accepts both `example(1)` and
`example(someMutVar)` without two overloads.

By contrast, a plain `mut a: MemExample` argument gets a fixed-mutability
origin (`mut a`) without the extra `Bool` — mutability is known from the
argument convention, not inferred.

## Origin attribute algebra

Origin *values* are compile-time `TypedAttr`s typed as `!lit.origin<mut>`. They
form a small closed algebra defined in `LITAttrs.td` and used on `!lit.ref`
types, origin parameters, and function metadata. This section describes what
each attribute *means*; normalization rules are covered in the next section.

Most origins name a variable or parameter whose lifetime governs a reference.
A few are singletons for special storage classes; combinators build new origins
from existing ones.

### Decl references

**`#kgen.param.decl.ref<"name">`** — the origin of a named
parameter or local variable defining an origin. The parser creates
one for each memory-resident argument (for example `mut a: T` becomes
`!lit.ref<T, mut *"a`">`). The verbose form of the attribute looks like this:

```mlir
#kgen.param.decl.ref<"user_defined_name">
#kgen.param.decl.ref<"arg`">
```

Sidenote: the backtick is inserted by the parser to avoid parameter name
conflicts, but makes the IR annoyingly verbose because we need the `*""` stuff,
it would be great to use the normal parameter mangling logic.

**`#lit.implicit.origin.ref<depth, index>`** — a reference to an implicit
origin parameter of an *enclosing* signature, before that parameter is bound to
a concrete name. Used in nested function types and generic signatures. The
`depth` counts enclosing signatures (0 = innermost); `index` selects which
implicit origin decl.

```mlir
#lit.implicit.origin.ref<1, 0> : !lit.origin<false>
```

Implicit origins are used for "read" and "mut" argument conventions: we'd like
to eventually remove them entirely, which should now be possible due to recent
improvements.

### Leaf and singleton origins

These attributes do not refer to a particular local variable. Each exists in
immutable and mutable forms (`!lit.origin<false>` vs `!lit.origin<true>`).

- **`#lit.static.origin`** — storage that lives for the entire program (string
  literal data, global variables). Mojo doesn't support mutable globals
  (because we don't have a threading model yet), so in practice these are always
  immutable (e.g. used for string literals).
- **`#lit.comptime.origin`** — memory visible only during comptime
  interpretation; keeps comptime references from leaking into runtime IR. Mojo
  prevents references to comptime memory from being materialized as runtime
  values (e.g. a comptime `Span` can't be materialized).
- **`#lit.any.origin`** — wildcard origin: the reference may alias any live
  value. Disables exclusivity and ASAP-destruction reasoning for affected
  scopes; intended as an escape hatch, not routine API design. We would like to
  eventually remove this entirely.

We have pretty names for these in the standard library, accessing these with
the `__mlir_attr` syntax. Users should always use the aliases, not the
attributes directly, because attributes are internal compiler implementation
details that we will never stabilize. For example:

- `StaticConstantOrigin` —
  `#lit.origin.field<#lit.static.origin, "__constants__">`.
- `ImmutAnyOrigin` / `MutAnyOrigin` — `#lit.any.origin` at the corresponding
  mutability.
- `UntrackedOrigin` — `#lit.origin.union<>` (empty union; no tracked alias).

### Derived origins

**`#lit.origin.field<base, "field">`** — a sub-origin of a struct or aggregate
origin. The base origin names the whole value; the field origin names one
member. Sibling fields get distinct origins, providing "field sensitivity" for
origin analysis. This lets the exclusivity checker allow concurrent mutable
access to `a.x` and `a.y` while still treating `a.x` and `a` as conflicting,
and allows CheckLifetimes to track uninitialized fields accurately.

Origin syntax uses postfix `->field` chains (`life->names`) or dotted paths in
`origin_of(self.names)`. Example:

```mlir
#lit.origin.field<
  #kgen.param.decl.ref<"self"> : !lit.origin<mut>,
  "names">
  : !lit.origin<mut>
```

**`#lit.interior.origin<base, userName>`** — an interior sub-origin of a base
origin. Unlike `#lit.origin.field`, which names a struct member stored inline
in the parent value, an interior origin names storage that is usually (but not
necessarily) embedded inside or one pointer indirection away from the base — for
example, an element reference returned from `List.__getitem__` that points into
heap memory it owns, or inlined data for an element in `Variant`'s storage.

The base origin (including any `->field` prefix) governs invalidation of the
interior origin. `CheckLifetimes` tracks interior origins separately from
ordinary value liveness: mutating an enclosing origin invalidates derived
interior references even when the base variable is still alive.

The string `userName` appears in diagnostics (for example, `list["element"]`)
and can identify the interior object when Mojo gains fine-grained invalidation
sets. Interior origins with different names are logically independent values
tracked in a "field sensitive" way.

Functions that introduce interior references mark themselves with the
`@__defines_interior_origins` decorator (which needs to be properly designed).
Methods that only read through nested origins without invalidating them can use
`@__unsafe_nested_origins_read_only` to opt out of blanket invalidation on
call. In the standard library, `Origin.get_owned_interior[name]` builds the
corresponding attribute; origin syntax uses postfix `["name"]` after field
chains (`list["element"]`, `self.names["item"]`).

Example:

```mlir
#lit.interior.origin<
  #kgen.param.decl.ref<"list"> : !lit.origin<mut>,
  "element">
  : !lit.origin<mut>
```

Nested forms like `list_of_lists["element"].first["element"]` combine field
sensitivity with interior tracking: mutating `.first` invalidates interior
references rooted under that field, not sibling fields. Like field selection,
interior selection distributes through `#lit.origin.union` and
`#lit.origin.mutcast` during canonicalization.

### Combinators

**`#lit.origin.union<op1, op2, …>`** — the union of two or more origins. A
reference with a union origin may point at storage governed by *any* member;
`CheckLifetimes` extends all member lifetimes while the reference is live.
Mutability of the union is parametric: the reference is mutable through the
union only if every member origin is mutable.

Used for `origin_of(a, b)` and `ref` returns that may refer to different
arguments on different paths:

```mojo
def pick(cond: Bool, mut a: String, mut b: String) -> ref[a, b] String:
    return a if cond else b
```

```mlir
!lit.ref<!String, mut #lit.origin.union<
  mut #kgen.param.decl.ref<"a">,
  mut #kgen.param.decl.ref<"b">>>
```

Suggestion for origin nerds: look at how
`def pick(cond: Bool, ref a: String, ref b: String) -> ref[a, b] String:` is
compiled. It allows `pick(cond, immString, mutString)` to
return an immutable reference, but allows `pick(cond, mutStr1, mutStr2)` to
return a mutable one.

**`#lit.origin.mutcast<operand>`** — an origin viewed at a different
mutability than its natural type. The operand keeps the same underlying origin;
only the `!lit.origin<mut>` type changes. Appears when parametric signatures
reconcile mutability (for example union members that must share a common
`!lit.origin<mut>` parameter) or when a `Bool` parameter constrains whether an
origin is mutable. The most common use is when a mutable reference is passed to
a read-only argument:

```mlir
#lit.origin.mutcast<
  #kgen.param.decl.ref<"a"> : !lit.origin<true>>
  : !lit.origin<false>
```

**`#lit.origin.eq<lhs, rhs>`** — a compile-time predicate: do two origins
denote the same value? Used in parser-evaluated `where` clauses on generic
signatures. Origins do not survive to elaboration, so this attribute is not
available at runtime or in `comptime if`.

## Origin attribute canonicalization

Given a closed algebra of attributes, every attribute builder in
`KGEN/lib/LITDialect/LITAttrs.cpp` normalizes its result on construction. The IR
you see in dumps is therefore always canonical, and the compiler relies on these
rules so downstream code (origin comparison, exclusivity, lifetime extension)
can treat syntactically different forms as equivalent.

The short version of the invariants is that you'll see attributes in the
following canonical form:

1) Unions (if present) on the outside.
2) MutCast within that.
3) Field references and interior origins within that.
4) Singletons and declaration references within that.

### Unions on the outside

`#lit.origin.field` and `#lit.origin.mutcast` distribute through each other
so that `#lit.origin.union` is never nested inside a field or cast. Field
selection into a union becomes a union of field selections:

```text
field(union(x, y), "f")  →  union(field(x, "f"), field(y, "f"))
```

`OriginMutCastAttr::get` on a union applies the cast to each member and
rebuilds the union. Nested unions flatten:
`union(x, union(y, z))` → `union(x, y, z)`.

### Union membership

When `#lit.origin.union<…>` is built:

- If any member is `#lit.any.origin`, the whole union becomes that wildcard.
- Operands are sorted into a stable order (comparison uses
  `OriginMutCastAttr::strip` to ignore outer casts).
- Duplicate members are removed after sorting.
- A one-element union collapses to its sole operand.
- The union's mutability is the logical **AND** of member mutabilities. When
  members disagree, each is wrapped in `#lit.origin.mutcast<…>` to the common
  mutability before the union is formed.

### Mutcast folding

- `mutcast(mutcast(x))` collapses to a single cast.
- If the operand already has the requested mutability, the cast is omitted.
- `#lit.any.origin` and `#lit.comptime.origin` never keep a cast wrapper; the
  builder returns the singleton at the target mutability.
- Mutcasts are pushed out of field references:
  `mutcast(x).field` → `mutcast(x.field)`.

For equality and exclusivity checks, use `OriginMutCastAttr::strip()` — outer
mutcasts are type sugar, not distinct origins.

### Origin equality folding

`#lit.origin.eq<lhs, rhs>` evaluates when possible: equal operands fold to
`true`; two distinct simple constants fold to `false`; otherwise the
comparison stays symbolic for the parser to resolve in a `where` clause. As
mentioned before, these can only be evaluated at parser time.

## Deep dive on interior origins

Many containers do not store their elements inline in the struct you hold in a
local variable. A `List` keeps its elements in heap storage; a `Variant` stores
its active value inside a discriminated buffer. When you take a reference into
that storage (`list[i]`, `variant[Int]`, and similar APIs), the reference points
at memory the container owns and may reallocate or overwrite.

That pattern is convenient, but it creates a classic memory-safety hole: the
container can change while an old reference still points at storage that is no
longer valid. Interior origins are how Mojo tracks those references at compile
time.

### A motivating example

```mojo
def example():
    var list = List[Int]()
    list.append(1)
    ref elt = list[0]
    elt += 4  # Valid: `list` has not invalidated this element reference.

    list.append(24)  # May reallocate; invalidates existing element refs.
    elt += 4         # Compile error: use of invalidated interior reference.
```

Here `elt` is not a reference to the `list` variable itself. It names an
element slot inside storage owned by `list`. After `append()` runs, the list may
move its buffer to a new address. The compiler rejects the second `elt += 4`
because the interior reference `list["element"]` was invalidated when the list
was mutated, even though `list` is still alive.

In C and C++, the same program is undefined behavior: `elt` may dangle after
reallocation, and nothing in the type system stops you from using it. Bugs like
this—iterator invalidation, use-after-reallocation, stale pointers into
`std::vector`—are a major source of security and reliability problems.

Rust largely prevents this class of bug with its borrow checker: while a
mutable borrow of a container is live, you cannot hold other references that
might alias the same storage. That model is sound, but it can feel restrictive.
Operations that reborrow or split borrows across fields often require careful
API design, and innocent-looking code can fail with borrow-check errors when
multiple handles into one collection are natural to write.

Mojo targets the same safety property with a different default: you may hold
references into container storage, but the compiler performs **flow-sensitive
invalidation** of those interior references. Mutating a container invalidates
interior references derived from it; using an invalidated reference is a
compile-time error, not undefined behavior at run time.

### What is an interior reference?

An **interior reference** is a `ref` to storage that lives inside (or behind) a
**base value** you already have in scope—a container, a variant buffer, a struct
field that owns heap data, and so on. The base value remains responsible for
allocating, moving, and destroying that storage.

Each interior reference carries an **interior origin**: a symbolic name for
“this reference points at interior storage owned by `base`.” Diagnostics render
that as a string tag on the base origin, for example `list["element"]` or
`v["value"]`. The tag identifies which logical slot inside the container the
reference names; it is not a run-time field name on your struct.

**`List`:** Indexing returns a reference into the list’s element buffer. The
interior origin is tied to the `list` variable; operations like `append()` that
can reallocate the buffer invalidate element references taken earlier.

```mojo
var list: List = [1, 2, 3]
ref first_ref = list[0]
first_ref = 10         # OK while the list has not invalidated `first_ref`.
list.append(4)         # Invalidates `first_ref` if reallocation occurs.
```

**`Variant`:** Typed access with `variant[T]` returns a reference to the active
variant payload inside the variant’s storage, not to the variant struct as a
whole.

```mojo
var v: Variant[Int, Float64] = 42
ref r = v[Int]
r = 100            # OK: `r` names the Int payload inside `v`.
v = 3.14           # Replacing the variant invalidates `r`.
```

In both cases the data you reach through the reference is fully owned and
managed by the container. Interior origin tracking connects the reference’s
lifetime to mutations on that owner.

Library authors who vend interior references use `@__defines_interior_origins`
on the accessor and derive the origin with helpers such as
`Origin._get_owned_interior[name]`. Application code typically just uses the
container API; the compiler attaches the interior origin automatically.
The specific decorator names are under discussion and expected to change, which
is why they start with double underscores. Similarly, the utility methods on
`Origin` and `Pointer` start with underscore to indicate they are still
evolving rapidly.

### Interior origins and flow sensitivity

Invalidation is **flow-sensitive**: the compiler tracks, at each point in your
function, whether a given interior reference is still valid. It merges that
information across control-flow joins the same way it tracks ordinary variable
liveness.

If a mutation might have run on any path to the current point, the interior
reference is treated as invalidated:

```mojo
def maybe_invalidate(cond: Bool):
    var list: List = [1, 2, 3]
    ref elt = list[0]
    elt = 5

    if cond:
        list.append(99)  # May invalidate `elt` on this path.

    # Error: `elt` might have been invalidated when `cond` was true.
    elt += 1
```

When no invalidating operation runs, uses remain valid—even across branches:

```mojo
def read_only_paths(cond: Bool):
    var list: List = [1, 2, 3]
    ref elt = list[0]
    var sum = 0

    if cond:
        sum += elt
        _ = len(list)   # Read-only; does not invalidate `elt`.
    else:
        sum += elt

    sum += elt          # OK: neither branch mutated storage in a way that
                        # invalidates element references.
```

You can always obtain a fresh interior reference after mutation. The new
reference is valid; the old one stays invalid:

```mojo
def refresh_after_mutation():
    var list: List = [1]
    ref old_elt = list[0]
    old_elt = 2

    list.append(99)     # Invalidates `old_elt`.
    ref new_elt = list[0]
    new_elt = 3         # OK: taken after the mutation.

    old_elt = 4         # Error: `old_elt` was not refreshed.
```

Interior references are control-flow aware, even when revived. For example,
Mojo is smart enough to know that valid references inside `if` or `else` bodies
do not revive a reference that was invalidated before the branch:

```mojo
def stale_after_branch(cond: Bool):
    var list = List(1)
    ref elt = list[0]
    elt = 1

    if cond:
        list.append(2)
        ref fresh = list[0]
        fresh = 3         # OK inside the branch.
    else:
        list.append(4)
        ref also_fresh = list[0]
        also_fresh = 5

    elt = 6                 # Error: `elt` was invalidated on both paths.
```

Flow sensitivity is what lets Mojo reject stale interior references without
banning every pattern where a container and a reference into it coexist. What
matters is whether an invalidating use might have happened on the path to each
use site. This follows Mojo's existing behavior that rejects uses of
uninitialized or transferred data.

### Interior origins and field sensitivity

Interior origins compose with Mojo’s existing **field-sensitive** origin
tracking. Sibling fields of a struct have distinct origins, and interior tags
are scoped under the path to the container that owns the storage.

```mojo
struct Pair:
    var left: List[Int]
    var right: List[Int]

def field_scoped_invalidation():
    var pair = Pair([1], [10])
    ref left_elt = pair.left[0]
    ref right_elt = pair.right[0]

    left_elt = 2
    right_elt = 20

    pair.left.append(99)   # Invalidates `left_elt` only.

    right_elt += 1         # OK: `right` was not mutated.
    left_elt += 1          # Error: invalidated interior reference.
```

Nested containers produce nested interior paths in diagnostics—for example
`outer["element"].left["element"]` when a list holds structs that themselves
contain lists. Invalidation at an inner container invalidates interior
references rooted under that container, not references rooted under sibling
fields.

Field sensitivity and flow sensitivity work together: the compiler knows both
**which** interior slot a reference names and **whether** that slot may still
be live at the current program point. That combination is what makes interior
references practical for standard containers like `List` and `Variant` without
 giving up the ASAP destruction and borrow checking guarantees described
 earlier in this document.
