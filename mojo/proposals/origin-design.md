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
3) Field references within that.
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
