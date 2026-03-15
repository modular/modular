# Comptime parameterized return type syntax

**Status**: Draft.

## Introduction

Functions in Mojo can be parameterized over compile-time values (parameters).
In some cases the *return type* of such a function should depend on those
parameters—for example, returning a value of type `T` when a boolean parameter
is true, and `None` when it is false. Today there is no direct, intuitive
syntax to express this; the manual documents no built-in way to choose the
return type from a comptime condition.

This proposal describes the desired behavior, the current workarounds and
their limitations, and suggests language syntax to support comptime-parameterized
return types in a clear and ergonomic way.

**Source**: Community discussion [Comptime parameterized return type](https://forum.modular.com/t/comptime-parameterized-return-type/2847).

## Problem statement

### Use case

A typical use case is a parser method that can optionally return the consumed
token:

```mojo
def expect[ReturnToken: Bool = False](mut self, t: Token.Type)
    -> ???:   # Token when ReturnToken else None

    ref cur = self.current()
    if unlikely(cur.type != t):
        raise Error("Unexpected token type ...")
    self.skip()
    if ReturnToken:
        return cur.copy()
    # else: return nothing / None
```

Callers want:

- `parser.expect[False](Token.LPAREN)` — returns `None` (or no value); type is `NoneType`.
- `parser.expect[True](Token.LPAREN)` — returns the matched `Token`; type is `Token`.

So the *return type* must be `Token` when `ReturnToken` is true and `NoneType`
when `ReturnToken` is false.

### Desired syntax (not valid today)

A natural way to express this would be a type-level conditional, e.g.:

```mojo
def expect[ReturnToken: Bool = False](mut self, t: Token.Type)
    -> Token if ReturnToken else None:
```

This does not work in current Mojo: `Token if ReturnToken else None` is not
accepted as a return type.

## Current workarounds

### 1. Two separate methods

Expose two APIs instead of one parameterized API:

```mojo
def expect(self, t: Token.Type) -> None: ...
def expect_and_return(self, t: Token.Type) -> Token: ...
```

**Pros**: Simple, no compiler magic, works on all Mojo versions.  
**Cons**: Duplicated logic, more surface area, callers must choose the method
name instead of a single parameter.

### 2. `ConditionalType` from `std.utils.type_functions`

The standard library (nightly) provides
[`ConditionalType`](https://github.com/modular/mojo/blob/main/stdlib/std/utils/type_functions.mojo),
a type function that selects between two types based on a boolean parameter:

```mojo
from std.utils.type_functions import ConditionalType

def expect[ReturnToken: Bool = False](mut self, t: Token.Type)
    -> ConditionalType[If=ReturnToken, Then=Token, Else=NoneType]:
```

**Limitations**:

- `ConditionalType` requires a `Trait` that both `Then` and `Else` conform to.
  For `Token` and `NoneType`, a common trait (e.g. `ImplicitlyCopyable`) must
  be specified; if one of the types does not conform, the pattern does not
  apply without wrapping or changing types.
- Using `ConditionalType` directly as the return type can hit compiler
  limitations; see below.

### 3. Parametric alias + `rebind` (workaround for compiler limits)

Forum feedback shows that using `ConditionalType` as the *direct* return type
can trigger errors (e.g. parameter `Else` having `AnyStruct[Token]` instead of
`NoneType`). A working pattern is to introduce a comptime alias and use
`rebind` in the implementation:

```mojo
from std.utils.type_functions import ConditionalType

comptime Cond[b: Bool] = ConditionalType[
    Trait=ImplicitlyCopyable, If=b, Then=Int, Else=NoneType
]

def expect[ReturnToken: Bool = False](t: Int) -> Cond[ReturnToken]:
    comptime if ReturnToken:
        return rebind[Cond[ReturnToken]](t)
    else:
        return rebind[Cond[ReturnToken]](None)
```

**Pros**: Single API, parameterized return type.  
**Cons**: Verbose; requires a shared `Trait`; use of `rebind` is non-obvious;
and the need for a separate alias and rebind is a compiler/implementation
limitation rather than a desired user-facing design.

## Proposal

### Goal

Enable a direct, intuitive way to declare that a function’s return type depends
on a compile-time condition, without requiring `ConditionalType`, a common
trait, or `rebind` in typical cases.

### Option A: Ternary type expression in return position

Allow a conditional type expression in return type position, using the same
semantics as a type-level “if/else”:

```mojo
def expect[ReturnToken: Bool = False](mut self, t: Token.Type)
    -> (Token if ReturnToken else None):
    ...
```

Grammar (conceptual): in a type position, allow `Type1 if Condition else Type2`
where `Condition` is a compile-time boolean expression (parameter, or
expression of parameters). The result is `Type1` when the condition is true
and `Type2` when false.

- **Pros**: Reads like the intended semantics; familiar from value-level
  conditionals.  
- **Cons**: May require grammar and type-checking changes; interaction with
  traits and inference needs to be specified.

### Option B: Standardize on `ConditionalType` and fix compiler/UX

Keep the return type as a type function application, but:

1. **Compiler**: Fix limitations that currently prevent using
   `ConditionalType[If=..., Then=Token, Else=NoneType]` directly as the
   return type (and improve error messages).
2. **Library**: Consider a variant or overload of `ConditionalType` that
   does not require a single `Trait` when both branches are valid (e.g. allow
   `AnyType` or a union of constraints), so that `Token` and `NoneType` can
   be used without forcing a common trait.
3. **Documentation**: Document the pattern (including any need for `rebind` or
   an alias) in the manual so that “comptime parameterized return type” is a
   first-class, supported pattern.

This improves the status quo without new syntax.

### Option C: Sugar for `ConditionalType` in return type

Introduce lightweight syntax that desugars to `ConditionalType` (or an
equivalent type function), e.g.:

```mojo
def expect[ReturnToken: Bool = False](mut self, t: Token.Type)
    -> if ReturnToken then Token else None:
```

So the compiler treats `if Param then T1 else T2` in return type position as
`ConditionalType[If=Param, Then=T1, Else=T2]` (with trait resolution as needed).

**Pros**: Single, readable form; implementation can align with existing
`ConditionalType` semantics.  
**Cons**: New syntax to maintain; must define trait/conformance rules for the
two branches.

## Design questions

1. **Trait / constraint handling**: For `T1 if Cond else T2`, should both types
   be required to conform to a common trait (for consistency with
   `ConditionalType`), or can the compiler infer a union of constraints?
2. **Scope**: Should this apply only to return types, or to any type position
   (e.g. variable annotations, parameter types)?
3. **Multiple parameters**: Should we support only a single boolean condition,
   or allow more general comptime expressions (e.g. multiple conditions,
   nested conditionals)?
4. **Interaction with `comptime`**: How does this interact with
   [comptime expression syntax](comptime-expr.md) and possible future
   `comptime if` / parameter control flow in return-type position?

## Summary

| Approach | New syntax | Relies on ConditionalType | Compiler/library fixes |
|----------|------------|---------------------------|-------------------------|
| A. Ternary type | `T1 if Cond else T2` | No | Type checker + grammar |
| B. Fix current approach | No | Yes | Yes (return type + optional trait relaxation) |
| C. Sugar | `if Cond then T1 else T2` | Desugar to it | Possibly |

This document is a draft to shape discussion. Feedback from the community and
the Mojo team on the preferred direction (A, B, C, or a combination) is
welcome.
