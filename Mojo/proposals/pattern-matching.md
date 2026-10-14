# Mojo Pattern Matching

**Status**: Implementation underway.

Date: Aug 29, 2026; Updated Oct 10, 2026.

This is a design to add "pattern matching" to Mojo, using the existing Python
`match` syntax, using it as a baseline for the core syntax and semantics with
a few minor extensions to make the behavior comparable to modern functional
languages, and providing support for pattern matching in `if` statements.

A separate [Enums for Mojo Notes](enums.md) doc builds on the pattern system
described here, but enums are not required to implement or motivate the core
pattern-matching design.

## Introduction

**Pattern matching** is a general mechanism for testing whether a value has a
particular structure and, when it does, extracting pieces of that value into new
bindings. Mojo follows Python's core syntax and extends it for language-specific
needs such as ownership handling.

Let's look at an example. Here we can see a pattern that simultaneously tests
parts of a tuple and bind others:

```mojo
def inspect(point: Tuple[Int, Int]):
    match point:
        case 0, 0:
            print("origin")
        case x, 0:
            print("on the x axis:", x)
        case 0, y:
            print("on the y axis:", y)
        case x, y:
            print("point:", x, y)
```

The essential operation here is structural: compare parts of a value against
patterns while binding other parts to names.

Patterns are useful well beyond `match` statements. Python has rich pattern
matching, but it is not alone. Languages such as Rust and
Swift treat patterns as a general language concept that can be reused across
declarations and control-flow constructs. For example, Rust uses patterns in
ordinary destructuring:

```rust
let (x, y) = point;

if let [first, second] = values.as_slice() {
    use_values(first, second);
}

let Point { x, y } = point;
```

Swift similarly uses patterns in constructs such as `switch`, `if case`, `guard
case`, and `for case`:

```swift
if case let (x, 0) = point {
    print("x axis:", x)
}
```

These languages therefore suggest the same useful decomposition for Mojo:

1. **Patterns describe values.** They provide matching, decomposition, and
   binding semantics independently of any particular control-flow construct.
2. **Patterns compose recursively.** Binding, wildcard, tuple, sequence, value,
   and struct patterns fit into a common model rather than being unrelated
   language features.
3. **Different language constructs can consume patterns differently.** A `match`
   statement may try several patterns, while an `if` may test one pattern and
   ordinary destructuring may require a particular structure.

The goal is therefore to establish **patterns as a first-class piece of the
language grammar and semantic model**, rather than designing `match` as an
isolated control-flow feature.

### Destructuring in Mojo Today

Mojo already supports rich **destructuring** in declarations and assignments.
For example:

```mojo
# Simple assignment
var x = get_value()

# Tuple destructuring
var (x, y) = get_pair()

# Generalized assignment targets
(var x, ref y, _) = get_values()
(a[i], b[j]) = get_pair()

# These nest arbitrarily
(var x, (ref y, _)) = get_nested_values()
```

The important observation is that Mojo already has a composable language for
describing how the components of a value are distributed into bindings and
assignment targets. `var x`, `ref y`, `_`, generalized lvalues, and `(p1, p2,
...)` can all participate recursively in destructuring. As with Python, tuple
destructuring doesn't even require parentheses in contexts where the grammar is
unambiguous.

Mojo also supports destructuring in other assignment-like contexts, such as
`for` loop targets. Note that `for` loop targets default to implicitly binding
names with "imm" bindings:

```mojo
for key, value in entries:
    ...
```

Today, destructuring is required to be statically known to succeed. For example,
given:

```mojo
(var x, var y) = value
```

the compiler must know that `value` has an appropriate two-element tuple
structure. A value with an incompatible structure is a compile-time error rather
than something that can fail dynamically.

### Destructuring vs. Match Patterns

Destructuring and pattern matching are closely related, but they are not quite
the same thing. This proposal follows Python's general design by keeping them as
distinct grammatical contexts.

A **destructuring target** describes where pieces of a value should go. A
**match pattern** instead asks whether a value satisfies some condition and may
introduce bindings when it does:

```mojo
match get_pair():
    # Bind x, check against 0.
    case x, 0:
        use(x)

    # Error: arbitrary expressions/lvalues aren't value patterns.
    case var x, a[2]:

    # Ok: Use == syntax to make comparison against a value explicit.
    case var x, ==a[2]:
```

Match patterns intentionally use a restricted grammar rather than treating
arbitrary Mojo expressions as equality tests. They can include literals,
contextual initializer lists, and other explicitly supported pattern forms, but
not arbitrary calls, operators, subscripts, or generalized lvalues. This avoids
surprising interactions with expressions that produce lvalues or references
and keeps patterns statically understandable.

Ordinary assignment continues to use Mojo's existing assignment-target rules:

```mojo
(var a, 0) = get_pair()   # Error: `0` isn't an assignment target.
0 = get_value()           # Error.

a[2] = get_value()        # assignment into generalized lvalue is ok.
```

Despite this difference, destructuring targets and match patterns share much of
the same recursive structure: tuple decomposition, `var`/`ref` bindings,
wildcards, sequence decomposition, and other forms can reuse the same underlying
compiler machinery.

## The `match` Statement

The most direct way to use match patterns is with a `match` statement. For
example, a dynamically-sized list can be classified by its shape. Here we use
the `[a, b, c]` pattern which matches against a list of a specific length:

```mojo
match values:
    case []:
        print("empty")

    case [x]:
        print("one element:", x)

    case [x, y]:
        print("two elements:", x, y)

    case _:
        print("many elements")
```

Conceptually, `match` evaluates its subject once and considers its cases in
source order.

For each `case`, the corresponding pattern is applied to the subject. If the
pattern succeeds, any bindings it introduces become available in the case body
and that body executes. If the pattern fails, execution proceeds to the next
case.

This introduces the important distinction between **irrefutable** and
**refutable** patterns.

An irrefutable pattern is statically guaranteed to match every value of its
input type. For example:

```mojo
case x:
case var y:  # Copy into a mutable local var.
case _:
```

A refutable pattern may or may not match depending on the runtime value. A
fixed-length sequence pattern is one simple example:

```mojo
case [x, y]:
```

If the subject is a dynamically-sized list, this succeeds only when it contains
exactly two elements.

Value patterns are another important refutable form:

```mojo
match point:
    case 0, 0:
        print("origin")

    case x, 0:
        print("on the x axis:", x)
```

Here `var x, 0` first decomposes the tuple, binds its first element to `x`,
and requires its second element to match `0`. Failure of either requirement
means that the next case is considered. Like Python, Mojo only supports constant
value patterns - constant integers, floating point, bool, strings, etc.

Patterns compose recursively, so refutability naturally composes as well:

```mojo
match values:
    case [(x, 0), a_pair]:
        handle_special_value(x, a_pair)

    case [x]:
        handle_one_value(x)

    case _:
        handle_other_lengths(values)
```

A pattern is therefore **irrefutable when the compiler can prove that it matches
every value of its input type; otherwise it is refutable**.

Patterns that can be proven to never match are diagnosed at compile time. For
example, matching a statically known two-element tuple with a three-element
tuple pattern is not a runtime match failure—it is simply invalid code.

> 💡 Patterns that can **NEVER** match are a compile-time error. We expect `case
> var (a, b):` to be a compile-time error if the matched value is a 3-element
> tuple.
>

`comptime match` applies the same pattern model when the subject is a
parameter expression and performs the match at compile time.

### Exhaustiveness and unreachability

`match` still evaluates the subject once and tries cases in source order. On
top of that, the implementation performs a **covering analysis** that:

- Warns when a `match` on a finite subject is not exhaustive.
- Warns when a later case is unreachable because earlier cases already cover
  every value it could match.
- Marks covered / concluding cases so lowering can cut false fallthrough off.

Cases with `if` guards never contribute to coverage: a guarded case may fail at
runtime, so it does not shrink the remaining space. Coverage is driven by the
subject's type and the pattern command list for each case:

1. **`EnumLike` roots (finite constructors).** The open set is the enum's
   constructor tags (for example `True`/`False`, `Optional.Some`/`None`). Simple
   root tag / Bool-style equal patterns clear one constructor. A match that
   leaves any constructor open is non-exhaustive; repeating an already-cleared
   constructor is unreachable. `EnumLike._enum_is_exhaustive` defaults to
   `True` for this closed-enum behavior.

2. **Open `EnumLike` roots.** A conforming type can set
   `_enum_is_exhaustive = False` when `_enum_case_names` lists known
   alternatives but other values may exist. Known alternatives remain
   mutually exclusive and duplicate cases are still diagnosed, but covering
   every known alternative does not make the match exhaustive or a following
   catch-all unreachable. A product containing an open `EnumLike` leaf is also
   open.

3. **Products of `EnumLike` leaves.** Tuples and structs whose leaves are all
   `EnumLike` are flattened into a cartesian cell bitset so correlated patterns
   cover the right combinations. Independent per-field bitsets would wrongly
   treat `True, True` plus `False, False` as total for a `Bool × Bool` subject;
   the cell model does not. Omitted struct fields and `_` in a position are
   wildcards over that dimension.

4. **OR patterns.** An `a | b` case expands to a union of alternatives. Each
   alternative is covered independently; the case is credited with the union.
   An irrefutable alternative (for example `_` or bind-only) closes the whole
   space.

5. **Enum payload refinements.** A constructor is covered only when its payload
   pattern is irrefutable. `Optional.Some(_)` covers `Some`, but
   `Optional.Some(0)` does not: it leaves other `Some` payloads open, so a later
   `Some(_)` remains reachable. In the opposite order, `Some(_)` makes a later
   `Some(0)` unreachable because the refined pattern is a subset of the
   constructor.

6. **Oversized products.** If the flattened cell count exceeds a fixed cap
   (currently 1024), the subject is **too complex** to track cell-by-cell. The
   compiler does not silently give up: it warns that exhaustivity cannot be
   checked and requires a catch-all `_` (or other irrefutable case).

7. **Open literal subjects (`Int`, `String`, …).** These are an open universe.
   The analysis tracks root equal-literal spellings (`4`, `"foo"`, …) so
   duplicate / already-covered literal cases warn, but covering literals never
   proves exhaustivity and never requires `_` by itself. A trailing irrefutable
   case still closes the space for unreachability of later arms.

Irrefutable unguarded patterns (`_`, bare bind) always close the remaining
space. After the space is closed, every subsequent case is unreachable.

```mojo
# Finite EnumLike: both ctors required (or a catch-all).
match flag:  # warning: not exhaustive; missing case for 'False'
    case True:
        ...

# Product of EnumLike: correlated wildcards can finish the space.
match flag, flag:
    case True, True:
        ...
    case False, True:
        ...
    case _, False:
        ...
    case _:  # warning: unreachable; previous cases cover every value
        ...

# Open literals: duplicates warn; no exhaustivity requirement.
match n:
    case 0:
        ...
    case 0:  # warning: '0' is already covered by a previous case
        ...
    case 1:
        ...

# Open EnumLike: known cases do not cover unknown values.
match status:
    case .ready:
        ...
    case .stopped:
        ...
    case _:
        ...  # reachable because Status._enum_is_exhaustive is False
```

#### Intentional Limitations

These are the main limitations that people may ask about:

- **Guards do not count.** `case True if cond:` does not cover `True` for
  exhaustivity, because the parser cannot reason about conditions (even if
  comptime expressions) at parser time.
- **Payload domains are not refined.** Refined payload patterns such as
  `Some(0)` receive no constructor-level coverage credit. The analysis does not
  prove that combinations of payload patterns cover a constructor or diagnose
  duplicate payload patterns.
- **Nested / product literals on open types are not tracked.** A subject such
  as `Tuple[Int, Int]` is not a finite `EnumLike` product, so it falls into the
  open-literal model, which only records **root** equal spellings. Duplicate
  `case (0, 1):` arms are not diagnosed today.
- **Literal identity is spelling-based.** `4` and `0x4` are different keys;
  concatenated string literals without a single stable spelling are not
  tracked. This could be addressed with increased implementation, but would be
  very niche.
- **Generic `EnumLike` types** case analysis is done by the Mojo parser, so
  matching over a generic `EnumLike` won't work.
- **Sequence, dict, and other future pattern forms** are outside this
  covering model until those patterns land.

## Pattern forms

Mojo already has several building blocks that naturally carry over into match
patterns: `var` / `ref` bindings, `_` wildcards, tuple decomposition, and
recursive nesting. The pattern grammar builds on this foundation with the
following forms:

1. **Literal value patterns** match against a literal:

    ```mojo
    case 0:
    case "hello":
    case 0, 1:  # two values in a tuple pattern
    ```

    A bare identifier introduces a binding, so `case Int:` binds the subject to
    a new name `Int` instead of comparing the subject to the type `Int`.

2. **Explicit equality patterns** compare the subject to the value of an
   expression:

    ```mojo
    case == expected:
    case (x, == expected):
    case == compute_expected():
    ```

    The explicit marker distinguishes comparison from the implicit binding
    introduced by a bare identifier. It also allows matching on types without
    introducing special type-pattern semantics:

    ```mojo
    comptime match T:
        case == Int:
            return "integer"
        case == String:
            return "string"
        case _:
            return "other"
    ```

    Types are compile-time values and equality is already defined on them, so
    `== Int` is an ordinary equality pattern whose expected value happens to be
    a type. PEP 635 records similar explicit constant markers considered for
    Python, and PEP 642 proposed the same `== expected` syntax.

3. **Sequence patterns** decompose sequence-like values:

    ```mojo
    case []:
    case [x]:
    case [x, y]:
    ```

    These are refutable when the runtime sequence may have a different length.
    The subject must conform to `Sized` and support integer subscripting. The
    same protocol supports ordinary list destructuring assignments.

4. **Variable-length sequence patterns** capture a prefix, suffix, or remainder
   (only a single `*` pattern is allowed):

    ```mojo
    case [first, *rest]:
    case [first, *middle, last]:
    ```

    The subject must also support `ContiguousSlice` subscripting. The starred
    pattern receives the slice between the fixed prefix and suffix.

5. **OR patterns** match when any of several alternatives match:

    ```mojo
    case 0 | 1:
    case [var x, 0] | [var x, 1]:
    ```

    Alternatives that introduce bindings must introduce compatible bindings on
    each path.

6. **AS patterns** apply a pattern while also retaining the complete matched
   value, producing an `imm` binding:

    ```mojo
    case [var x, var y] as value:
        ...
    ```

7. **Struct patterns** decompose struct-like values:

    ```mojo
    case Point(x=x_value, y=y_value):
        print(x_value, y_value)
    ```

    Each element of the pattern must be a keyword argument denoting a valid
    field name of the struct. Field names may be omitted but not duplicated.

8. **Initializer-list patterns** are struct patterns that implicitly use the
   type of the subject to know which struct type is being inspected.

    ```mojo
    case {x=1, y=y_value}: print(y_value)
    ```

9. **Mapping patterns** match particular keys and decompose their values:

    ```mojo
    case {"name": name, "age": age}:
        ...
    ```

Finally, **match guards** are useful but are not themselves patterns:

```mojo
case [x, y] if x < y:
    ...
```

The pattern matches first and introduces its bindings; the guard is then
evaluated to determine whether the case is selected.

## Conditional Pattern Bindings: “`if case`” for Mojo

Pattern matching also fits naturally into Mojo's existing `if` statement, and it
makes sense to allow the core "check a pattern matches and bind the expression
in the body of the "then" block. To avoid ambiguity, this is introduced with
"if case" syntax. For example:

```mojo
if case [a, 0] = get_list():
    use(a)
```

The right-hand side is evaluated once and matched against `[a, 0]`. The
condition succeeds only if the list contains exactly two elements and its second
element matches `0`. On success, `a` is bound and the body executes; otherwise
the condition is false.

This is intentionally more expressive than ordinary destructuring:

```mojo
# Ordinary destructuring: valid, but throws if the length is wrong.
var [a, b] = get_list()

# Ordinary destructuring: invalid because `0` isn't an assignment target.
var [a, 0] = get_list()

# Conditional matching: valid; failure takes the false path.
if case [a, 0] = get_list():
    use(a)
else:
    handle_other_shape()
```

Bindings introduced by a successful conditional match are scoped to the
corresponding `if` body:

```mojo
if case [first, second] = get_list():
    use(first, second)

# `first` and `second` are not available here.
```

This provides the capability commonly spelled `if let` in languages such as Rust
and Swift:

```mojo
if case (x, [y, 0]) = get_value():
    use(x, y)
```

Irrefutable patterns can technically appear in this position as well:

```mojo
if case x = get_value():
    use(x)
```

but such a condition is statically known to succeed and is likely a mistake. The
compiler should probably warn about this, just as it may diagnose other
conditions known to be constant.

This gives `if` conditions two natural forms:

- A boolean expression succeeds when it evaluates to `True`.
- A conditional pattern match succeeds when its pattern matches.

The same treatment should apply to `while` loop conditions:

```mojo
while case [item, 0] = get_next():
    use(item)
```

### Alternative considered

**`if pattern = expr` with no `case keyword**

We considered eliding the `case` keyword in `if case`. Note that `=` is a
**statement**, rather than an expression. This means that we could have allowed
the user to write:

```mojo
if expression:
    ...

if pattern = expression:
    ...
```

In the second form, the left-hand side is parsed using the full **match-pattern
grammar**. We can consider enabling this syntax in the future, but it admits a
number of very confusing things, including things like `if 4 = foo():` and
`if (True, x) = bar():` which visually look like they are comparison, not
assignments. Requiring the `case` keyword makes these visually more distinct
and obvious to readers.

**Trailing condition on `if case`**

We could allow trailing "if" conditions on `if case` following the functionality
of `case` in a match:

```mojo
if case .Some(x) = get_optional_int() if x != 42:
    print(x)
```

However, repeating `if` within a single condition is visually awkward. In
`if case ... if ...`, the second `if` can look like the start of a nested
statement, and the syntax gives the trailing boolean condition a special
position that cannot naturally be followed by another pattern match.

If we want more expressive capability to reduce nested `if`s, we should consider
a Swift-style comma-separated condition list instead. This allows each clause to
be either a boolean condition or a conditional pattern match:

```mojo
if case .Some(x) = get_first(), x != 42, case .Some(y) = get_second(x):
    use(x, y)
```

This form flows lexically from left to right and naturally supports sequences
such as pattern match, boolean condition, and another pattern match. Bindings
introduced by a `case` clause would be available to the clauses that follow it.

## Implementation status

The core proposal is fully implemented and expected to work correctly. The
following parts of this proposal are not yet implemented:

- **Mapping patterns** — for example `case {"name": name, "age": age}:`.
