# Enums and unions for Mojo

**Status**: Refined proposal, but not finalized.

Date: Nov 7, 2026

This document describes a design for enums and unions in Mojo. This builds on
the [Mojo Pattern Matching Notes](pattern-matching.md) doc - make sure you have
that in context before digging into this.

## Introduction

This proposal adds two related nominal types to Mojo: **enums** and **unions**.
Both declare a closed set of named cases, but they serve different purposes:

- An enum case never carries a payload. Enums represent a fixed set of named
  values and may eventually support explicit integer storage types and values.
- A union case always carries exactly one payload. The payload may be the unit
  type, written as an empty tuple `()`, when a case has no associated data.

For example, `Color` is an enum whose cases are plain named values:

```mojo
enum Color:
    case red, green, blue
```

Enums permit comma-separated case declarations because the cases have no
independent payload declarations. By contrast, `Result` is a union whose cases
carry different payload types:

```mojo
union Result[T: Movable, E: Movable](Movable):
    case success(T)
    case failure(E)
```

A `Result[T, E]` value contains either a `T` or an `E`, together with the case
that identifies the active payload. Union cases are declared separately rather
than in comma-separated lists because each case defines its own payload type.

This split follows the broad C and C++ precedent: enums model named,
integer-like values, while unions provide storage for one of several payload
types. Mojo unions are safe, tagged unions rather than C-style untagged unions,
so the active case is known and participates in pattern matching. The overall
model also has extensive prior art in functional-language algebraic data types,
Swift enums, Rust enums, and C++ facilities such as `std::variant`. Swift and
Rust use one construct for both payload-free and payload-bearing cases; this
proposal separates them to preserve the useful distinction between enums and
unions and to leave room for integer-backed enums, and unsafe untagged C-style
unions.

Enums and unions otherwise behave like normal Mojo nominal types. They may have
methods, conform to traits, be generic, contain nested declarations, and
participate in Mojo's ownership system. They complement structs: a `struct`
contains a product of its stored values, while `enum` and `union` declarations
contains one alternative from a closed set.

## Algebraic data type model

Enums and unions are two kinds of nominal **algebraic data types**. Each has a
statically known, closed set of cases, and each value has exactly one active
case. They differ in whether those cases contain associated state:

```mojo
enum Color:
    case red, green, blue

union Result[T: Movable, E: Movable](Movable):
    case success(T)
    case failure(E)
```

A `Color` value is one of three payload-free cases. A `Result[T, E]` value is
either a `success` containing a `T` or a `failure` containing an `E`.

Enums and unions may have methods, conform to traits, declare comptime values,
and use generic parameters. They may not declare stored instance fields outside
their cases:

```mojo
union WeirdResult[T: Movable, E: Movable](Movable):
    var timestamp: Int  # error: unions may not have additional stored state

    case success(T)
    case failure(E)
```

Mojo does not allow this because shared stored state can already be expressed
with composition. Allowing it directly would also introduce unnecessary
questions about initialization, destruction, and access across cases. Swift and
Rust follow the same restriction for their algebraic data types.

## Enums

An enum represents a fixed set of payload-free named values. Cases are
introduced with `case` and may be grouped in comma-separated lists:

```mojo
enum Color:
    case red, green, blue
```

Because no case has stored payload state, every enum automatically conforms to
the standard value traits: `Deinitable`, `Movable`, `Copyable`,
and `ImplicitlyCopyable` - you can opt-out of one of these with
(e.g.) `not Movable` conformance if there is a reason to.

Enums can synthesize `Equatable`, `Comparable`, `Hashable`, and `Writable`
conformances when requested in the conformance list. These conformances use the
active case's identity: equality and ordering compare cases, hashing hashes the
case, and formatting writes its case name.

Each enum case is a value, not a constructor. Code refers to it without
parentheses:

```mojo
var color = Color.red
set_brush_color(.green)
```

Writing `Color.red()` is an error. Parentheses imply construction of a payload,
and enum cases never have payloads.

Enums are likely to grow integer-oriented capabilities familiar from C and
C++, including explicit backing integer types and explicit values for cases.
Those extensions do not change the core rule that an enum case is a
payload-free value.

## Unions

A union represents a choice among payload-bearing alternatives. Every case
declares exactly one payload type (though it may be a tuple) and each case must
be declared separately:

```mojo
union Result[T: Movable, E: Movable](Movable):
    case success(T)
    case failure(E)
```

Union cases always use parentheses in declarations and construction. An empty
parenthesis list is the empty tuple payload:

```mojo
union Poll[T: Movable](Movable):
    case pending()
    case ready(T)

var pending = Poll[Int].pending()
var ready = Poll[Int].ready(42)
```

Writing `Poll.pending` without parentheses produces an unapplied function for
the constructor. This keeps union cases uniform: every case is a constructor and
every constructed value contains a payload, even when that payload is the empty
tuple.

The parenthesis list is tuple-like: its elements define the payload type, and
the case constructor accepts the corresponding values as arguments. The payload
is one unlabeled type. As a convenience, several types inside the parentheses
form a tuple payload:

```mojo
union Token(Movable):
    case eof()  # Payload has type Tuple[]
    case identifier(String)  # Payload has type String
    case integer(Int)  # Int
    case source_range(Int, Int)  # Tuple[Int, Int]
```

Payloads are positional and do not support keyword labels. For example,
`case source_range(start: Int, end: Int)` is not part of the model. When a
payload needs named fields, define a struct and use that struct as the payload:

```mojo
@fieldwise_init
struct SourceRange(Movable):
    var start: Int
    var end: Int

union Token(Movable):
    case eof()
    case source_range(SourceRange)
```

Generic types and constraints follow the normal rules for nominal types. The
payload type describes the state associated with that case; it does not declare
independent fields on the enclosing union.

## Matching and destructuring cases

Enums and unions use the same case-pattern machinery. A case pattern first
tests the active case. An enum pattern is complete at that point because an
enum has no payload:

```mojo
match color:
case .red:
    print("red")

case .green:
    print("green")

case .blue:
    print("blue")
```

A union pattern also recursively applies its payload pattern to the active
case's associated value:

```mojo
match result:
case .success(var value):
    use(value)
case .failure(var error):
    handle(error)
```

Multi-element union payloads are tuples, so ordinary tuple patterns decompose
them positionally:

```mojo
match some_token:
case .eof():
    ...
case .identifier(value):
    ...
case .integer(value):
    ...
case .source_range(start, end):
    ...  # tuple pattern against Tuple[Int, Int]
```

Keyword labels on the payload are not allowed: there are no labeled fields to
bind:

```mojo
case .source_range(start=x, end=y):  # error: not allowed
```

To destructure by name, store a named struct as the payload and use a struct
pattern:

```mojo
@fieldwise_init
struct SourceRange(Movable):
    var start: Int
    var end: Int

union Token(Movable):
    case eof()
    case source_range(SourceRange)

match some_token:
case .source_range({start=x, end=y}):
    ...
```

The case pattern's payload subpatterns are ordinary **refutable patterns**. They
are not specific to the `match` statement and work with `if case` syntax:

```mojo
if case .success(var value) = result:
    use(value)
```

Case patterns compose recursively with all other pattern forms:

```mojo
case .success([var first, *rest]):
    ...
```

Likewise, union payloads may contain enums, unions, or other destructurable
types.

The [pattern-matching proposal](pattern-matching.md) defines the general
mechanics of pattern success, failure, binding, and control flow. Enums and
unions add another refutable pattern form to that system, implemented against
the `SumType` family of traits described below.

### How pattern matching works

From the programmer's point of view, there are two surface forms:

1. **Enum patterns** name a case without parentheses, such as `.red` or
   `Color.blue`. They succeed when the subject's active case matches the name.
2. **Union patterns** always include a payload pattern in parentheses, such as
   `.success(ref value)` or `.pending()`. They succeed when the active case
   matches and the payload pattern matches the associated value. A
   multi-element payload is one tuple value, so several subpatterns form a
   tuple pattern.

A few rules follow from that split:

- Enum cases reject parentheses because there is no payload to destructure.
- Union cases require parentheses. Use `_` to ignore a payload or an empty
  parenthesis list to match an empty-tuple payload.
- Payload patterns are positional. There are no `name=` bindings against the
  case itself; use a struct payload when named fields are required.
- Leading-dot forms (`.success(ref x)`) resolve the case name against the
  subject's type, the same way contextual member lookup works for
  construction.
- Qualified forms (`Result.success(ref x)`) require the type prefix to be the
  subject's nominal type (unbound `Result` may match `Result[Int, Error]`).

Under the hood, the compiler does not need separate matching algorithms for
`enum` and `union`. It asks whether the **subject type** conforms to `SumType`,
then:

1. Resolves the written case name (`.success` / `Result.success`) to a case
   index by searching `_sum_case_names`.
2. Emits a predicate comparing `_get_sum_discriminant()` to that index.
3. For a union pattern, verifies `UnionLike`, projects the active payload with
   `_unsafe_get_union_payload[id]()` and recursively matches the payload
   pattern against that projected value (short-circuiting so the payload is
   only projected when the discriminant matches).

Payload bindings use the same `var` / `ref` / bare-binding rules as every
other pattern. For example:

```mojo
match value:
case .some(ref element):
    use(element)  # borrows the payload; mutability follows the subject
```

borrow the payload rather than copying or moving it, subject to the usual
ownership and exclusivity rules.

Unknown case names (`Optional.Nope`) are diagnosed against `_sum_case_names`
rather than ordinary member lookup: logical cases need not exist as real members
of the type for matching to work (hand-written `SumType` conformances can
expose logical cases that are not constructor APIs).

## Methods, traits, and generics

Enums and unions are full nominal Mojo types. They support methods in the same
way as structs:

```mojo
enum Color(Equatable):
    case red, green, blue

    def is_red(self) -> Bool:
        return self == .red  # Use the generated Equatable conformance.
```

They also support explicit trait conformance:

```mojo
union Things(Fooable):
    case thing1(Int)
    case thing2(String)

    def foo(self):
        print("foo'd")
```

Generic parameters and constraints follow the normal rules for nominal types.

More generally, enum and union declarations support the same capabilities as
struct declarations wherever doing so is semantically meaningful:

- instance and static methods;
- trait conformance and default implementation inheritance;
- generic parameters and constraints;
- comptime values and type aliases;
- computed properties and nested declarations.

The important restriction is on **stored instance state**, not on the
capabilities of the nominal type itself.

This gives Mojo a simple guiding rule:

> A struct contains all of its stored fields simultaneously. An enum contains
> one payload-free named value. A union contains the payload of exactly one
> active case.

Everything else should reuse the normal Mojo type system as much as possible.

## Ownership and mutation

Enums and unions participate naturally in Mojo's ownership system. Enums have
no payload ownership to manage. A union's ownership properties depend on the
payloads that its cases may contain. For example:

```mojo
union Result[T: Movable, E: Movable](Movable):
    case success(T)
    case failure(E)
```

Moving a union moves the active payload. Destroying a union destroys the payload
of the active case. Copyability similarly depends on whether all possible
payloads support the required copying semantics.

A union value may also be replaced with a value containing a different case:

```mojo
var result = Result.success(value)
result = Result.failure(error)
```

This destroys the old active payload and initializes the new one according to
Mojo's normal assignment semantics.

Pattern matching should provide the normal Mojo binding conventions for
associated state. For example:

```mojo
match result:
case .success(ref value):
    use(value)
case .failure(ref error):
    handle(error)
```

should borrow the payload rather than copying or moving it.

Similarly, mutable access to a union payload should follow the same ownership
and exclusivity rules as mutable access to ordinary stored state.

## The `SumType`, `EnumLike`, and `UnionLike` traits

The `enum` and `union` syntax both introduce closed sets of cases, but the
underlying semantic abstraction is broader than either syntax. Mojo captures it
with three built-in traits:

```mojo
# All enums and unions conform to this trait.
trait SumType:
    comptime _sum_case_names: _MLIR.KGENParamListType[KGENString]

    def _get_sum_discriminant(self) -> Int:
        ...

# Enums conform to this trait.
trait EnumLike(SumType):
    pass

# Unions conform to this treait.
trait UnionLike(SumType):
    comptime _union_case_types: _MLIR.KGENParamListType[AnyType]

    comptime _union_elt_type_for_case[id: Int]: AnyType = TypeList[
        Self._union_case_types
    ]()[id]

    # FIXME: Prefer an interior origin so payload refs preserve subject
    # mutability cleanly.
    def _unsafe_get_union_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._union_case_types]()[id]:
        ...
```

`SumType` describes the common behavior: a finite set of named alternatives
and the active case's discriminant. Code generic over `SumType` can inspect or
format the active case without caring whether the value was declared as an enum
or union. `EnumLike` and `UnionLike` refine that common interface to preserve
syntax-specific information for clients that need it, such as a pretty printer
that prints a union payload but never tries to project one from an enum.

The compiler synthesizes `SumType` and `EnumLike` conformances for every enum,
and `SumType` and `UnionLike` conformances for every union. The traits are not
restricted to compiler-generated algebraic data types. User-defined types may
conform when they provide equivalent semantics. `Optional` is the motivating
example: it is a struct that can conform to `UnionLike` so pattern matching can
treat `None` and `Some` as logical cases:

```mojo
struct Optional[T: Movable](Copyable, UnionLike):
    ...
    # These lists must have the same length, and all discriminators must be
    # within the range.
    comptime _sum_case_names = ParameterList.of[
        "None".value, "Some".value
    ].values
    comptime _union_case_types = TypeList.of[Tuple[], Self.T].values

    def _get_sum_discriminant(self) -> Int: ...
    def _unsafe_get_union_payload[id: Int](ref self) -> ...: ...
```

The runtime discriminant selects the active alternative:

```mojo
Color.red._get_sum_discriminant() == 0
Color.green._get_sum_discriminant() == 1
Color.blue._get_sum_discriminant() == 2
Result[Int, Error].success(42)._get_sum_discriminant() == 0
Result[Int, Error].failure(error)._get_sum_discriminant() == 1
```

The discriminant is a semantic case index, not necessarily a promise about the
physical representation or ABI of the type. An implementation may use an
explicit tag, a niche representation, or some other encoding while presenting
the same `SumType` interface.

### `SumType` as the basis for pattern matching

Syntactic enums and unions should not be privileged by the pattern-matching
implementation. The compiler implements case patterns in terms of `SumType`
and, for union payload patterns, `UnionLike`.

When matching a subject whose type conforms to `SumType`, for example:

```mojo
if case .success(value) = result:
    use(value)
```

or:

```mojo
match opt:
case Optional.Some(elt):
    use(elt)
case Optional.None():
    handle_missing()
```

the compiler performs the steps in [How pattern matching
works](#how-pattern-matching-works): resolve the case name through
`_sum_case_names`, compare `_get_sum_discriminant()`, and, for a union payload
pattern, project with `_unsafe_get_union_payload[id]()` before recursively
matching payload patterns.

This means the compiler's pattern-matching machinery operates on **sum types**,
rather than special-casing declarations spelled with `enum` or `union`.

### Reflection and derived conformances

`SumType` provides a useful foundation for generic reflection. For example, an
`Equatable` default implementation can first compare the active cases:

```mojo
if lhs._get_sum_discriminant() != rhs._get_sum_discriminant():
    return False
```

When both values conform to `UnionLike` and their discriminants match,
`_union_case_types` identifies the payload type, allowing the implementation to
compare associated state when that state is itself `Equatable`.

Similarly, `_sum_case_names` enables generic formatting, debugging,
serialization, or inspection:

```mojo
comptime name = T._sum_case_names[index]
```

Other derived behaviors such as hashing or serialization can use the same
metadata.

This gives the language a useful separation of concerns:

- `enum` declares a closed set of payload-free values and conforms to
  `EnumLike`.
- `union` declares a closed set of payload-bearing alternatives and conforms
  to `UnionLike`.
- Both refine `SumType`, which defines the common semantic and reflective
  interface for finite alternatives.
- Pattern matching is implemented against `SumType` and `UnionLike`, rather
  than giving syntactic enums or unions special treatment.
- User-defined representations may participate in the same system by
  conforming to the appropriate trait or traits.

This is consistent with the broader implementation philosophy of the proposal:
enums and unions should introduce useful high-level syntax and synthesized
machinery without creating a parallel type system underneath Mojo.

## Other implementation details

At the language implementation level, enums and unions should be treated as
**syntactic sugar for structs with compiler-synthesized state and members**.
They should become `StructDeclOp` operations in the compiler; they do not
require distinct declaration operations. Swift's use of distinct declaration
machinery caused avoidable implementation complexity.

For example:

```mojo
union Optional[T: Movable](Movable):
    case none()
    case some(T)

    def has_value(self) -> Bool:
        if case .some(_) = self:
            return True
        return False
```

can conceptually be viewed as defining a nominal struct-like type containing
compiler-managed representation state together with synthesized operations
corresponding to its cases:

```mojo
struct Optional[T]:
    var _state: `!kgen.variant<...>`

    <synthesized case constructors>
    <synthesized destruction/move/copy behavior>
    <synthesized case inspection/decomposition support>

    def has_value(self) -> Bool: ...
```

The important point is the **semantic lowering model**: an enum or union does
not require an entirely separate kind of nominal type throughout the compiler.

This implementation strategy has several advantages:

- Existing machinery for nominal types, generics, trait conformance, methods,
  visibility, and member lookup can be reused unchanged.
- Ownership semantics can be expressed through synthesized operations rather
  than building a parallel ownership system for unions.
- Improvements to struct capabilities naturally apply to enums and unions
  where appropriate.
- Algebraic-data-type-specific compiler machinery can remain focused on case
  representation, case transitions, and pattern decomposition.
- The surface language maintains a strong distinction among structs, enums,
  and unions even though their implementations share substantial
  infrastructure.
- This is easy to implement!

In this sense, `enum` and `union` are analogous to other high-level language
features that synthesize otherwise tedious or unsafe implementation details.
The syntax provides concise and semantically meaningful ways to declare
algebraic data types while allowing the compiler to lower them onto Mojo's
existing nominal-type infrastructure.

### Future consideration: control over layout and discriminators

The initial proposal leaves enum representation unspecified. A subsequent
extension should support the useful C and C++ capabilities of explicit backing
integer types, explicit case values, and efficient integer conversions. For
example:

```C++
enum MyColors : int32_t {
    red   = 0xFF0000,
    green = 0x00FF00,
    blue  = 0x0000FF
}
```

These capabilities can be layered on without changing the rule that enum cases
never carry payloads. Mojo does not currently provide fine-grained control of
struct layout; that broader problem should be addressed before exposing layout
control for enums.

Note that C also supports "enums as random collections of values", such as:

```C++
enum {
  Value1 = 42,
  Value2 = 17,
  Value3 = 42
};
```

This is a fundamentally different construct than an algebraic data type. Mojo
supports such concepts with existing `comptime` values, enums do not need to
provide support for this use-case.

### Future consideration: recursive and indirect union cases

The initial proposal doesn't provide support for
[Swift-style indirect cases](https://www.hackingwithswift.com/example-code/language/what-are-indirect-enums).

We can evaluate adding such a feature in the future, but existing language
features are preferable when they can express the same structure:

```mojo
union LinkedListItem[T: Copyable](Copyable):
    case end_point(T)
    case link_node(T, HeapBox[LinkedListItem])
```

### Future consideration: caseless enums and `Never`

An enum with no cases has no possible value:

```mojo
enum Never:
    pass
```

This is the natural language-level representation of an impossible value. It
could replace the current compiler-specific `kgen` handling of `Never`, letting
the language and standard library model non-returning functions, unreachable
control-flow paths, and impossible pattern branches with an ordinary caseless
enum. A caseless enum cannot synthesize `Defaultable` because no default value
exists.

### Unlikely future consideration: anonymous sum types

Note: This is specifically NOT part of the initial proposal. This is included to
address a frequently discussed possible next step that can only be considered
after the basic proposal lands and settles.

One possible extension is to allow **anonymous sum types** directly in type
position, rather than requiring every sum type to have a nominal enum or union
declaration.

For example, we could imagine writing:

```mojo
def activate_light(state: .on | .off):
    ...
```

rather than declaring a named type:

```mojo
enum LightState:
    case on, off

def activate_light(state: LightState):
    ...
```

This is appealing for small, locally useful sets of alternatives. It is
analogous to anonymous structural types in other parts of the type system and
avoids introducing a declaration whose only purpose is to name two or three
cases.

One possible implementation model would be to define a generic library
type—conceptually something like `Enum[...]`—and desugar the anonymous syntax
into an instantiation of it: `.on | .off`.  This could conceptually become
something like: `Enum["on", "off"]`.

However, this becomes substantially more complicated once the cases are
expected to behave like normal enum or union cases.

For example:

```mojo
def activate_light(state: .on | .off):
    activate_light(.on)
```

requires `.on` to participate in contextual member lookup even though there is
no nominal declaration containing an `on` member. This can be supported with a
suitable implementation of `Enum.__getattr__`, but would be complicated.

Payload-bearing cases make this more challenging:

```mojo
def handle(value: .none | .some(Int)):
    ...

handle(.some(42))
```

The compiler must somehow discover that `.some` is a constructor available on
this particular structural sum type, understand its parameter list, produce
useful diagnostics for misspelled cases, and make the same information available
to pattern matching and reflection.

Anonymous sum types also raise broader questions that nominal enums and unions
answer naturally:

- What is the concrete syntax, how do we handle the ambiguities?
- What determines whether two anonymous sum types are the same type?
- Does case ordering matter for type identity?
- Are `.on | .off` and `.off | .on` equivalent?
- How are case names represented in generic signatures and mangling?
- How do documentation and diagnostics refer to the type?
- Where do methods or trait conformances live?
- How does contextual lookup find case constructors?
- How do anonymous cases participate in reflection and `SumType` metadata?
- How do source and ABI stability work when an alternative is added or
  reordered?

None of these questions make anonymous sum types inherently undesirable. In
fact, they could eventually be a useful lightweight facility, particularly for
APIs whose alternatives have no meaningful identity outside of a single
signature.

However, they are orthogonal to the fundamental enum and union features and
require considerably more type-system and member-lookup machinery than nominal
types. If there is enough well-motivated demand, we can investigate adding this
in the future.
