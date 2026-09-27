# Struct annotations

**Status**: Experimental implementation landed. The `@__annotation` decorator
and the `reflect[T].annotations()` / `reflect[T].field_annotations[i]()`
readers are in the compiler and stdlib behind a double-underscore spelling,
and carry no stability guarantees. The final syntax is still open.

Date: September 10, 2026. Updated September 27, 2026 to match the landed
implementation.

## Motivation

Mojo is going to need to solve the issue of serialization and deserialization
sooner rather than later. Once we have `async`, users will expect to use Mojo in
web-servers and data intensive applications. In order to win this space, we need
to offer state of the art ergonomics such that users can match what they would
expect in Rust, C++, or Go (the dominant languages in this space). Importantly,
it should be easy to specify serialization transformations without needing the
boilerplate of a full trait implementation. The way this is solved in all three
languages is with **annotations**.

## TL/DR

Annotations are Mojo compile-time values that are syntactically attached to a
struct or to one of its fields and read back as a tuple through the `reflect`
API:

```mojo
@fieldwise_init
struct Tag(Deinitable, Movable):
    var name: StaticString

# Any comptime value works, including the result of a call.
def tag(name: StaticString) -> Tag:
    return Tag(name)

# This attaches values to `Target` and to its `value` field.
@__annotation(tag("a"))
@__annotation(tag("b"), 2)
struct Target:
    @__annotation(tag("c"))
    var value: String

def main():
    # This reads them back as ordinary tuples, in source order.
    var attrs = reflect[Target].annotations()      # Tuple[Tag, Tag, Int]
    print(len(attrs), attrs[0].name, attrs[2])     # 3 a 2
    comptime assert type_of(attrs).Ts[2] == Int

    var field_attrs = reflect[Target].field_annotations[0]()  # Tuple[Tag]
    print(field_attrs[0].name)                                # c
```

## What landed

The implementation is split between the parser, which stores the values, and
the reflection library, which reads them.

### The decorator

`@__annotation(value, ...)` takes one or more positional comptime values.
Keyword arguments, unpacked arguments (`*pack`, `**pack`), and types
(`@__annotation(Int)`) are rejected with a diagnostic. Repeated decorators
accumulate into one list, top to bottom and left to right within a decorator.

The decorator is only accepted on structs and struct fields. Functions,
methods, traits, trait members and `comptime` aliases reject it. Extending it
to those declarations is possible later but is not part of this design.

Values are written in the decorated struct's own scope and stored unevaluated.
A value can therefore name the struct's parameters (`Self.N`), a `comptime`
alias declared anywhere in the struct (including after the field), a module
level alias, or a value taken from a trait bound (`Self.T.tag`). Each
instantiation of a parametric struct reads back its own values:

```mojo
@__annotation(Self.N, Self.N + 1)
struct Param[N: Int]:
    @__annotation(Self.N * 2)
    var value: Int

def main():
    print(reflect[Param[3]].annotations()[1])          # 4
    print(reflect[Param[5]].field_annotations[0]()[0]) # 10
```

A literal is stored in its materialized form, so `@__annotation(9)` reads back
as an `Int` and `@__annotation("x")` as a `String`. A constructor call such as
`Tag("x")` reads back as a `Tag`.

### The readers

`reflect[T]` has two static methods:

- `annotations()` returns the struct's values as a `Tuple`.
- `field_annotations[field_index]()` returns the values on the field at
  `field_index` as a `Tuple`.

The tuple carries everything else. Its `Ts` parameter is the list of
annotation types, `len()` is the count, and indexing reads one value. A struct
or field with no annotations yields an empty tuple rather than an error. Each
value is materialized from its compile-time form, so the result is an ordinary
runtime tuple.

Every annotation type must conform to `Movable & Deinitable`. Storage erases
each type to `AnyType`, and the reader downcasts to that bound so the value can
be materialized into a `Tuple` element and destroyed with the tuple.

In generic code the element types depend on `T`, so a caller dispatches on
`type_of(...).Ts[i]` and names the concrete type with `rebind`:

```mojo
def first_int_annotation[T: AnyType]() -> Int:
    var attrs = reflect[T].annotations()
    comptime assert type_of(attrs).Ts[0] == Int
    return rebind[Int](attrs[0])
```

## Examples

When one is writing a Mojo struct, they should be able to annotate the struct
itself, as well as any field. Annotations are **compile time values** associated
with a struct itself or any of its fields. This concept is easier to explain
with examples, so we will start with motivating code samples below.

The examples use the current `@__annotation` spelling. The double underscore
marks it as experimental; see the design decisions for the open bikeshed on
the final syntax.

**CLI Argparse**

Let’s consider an API similar to the Rust clap library or the C++ reflection
example (from which these examples are adapted).

With annotations, it would simply be:

```mojo
struct Args:
    @__annotation(clap.help("Name of the person to greet"))
    @__annotation(clap.short, clap.long)
    var name: String

    @__annotation(clap.help("Number of times to greet"))
    @__annotation(clap.short, clap.long)
    var count: Int

def main() raises:
    var args = parse[Args](argv())
    for _ in range(args.count):
        print("Hello", args.name, "!")
```

And the clap implementation code would be this (the metaprogramming queries
are the `field_annotations` read and the `comptime for` over its `Ts`):

```mojo
from std.collections import List
from std.utils import Variant
from std.sys import argv
from std.reflection import reflect

@fieldwise_init
struct Help(Deinitable, ImplicitlyCopyable, Movable):
    var text: StaticString

@fieldwise_init
struct ShortArg(Deinitable, ImplicitlyCopyable, Movable):
    var value: Optional[UInt8]

    def __call__(self, c: UInt8) -> ShortArg:
        return ShortArg(c)

    @staticmethod
    def derived() -> ShortArg:
        return ShortArg(None)

@fieldwise_init
struct LongArg(Deinitable, ImplicitlyCopyable, Movable):
    var enabled: Bool

def help(text: StaticString) -> Help:
    return Help(text)

def short_for(c: UInt8) -> ShortArg:
    return ShortArg(Optional[UInt8](c))

comptime short = ShortArg.derived()
comptime long = LongArg(True)

def _char_string(u: UInt8) -> String:
    return String(Codepoint(u))

def _first_codepoint(name: String) -> UInt8:
    for cp in name.codepoints():
        return UInt8(Int(cp))
    return 0

struct Option(ImplicitlyCopyable, Movable):
    var short_flag: Optional[String]
    var long_name: String
    var help_text: String

    def __init__(out self):
        self.short_flag = Optional[String]()
        self.long_name = ""
        self.help_text = ""

    def __init__(out self, *, long_name: String):
        self.short_flag = Optional[String]()
        self.long_name = long_name
        self.help_text = ""

    # The annotation arrives as a runtime tuple element whose type is known
    # at compile time, so dispatch happens on `T` and the value is rebound.
    def apply_annotation[
        T: Movable & Deinitable, //, field_name: StaticString
    ](mut self, attr: T):
        comptime if T == Help:
            self.help_text = String(rebind[Help](attr).text)
        elif T == ShortArg:
            var s = rebind[ShortArg](attr)
            if s.value is not None:
                self.short_flag = Optional[String](
                    _char_string(s.value.value())
                )
            else:
                self.short_flag = Optional[String](
                    _char_string(_first_codepoint(String(field_name)))
                )
        elif T == LongArg:
            if rebind[LongArg](attr).enabled:
                self.long_name = String(field_name)

def _match_value(
    input: Span[StaticString, ImmStaticOrigin],
    opt: Option,
) raises -> Optional[String]:
    """Scan argv for this option's flag. Returns the value, or `None`.

    Accepts both `--name value` / `-n value` and `--name=value`.
    """
    var long_flag = "--" + opt.long_name
    var use_short = opt.short_flag is not None
    var short_flag = ""
    if use_short:
        short_flag = "-" + opt.short_flag.value()

    var i = 0
    var n = len(input)
    while i < n:
        var tok = String(input[i])

        if tok == long_flag and i + 1 < n:
            return Optional[String](String(input[i + 1]))

        if use_short and tok == short_flag and i + 1 < n:
            return Optional[String](String(input[i + 1]))

        var long_prefix = long_flag + "="
        if tok.startswith(long_prefix):
            return Optional[String](tok[len(long_prefix):])

        if use_short:
            var short_prefix = short_flag + "="
            if tok.startswith(short_prefix):
                return Optional[String](tok[len(short_prefix):])

        i += 1

    return Optional[String]()

def parse[
    Args: Movable
](input: Span[StaticString, ImmStaticOrigin]) raises -> Args:
    var result = Args()

    comptime count = reflect[Args].field_count()
    comptime names = reflect[Args].field_names()
    comptime types = reflect[Args].field_types()

    comptime for idx in range(count):

        var opt = Option(long_name=String(names[idx]))
        # This is the new part: the field's annotations come back as a tuple,
        # and its `Ts` parameter drives the dispatch.
        var attrs = reflect[Args].field_annotations[idx]()
        comptime for i in range(type_of(attrs).Ts.length):
            opt.apply_annotation[names[idx]](attrs[i])

        var maybe_value = _match_value(input, opt)
        comptime FieldType = types[idx]
        if maybe_value is not None:
            comptime if conforms_to(FieldType, ImplicitlyCopyable & Deinitable):
                ref dest = reflect[Args].field_ref[idx](result)
                var raw = maybe_value.value()
                comptime if FieldType == Int:
                    dest = rebind[FieldType](Int(raw))
                elif FieldType == String:
                    dest = rebind[FieldType](raw^)
                else:
                    raise Error(
                        "clap: unsupported field type for '" + names[idx] + "'"
                    )
            else:
                raise Error(t"clap: field '{names[idx]}' is not copyable")

    return result^

```

Without annotations, this code would be at minimum difficult to write
generically at compile time, if not impossible. One would have to directly map
the annotations associated with fields to a dictionary of attributes.

### Serde

For an example our users are also concerned with, consider serialization and
deserialization, right now one might write:

```mojo
import serde
from serde import to_json
from std.collections import List

@__annotation(serde.rename_all_camel)
@fieldwise_init
struct User:
    @__annotation(serde.rename_field("identifier"))  # explicit per-field rename
    var id: Int
    var first_name: String               # -> "firstName" via rename_all
    var last_name: String                # -> "lastName"
    @__annotation(serde.skip_if_none)    # omit the key when None
    var middle_name: Optional[String]
    var age: Int                         # -> "age"
    var active: Bool                     # -> "active"

def main() raises:
    var with_middle = User(
        id=1,
        first_name="Ada",
        last_name="Lovelace",
        middle_name=String("Augusta"),
        age=36,
        active=True,
    )
    print(to_json(with_middle))

    var no_middle = User(
        id=2,
        first_name="Grace",
        last_name="Hopper",
        middle_name=None,
        age=85,
        active=False,
    )
    print(to_json(no_middle))

```

where `serde.mojo` looks like:

```mojo
from std.reflection import reflect
from std.collections import List
from std.utils import Variant

@fieldwise_init
struct RenameCase(Deinitable, Equatable, ImplicitlyCopyable, Movable):
    var mode: UInt8
    comptime camel = Self(0)
    comptime pascal = Self(1)
    comptime snake = Self(2)
    comptime kebab = Self(3)

@fieldwise_init
struct RenameField(Deinitable, ImplicitlyCopyable, Movable):
    var to: StaticString

@fieldwise_init
struct SkipIfNone(Deinitable, ImplicitlyCopyable, Movable):
    pass

comptime rename_all_camel = RenameCase.camel
comptime rename_all_pascal = RenameCase.pascal
comptime rename_all_snake = RenameCase.snake
comptime rename_all_kebab = RenameCase.kebab
comptime skip_if_none = SkipIfNone()

def rename_field(to: StaticString) -> RenameField:
    return RenameField(to)

def _ascii_up(cp: Codepoint) -> Codepoint:
    var u = Int(cp)
    if u >= 97 and u <= 122:
        u = u - 32
    return Codepoint(UInt8(u))

def _ascii_lo(cp: Codepoint) -> Codepoint:
    var u = Int(cp)
    if u >= 65 and u <= 90:
        u = u + 32
    return Codepoint(UInt8(u))

def _apply_case(mode: RenameCase, name: String) -> String:
    if mode == .snake:
        return name
    if mode == .kebab:
        var out = String()
        for cp in name.codepoints():
            if cp == Codepoint.ord("_"):
                out.append(Codepoint.ord("-"))
            else:
                out.append(cp)
        return out^

    # `camel` (0) lowercases the first segment; `pascal` (1) capitalizes it.
    var out = String()
    var cap_next = mode == .pascal
    for cp in name.codepoints():
        if cp == Codepoint.ord("_"):
            cap_next = True
            continue
        if cap_next:
            out.append(_ascii_up(cp))
            cap_next = False
        else:
            out.append(_ascii_lo(cp))
    return out^

def ri(v: Int) -> String:
    return String(v)

def rb(v: Bool) -> String:
    return "true" if v else "false"

def rs(v: String) -> String:
    var out = String()
    out.append(Codepoint.ord('"'))
    for cp in v.codepoints():
        if cp == Codepoint.ord("\\"):
            out += "\\\\"
        elif cp == Codepoint.ord('"'):
            out += '\\"'
        elif cp == Codepoint.ord("\n"):
            out += "\\n"
        elif cp == Codepoint.ord("\t"):
            out += "\\t"
        elif cp == Codepoint.ord("\r"):
            out += "\\r"
        else:
            out.append(cp)
    out.append(Codepoint.ord('"'))
    return out^

def ro_s(v: Optional[String]) -> String:
    if v is None:
        return "null"
    return rs(v.value())

def ro_i(v: Optional[Int]) -> String:
    if v is None:
        return "null"
    return String(v.value())

def ro_b(v: Optional[Bool]) -> StaticString:
    if v is None:
        return "null"
    return "true" if v.value() else "false"

def render[T: AnyType](v: T) raises -> String:
    comptime if T == Int:
        return ri(rebind[Int](v))
    elif T == Bool:
        return rb(rebind[Bool](v))
    elif T == String:
        return rs(rebind[String](v))
    elif T == Optional[String]:
        return ro_s(rebind[Optional[String]](v))
    elif T == Optional[Int]:
        return ro_i(rebind[Optional[Int]](v))
    elif T == Optional[Bool]:
        return ro_b(rebind[Optional[Bool]](v))
    else:
        return '"?<unsupported>?"'

def to_json[T: AnyType](value: T) raises -> String:
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    comptime count = reflect[T].field_count()

    # Resolve the struct-level `rename_all`, if any. The tuple's `Ts` lists
    # the annotation types, so the match is a compile-time type comparison.
    var attrs = reflect[T].annotations()
    comptime AttrTypes = type_of(attrs).Ts
    var type_case = Optional[RenameCase]()
    comptime for i in range(AttrTypes.length):
        comptime if AttrTypes[i] == RenameCase:
            type_case = Optional[RenameCase](rebind[RenameCase](attrs[i]))

    var parts = List[String]()
    comptime for idx in range(count):
        comptime FieldType = types[idx]
        var field_name = String(names[idx])

        var key = field_name
        if type_case is not None:
            key = _apply_case(type_case.value(), field_name)

        var skip_if_none_field = False
        var field_attrs = reflect[T].field_annotations[idx]()
        comptime FieldAttrTypes = type_of(field_attrs).Ts
        comptime for j in range(FieldAttrTypes.length):
            comptime if FieldAttrTypes[j] == RenameField:
                key = String(rebind[RenameField](field_attrs[j]).to)
            elif FieldAttrTypes[j] == SkipIfNone:
                skip_if_none_field = True

        ref fld = reflect[T].field_ref[idx](value)

        var omit = False
        comptime if conforms_to(FieldType, Boolable) and (
            reflect[FieldType].base_name() == "Optional"
        ):
            if skip_if_none_field and not fld:
                omit = True

        if not omit:
            parts.append('"' + key + '": ' + render(fld))

    return "{" + ", ".join(parts) + "}"

```

## Design decisions

### Annotations are tuples of values, not dictionaries

This is decided and is what shipped. An annotation list is a positional tuple
of values: `annotations()` and `field_annotations[i]()` return `Tuple[*Ts]`,
where `Ts` is the list of annotation types in source order. The parser rejects
keyword arguments, so there is no name to key on; the type of each value is the
discriminator, as the clap and serde examples show. This behavior aligns with
how C++ handles annotations and reflection. The main benefit of this approach
is that we do not have to design with the fear of namespace collisions, and an
annotation is scoped by the module that defines its type.

A consequence worth naming: two annotations of the same type on one field are
both kept, and the reader makes no attempt to deduplicate or to pick one. A
library that wants "the `Rename` on this field" walks the tuple and decides
for itself what a repeat means.

### What is the syntax for annotating a field or structure?

The current spelling is `@__annotation(...)`. The double underscore is
deliberate: it marks the decorator as experimental and keeps the eventual
user-facing name free. Here are a couple of samples of syntax from other
languages:

```text
struct User { #[serde(rename="firstName")] fist_name: String } - Rust
struct User { [[=serde::rename("firstName")]] std::string first_name } - C++
type User struct { FirstName string `json:"firstName"`} - Golang
```

Users probably prefer something shorter than `@annotation`. Should this use
the decorator syntax? Should it use something else? Maybe just `@(...)`? This
remains open.

Placement is also narrower than the original sketch, which mentioned
functions and methods. Only structs and struct fields accept the decorator
today; the parser has nowhere to store a value on the other declarations.

### Should we support linear types?

On the one hand, supporting arbitrary metadata could be useful, but on the other
hand, everything that exists in an annotation is logically a `comptime`
expression. Will users actually understand this nuance? This matters for whether
`annotations()` returns a `Tuple[*Movable]` or a
`Tuple[*Movable & Deinitable]`. The readers that landed require
`Movable & Deinitable` today, but the question stays open.

### Open questions

- The final spelling of the decorator, per the bikeshed above.
- Whether to accept the decorator on functions, methods and parameters.
- Whether the reflection API should grow helpers over the tuple, such as
  "the first annotation of type `X`" or "does this field carry an `X`", or
  whether the `comptime for` over `Ts` in the examples is enough.
- Whether a reader that returns comptime values, without materializing them,
  is wanted alongside the tuple readers.
