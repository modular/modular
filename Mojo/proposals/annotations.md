# Struct Annotations

**Status**: Concept proposal.

Date: September 10, 2026

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

Annotations are Mojo compile-time values that are syntactically attached to Mojo
decls (structs, functions, methods etc.) and can be introspected through a
comptime reflection API. Ignore the syntax but see the meaning:

```jsx
# This defines a function (or any other comptime value)
def myfunc(Int x)->Int:
    return x

# This attaches that value to `target`
@__annotation(myfunc(1))
@__annotation(myfunc(2), myfunc(3))
struct Target:
  @__annotation(myfunc(4))
  var value: String

# This reads the annotation value via metaprogramming
...
comptime x: Tuple[Int, Int, Int] = reflect[Target].attributes()
comptime assert x == (myfunc(1), myfunc(2), myfunc(3))
comptime y: Tuple[Int] = reflect[Target].field_attributes[0]()
comptime assert y == (myfunc(4),)
```

## Examples

When one is writing a Mojo struct, they should be able to annotate the struct
itself, as well as any field. Annotations are **compile time values** associated
with a struct itself or any of its fields. This concept is easier to explain
with examples, so we will start with motivating code samples below.

<aside>
💡

For the purposes of this document I will use `@annotation` as the spelling for
this feature. I don’t like the name, and at some point we should have a bike
shed conversation about the final syntax to specify annotations.

</aside>

**CLI Argparse**

Let’s consider an API similar to the Rust clap library or the C++ reflection
example (from which these examples are adapted).

With annotations, it would simply be:

```python
struct Args:
    @annotation(clap.help("Name of the person to greet"))
    @annotation(clap.short, clap.long)
    var name: String

    @annotation(clap.help("Number of times to greet"))
    @annotation(clap.short, clap.long)
    var count: Int

def main() raises:
    var args = parse[Args](argv())
    for _ in range(args.count):
        print("Hello", args.name, "!")
```

And the clap implementation code would be this (see 2 bolded lines for the
metaprogramming queries):

```python
from std.collections import List
from std.utils import Variant
from std.sys import argv
from std.reflection import reflect

@fieldwise_init
struct Help(ImplicitlyCopyable, Movable):
    var text: StaticString

@fieldwise_init
struct ShortArg(ImplicitlyCopyable, Movable):
    var value: Optional[UInt8]

    def __call__(self, c: UInt8) -> ShortArg:
        return ShortArg(c)

    @staticmethod
    def derived() -> ShortArg:
        return ShortArg(None)

@fieldwise_init
struct LongArg(ImplicitlyCopyable, Movable):
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

    def apply_annotation[T: ImplicitlyCopyable, //, attr: T, field_name: StaticString](mut self):
        comptime if T == Help:
            self.help_text = comptime(rebind[Help](attr).text)
        elif T == ShortArg:
            comptime s = rebind[ShortArg](attr)
            if s.value is not None:
                self.short_flag = Optional[String](
                    _char_string(s.value.value())
                )
            else:
                self.short_flag = Optional[String](
                    _char_string(_first_codepoint(String(field_name)))
                )
        elif T == LongArg:
            if comptime (rebind[LongArg](attr)).enabled:
                self.long_name = field_name

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

def parse[Args: Movable](input: Span[StaticString, ImmStaticOrigin]) raises -> Args:
    var result = Args()

    comptime count = reflect[Args].field_count()
    comptime names = reflect[Args].field_names()
    comptime types = reflect[Args].field_types()

    comptime for idx in range(count):

        var opt = Option(long_name=names[idx])
        # This is the new part
        # we can get a tuple of the annotation values.
        comptime attrs: Tuple[...] = reflect[Args].field_attributes[idx]()
        comptime for i in range(attrs.length):
            opt.apply_annotation[attrs[i], names[idx]]()

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

```python
from serde import (
    Serialize,
    to_json,
    Attribute,
    rename_all_camel,
    rename_field,
    skip_if_none,
)
from std.collections import List

@annotation(serde.rename_all_camel)
@fieldwise_init
struct User:
    @field: serde.rename("identifier")   # explicit per-field rename
    var id: Int
    var first_name: String               # -> "firstName" via rename_all
    var last_name: String                # -> "lastName"
    @field: serde.skip_if_none           # omit the key when None
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

```python
from std.reflection import reflect
from std.collections import List
from std.utils import Variant

@fieldwise_init
struct RenameCase(Equatable, ImplicitlyCopyable, Movable):
    var mode: UInt8
    comptime camel = Self(0)
    comptime pascal = Self(1)
    comptime snake = Self(2)
    comptime kebab = Self(3)

@fieldwise_init
struct RenameField(ImplicitlyCopyable, Movable):
    var to: String

@fieldwise_init
struct SkipIfNone(ImplicitlyCopyable, Movable):
    pass

comptime rename_all_camel = RenameCase.camel
comptime rename_all_pascal = RenameCase.pascal
comptime rename_all_snake = RenameCase.snake
comptime rename_all_kebab = RenameCase.kebab
comptime skip_if_none = SkipIfNone()

def rename_field(to: String) -> RenameField:
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

def to_json[T: Serialize](value: T) raises -> String:
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    comptime count = reflect[T].field_count()

    # Resolve the struct-level `rename_all`, if any.
    comptime attrs = reflect[T].attributes()
    var type_case = Optional[RenameCase]()
    comptime for i in range(attrs.size):
        if type_of(attrs[i]) == RenameCase:
            type_case = rebind[RenameCase](attrs[i])

    var parts = List[String]()
    comptime for idx in range(count):
        comptime FieldType = types[idx]
        var field_name = String(comptime (names[idx]))

        var key = field_name
        if type_case is not None:
            key = _apply_case(type_case.value(), field_name)

        var skip_if_none_field = False
        comptime field_attrs = reflect[T].field_attributes[idx]()
        comptime for field_idx in range(field_attrs.size):
            comptime attr = field_attrs[field_idx]
            if type_of(attr) == RenameField:
                key = comptime(rebind[RenameField](attr).to)
            elif type_of(attr) == SkipIfNone:
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

## Design Decisions

### Are annotations tuples of values (recommended) or dictionaries?

One difference from a language such as Rust is that annotations are not
arbitrary dictionaries, one can think of the annotations as tuples of values (in
fact this might be the return type of `field_metadata`), or possibly a tuple of
pairs (”annotation_name”, value). This behavior aligns with how C++ handles
annotations and reflection. The main benefit of this approach is that we do not
have to design with the fear of namespace collisions, and we can scope an
annotated type by the module it originates from.

### What is the syntax for annotating a field or structure?

One bikeshed highlighted above is that we need a syntax for annotations. Here
are a couple of samples of syntax

```text
struct User { #[serde(rename="firstName")] fist_name: String } - Rust
struct User { [[=serde::rename("firstName")]] std::string first_name } - C++
type User struct { FirstName string `json:"firstName"`} - Golang
```

I think users probably prefer something shorter than `@annotation`. Should this
use the decorator syntax? Should it use something else? Maybe just `@(...)`?

### Should we support linear types?

On the one hand, supporting arbitrary metadata could be useful, but on the other
hand, everything that exists in an annotation is logically a `comptime`
expression. Will users actually understand this nuance? This matters for whether
the `__annotations_of(T)` returns a `Tuple[*Movable]` or a
`Tuple[*Movable & Deinitable]`.
