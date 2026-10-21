# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

"""End-to-end `__enum`: construction, `match` over the synthesized
`EnumLike` conformance, ownership of payloads, and conformances."""

from std.hash import Hasher
from std.iter import map


struct Tracked(ImplicitlyCopyable, Movable, Writable):
    """A payload that reports how it is copied, moved and destroyed."""

    var id: Int

    def __init__(out self, id: Int):
        self.id = id
        print("init", id)

    def __init__(out self, *, copy: Self):
        self.id = copy.id + 10
        print("copy", self.id)

    def __init__(out self, *, deinit move: Self):
        self.id = move.id
        print("move", self.id)

    def __deinit__(deinit self):
        print("destroy", self.id)


# TODO: `ImplicitlyCopyable` only so that `Token.eof`, a `comptime` member, can
# be used as a runtime value: a `comptime` value of a type that is not
# `ImplicitlyCopyable` does not materialize, although a case without a payload
# owns nothing. Exempt it, or give cases their own types.
__enum Token(ImplicitlyCopyable, Writable):
    case eof
    case identifier(String)
    case integer(Int)
    case span(Int, Int)
    case keyword(String)
    case tracked(Tracked)


def describe(token: Token) -> String:
    match token:
        case .eof:
            return "eof"
        case .identifier(name) | .keyword(name):
            return String("word:", name)
        case .integer(value):
            return String("int:", value)
        case .span((start, end)):
            return String("span:", start, "..", end)
        case .tracked(t):
            return String("tracked:", t.id)


# CHECK-LABEL: == construction
def test_construction():
    print("== construction")
    # CHECK-NEXT: eof 0
    print(describe(Token.eof), Token.eof._get_enum_discriminant())
    # CHECK-NEXT: word:main 1
    var ident: Token = .identifier("main")
    print(describe(ident), ident._get_enum_discriminant())
    # CHECK-NEXT: int:42 2
    print(
        describe(Token.integer(42)), Token.integer(42)._get_enum_discriminant()
    )
    # CHECK-NEXT: span:3..7 3
    print(describe(Token.span(3, 7)), Token.span(3, 7)._get_enum_discriminant())
    # Two cases may share a payload type.
    # CHECK-NEXT: word:def 4
    var kw: Token = .keyword("def")
    print(describe(kw), kw._get_enum_discriminant())


def take(token: Token) -> String:
    return describe(token)


# CHECK-LABEL: == contextual
def test_contextual():
    print("== contextual")
    # CHECK-NEXT: eof int:1 span:1..2
    print(take(.eof), take(.integer(1)), take(.span(1, 2)))


# CHECK-LABEL: == ref binding
def test_ref_binding():
    print("== ref binding")
    var token: Token = .identifier("count")
    match token:
        case .identifier(ref name):
            name += "er"
        case _:
            pass
    # CHECK-NEXT: word:counter
    print(describe(token))
    var copied: Token = .integer(1)
    match copied:
        case .integer(var value):
            value += 1
            # CHECK-NEXT: local 2
            print("local", value)
        case _:
            pass
    # CHECK-NEXT: int:1
    print(describe(copied))


# CHECK-LABEL: == ownership
def test_ownership():
    print("== ownership")
    # CHECK-NEXT: init 1
    # CHECK-NEXT: move 1
    var token: Token = .tracked(Tracked(1))
    # CHECK-NEXT: copy 11
    var copy = token.copy()
    # A binding without `var` borrows the payload; `copy` ends after its
    # last use.
    # CHECK-NEXT: destroy 11
    # CHECK-NEXT: tracked:11
    print(describe(copy))
    # CHECK-NEXT: move 1
    var moved = token^
    # CHECK-NEXT: destroy 1
    # CHECK-NEXT: tracked:1
    print(describe(moved))
    # CHECK-NEXT: init 2
    # CHECK-NEXT: move 2
    var reassigned: Token = .tracked(Tracked(2))
    # CHECK-NEXT: destroy 2
    # CHECK-NEXT: tracked:2
    print(describe(reassigned))
    reassigned = .eof
    # CHECK-NEXT: eof
    print(describe(reassigned))


# CHECK-LABEL: == writable
def test_writable():
    print("== writable")
    # CHECK-NEXT: Token.eof Token.integer(5) Token.span((1, 2))
    print(String(Token.eof), String(Token.integer(5)), String(Token.span(1, 2)))
    # CHECK-NEXT: Token.identifier('x')
    print(repr(Token.identifier("x")))


# CHECK-LABEL: == function value
def test_function_value():
    print("== function value")
    var tokens = List[Token]()
    for token in map[Token.integer]([1, 2, 3]):
        tokens.append(token.copy())
    # CHECK-NEXT: 3 int:3
    print(len(tokens), describe(tokens[2]))


__enum Result[T: Copyable & Deinitable, E: Copyable & Deinitable](
    Copyable,
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable) and conforms_to(
        E, ImplicitlyCopyable
    ),
):
    case success(Self.T)
    case failure(Self.E)

    def value_or(self, default: Self.T) -> Self.T:
        match self:
            case .success(value):
                return value.copy()
            case .failure(_):
                return default.copy()


def implicit_copy[T: ImplicitlyCopyable](value: T) -> T:
    return value


# CHECK-LABEL: == generic
def test_generic():
    print("== generic")
    var ok: Result[Int, String] = .success(1)
    var err: Result[Int, String] = .failure("bad")
    # The enum is `ImplicitlyCopyable` when both payloads are.
    var ok_copy = implicit_copy(ok)
    # CHECK-NEXT: 1 7
    print(ok_copy.value_or(7), err.value_or(7))
    var list: Result[List[Int], String] = .success([1, 2])
    # CHECK-NEXT: 2
    print(len(list.copy().value_or([])))
    # CHECK-NEXT: 3 8
    print(
        Result[Int, String].success(3).value_or(7),
        implicit_copy(Result[Int, String].failure("bad")).value_or(8),
    )


__enum Shape(ImplicitlyCopyable, RegisterPassable):
    case empty
    case square(Int)
    case rect(Int, Int)

    @implicit
    def __init__(out self, side: Int):
        self = .square(side)

    def area(self) -> Int:
        match self:
            case .empty:
                return 0
            case .square(side):
                return side * side
            case .rect((width, height)):
                return width * height


# CHECK-LABEL: == register passable
def test_register_passable():
    print("== register passable")
    var shapes: List[Shape] = [Shape.empty, Shape.rect(2, 3), 4]
    # CHECK-NEXT: 0 6 16
    print(shapes[0].area(), shapes[1].area(), shapes[2].area())


# CHECK-LABEL: == comptime match
def test_comptime_match():
    print("== comptime match")
    comptime shape: Shape = .rect(2, 5)
    comptime match shape:
        case .rect((width, height)):
            # CHECK-NEXT: rect 2 5
            print("rect", width, height)
        case _:
            print("other")


__enum Key(Equatable, Hashable, ImplicitlyCopyable, Writable):
    case named(String)
    case numbered(Int)
    case anonymous

    def __eq__(self, other: Self) -> Bool:
        return String(self) == String(other)

    def __hash__[H: Hasher](self, mut hasher: H):
        String(self).__hash__(hasher)


# CHECK-LABEL: == user-written equality and hashing
def test_user_equality_and_hashing() raises:
    print("== user-written equality and hashing")
    # CHECK-NEXT: True False True
    print(
        Key.named("a") == Key.named("a"),
        Key.named("a") == Key.numbered(1),
        Key.anonymous != Key.numbered(1),
    )
    var counts = Dict[Key, Int]()
    counts[Key.named("a")] = 1
    counts[Key.numbered(2)] = 2
    counts[Key.anonymous] = 3
    counts[Key.named("a")] += 10
    # CHECK-NEXT: 3 11 2 3
    print(
        len(counts),
        counts[Key.named("a")],
        counts[Key.numbered(2)],
        counts[Key.anonymous],
    )


# TODO: `ImplicitlyCopyable` for the same reason as `Token`.
__enum Level(ImplicitlyCopyable):
    case low
    case high

    comptime _enum_is_exhaustive = False


def describe_level(level: Level) -> String:
    match level:
        case .low:
            return "low"
        case .high:
            return "high"
        case _:
            return "other"


# CHECK-LABEL: == open enum
def test_open_enum():
    print("== open enum")
    # CHECK-NEXT: low high
    print(describe_level(Level.low), describe_level(Level.high))


# An enum without cases has no values, but its type works like any other.
__enum Uninhabited(Copyable, Writable):
    pass


def describe_uninhabited(uninhabited: Uninhabited) -> String:
    match uninhabited:
        case _:
            return "unreachable"


# CHECK-LABEL: == caseless enum
def test_caseless_enum():
    print("== caseless enum")
    var values = List[Uninhabited]()
    var copies = values.copy()
    # CHECK-NEXT: 0 []
    print(len(copies), values)
    var describe = describe_uninhabited
    _ = describe


def main() raises:
    test_construction()
    test_contextual()
    test_ref_binding()
    test_ownership()
    test_writable()
    test_function_value()
    test_generic()
    test_register_passable()
    test_comptime_match()
    test_user_equality_and_hashing()
    test_open_enum()
    test_caseless_enum()
