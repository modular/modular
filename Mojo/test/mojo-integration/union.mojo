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

# Checks:
# 1. Construct cases and use their constructors as function values.
# 2. Borrow, mutate, copy and move stored payloads.
# 3. Derive formatting, equality, hashing and conditional conformances.
# 4. Instantiate generic and trivial unions, and open and caseless enums.

from std.hash import Hasher
from std.iter import map
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)


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


__union Token(ImplicitlyCopyable, Writable):
    case eof()
    case identifier(String)
    case integer(Int)
    case span(Int, Int)
    case keyword(String)
    case tracked(Tracked)


def describe(token: Token) -> String:
    match token:
        case .eof():
            return "eof"
        case .identifier(name) | .keyword(name):
            return String("word:", name)
        case .integer(value):
            return String("int:", value)
        case .span(start, end):
            return String("span:", start, "..", end)
        case .tracked(t):
            return String("tracked:", t.id)


# CHECK-LABEL: == construction
def test_construction():
    print("== construction")
    # CHECK-NEXT: construction: eof 0
    print(
        "construction:",
        describe(Token.eof()),
        Token.eof()._get_enum_discriminant(),
    )
    # CHECK-NEXT: construction: word:main 1
    var ident: Token = .identifier("main")
    print("construction:", describe(ident), ident._get_enum_discriminant())
    # CHECK-NEXT: construction: int:42 2
    print(
        "construction:",
        describe(Token.integer(42)),
        Token.integer(42)._get_enum_discriminant(),
    )
    # CHECK-NEXT: construction: span:3..7 3
    print(
        "construction:",
        describe(Token.span(3, 7)),
        Token.span(3, 7)._get_enum_discriminant(),
    )
    # Two cases may share a payload type.
    # CHECK-NEXT: construction: word:def 4
    var kw: Token = .keyword("def")
    print("construction:", describe(kw), kw._get_enum_discriminant())


def take(token: Token) -> String:
    return describe(token)


# CHECK-LABEL: == contextual
def test_contextual():
    print("== contextual")
    # CHECK-NEXT: contextual: eof int:1 span:1..2
    print("contextual:", take(.eof()), take(.integer(1)), take(.span(1, 2)))


# CHECK-LABEL: == ref binding
def test_ref_binding():
    print("== ref binding")
    var token: Token = .identifier("count")
    match token:
        case .identifier(ref name):
            name += "er"
        case _:
            pass
    # CHECK-NEXT: ref_binding: word:counter
    print("ref_binding:", describe(token))
    var copied: Token = .integer(1)
    match copied:
        case .integer(var value):
            value += 1
            # CHECK-NEXT: local 2
            print("local", value)
        case _:
            pass
    # CHECK-NEXT: ref_binding: int:1
    print("ref_binding:", describe(copied))


# CHECK-LABEL: == ownership
def test_ownership():
    print("== ownership")
    # CHECK-NEXT: init 1
    # CHECK-NEXT: move 1
    var token: Token = .tracked(Tracked(1))
    # CHECK-NEXT: copy 11
    var copy = token.copy()
    # The borrowed payload keeps its owner alive through the last use.
    # CHECK-NEXT: destroy 11
    # CHECK-NEXT: ownership: tracked:11
    print("ownership:", describe(copy))
    # CHECK-NEXT: move 1
    var moved = token^
    # CHECK-NEXT: destroy 1
    # CHECK-NEXT: ownership: tracked:1
    print("ownership:", describe(moved))
    # CHECK-NEXT: init 2
    # CHECK-NEXT: move 2
    var reassigned: Token = .tracked(Tracked(2))
    # CHECK-NEXT: destroy 2
    # CHECK-NEXT: ownership: tracked:2
    print("ownership:", describe(reassigned))
    reassigned = .eof()
    # CHECK-NEXT: ownership: eof
    print("ownership:", describe(reassigned))


# CHECK-LABEL: == function value
def test_function_value():
    print("== function value")
    var tokens = List[Token]()
    for token in map([1, 2, 3], Token.integer):
        tokens.append(token.copy())
    # CHECK-NEXT: function_value: 3 int:3
    print("function_value:", len(tokens), describe(tokens[2]))
    var generic = List[Result[Int, String]]()
    for result in map([5, 6], Result[Int, String].success):
        generic.append(result.copy())
    # CHECK-NEXT: generic constructor: 6
    print("generic constructor:", generic[1].value_or(0))


__union Result[T: Copyable & Deinitable, E: Copyable & Deinitable](
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
    # The union is `ImplicitlyCopyable` when both payloads are.
    var ok_copy = implicit_copy(ok)
    # CHECK-NEXT: generic: 1 7
    print("generic:", ok_copy.value_or(7), err.value_or(7))
    var list: Result[List[Int], String] = .success([1, 2])
    # CHECK-NEXT: generic: 2
    print("generic:", len(list.copy().value_or([])))
    # CHECK-NEXT: generic: 3 8
    print(
        "generic:",
        Result[Int, String].success(3).value_or(7),
        implicit_copy(Result[Int, String].failure("bad")).value_or(8),
    )


__union Shape(ImplicitlyCopyable, RegisterPassable):
    case empty()
    case square(Int)
    case rect(Int, Int)

    @implicit
    def __init__(out self, side: Int):
        self = .square(side)

    def area(self) -> Int:
        match self:
            case .empty():
                return 0
            case .square(side):
                return side * side
            case .rect(width, height):
                return width * height


@fieldwise_init
struct Pixel(TrivialRegisterPassable):
    var x: Int
    var y: Int


__union Trivial[T: TrivialRegisterPassable](TrivialRegisterPassable):
    case none()
    case some(Self.T)
    case pixel(Pixel)
    case pair(Int, Float64)


def describe_trivial(value: Trivial[Int]) -> String:
    match value:
        case .none():
            return "none"
        case .some(x):
            return String("some:", x)
        case .pixel(p):
            return String("pixel:", p.x, ",", p.y)
        case .pair(n, f):
            return String("pair:", n, ",", f)


comptime Callback = def(Int) thin -> Int


def twice(x: Int) -> Int:
    return 2 * x


# A payload type that is not a struct is `TrivialRegisterPassable` as well.
__union Primitive(TrivialRegisterPassable):
    case index(__mlir_type.index)
    case call(Callback)


def describe_primitive(value: Primitive) -> String:
    match value:
        case .index(i):
            return String("index:", Int(mlir_value=i))
        case .call(f):
            return String("call:", f(21))


def require_trivial[T: TrivialRegisterPassable](value: T) -> T:
    return value


struct TrivialHolder[value: Trivial[Int]]:
    @staticmethod
    def describe() -> String:
        return describe_trivial(Self.value)


# CHECK-LABEL: == trivial register passable
def test_trivial_register_passable():
    print("== trivial register passable")
    var original: Trivial[Int] = .pixel(Pixel(1, 2))
    var copy = original
    match copy:
        case .pixel(ref p):
            p.x = 3
        case _:
            pass
    # CHECK-NEXT: trivial_register_passable: pixel:1,2 pixel:3,2
    print(
        "trivial_register_passable:",
        describe_trivial(original),
        describe_trivial(copy),
    )
    # CHECK-NEXT: trivial_register_passable: some:4 pair:5,0.5 none
    print(
        "trivial_register_passable:",
        describe_trivial(require_trivial(Trivial[Int].some(4))),
        describe_trivial(.pair(5, 0.5)),
        describe_trivial(.none()),
    )
    var call: Primitive = .call(twice)
    var call_copy = call
    # CHECK-NEXT: trivial_register_passable: index:5 call:42 call:42
    print(
        "trivial_register_passable:",
        describe_primitive(.index(Int(5).__mlir_index__())),
        describe_primitive(call_copy),
        describe_primitive(require_trivial(call)),
    )
    comptime constant = Trivial[Int].some(6)
    # CHECK-NEXT: trivial_register_passable: some:6 some:7
    print(
        "trivial_register_passable:",
        describe_trivial(constant),
        TrivialHolder[Trivial[Int].some(7)].describe(),
    )
    # CHECK-NEXT: trivial_register_passable: True True True
    print(
        "trivial_register_passable:",
        IsTriviallyCopyable[Trivial[Pixel]],
        IsTriviallyMovable[Trivial[Pixel]],
        IsTriviallyDeinitable[Trivial[Pixel]],
    )


# CHECK-LABEL: == register passable
def test_register_passable():
    print("== register passable")
    var shapes: List[Shape] = [Shape.empty(), Shape.rect(2, 3), 4]
    # CHECK-NEXT: register_passable: 0 6 16
    print(
        "register_passable:",
        shapes[0].area(),
        shapes[1].area(),
        shapes[2].area(),
    )
    # Copying, moving and destroying a union are each trivial when they are for
    # every payload: `Key` has a `String` payload.
    # CHECK-NEXT: register_passable: True True True
    print(
        "register_passable:",
        IsTriviallyCopyable[Shape],
        IsTriviallyMovable[Shape],
        IsTriviallyDeinitable[Shape],
    )
    # CHECK-NEXT: register_passable: False True False
    print(
        "register_passable:",
        IsTriviallyCopyable[Key],
        IsTriviallyMovable[Key],
        IsTriviallyDeinitable[Key],
    )


struct BitwiseTracked(ImplicitlyCopyable, Movable):
    """A payload that declares a trivial copy and move, so storage copies it
    bitwise and its constructors print only if storage calls them anyway."""

    comptime __copy_ctor_is_trivial = True
    comptime __move_ctor_is_trivial = True

    var id: Int

    def __init__(out self, id: Int):
        self.id = id

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        print("copy", self.id)

    def __init__(out self, *, deinit move: Self):
        self.id = move.id
        print("move", self.id)


__union Bitwise(ImplicitlyCopyable):
    case empty()
    case one(BitwiseTracked)
    case pair(BitwiseTracked, Int)


def describe(value: Bitwise) -> String:
    match value:
        case .empty():
            return "empty"
        case .one(t):
            return String("one:", t.id)
        case .pair(t, n):
            return String("pair:", t.id, ",", n)


# CHECK-LABEL: == trivial lifecycle
def test_trivial_lifecycle():
    print("== trivial lifecycle")
    var one: Bitwise = .one(BitwiseTracked(1))
    var pair: Bitwise = .pair(BitwiseTracked(2), 3)
    # CHECK: copying
    print("copying")
    var one_copy = one.copy()
    var pair_copy = pair.copy()
    # CHECK-NEXT: moving
    print("moving")
    var one_moved = one_copy^
    var pair_moved = pair_copy^
    # CHECK-NEXT: trivial_lifecycle: one:1 pair:2,3
    print("trivial_lifecycle:", describe(one_moved), describe(pair_moved))
    _ = one
    _ = pair


# CHECK-LABEL: == comptime match
def test_comptime_match():
    print("== comptime match")
    comptime shape: Shape = .rect(2, 5)
    comptime match shape:
        case .rect(width, height):
            # CHECK-NEXT: rect 2 5
            print("rect", width, height)
        case _:
            print("other")


__union Key(Equatable, Hashable, ImplicitlyCopyable, Writable):
    case named(String)
    case numbered(Int)
    case anonymous()

    def __eq__(self, other: Self) -> Bool:
        return String(self) == String(other)

    def __hash__[H: Hasher](self, mut hasher: H):
        String(self).__hash__(hasher)


__union Derived(Copyable, Equatable, Hashable, Writable):
    case none()
    case one(Int)
    case many(List[Int])
    case pair(Int, String)


# CHECK-LABEL: == derived equality and hashing
def test_derived_equality_and_hashing() raises:
    print("== derived equality and hashing")
    # The defaults compare and hash the active case and its payload.
    # Evaluate the empty constructor in a parameter context.
    # CHECK-NEXT: derived_equality_and_hashing: True True True True
    print(
        "derived_equality_and_hashing:",
        materialize[Derived.none()]() == materialize[Derived.none()](),
        Derived.one(1) == Derived.one(1),
        Derived.many([1, 2]) == Derived.many([1, 2]),
        Derived.pair(1, "a") == Derived.pair(1, "a"),
    )
    # CHECK-NEXT: derived_equality_and_hashing: False False False False
    print(
        "derived_equality_and_hashing:",
        materialize[Derived.none()]() == Derived.one(1),
        Derived.one(1) == Derived.one(2),
        Derived.many([1]) == Derived.many([1, 2]),
        Derived.pair(1, "a") == Derived.pair(1, "b"),
    )
    # CHECK-NEXT: derived_equality_and_hashing: True True True True
    print(
        "derived_equality_and_hashing:",
        hash(Derived.one(1)) == hash(Derived.one(1)),
        hash(Derived.many([1, 2])) == hash(Derived.many([1, 2])),
        hash(Derived.none()) == hash(Derived.none()),
        hash(Derived.pair(1, "a")) == hash(Derived.pair(1, "a")),
    )

    var counts = Dict[Derived, Int]()
    counts[Derived.one(1)] = 1
    counts[Derived.none()] = 2
    counts[Derived.one(1)] += 10
    # CHECK-NEXT: derived_equality_and_hashing: 2 11 2
    print(
        "derived_equality_and_hashing:",
        len(counts),
        counts[Derived.one(1)],
        counts[Derived.none()],
    )
    # CHECK-NEXT: derived_equality_and_hashing: Derived.none() Derived.many([1, 2]) Derived.pair(1, a)
    print(
        "derived_equality_and_hashing:",
        Derived.none(),
        Derived.many([1, 2]),
        Derived.pair(1, "a"),
    )


# CHECK-LABEL: == user-written equality and hashing
def test_user_equality_and_hashing() raises:
    print("== user-written equality and hashing")
    # CHECK-NEXT: user_equality_and_hashing: True False True
    print(
        "user_equality_and_hashing:",
        Key.named("a") == Key.named("a"),
        Key.named("a") == Key.numbered(1),
        Key.anonymous() != Key.numbered(1),
    )
    var counts = Dict[Key, Int]()
    counts[Key.named("a")] = 1
    counts[Key.numbered(2)] = 2
    counts[Key.anonymous()] = 3
    counts[Key.named("a")] += 10
    # CHECK-NEXT: user_equality_and_hashing: 3 11 2 3
    print(
        "user_equality_and_hashing:",
        len(counts),
        counts[Key.named("a")],
        counts[Key.numbered(2)],
        counts[Key.anonymous()],
    )


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
    # CHECK-NEXT: open_enum: low high
    print("open_enum:", describe_level(Level.low), describe_level(Level.high))


# A union without cases has no values, but its type works like any other.
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
    # CHECK-NEXT: caseless_enum: 0 []
    print("caseless_enum:", len(copies), values)
    var describe = describe_uninhabited
    _ = describe


__union Pair:
    case ints(Int, Int)
    case maybe(Optional[Int], Int)
    case tracked(Tracked, Tracked)


def classify(pair: Pair) -> String:
    match pair:
        case .ints(0, n) | .ints(n, 0):
            return String("axis:", n)
        case .ints(x, y) if x == y:
            return String("diagonal:", x)
        case .ints(x, _):
            return String("ints:", x)
        case .maybe(.Some(x), n):
            return String("some:", x, ",", n)
        case .maybe(_, n):
            return String("none:", n)
        case .tracked(a, b):
            return String("tracked:", a.id, ",", b.id)


# CHECK-LABEL: == element-wise
def test_elementwise():
    print("== element-wise")
    # A case pattern takes one subpattern per payload element.
    # CHECK-NEXT: elementwise: axis:3 axis:4 diagonal:2 ints:5
    print(
        "elementwise:",
        classify(Pair.ints(0, 3)),
        classify(Pair.ints(4, 0)),
        classify(Pair.ints(2, 2)),
        classify(Pair.ints(5, 6)),
    )
    # CHECK-NEXT: elementwise: some:7,8 none:9
    print(
        "elementwise:",
        classify(Pair.maybe(7, 8)),
        classify(Pair.maybe(None, 9)),
    )
    var pair = Pair.ints(1, 2)
    match pair:
        case .ints(ref a, ref b):
            a = 10
            b += 5
        case _:
            pass
    if case .ints(a, b) = pair:
        # CHECK-NEXT: if case 10 7
        print("if case", a, b)
    # Bindings without `var` borrow the elements.
    # CHECK-NEXT: init 1
    # CHECK-NEXT: init 2
    # CHECK-NEXT: move 1
    # CHECK-NEXT: move 2
    # CHECK-NEXT: destroy 1
    # CHECK-NEXT: destroy 2
    # CHECK-NEXT: elementwise: tracked:1,2
    print("elementwise:", classify(Pair.tracked(Tracked(1), Tracked(2))))
    # A qualified case pattern resolves and runs like its short form.
    match Token.identifier("x"):
        case Token.eof():
            print("explicit-eof")
        case .identifier(name):
            # CHECK-NEXT: explicit-id: x
            print("explicit-id:", name)
        case Token.integer(_):
            print("explicit-int")
        case _:
            pass


__union EmptyConstructor(Copyable, Writable):
    case empty()


def test_empty_constructor():
    var make = EmptyConstructor.empty
    var inferred: EmptyConstructor = .empty()
    # CHECK: empty constructor: EmptyConstructor.empty() EmptyConstructor.empty()
    print("empty constructor:", make(), inferred)


__union Action(not Movable):
    case call(Callback)

    def __deinit__(deinit self):
        pass


def test_callback():
    var action = Action.call(twice)
    match action:
        case .call(f):
            # CHECK: callback: 42
            print("callback:", f(21))


__union MoveResult[T: Movable & Deinitable](Movable):
    case value(Self.T)


def test_move_bound():
    var result = MoveResult[Error].value(Error("bad input"))
    match result:
        case .value(ref error):
            # CHECK: move bound: bad input
            print("move bound:", error)


__union Color(ImplicitlyCopyable):
    case rgb(Int, Int, Int)

    def is_grayscale(self) -> Bool:
        match self:
            case .rgb(var r, var g, var b):
                return r == g == b


def test_chain():
    # CHECK: chain: True False False False
    print(
        "chain:",
        Color.rgb(7, 7, 7).is_grayscale(),
        Color.rgb(7, 7, 8).is_grayscale(),
        Color.rgb(8, 7, 7).is_grayscale(),
        Color.rgb(7, 8, 7).is_grayscale(),
    )


def main() raises:
    test_construction()
    test_contextual()
    test_ref_binding()
    test_ownership()
    test_function_value()
    test_generic()
    test_trivial_register_passable()
    test_register_passable()
    test_trivial_lifecycle()
    test_comptime_match()
    test_derived_equality_and_hashing()
    test_user_equality_and_hashing()
    test_open_enum()
    test_caseless_enum()
    test_elementwise()
    test_empty_constructor()
    test_callback()
    test_move_bound()
    test_chain()
