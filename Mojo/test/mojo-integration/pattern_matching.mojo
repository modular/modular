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

"""End-to-end pattern matching: literals, tuples, structs, Optional, EnumLike."""

from std.collections import Optional
from std.utils import Variant
from std.os import abort


@fieldwise_init
struct _Eof(TrivialRegisterPassable):
    pass


# FIXME: Adopt "enum" syntax when it is implemented in the future.
struct Token(Copyable, Deinitable, EnumLike, Movable):
    """A tiny lexer token with EnumLike cases for pattern matching."""

    # Should all be synthesized by the compiler.
    comptime _enum_case_length = 3
    comptime _enum_case_names = ParameterList.of[
        "eof".value, "identifier".value, "integer".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, String, Int
    ].values
    var _value: Variant[_Eof, String, Int]

    def __init__(out self, *, eof: NoneType):
        self._value = Variant[_Eof, String, Int](_Eof())

    def __init__(out self, *, var identifier: String):
        self._value = Variant[_Eof, String, Int](identifier^)

    def __init__(out self, *, integer: Int):
        self._value = Variant[_Eof, String, Int](integer)

    @staticmethod
    def eof() -> Self:
        return Self(eof=None)

    @staticmethod
    def identifier(var name: String) -> Self:
        return Self(identifier=name^)

    @staticmethod
    def integer(value: Int) -> Self:
        return Self(integer=value)

    def _get_enum_discriminant(self) -> Int:
        if self._value.isa[_Eof]():
            return 0
        if self._value.isa[String]():
            return 1
        return 2

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        comptime assert id != 0, "eof has no payload"
        comptime if id == 1:
            return rebind[TypeList[Trait=AnyType, Self._enum_case_types]()[id]](
                Pointer(to=self._value.unsafe_get[String]()).unsafe_origin_cast[
                    origin_of(self)
                ]()[]
            )
        else:
            return rebind[TypeList[Trait=AnyType, Self._enum_case_types]()[id]](
                Pointer(to=self._value.unsafe_get[Int]()).unsafe_origin_cast[
                    origin_of(self)
                ]()[]
            )


@fieldwise_init
struct Point(Copyable, ImplicitlyCopyable, Movable):
    var x: Int
    var y: Int


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def describe_optional(opt: Optional[Int]) -> String:
    __match opt:
    case .Some(value):
        return String("some:", value)
    case .None:
        return "none"
    case _:
        return "unreachable"


def describe_token(tok: Token) -> String:
    __match tok:
    case .eof:
        return "eof"
    case .identifier(name):
        return String("id:", name)
    case .integer(value):
        return String("int:", value)
    case _:
        return "unreachable"


def classify_int(x: Int) -> String:
    __match x:
    case 0:
        return "zero"
    case 1 | 2:
        return "small"
    case n if n < 0:
        return "negative"
    case n:
        return String("other:", n)


def classify_point(p: Point) -> String:
    __match p:
    case Point(x=0, y=0):
        return "origin"
    case Point(x=var x, y=0):
        return String("x-axis:", x)
    case Point(x=0, y=var y):
        return String("y-axis:", y)
    case Point(x=x, y=y):
        return String("point:", x, ",", y)


def classify_pair(pair: Tuple[Int, Int]) -> String:
    __match pair:
    case (0, 0):
        return "origin"
    case (var x, 0):
        return String("x-axis:", x)
    case (0, var y):
        return String("y-axis:", y)
    case (var x, var y):
        return String("pair:", x, ",", y)


def classify_or_bind_var(pair: Tuple[Int, Int]) -> String:
    __match pair:
    case (0, var x) | (var x, 1):
        return String("var:", x)
    case _:
        return "miss"


def classify_or_bind_ref(var pair: Tuple[Int, Int]) -> String:
    __match pair:
    case (2, ref x) | (ref x, 3):
        return String("ref:", x)
    case _:
        return "miss"


# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_optional():
    # CHECK-LABEL: == test_optional
    print("== test_optional")

    # CHECK: some:42
    print(describe_optional(Optional(42)))
    # CHECK: none
    print(describe_optional(Optional[Int](None)))

    var mut_opt = Optional(7)
    __match mut_opt:
    case .Some(ref value):
        # CHECK: mut-some: 7
        print("mut-some:", value)
    case .None:
        print("mut-none")

    __match Optional[Int](None):
    case .Some:
        print("tag-some")
    case .None:
        # CHECK: tag-none
        print("tag-none")


def test_enum_like_token():
    # CHECK-LABEL: == test_enum_like_token
    print("== test_enum_like_token")

    # CHECK: eof
    print(describe_token(Token.eof()))
    # CHECK: id:main
    print(describe_token(Token.identifier("main")))
    # CHECK: int:99
    print(describe_token(Token.integer(99)))

    __match Token.identifier("x"):
    case Token.eof:
        print("explicit-eof")
    case .identifier(var name):
        # CHECK: explicit-id: x
        print("explicit-id:", name)
    case Token.integer(_):
        print("explicit-int")


def test_literals_guards_or():
    # CHECK-LABEL: == test_literals_guards_or
    print("== test_literals_guards_or")

    # CHECK: zero
    print(classify_int(0))
    # CHECK: small
    print(classify_int(1))
    # CHECK: small
    print(classify_int(2))
    # CHECK: negative
    print(classify_int(-3))
    # CHECK: other:10
    print(classify_int(10))


def test_tuples_and_structs():
    # CHECK-LABEL: == test_tuples_and_structs
    print("== test_tuples_and_structs")

    # CHECK: origin
    print(classify_pair((0, 0)))
    # CHECK: x-axis:4
    print(classify_pair((4, 0)))
    # CHECK: y-axis:5
    print(classify_pair((0, 5)))
    # CHECK: pair:1,2
    print(classify_pair((1, 2)))

    # CHECK: origin
    print(classify_point(Point(0, 0)))
    # CHECK: x-axis:3
    print(classify_point(Point(3, 0)))
    # CHECK: y-axis:8
    print(classify_point(Point(0, 8)))
    # CHECK: point:9,1
    print(classify_point(Point(9, 1)))


def test_or_bindings():
    # CHECK-LABEL: == test_or_bindings
    print("== test_or_bindings")

    # Left alternative: (0, var x)
    # CHECK: var:5
    print(classify_or_bind_var((0, 5)))
    # Right alternative: (var x, 1)
    # CHECK: var:7
    print(classify_or_bind_var((7, 1)))
    # Both alternatives match; first wins with x from the left.
    # CHECK: var:1
    print(classify_or_bind_var((0, 1)))
    # CHECK: miss
    print(classify_or_bind_var((4, 4)))

    # Left alternative: (2, ref x)
    # CHECK: ref:9
    print(classify_or_bind_ref((2, 9)))
    # Right alternative: (ref x, 3)
    # CHECK: ref:8
    print(classify_or_bind_ref((8, 3)))
    # CHECK: miss
    print(classify_or_bind_ref((0, 0)))


def main():
    test_optional()
    test_enum_like_token()
    test_literals_guards_or()
    test_tuples_and_structs()
    test_or_bindings()
