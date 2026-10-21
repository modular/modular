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

"""End-to-end pattern matching: literals, tuples, structs, Optional, enums,
and comptime match."""

from std.collections import List, Optional
from std.os import abort


__enum Token(ImplicitlyCopyable):
    """A tiny lexer token for pattern matching."""

    case eof
    case identifier(String)
    case integer(Int)


@fieldwise_init
struct Point(ImplicitlyCopyable):
    var x: Int
    var y: Int


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def describe_optional(opt: Optional[Int]) -> String:
    match opt:
        case .Some(value):
            return String("some:", value)
        case .None:
            return "none"


def describe_token(tok: Token) -> String:
    match tok:
        case .eof:
            return "eof"
        case .identifier(name):
            return String("id:", name)
        case .integer(value):
            return String("int:", value)


def classify_int(x: Int) -> String:
    match x:
        case 0:
            return "zero"
        case 1 | 2:
            return "small"
        case n if n < 0:
            return "negative"
        case n:
            return String("other:", n)


def classify_point(p: Point) -> String:
    match p:
        case Point(x=0, y=0):
            return "origin"
        case Point(x=var x, y=0):
            return String("x-axis:", x)
        case Point(x=0, y=var y):
            return String("y-axis:", y)
        case Point(x=x, y=y):
            return String("point:", x, ",", y)


def classify_pair(pair: Tuple[Int, Int]) -> String:
    match pair:
        case (0, 0):
            return "origin"
        case (var x, 0):
            return String("x-axis:", x)
        case (0, var y):
            return String("y-axis:", y)
        case (var x, var y):
            return String("pair:", x, ",", y)


def classify_list(values: List[Int]) -> String:
    match values:
        case []:
            return "empty"
        case [0]:
            return "zero"
        case [first, *middle, last]:
            return String("middle:", first, ":", len(middle), ":", last)
        case [first, *rest]:
            return String("rest:", first, ":", len(rest))
        case _:
            abort("unreachable")


def classify_expected(value: Int, expected: Int) -> String:
    match value:
        case == expected:
            return "expected"
        case _:
            return "other"


def classify_expected_pair(pair: Tuple[Int, Int], expected: Int) -> String:
    match pair:
        case (_, == expected):
            return "expected-second"
        case _:
            return "other"


def classify_computed_expected(value: Int, expected: Int) -> String:
    match value:
        case == (expected + 1):
            return "computed"
        case _:
            return "other"


def classify_conditional_expected(
    value: Int, expected: Int, alternate: Int, use_expected: Bool
) -> String:
    match value:
        case == (expected if use_expected else alternate):
            return "conditional"
        case _:
            return "other"


def classify_type[T: AnyType]() -> String:
    comptime match T:
        case == Int:
            return "integer"
        case == String:
            return "string"
        case _:
            return "other"


def classify_if_case_type[T: AnyType]() -> String:
    comptime if case == Int = T:
        return "integer"
    elif case == String = T:
        return "string"
    else:
        return "other"


def classify_comptime_if_case_pair[pair: Tuple[Int, Int]]() -> String:
    comptime if case (0, value) = pair:
        return String("first:", value)
    elif case (value, 1) = pair:
        return String("second:", value)
    else:
        return "miss"


def classify_or_bind_var(pair: Tuple[Int, Int]) -> String:
    match pair:
        case (0, var x) | (var x, 1):
            return String("var:", x)
        case _:
            return "miss"


def classify_or_bind_ref(var pair: Tuple[Int, Int]) -> String:
    match pair:
        case (2, ref x) | (ref x, 3):
            return String("ref:", x)
        case _:
            return "miss"


def classify_if_case(pair: Tuple[Int, Int], fallback: Bool) -> String:
    if case (0, value as same_value) = pair:
        _ = same_value
        return String("first:", value)
    elif fallback:
        return "fallback"
    elif case (value, 1) = pair:
        return String("second:", value)
    else:
        return "miss"


def classify_if_case_or(pair: Tuple[Int, Int]) -> String:
    if case ((0, value) | (value, 1)) as matched = pair:
        _ = matched
        return String("or:", value)
    else:
        return "miss"


def classify_if_case_nested(value: Optional[Point]) -> String:
    if case .Some(Point(x=0, y=y)) = value:
        return String("y-axis:", y)
    elif case .Some(Point(x=x, y=0)) = value:
        return String("x-axis:", x)
    elif case .None = value:
        return "none"
    else:
        return "other"


def classify_while_case(var pair: Tuple[Int, Int]) -> String:
    var result = String("miss")
    while case ((0, value) | (value, 1)) as matched = pair:
        _ = matched
        result = String("hit:", value)
        pair = (4, 4)
    else:
        result += ":done"
    return result^


def classify_while_case_break(pair: Tuple[Int, Int]) -> String:
    var result = String("miss")
    while case (0, value) = pair:
        result = String("break:", value)
        break
    else:
        result += ":done"
    return result^


def classify_comptime_int[x: Int]() -> String:
    comptime match x:
        case 0:
            return "zero"
        case 1 | 2:
            return "small"
        case n if n < 0:
            return "negative"
        case n:
            return String("other:", n)


def classify_comptime_bool[flag: Bool]() -> String:
    comptime match flag:
        case True:
            return "yes"
        case False:
            return "no"


def classify_comptime_or_bind[pair: Tuple[Int, Int]]() -> String:
    comptime match pair:
        case (0, y) | (y, 1):
            return String("hit:", y)
        case _:
            return "miss"


def classify_comptime_as[x: Int]() -> String:
    comptime match x:
        case 0 as n:
            return String("zero:", n)
        case _:
            return "other"


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
    match mut_opt:
        case .Some(ref value):
            # CHECK: mut-some: 7
            print("mut-some:", value)
        case .None:
            print("mut-none")

    match Optional[Int](None):
        case .Some:
            print("tag-some")
        case .None:
            # CHECK: tag-none
            print("tag-none")


def test_enum_like_token():
    # CHECK-LABEL: == test_enum_like_token
    print("== test_enum_like_token")

    # CHECK: eof
    print(describe_token(Token.eof))
    # CHECK: id:main
    print(describe_token(Token.identifier("main")))
    # CHECK: int:99
    print(describe_token(Token.integer(99)))

    match Token.identifier("x"):
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


def test_lists():
    # CHECK-LABEL: == test_lists
    print("== test_lists")

    # CHECK: empty
    print(classify_list([]))
    # CHECK: zero
    print(classify_list([0]))
    # CHECK: middle:1:1:3
    print(classify_list([1, 2, 3]))
    # CHECK: middle:4:0:5
    print(classify_list([4, 5]))
    # CHECK: rest:4:0
    print(classify_list([4]))


def test_explicit_value_patterns():
    # CHECK-LABEL: == test_explicit_value_patterns
    print("== test_explicit_value_patterns")

    # CHECK: expected
    print(classify_expected(42, 42))
    # CHECK: other
    print(classify_expected(7, 42))
    # CHECK: expected-second
    print(classify_expected_pair((1, 42), 42))
    # CHECK: other
    print(classify_expected_pair((42, 1), 42))
    # CHECK: computed
    print(classify_computed_expected(43, 42))
    # CHECK: other
    print(classify_computed_expected(42, 42))
    # A nested ternary remains part of the explicit value pattern.
    # CHECK: conditional
    print(classify_conditional_expected(42, 42, 7, True))
    # CHECK: conditional
    print(classify_conditional_expected(7, 42, 7, False))
    # CHECK: other
    print(classify_conditional_expected(42, 42, 7, False))

    # Types are ordinary compile-time values.
    # CHECK: integer
    print(classify_type[Int]())
    # CHECK: string
    print(classify_type[String]())
    # CHECK: other
    print(classify_type[Bool]())

    # The same type switch expressed as conditional patterns.
    # CHECK: if-case-integer
    print("if-case-" + classify_if_case_type[Int]())
    # CHECK: if-case-string
    print("if-case-" + classify_if_case_type[String]())
    # CHECK: if-case-other
    print("if-case-" + classify_if_case_type[Bool]())

    # Bindings introduced by a comptime conditional pattern are parameters.
    # CHECK: first:5
    print(classify_comptime_if_case_pair[(0, 5)]())
    # CHECK: second:7
    print(classify_comptime_if_case_pair[(7, 1)]())
    # CHECK: miss
    print(classify_comptime_if_case_pair[(4, 4)]())


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


def test_if_case():
    # CHECK-LABEL: == test_if_case
    print("== test_if_case")

    # CHECK: first:5
    print(classify_if_case((0, 5), False))
    # CHECK: fallback
    print(classify_if_case((7, 1), True))
    # CHECK: second:7
    print(classify_if_case((7, 1), False))
    # CHECK: miss
    print(classify_if_case((4, 4), False))

    # An OR pattern emits a nested hlcf.match operation.
    # CHECK: or:5
    print(classify_if_case_or((0, 5)))
    # CHECK: or:7
    print(classify_if_case_or((7, 1)))
    # CHECK: or:1
    print(classify_if_case_or((0, 1)))
    # CHECK: miss
    print(classify_if_case_or((4, 4)))

    # Nested enum and struct patterns.
    # CHECK: y-axis:8
    print(classify_if_case_nested(Optional(Point(0, 8))))
    # CHECK: x-axis:3
    print(classify_if_case_nested(Optional(Point(3, 0))))
    # CHECK: none
    print(classify_if_case_nested(Optional[Point](None)))
    # CHECK: other
    print(classify_if_case_nested(Optional(Point(3, 4))))


def test_while_case():
    # CHECK-LABEL: == test_while_case
    print("== test_while_case")

    # CHECK: hit:5:done
    print(classify_while_case((0, 5)))
    # CHECK: hit:7:done
    print(classify_while_case((7, 1)))
    # The first OR alternative wins when both match.
    # CHECK: hit:1:done
    print(classify_while_case((0, 1)))
    # CHECK: miss:done
    print(classify_while_case((4, 4)))

    # An explicit break skips the loop's else block.
    # CHECK: break:5
    print(classify_while_case_break((0, 5)))
    # CHECK: miss:done
    print(classify_while_case_break((4, 4)))


def test_comptime_match():
    # CHECK-LABEL: == test_comptime_match
    print("== test_comptime_match")

    # Literals, or-patterns, guards, and bare bindings.
    # CHECK: zero
    print(classify_comptime_int[0]())
    # CHECK: small
    print(classify_comptime_int[1]())
    # CHECK: small
    print(classify_comptime_int[2]())
    # CHECK: negative
    print(classify_comptime_int[-3]())
    # CHECK: other:10
    print(classify_comptime_int[10]())

    # Exhaustive Bool match.
    # CHECK: yes
    print(classify_comptime_bool[True]())
    # CHECK: no
    print(classify_comptime_bool[False]())

    # Or-pattern bindings Cond-select across alternatives.
    # CHECK: hit:5
    print(classify_comptime_or_bind[(0, 5)]())
    # CHECK: hit:7
    print(classify_comptime_or_bind[(7, 1)]())
    # Both match; first alternative wins (y from the right element).
    # CHECK: hit:1
    print(classify_comptime_or_bind[(0, 1)]())
    # CHECK: miss
    print(classify_comptime_or_bind[(4, 4)]())

    # `as` binding after a value test.
    # CHECK: zero:0
    print(classify_comptime_as[0]())
    # CHECK: other
    print(classify_comptime_as[3]())


def main():
    test_optional()
    test_enum_like_token()
    test_literals_guards_or()
    test_tuples_and_structs()
    test_lists()
    test_explicit_value_patterns()
    test_or_bindings()
    test_if_case()
    test_while_case()
    test_comptime_match()
