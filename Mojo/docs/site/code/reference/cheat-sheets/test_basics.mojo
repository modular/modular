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
"""Tests for the Mojo "Basics" cheat-sheet card.

The module docstring itself exercises the Comments & docstrings panel.
"""
# Not tested (compile-time errors: a runtime test cannot assert that code
# FAILS to compile; these are verified by hand and belong in a lit
# expected-error test if we want them automated):
#   - no implicit numeric conversion: `var f: Float64 = i` (i: Int) must error
#   - a bare bracket literal is an `Array`, so `[1, 2, 3].append(4)` must error
#   - `SIMD` width must be a power of two: `SIMD[DType.float32, 3]` must error
#
# Not tested (outside a runtime test's reach):
#   - `mojo build hello.mojo` creates a `hello` executable
#   - `print()` output formatting (asserted through `String(...)` instead,
#     which shares the same `sep` keyword)
from std.math import sqrt
from std.math import sqrt as root
from std.testing import assert_equal, assert_false, assert_true


def risky() raises:
    raise Error("boom")


def add_two(a: Int, b: Int) -> Int:
    return a + b


def describe(x: Int) -> String:
    __match x:
    case 2:
        return "is exactly two"
    case _ if x.is_power_of_two():
        return "power of two"
    case _:
        return "not power of two"


@fieldwise_init
struct Point:
    var x: Int
    var y: Int


struct Counter:
    comptime START = 0
    var n: Int

    def __init__(out self):
        self.n = Self.START

    def bump(mut self):
        self.n += 1

    @staticmethod
    def origin() -> Point:
        return Point(0, 0)


def test_ref_writes_through() raises:
    var data: List = [1, 2, 3]
    ref view = data[0]
    view = 99
    assert_equal(data[0], 99)


def test_number_types() raises:
    var f: SIMD[DType.float32, 1] = Float32(1.5)
    var s: Scalar[DType.float32] = f
    assert_equal(s, 1.5)

    # Dot notation must name the same type as the spelled-out `DType` form,
    # or this assignment fails to compile.
    var v: SIMD[DType.float32, 8] = SIMD[.float32, 8](2.0)
    assert_equal(v.reduce_add(), 16.0)


def test_conversions() raises:
    var i = 42
    assert_equal(Float64(i), 42.0)
    assert_equal(Int(Float64(2.9)), 2)
    assert_equal(String(i), "42")
    assert_equal(SIMD[DType.int32, 4](7).cast[DType.float32]()[0], 7.0)


def test_operators() raises:
    assert_equal(2**10, 1024)
    assert_equal(pow(2, 10), 1024)
    assert_equal(7 // 2, 3)
    assert_equal(7 % 2, 1)
    var a = 1
    var b = 2
    var c = 2
    assert_true(a < b <= c)
    assert_false(a < b < c)


def test_other_types() raises:
    var arr = [1, 2, 3]
    # Only compiles if the bare literal inferred `Array[Int, 3]`.
    var fixed: Array[Int, 3] = arr^
    assert_equal(len(fixed), 3)


def test_strings() raises:
    var who = "Mojo"
    assert_equal("Hi, " + who, "Hi, Mojo")
    assert_equal(String(t"Hi, {who}!"), "Hi, Mojo!")
    assert_equal(String(1, 2, 3, sep=": "), "1: 2: 3")
    assert_equal(String(t"x = {1 + 1}"), "x = 2")
    var raw = r"C:\path"
    assert_equal(raw.byte_length(), 7)


def test_control_flow() raises:
    assert_equal(describe(2), "is exactly two")
    assert_equal(describe(8), "power of two")
    assert_equal(describe(6), "not power of two")
    var x = 3
    var kind = "even" if x % 2 == 0 else "odd"
    assert_equal(kind, "odd")


def test_loops() raises:
    var seen: List[Int] = []
    for item in [10, 20, 30]:
        if item == 20:
            continue
        if item == 30:
            break
        seen.append(item)
    assert_equal(len(seen), 1)
    assert_equal(seen[0], 10)

    var countdown: List[Int] = []
    var n = 4
    while n > 0:
        countdown.append(n)
        n -= 1
    assert_equal(countdown, [4, 3, 2, 1])


def test_functions() raises:
    assert_equal(add_two(2, 3), 5)

    var c = 10

    def add_c(a: Int) {imm c} -> Int:
        return a + c

    assert_equal(add_c(5), 15)

    var anon = lambda (a: Int) {imm c} -> Int: a + c
    assert_equal(anon(10), 20)


def test_imports() raises:
    assert_equal(sqrt(16.0), 4.0)
    assert_equal(root(16.0), 4.0)


def test_list_ops() raises:
    var xs: List = [1, 2, 3]
    xs.append(4)
    assert_equal(xs[0], 1)
    assert_equal(len(xs), 4)


def test_dict_ops() raises:
    var xs: Dict = {"a": 1, "b": 2}
    xs["c"] = 3
    assert_equal(xs.get("a", 0), 1)
    assert_equal(xs.get("d", 0), 0)
    assert_equal(len(xs), 3)

    var raised = False
    try:
        _ = xs["d"]
    except:
        raised = True
    assert_true(raised)


def test_structs() raises:
    var p = Point(3, 4)
    assert_equal(p.x, 3)
    assert_equal(p.y, 4)

    var counter = Counter()
    counter.bump()
    assert_equal(counter.n, 1)
    assert_equal(Counter.START, 0)
    assert_equal(Counter.origin().x, 0)


def test_errors_try_except() raises:
    var caught = String("")
    try:
        risky()
    except e:
        caught = String(e)
    assert_equal(caught, "boom")


def test_explicit_copy() raises:
    var a: List[Int] = [1, 2]
    var b = a.copy()
    b.append(3)
    assert_equal(len(a), 2)
    assert_equal(len(b), 3)


def test_paren_line_continuation() raises:
    # fmt: off
    var total = (1 +
        2 +
        3)
    # fmt: on
    assert_equal(total, 6)


def test_semicolons() raises:
    # fmt: off
    var a = 1; a += 1
    # fmt: on
    assert_equal(a, 2)


def main() raises:
    test_ref_writes_through()
    test_number_types()
    test_conversions()
    test_operators()
    test_other_types()
    test_strings()
    test_control_flow()
    test_loops()
    test_functions()
    test_imports()
    test_list_ops()
    test_dict_ops()
    test_structs()
    test_errors_try_except()
    test_explicit_copy()
    test_paren_line_continuation()
    test_semicolons()
