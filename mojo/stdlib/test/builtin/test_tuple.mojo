# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from testing import assert_equal, assert_false, assert_true, TestSuite
from test_utils import CopyCounter


def test_tuple_contains():
    var a = (123, True, StaticString("Mojo is awesome"))

    assert_true(StaticString("Mojo is awesome") in a)
    assert_true(a.__contains__(StaticString("Mojo is awesome")))

    assert_false(StaticString("Hello world") in a)
    assert_false(a.__contains__(StaticString("Hello world")))

    assert_true(123 in a)
    assert_true(a.__contains__(123))

    assert_true(True in a)
    assert_true(a.__contains__(True))

    assert_false(False in a)
    assert_false(a.__contains__(False))

    assert_false(a.__contains__(1))
    assert_false(a.__contains__(0))
    assert_false(1 in a)
    assert_false(0 in a)

    var b = (False, True)
    assert_true(True in b)
    assert_true(b.__contains__(True))
    assert_true(False in b)
    assert_true(b.__contains__(False))
    assert_false(b.__contains__(1))
    assert_false(b.__contains__(0))

    var c = (1, 0)
    assert_false(c.__contains__(True))
    assert_false(c.__contains__(False))
    assert_false(True in c)
    assert_false(False in c)

    var d = (123, True, "Mojo is awesome")

    assert_true("Mojo is awesome" in d)
    assert_false(StaticString("Mojo is awesome") in d)
    assert_true(d.__contains__("Mojo is awesome"))

    assert_false("Hello world" in d)
    assert_false(d.__contains__("Hello world"))

    comptime a_alias = (123, True, StaticString("Mojo is awesome"))

    assert_true(StaticString("Mojo is awesome") in a_alias)
    assert_true(a_alias.__contains__(StaticString("Mojo is awesome")))

    assert_false(StaticString("Hello world") in a_alias)
    assert_false(a_alias.__contains__(StaticString("Hello world")))

    assert_true(123 in a_alias)
    assert_true(a_alias.__contains__(123))

    assert_true(True in a_alias)
    assert_true(a_alias.__contains__(True))

    assert_false(False in a_alias)
    assert_false(a_alias.__contains__(False))

    assert_false(a_alias.__contains__(1))
    assert_false(a_alias.__contains__(0))
    assert_false(1 in a_alias)
    assert_false(0 in a_alias)

    comptime b_alias = (False, True)
    assert_true(True in b_alias)
    assert_true(b_alias.__contains__(True))
    assert_true(False in b_alias)
    assert_true(b_alias.__contains__(False))
    assert_false(b_alias.__contains__(1))
    assert_false(b_alias.__contains__(0))

    comptime c_alias = (1, 0)
    assert_false(c_alias.__contains__(True))
    assert_false(c_alias.__contains__(False))
    assert_false(True in c_alias)
    assert_false(False in c_alias)

    comptime d_alias = (123, True, "Mojo is awesome")
    # Ensure `contains` itself works in comp-time domain
    comptime ok = 123 in d_alias
    assert_true(ok)

    assert_true("Mojo is awesome" in d_alias)
    assert_true(d_alias.__contains__("Mojo is awesome"))

    assert_false("Hello world" in d_alias)
    assert_false(d_alias.__contains__("Hello world"))


def test_tuple_unpack():
    (var list) = [a + b for a, b in [(1, 2), (3, 4)]]
    assert_equal(list, [3, 7])

    var list2 = [a + b for a, b in [(1, 2), (3, 4)]]
    assert_equal(list2, [3, 7])


def test_tuple_default():
    var t: Tuple[Int, String, Float32] = {}
    assert_equal(t[0], 0)
    assert_equal(t[1], "")
    assert_equal(t[2], 0.0)


def test_tuple_comparison():
    assert_true((1, 2, 3) == (1, 2, 3))
    assert_false((1, 2, 3) != (1, 2, 3))
    assert_false((1, 2, 3) < (1, 2, 3))
    assert_false((1, 2, 3) > (1, 2, 3))
    assert_true((1, 2, 3) <= (1, 2, 3))
    assert_true((1, 2, 3) >= (1, 2, 3))
    assert_true((1, 2, 3) < (1, 2, 4))
    assert_true((1, 2, 3) > (1, 2, 2))
    assert_true((1, 2, 3) <= (1, 2, 4))
    assert_true((1, 2, 3) >= (1, 2, 2))
    assert_false((1, 2, 3) < (1, 2, 2))
    assert_false((1, 2, 3) > (1, 2, 4))
    assert_true((1, 2, 3) <= (1, 2, 4))
    assert_true((1, 2, 3) >= (1, 2, 2))


def test_tuple_comparison_different_types():
    assert_false((1, "foo") == (1, "bar"))
    assert_true((1, "foo") != (1, "bar"))
    assert_false((1, "foo") < (1, "bar"))
    assert_true((1, "foo") > (1, "bar"))


def test_tuple_comparison_different_lengths():
    assert_false((1, 2, 3) == (1, 2))
    assert_true((1, 2, 3) != (1, 2))
    assert_false((1, 2, 3) < (1, 2))
    assert_true((1, 2, 3) > (1, 2))
    assert_false((1, 2, 3) <= (1, 2))
    assert_true((1, 2, 3) >= (1, 2))


def test_tuple_comparison_different_types_and_lengths():
    assert_false((1, "foo") == (1, "bar", "baz"))
    assert_true((1, "foo") != (1, "bar", "baz"))
    assert_false((1, "foo") < (1, "bar", "baz"))
    assert_true((1, "foo") > (1, "bar", "baz"))
    assert_false((1, "foo") <= (1, "bar", "baz"))
    assert_true((1, "foo") >= (1, "bar", "baz"))


def test_tuple_reverse_odd():
    var t = ("hi", 1, 4.5)
    var reversed_t = t^.reverse()
    assert_equal(reversed_t[0], 4.5)
    assert_equal(reversed_t[1], 1)
    assert_equal(reversed_t[2], "hi")


def test_tuple_reverse_empty():
    var t = Tuple[]()
    var t_reversed = t^.reverse()
    assert_true(t_reversed == ())


def test_tuple_reverse_even():
    var t = (Bool(True), Int(42))
    var t_reversed = t^.reverse()
    assert_equal(t_reversed[0], Int(42))
    assert_equal(t_reversed[1], Bool(True))


def test_tuple_reverse_copy_count():
    var t = (CopyCounter(),)
    var t2 = t^.reverse()
    assert_equal(t2[0].copy_count, 0)


def test_tuple_concat():
    var t = ("hi", "hey", 1)
    var t2 = (4.5, "hello")
    var concatted = t^.concat(t2^)
    assert_equal(concatted[0], "hi")
    assert_equal(concatted[1], "hey")
    assert_equal(concatted[2], 1)
    assert_equal(concatted[3], 4.5)
    assert_equal(concatted[4], "hello")


def test_tuple_empty_concat():
    var t = ()
    var t2 = ()
    var concatted = t^.concat(t2^)
    assert_true(concatted == ())


def test_tuple_identity_concat():
    var t = (Bool(True),)
    var t2 = ()
    var concatted = t^.concat(t2^)
    assert_true(concatted == (Bool(True),))


def test_tuple_concat_copy_count():
    var t = (CopyCounter(),)
    var t2 = (String(""),)
    var t3 = t^.concat(t2^)
    assert_equal(t3[0].copy_count, 0)


# This test doesn't need to run, it just needs to compile
def test_tuple_size_parse_time():
    fn func_with_where_clause(t: Tuple) where type_of(t).__len__() < 4:
        pass

    func_with_where_clause((1, 3, 2))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
