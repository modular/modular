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
"""Tests for `TriviallyMovable` / `TriviallyCopyable` / `TriviallyDeinitable`."""

from std.traits import (
    TriviallyCopyable,
    TriviallyDeinitable,
    TriviallyMovable,
)
from std.testing import TestSuite, assert_false, assert_true
from test_utils import ConfigureTrivial


@fieldwise_init
struct AllTrivial(Copyable):
    """A struct whose move/copy/deinit are all trivial."""

    var value: Int


struct NoneTrivial(Copyable):
    """A struct whose move/copy/deinit are all non-trivial because of user-defined
    lifecycle methods."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.value = move.value

    def __deinit__(deinit self):
        pass


struct NonMovable(Movable where False):
    pass


struct NonCopyable(Movable):
    pass


struct NonDeinitable(Deinitable where False, Movable):
    pass


def test_builtin_scalar_types() raises:
    assert_true(TriviallyMovable[Int])
    assert_true(TriviallyCopyable[Int])
    assert_true(TriviallyDeinitable[Int])

    assert_true(TriviallyMovable[Bool])
    assert_true(TriviallyCopyable[Bool])
    assert_true(TriviallyDeinitable[Bool])

    assert_true(TriviallyMovable[Float64])
    assert_true(TriviallyCopyable[Float64])
    assert_true(TriviallyDeinitable[Float64])


def test_string_is_non_trivial_copy_and_del() raises:
    # `String` owns a heap buffer, so copy and destruction are not trivial.
    assert_false(TriviallyCopyable[String])
    assert_false(TriviallyDeinitable[String])
    # Moves remain a bit-copy though.
    assert_true(TriviallyMovable[String])


def test_struct_with_only_trivial_fields() raises:
    assert_true(TriviallyMovable[AllTrivial])
    assert_true(TriviallyCopyable[AllTrivial])
    assert_true(TriviallyDeinitable[AllTrivial])


def test_struct_with_user_defined_lifecycle() raises:
    assert_false(TriviallyMovable[NoneTrivial])
    assert_false(TriviallyCopyable[NoneTrivial])
    assert_false(TriviallyDeinitable[NoneTrivial])


def test_non_movable_type_is_never_trivial() raises:
    assert_false(TriviallyMovable[NonMovable])
    assert_false(TriviallyCopyable[NonMovable])


def test_non_copyable_type_is_trivially_movable_but_not_copyable() raises:
    assert_true(TriviallyMovable[NonCopyable])
    assert_false(TriviallyCopyable[NonCopyable])


def test_non_deinitable_type_is_never_trivially_deletable() raises:
    assert_false(TriviallyDeinitable[NonDeinitable])
    assert_true(TriviallyMovable[NonDeinitable])


def test_configure_trivial_flags() raises:
    # Each flag can be toggled independently.
    comptime AllOn = ConfigureTrivial[
        del_is_trivial=True,
        copyinit_is_trivial=True,
        moveinit_is_trivial=True,
    ]
    assert_true(TriviallyMovable[AllOn])
    assert_true(TriviallyCopyable[AllOn])
    assert_true(TriviallyDeinitable[AllOn])

    comptime OnlyMove = ConfigureTrivial[
        del_is_trivial=False,
        copyinit_is_trivial=False,
        moveinit_is_trivial=True,
    ]
    assert_true(TriviallyMovable[OnlyMove])
    assert_false(TriviallyCopyable[OnlyMove])
    assert_false(TriviallyDeinitable[OnlyMove])

    comptime OnlyCopy = ConfigureTrivial[
        del_is_trivial=False,
        copyinit_is_trivial=True,
        moveinit_is_trivial=False,
    ]
    assert_false(TriviallyMovable[OnlyCopy])
    assert_true(TriviallyCopyable[OnlyCopy])
    assert_false(TriviallyDeinitable[OnlyCopy])

    comptime OnlyDel = ConfigureTrivial[
        del_is_trivial=True,
        copyinit_is_trivial=False,
        moveinit_is_trivial=False,
    ]
    assert_false(TriviallyMovable[OnlyDel])
    assert_false(TriviallyCopyable[OnlyDel])
    assert_true(TriviallyDeinitable[OnlyDel])


def test_helpers_match_underlying_flags() raises:
    # The helpers must agree with the raw trait fields they wrap.
    assert_equal_bool(TriviallyMovable[Int], Int.__move_ctor_is_trivial)
    assert_equal_bool(TriviallyCopyable[Int], Int.__copy_ctor_is_trivial)
    assert_equal_bool(TriviallyDeinitable[Int], Int.__del__is_trivial)
    assert_equal_bool(TriviallyMovable[String], String.__move_ctor_is_trivial)
    assert_equal_bool(TriviallyCopyable[String], String.__copy_ctor_is_trivial)
    assert_equal_bool(TriviallyDeinitable[String], String.__del__is_trivial)


def assert_equal_bool(a: Bool, b: Bool) raises:
    assert_true(a == b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
