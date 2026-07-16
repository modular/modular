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

from std.memory import (
    is_trivially_copyable,
    is_trivially_deletable,
    is_trivially_movable,
)
from std.testing import TestSuite, assert_false, assert_true
from test_utils import ConfigureTrivial


@fieldwise_init
struct AllTrivial(Copyable):
    """A struct whose move/copy/del are all trivial."""

    var value: Int


struct NoneTrivial(Copyable):
    """A struct whose move/copy/del are all non-trivial because of user-defined
    lifecycle methods."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.value = move.value

    def __del__(deinit self):
        pass


def test_builtin_scalar_types() raises:
    assert_true(is_trivially_movable[Int]())
    assert_true(is_trivially_copyable[Int]())
    assert_true(is_trivially_deletable[Int]())

    assert_true(is_trivially_movable[Bool]())
    assert_true(is_trivially_copyable[Bool]())
    assert_true(is_trivially_deletable[Bool]())

    assert_true(is_trivially_movable[Float64]())
    assert_true(is_trivially_copyable[Float64]())
    assert_true(is_trivially_deletable[Float64]())


def test_string_is_non_trivial_copy_and_del() raises:
    # `String` owns a heap buffer, so copy and destruction are not trivial.
    assert_false(is_trivially_copyable[String]())
    assert_false(is_trivially_deletable[String]())
    # Moves remain a bit-copy though.
    assert_true(is_trivially_movable[String]())


def test_struct_with_only_trivial_fields() raises:
    assert_true(is_trivially_movable[AllTrivial]())
    assert_true(is_trivially_copyable[AllTrivial]())
    assert_true(is_trivially_deletable[AllTrivial]())


def test_struct_with_user_defined_lifecycle() raises:
    assert_false(is_trivially_movable[NoneTrivial]())
    assert_false(is_trivially_copyable[NoneTrivial]())
    assert_false(is_trivially_deletable[NoneTrivial]())


def test_configure_trivial_flags() raises:
    # Each flag can be toggled independently.
    comptime AllOn = ConfigureTrivial[
        del_is_trivial=True,
        copyinit_is_trivial=True,
        moveinit_is_trivial=True,
    ]
    assert_true(is_trivially_movable[AllOn]())
    assert_true(is_trivially_copyable[AllOn]())
    assert_true(is_trivially_deletable[AllOn]())

    comptime OnlyMove = ConfigureTrivial[
        del_is_trivial=False,
        copyinit_is_trivial=False,
        moveinit_is_trivial=True,
    ]
    assert_true(is_trivially_movable[OnlyMove]())
    assert_false(is_trivially_copyable[OnlyMove]())
    assert_false(is_trivially_deletable[OnlyMove]())

    comptime OnlyCopy = ConfigureTrivial[
        del_is_trivial=False,
        copyinit_is_trivial=True,
        moveinit_is_trivial=False,
    ]
    assert_false(is_trivially_movable[OnlyCopy]())
    assert_true(is_trivially_copyable[OnlyCopy]())
    assert_false(is_trivially_deletable[OnlyCopy]())

    comptime OnlyDel = ConfigureTrivial[
        del_is_trivial=True,
        copyinit_is_trivial=False,
        moveinit_is_trivial=False,
    ]
    assert_false(is_trivially_movable[OnlyDel]())
    assert_false(is_trivially_copyable[OnlyDel]())
    assert_true(is_trivially_deletable[OnlyDel]())


def test_helpers_match_underlying_flags() raises:
    # The helpers must agree with the raw trait fields they wrap.
    assert_equal_bool(is_trivially_movable[Int](), Int.__move_ctor_is_trivial)
    assert_equal_bool(is_trivially_copyable[Int](), Int.__copy_ctor_is_trivial)
    assert_equal_bool(is_trivially_deletable[Int](), Int.__del__is_trivial)
    assert_equal_bool(
        is_trivially_movable[String](), String.__move_ctor_is_trivial
    )
    assert_equal_bool(
        is_trivially_copyable[String](), String.__copy_ctor_is_trivial
    )
    assert_equal_bool(
        is_trivially_deletable[String](), String.__del__is_trivial
    )


def assert_equal_bool(a: Bool, b: Bool) raises:
    assert_true(a == b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
