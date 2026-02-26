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

from memory import UnsafeMaybeUninit, memcmp
from sys import size_of
from test_utils import (
    AbortOnDel,
    ConfigureTrivial,
    CopyCounter,
    DelRecorder,
    MoveCounter,
)
from testing import *


def test_maybe_uninitialized():
    # Every time an Int is destroyed, it's going to be recorded here.
    var destructor_recorder = List[Int]()

    var ptr = UnsafePointer(to=destructor_recorder).as_immutable()
    var a = UnsafeMaybeUninit[DelRecorder[ptr.origin]]()
    a.init_from(DelRecorder(42, ptr))

    assert_equal(a.unsafe_assume_init_ref().value, 42)
    assert_equal(len(destructor_recorder), 0)

    assert_equal(a.unsafe_ptr()[].value, 42)
    assert_equal(len(destructor_recorder), 0)

    a.unsafe_assume_init_destroy()
    assert_equal(len(destructor_recorder), 1)
    assert_equal(destructor_recorder[0], 42)
    _ = a

    # Last use of a, but the destructor should not have run
    # since we assume uninitialized memory
    assert_equal(len(destructor_recorder), 1)


def test_write_does_not_trigger_destructor():
    var a = UnsafeMaybeUninit[AbortOnDel]()
    a.init_from(AbortOnDel(42))

    # Using the initializer should not trigger the destructor too.
    _ = UnsafeMaybeUninit[AbortOnDel](AbortOnDel(42))

    # The destructor of a and b have already run at this point, and it shouldn't have
    # caused a crash since we assume uninitialized memory.


def test_init_from():
    var a = "hello"
    var uninit = UnsafeMaybeUninit[String]()
    uninit.init_from(a^)
    assert_equal(uninit.unsafe_assume_init_ref(), "hello")
    uninit.unsafe_assume_init_destroy()


def test_take():
    var a = MoveCounter(0)
    var uninit = UnsafeMaybeUninit[MoveCounter[Int]](a^)
    var moved = uninit.unsafe_assume_init_take()
    assert_equal(moved.move_count, 2)


def test_zeroed():
    # For Int, zeroed memory is valid and should be 0.
    var a = UnsafeMaybeUninit[Int].zeroed()
    assert_equal(a.unsafe_assume_init_ref(), 0)

    # For UInt64, zeroed memory should also be 0.
    var b = UnsafeMaybeUninit[UInt64].zeroed()
    assert_equal(b.unsafe_assume_init_ref(), 0)

    var c = UnsafeMaybeUninit[String].zeroed()
    var arr = InlineArray[Byte, size_of[String]()](fill=0)
    assert_equal(
        memcmp(
            c.unsafe_ptr().bitcast[Byte](), arr.unsafe_ptr(), size_of[String]()
        ),
        0,
    )


def test_triviality():
    comptime Trivial = UnsafeMaybeUninit[Int]
    comptime NotTrivial = UnsafeMaybeUninit[
        ConfigureTrivial[
            copyinit_is_trivial=False,
            moveinit_is_trivial=False,
        ]
    ]

    assert_true(Trivial.__copy_ctor_is_trivial)
    assert_true(Trivial.__move_ctor_is_trivial)
    assert_true(Trivial.__del__is_trivial)

    assert_false(NotTrivial.__copy_ctor_is_trivial)
    assert_false(NotTrivial.__move_ctor_is_trivial)
    # UnsafeMaybeUninit always has a trivial destructor
    assert_true(NotTrivial.__del__is_trivial)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
