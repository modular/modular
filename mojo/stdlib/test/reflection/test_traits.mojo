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
"""Tests for trait-checking meta functions in reflection.traits."""

from reflection.traits import (
    AllWritable,
    AllMovable,
    AllCopyable,
    AllImplicitlyCopyable,
    AllDefaultable,
    AllEquatable,
)
from testing import assert_true, assert_false
from testing import TestSuite


struct NoConformances:
    pass


def test_all_writable():
    assert_true(comptime (AllWritable[Int]))
    assert_true(comptime (AllWritable[Int, String, Float64]))
    assert_false(comptime (AllWritable[Int, NoConformances]))
    assert_false(comptime (AllWritable[NoConformances]))


def test_all_movable():
    assert_true(comptime (AllMovable[Int]))
    assert_true(comptime (AllMovable[Int, String, Float64]))
    assert_false(comptime (AllMovable[Int, NoConformances]))
    assert_false(comptime (AllMovable[NoConformances]))


def test_all_copyable():
    assert_true(comptime (AllCopyable[Int]))
    assert_true(comptime (AllCopyable[Int, String, Float64]))
    assert_false(comptime (AllCopyable[Int, NoConformances]))
    assert_false(comptime (AllCopyable[NoConformances]))


def test_all_implicitly_copyable():
    assert_true(comptime (AllImplicitlyCopyable[Int]))
    assert_true(comptime (AllImplicitlyCopyable[Int, Float64, Bool]))
    assert_false(comptime (AllImplicitlyCopyable[Int, NoConformances]))
    assert_false(comptime (AllImplicitlyCopyable[NoConformances]))


def test_all_defaultable():
    assert_true(comptime (AllDefaultable[Int]))
    assert_true(comptime (AllDefaultable[Int, Float64, Bool]))
    assert_false(comptime (AllDefaultable[Int, NoConformances]))
    assert_false(comptime (AllDefaultable[NoConformances]))


def test_all_equatable():
    assert_true(comptime (AllEquatable[Int]))
    assert_true(comptime (AllEquatable[Int, String, Bool]))
    assert_false(comptime (AllEquatable[Int, NoConformances]))
    assert_false(comptime (AllEquatable[NoConformances]))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
