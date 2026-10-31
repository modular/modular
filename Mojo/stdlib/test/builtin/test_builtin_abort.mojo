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
#
# Verifies that failing `debug_assert`s and the `range` length overflow check
# abort with the expected message and source location.
#
# ===----------------------------------------------------------------------=== #

from std.reflection import SourceLocation, call_location
from std.testing import _assert_aborts, TestSuite


def test_debug_assert_default_error() raises:
    def trigger() raises {} -> None:
        debug_assert[assert_mode="safe"](
            False, "formatted failure message: ", 2, ", ", UInt8(4)
        )

    _assert_aborts(trigger, contains="formatted failure message: 2, 4")


def test_debug_assert_mode_all_error() raises:
    def trigger() raises {} -> None:
        debug_assert(False, "formatted failure message: ", 2, ", ", UInt8(4))

    _assert_aborts(trigger, contains="formatted failure message: 2, 4")


# Makes sure that passing call_location() with the default inline_level of 1
# keeps reporting the caller's location, so assert errors like bounds checks
# keep pointing at the source location where the user passed the bad index.
def test_debug_assert_location() raises:
    def trigger() raises {} -> None:
        _location_outer()

    _assert_aborts(
        trigger,
        contains="test_builtin_abort.mojo:44:24: Assert Error: forcing failure",
    )


@inline(.always)
def _location_outer():
    _location_inner(call_location())


@inline(.always)
def _location_inner(location: SourceLocation):
    debug_assert(False, "forcing failure", location=location)


# A `debug_assert` with no message arguments reports a default message, and
# its location must stay at the caller.
def test_debug_assert_no_message() raises:
    def trigger() raises {} -> None:
        _no_message_outer()

    _assert_aborts(
        trigger,
        contains=(
            "test_builtin_abort.mojo:66:26: Assert Error: assertion failed"
        ),
    )


@inline(.always)
def _no_message_outer():
    _no_message_inner(call_location())


@inline(.always)
def _no_message_inner(location: SourceLocation):
    debug_assert(False, location=location)


def test_range_len_overflow_abort() raises:
    def trigger() raises {} -> None:
        _ = range(UInt(0), UInt.MAX).__len__()

    _assert_aborts(trigger, contains="range length exceeds Int.MAX")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
