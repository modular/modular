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

from std.memory import MaybeUninit
from std.testing import _assert_aborts, TestSuite


def test_copy_from_source_too_long() raises:
    def trigger() raises {} -> None:
        var source: Array[Int, 3] = [1, 2, 3]
        var storage = Array[MaybeUninit[Int], 2]()
        _ = Span(storage).unsafe_init_copy_from(source)

    _assert_aborts(
        trigger,
        contains=(
            "Span.unsafe_init_copy_from: source span length does not match"
        ),
    )


def test_copy_from_source_too_short() raises:
    def trigger() raises {} -> None:
        var source: Array[Int, 2] = [1, 2]
        var storage = Array[MaybeUninit[Int], 3]()
        _ = Span(storage).unsafe_init_copy_from(source)

    _assert_aborts(
        trigger,
        contains=(
            "Span.unsafe_init_copy_from: source span length does not match"
        ),
    )


def test_move_from_source_too_long() raises:
    def trigger() raises {} -> None:
        var source: Array[Int, 3] = [1, 2, 3]
        var storage = Array[MaybeUninit[Int], 2]()
        _ = Span(storage).unsafe_init_move_from(source)

    _assert_aborts(
        trigger,
        contains=(
            "Span.unsafe_init_move_from: source span length does not match"
        ),
    )


def test_move_from_source_too_short() raises:
    def trigger() raises {} -> None:
        var source: Array[Int, 2] = [1, 2]
        var storage = Array[MaybeUninit[Int], 3]()
        _ = Span(storage).unsafe_init_move_from(source)

    _assert_aborts(
        trigger,
        contains=(
            "Span.unsafe_init_move_from: source span length does not match"
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
