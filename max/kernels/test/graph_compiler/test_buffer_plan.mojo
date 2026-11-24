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
"""Pins the digit-string encoding of the `mgp.buffer.plan` sharing matrix.

The compiler emits `can_share` as one '0'/'1' digit per matrix entry instead
of a comptime `Array` list literal. These tests check that the decoded matrix
drives allocation reuse, so the shared digits and the planner cannot drift
apart.
"""

from builtin_primitives.primitives import mgp_buffer_plan

from std.testing import TestSuite, assert_equal


def test_sharing_digits_allow_reuse() raises:
    """Adjacent allocations share a block when their digit is 1."""
    comptime alignments: Array[Int, 2] = [64, 64]
    comptime static_sizes: Array[Int, 2] = [100, 50]
    var runtime_sizes: Array[Int, 0] = []

    # Rows are "01" and "10": allocation 1 fits inside allocation 0's block.
    var result = mgp_buffer_plan[alignments, "0110", static_sizes](
        runtime_sizes
    )

    assert_equal(result[0], 100)
    assert_equal(result[1][0], 0)
    assert_equal(result[1][1], 0)


def test_zero_digits_keep_allocations_apart() raises:
    """With no sharing digits set, each allocation gets its own block."""
    comptime alignments: Array[Int, 2] = [64, 64]
    comptime can_share: StaticString = "0000"
    comptime static_sizes: Array[Int, 2] = [100, 50]
    var runtime_sizes: Array[Int, 0] = []

    var result = mgp_buffer_plan[alignments, can_share, static_sizes](
        runtime_sizes
    )

    assert_equal(result[0], 178)
    assert_equal(result[1][0], 0)
    assert_equal(result[1][1], 128)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
