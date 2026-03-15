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

from std.testing import TestSuite
from std.testing import assert_equal, assert_raises


def test_max() raises:
    expected_result = SIMD[DType.bool, 4](True, True, False, True)
    actual_result = max(
        SIMD[DType.bool, 4](
            True,
            True,
            False,
            False,
        ),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(actual_result, expected_result)


def test_max_iterable() raises:
    var expected_result = 10
    var l = [1, 2, 10, 4, 5, 6]
    assert_equal(max(l), expected_result)

    assert_equal(max(range(20)), 19)

    expected_result = 10
    var t = {1, 2, 8, 4, 10, 6}
    assert_equal(max(t), expected_result)

    with assert_raises():
        l = []
        _ = max(l)


def test_max_scalar() raises:
    assert_equal(max(Bool(True), Bool(False)), Bool(True))
    assert_equal(max(Bool(False), Bool(True)), Bool(True))
    assert_equal(max(Bool(False), Bool(False)), Bool(False))
    assert_equal(max(Bool(True), Bool(True)), Bool(True))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
