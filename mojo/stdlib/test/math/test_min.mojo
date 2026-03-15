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


def test_min() raises:
    expected_result = SIMD[DType.bool, 4](False, True, False, False)
    actual_result = min(
        SIMD[DType.bool, 4](
            True,
            True,
            False,
            False,
        ),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(actual_result, expected_result)


def test_min_scalar() raises:
    assert_equal(min(Bool(True), Bool(False)), Bool(False))
    assert_equal(min(Bool(False), Bool(True)), Bool(False))
    assert_equal(min(Bool(False), Bool(False)), Bool(False))
    assert_equal(min(Bool(True), Bool(True)), Bool(True))


def test_min_iterable() raises:
    var expected_result = 0
    var l = [1, 2, 10, 4, 0, 6]
    assert_equal(min(l), expected_result)

    assert_equal(min(range(20)), 0)

    expected_result = 0
    var t = {1, 2, 8, 4, 0, 6}
    assert_equal(min(t), expected_result)

    with assert_raises():
        l = []
        _ = min(l)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
