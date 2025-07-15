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

from testing import assert_equal


def test_min():
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


def test_min_scalar():
    assert_equal(min(Bool(True), Bool(False)), Bool(False))
    assert_equal(min(Bool(False), Bool(True)), Bool(False))
    assert_equal(min(Bool(False), Bool(False)), Bool(False))
    assert_equal(min(Bool(True), Bool(True)), Bool(True))


def main():
    test_min()
    test_min_scalar()
