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
from bit._mask import is_negative, is_true
from sys.info import bitwidthof


def test_is_negative():
    alias dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.index,
    )
    alias widths = (1, 2, 4, 8)

    @parameter
    for i in range(len(dtypes)):
        alias D = dtypes[i]
        var last_value = 2 ** (bitwidthof[D]() - 1) - 1
        var values = [1, 2, last_value - 1, last_value]

        @parameter
        for j in range(len(widths)):
            alias S = SIMD[D, widths[j]]

            for k in values:
                assert_equal(S(-1), is_negative(S(-k)))
                assert_equal(S(0), is_negative(S(k)))


def test_is_true():
    alias dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.index,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    alias widths = (1, 2, 4, 8)

    @parameter
    for i in range(len(dtypes)):
        alias D = dtypes[i]

        @parameter
        for j in range(len(widths)):
            alias w = widths[j]
            alias B = SIMD[DType.bool, w]
            assert_equal(SIMD[D, w](-1), is_true[D](B(True)))
            assert_equal(SIMD[D, w](0), is_true[D](B(False)))


def test_compare():
    alias dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.index,
    )
    alias widths = (1, 2, 4, 8)

    @parameter
    for i in range(len(dtypes)):
        alias D = dtypes[i]
        var last_value = 2 ** (bitwidthof[D]() - 1) - 1
        var values = [1, 2, last_value - 1, last_value]

        @parameter
        for j in range(len(widths)):
            alias S = SIMD[D, widths[j]]

            for k in values:
                var s_k = S(k)
                var s_k_1 = S(k - 1)
                assert_equal(S(-1), is_true[D](s_k == s_k))
                assert_equal(S(-1), is_true[D](-s_k == -s_k))
                assert_equal(S(-1), is_true[D](s_k != s_k_1))
                assert_equal(S(-1), is_true[D](-s_k != s_k_1))
                assert_equal(S(-1), is_true[D](s_k > s_k_1))
                assert_equal(S(-1), is_true[D](s_k_1 > -s_k))
                assert_equal(S(-1), is_true[D](-s_k >= -s_k))
                assert_equal(S(-1), is_true[D](-s_k < s_k_1))
                assert_equal(S(-1), is_true[D](-s_k <= -s_k))


def main():
    test_is_negative()
    test_is_true()
    test_compare()
