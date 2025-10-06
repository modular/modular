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

from sys.info import bit_width_of

from bit._mask import is_negative, splat
from testing import assert_equal
from testing import TestSuite


def test_is_negative():
    alias dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int,
    )
    alias widths = (1, 2, 4, 8)

    @parameter
    for i in range(len(dtypes)):
        alias D = dtypes[i]
        var last_value = 2 ** (D.bit_width() - 1) - 1
        var values = [1, 2, last_value - 1, last_value]

        @parameter
        for j in range(len(widths)):
            alias S = SIMD[D, widths[j]]

            for k in values:
                assert_equal(S(-1), is_negative(S(-k)))
                assert_equal(S(0), is_negative(S(k)))


def test_splat():
    alias dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int,
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
            assert_equal(SIMD[D, w](-1), splat[D](B(fill=True)))
            assert_equal(SIMD[D, w](0), splat[D](B(fill=False)))


def test_compare():
    alias dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int,
    )
    alias widths = (1, 2, 4, 8)

    @parameter
    for i in range(len(dtypes)):
        alias D = dtypes[i]
        var last_value = 2 ** (D.bit_width() - 1) - 1
        var values = [1, 2, last_value - 1, last_value]

        @parameter
        for j in range(len(widths)):
            alias S = SIMD[D, widths[j]]

            for k in values:
                var s_k = S(k)
                var s_k_1 = S(k - 1)
                assert_equal(S(-1), splat[D](s_k.eq(s_k)))
                assert_equal(S(-1), splat[D]((-s_k).eq(-s_k)))
                assert_equal(S(-1), splat[D](s_k.ne(s_k_1)))
                assert_equal(S(-1), splat[D]((-s_k).ne(s_k_1)))
                assert_equal(S(-1), splat[D](s_k.gt(s_k_1)))
                assert_equal(S(-1), splat[D](s_k_1.gt(-s_k)))
                assert_equal(S(-1), splat[D]((-s_k).ge(-s_k)))
                assert_equal(S(-1), splat[D]((-s_k).lt(s_k_1)))
                assert_equal(S(-1), splat[D]((-s_k).le(-s_k)))


def main():
    var suite = TestSuite()

    suite.test[test_is_negative]()
    suite.test[test_splat]()
    suite.test[test_compare]()

    suite^.run()
