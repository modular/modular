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

from internal_utils import kl_div
from internal_utils._testing import assert_with_measure
from std.testing import assert_almost_equal


def test_assert_with_custom_measure() raises:
    var t0 = List(length=100, fill=Float32(1.0))
    var t1 = List(length=100, fill=Float32(1.0))

    def always_zero[
        dtype: DType
    ](
        lhs: ImmPointer[Scalar[dtype], _],
        rhs: ImmPointer[Scalar[dtype], _],
        n: Int,
    ) -> Float64:
        return 0

    assert_with_measure[always_zero](t0.unsafe_ptr(), t1.unsafe_ptr(), 100)


def test_kl_div() raises:
    comptime dtype = DType.float32
    comptime out_dtype = DType.float64
    comptime len = 10

    var a = Array[Scalar[dtype], len](fill=Scalar[dtype](1 / Float64(len)))
    var b = Array[Scalar[dtype], len](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Scalar[dtype](
            2 * Float64(i + 1) / (len * (len + 1))
        )
    )

    var aa = kl_div[out_type=out_dtype](a.unsafe_ptr(), a.unsafe_ptr(), len)
    var ab = kl_div[out_type=out_dtype](a.unsafe_ptr(), b.unsafe_ptr(), len)
    assert_almost_equal(0.0, aa)
    # exact value computed using Mathematica
    assert_almost_equal(0.19430683493087375, ab)


def main() raises:
    test_assert_with_custom_measure()
    test_kl_div()
