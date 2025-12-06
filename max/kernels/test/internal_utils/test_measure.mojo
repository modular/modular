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

from internal_utils import correlation, kl_div
from internal_utils._testing import _assert_with_measure_impl
from itertools import product
from memory import LegacyUnsafePointer as UnsafePointer
from testing import assert_almost_equal


fn test_assert_with_custom_measure() raises:
    var t0 = UnsafePointer[Float32].alloc(100)
    var t1 = UnsafePointer[Float32].alloc(100)
    for i in range(100):
        t0[i] = 1.0
        t1[i] = 1.0

    fn always_zero[
        dtype: DType
    ](
        lhs: UnsafePointer[Scalar[dtype], mut=False],
        rhs: UnsafePointer[Scalar[dtype], mut=False],
        n: Int,
    ) -> Float64:
        return 0

    _assert_with_measure_impl[always_zero](t0, t1, 100)

    t0.free()
    t1.free()


fn test_correlation() raises:
    var a = 10
    var b = 10
    var len = a * b
    var u = UnsafePointer[Float32].alloc(len)
    var v = UnsafePointer[Float32].alloc(len)
    var x = UnsafePointer[Float32].alloc(len)
    for i in range(len):
        u.store(i, (0.01 * i).cast[DType.float32]())
        v.store(i, (-0.01 * i).cast[DType.float32]())
    for i, j in product(range(a), range(b)):
        x.store(b * i + j, (0.1 * i + 0.1 * j).cast[DType.float32]())

    assert_almost_equal(1.0, correlation[out_type = DType.float64](u, u, len))
    assert_almost_equal(-1.0, correlation[out_type = DType.float64](u, v, len))
    # +/- 0.773957299203321 is the exactly rounded fp64 answer calculated using mpfr
    assert_almost_equal(
        0.773957299203321, correlation[out_type = DType.float64](u, x, len)
    )
    assert_almost_equal(
        -0.773957299203321, correlation[out_type = DType.float64](v, x, len)
    )
    u.free()
    v.free()
    x.free()


fn test_kl_div() raises:
    comptime dtype = DType.float32
    comptime out_dtype = DType.float64
    comptime len = 10

    var a = InlineArray[Scalar[dtype], len](uninitialized=True)
    var b = InlineArray[Scalar[dtype], len](uninitialized=True)
    for i in range(len):
        a[i] = Scalar[dtype](1 / len)
        b[i] = Scalar[dtype](2 * (i + 1) / (len * (len + 1)))

    var aa = kl_div[out_type=out_dtype](a.unsafe_ptr(), a.unsafe_ptr(), len)
    var ab = kl_div[out_type=out_dtype](a.unsafe_ptr(), b.unsafe_ptr(), len)
    assert_almost_equal(0.0, aa)
    # exact value computed using Mathematica
    assert_almost_equal(0.19430683493087375, ab)


def main():
    test_assert_with_custom_measure()
    test_correlation()
    test_kl_div()
