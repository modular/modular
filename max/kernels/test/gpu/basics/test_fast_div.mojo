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


from std.algorithm.functional import elementwise
from std.gpu import *
from std.gpu.host import DeviceContext
from std.testing import *

from std.utils.fast_div import FastDiv
from std.utils.index import Index, IndexList

from layout import TileTensor, Idx, row_major


def test_fast_div() raises:
    var divisor = 7
    var fast_div = FastDiv[DType.uint32](divisor)

    for i in range(1000):
        assert_equal(
            Scalar[fast_div.uint_type](i) / fast_div,
            Scalar[fast_div.uint_type](i // divisor),
            msg=String(t"mismatch for {i}/{divisor}"),
        )


def test_fast_div_uint64() raises:
    var divisor = 7
    var fast_div = FastDiv[DType.uint64](divisor)

    for i in range(1000):
        assert_equal(
            Scalar[fast_div.uint_type](i) / fast_div,
            Scalar[fast_div.uint_type](i // divisor),
            msg=String(t"uint64 mismatch for {i}/{divisor}"),
        )

    # Test with large values that exceed uint32 range.
    comptime large_base = 1 << 40
    for i in range(1000):
        var val = large_base + i
        assert_equal(
            Scalar[fast_div.uint_type](val) / fast_div,
            Scalar[fast_div.uint_type](val // divisor),
            msg=String(t"uint64 large mismatch for {val}/{divisor}"),
        )

    # Test power-of-2 divisor with uint64.
    var fast_div_pow2 = FastDiv[DType.uint64](16)
    for i in range(1000):
        var val = large_base + i
        assert_equal(
            Scalar[fast_div_pow2.uint_type](val) / fast_div_pow2,
            Scalar[fast_div_pow2.uint_type](val // 16),
            msg=String(t"uint64 pow2 mismatch for {val}/16"),
        )


def test_fast_div_print() raises:
    var fast_div = FastDiv[DType.uint32](33)
    assert_equal(
        """div: 33
mprime: 4034666248
sh1: 1
sh2: 5
is_pow2: False
log2_shift: 6
""",
        String(fast_div),
    )


def run_elementwise[type: DType](ctx: DeviceContext) raises:
    comptime length = 256

    var divisors_stack = InlineArray[Scalar[type], length](uninitialized=True)
    var divisors = TileTensor(divisors_stack, row_major[length]())
    var remainders_stack = InlineArray[Scalar[type], length](uninitialized=True)
    var remainders = TileTensor(remainders_stack, row_major[length]())

    var out_divisors = ctx.enqueue_create_buffer[type](length)
    var out_remainders = ctx.enqueue_create_buffer[type](length)

    var out_divisors_buffer = TileTensor(out_divisors, row_major(Idx(length)))
    var out_remainders_buffer = TileTensor(
        out_remainders, row_major(Idx(length))
    )

    @always_inline
    @__copy_capture(out_divisors_buffer, out_remainders_buffer)
    @parameter
    def func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        comptime fast_div = FastDiv[type](4)
        var idx = idx0[0]

        out_divisors_buffer[idx] = (
            Scalar[fast_div.uint_type](idx) / fast_div
        ).cast[type]()
        out_remainders_buffer[idx] = (
            Scalar[fast_div.uint_type](idx) % fast_div
        ).cast[type]()

    elementwise[func, simd_width=1, target="gpu"](Index(length), ctx)

    ctx.enqueue_copy(divisors.ptr, out_divisors)
    ctx.enqueue_copy(remainders.ptr, out_remainders)

    ctx.synchronize()

    for i in range(length):
        print(divisors[i], remainders[i])
        assert_equal(
            divisors[i], Scalar[type](i // 4), msg="the divisor is not correct"
        )
        assert_equal(
            remainders[i],
            Scalar[type](i % 4),
            msg="the remainder is not correct",
        )


def main() raises:
    test_fast_div()
    test_fast_div_uint64()
    test_fast_div_print()
    with DeviceContext() as ctx:
        run_elementwise[DType.uint32](ctx)
        run_elementwise[DType.uint64](ctx)
