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

from std.math import align_up, inf, tanh
from std.memory import bitcast
from std.sys import simd_width_of
from std.sys.info import bit_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_true, TestSuite
from test_utils.math import libm_call, ulp_diff

from std.utils.coord import Coord


def _max_ulp[dtype: DType]() -> Int where dtype.is_floating_point():
    """Returns the worst case on MI355 and B200.

    NVIDIA uses `tanh.approx`, which the PTX ISA bounds at 2^-11 relative
    error for float32.
    """
    comptime if default_accelerator().is_amd_gpu():
        return 4 if dtype == .float32 else 0
    return 123 if dtype == .float32 else 1


def _check[
    dtype: DType
](
    ctx: DeviceContext, xs: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var length = align_up(len(xs), pack_size)

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = xs[i] if i < len(xs) else Scalar[dtype](0)

    var in_ptr = in_device.unsafe_ptr()
    var out_ptr = out_device.unsafe_ptr()

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_ptr, var in_ptr}:
        var idx = Int(idx0[0])
        var val = in_ptr.unsafe_load[width=simd_width](idx)
        out_ptr.unsafe_store[width=simd_width](idx, tanh(val))

    elementwise[pack_size](func, Coord(length), ctx)

    var worst = 0
    var worst_msg = String()
    with out_device.map_to_host() as out_host:
        for i in range(len(xs)):
            var x = xs[i]
            var got = out_host[i]
            var want = libm_call["tanhf", "tanh"](x.cast[.float64]()).cast[
                dtype
            ]()
            var ulp = ulp_diff(got, want)
            if ulp > worst:
                worst = ulp
                worst_msg = String(t"tanh[{dtype}]({x}) = {got}, libm {want}")
    comptime max_ulp = _max_ulp[dtype]()
    assert_true(
        worst <= max_ulp, String(t"{worst} ulp > {max_ulp}: ") + worst_msg
    )


def _all_16bit[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    comptime assert bit_width_of[dtype]() == 16
    var xs = List[Scalar[dtype]]()
    for bits in range(1 << 16):
        var x = bitcast[dtype](UInt16(bits))
        if x == x:
            xs.append(x)
    return xs^


def _f32_inputs() -> List[Float32]:
    var xs: List[Float32] = [
        0.0,
        inf[.float32](),
        -inf[.float32](),
        Float32.MAX_FINITE,
        Float32.MIN_FINITE,
    ]
    comptime n = 1 << 15
    for i in range(-n, n + 1):
        xs.append(Float32(i) / Float32(n) * 10)
    # Toward zero, where tanh(x) rounds to x.
    var t: Float32 = 1.0
    for _ in range(120):
        t *= 0.7
        xs.append(t)
        xs.append(-t)
    return xs^


def test_tanh_bf16() raises:
    with DeviceContext() as ctx:
        _check(ctx, _all_16bit[.bfloat16]())


def test_tanh_f16() raises:
    with DeviceContext() as ctx:
        _check(ctx, _all_16bit[.float16]())


def test_tanh_f32() raises:
    with DeviceContext() as ctx:
        _check(ctx, _f32_inputs())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
