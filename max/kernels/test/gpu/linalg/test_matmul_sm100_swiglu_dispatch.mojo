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

from std.collections import OptionalReg
from std.testing import assert_equal, assert_true
from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from max.gpu.host import DeviceContext
from linalg.swiglu_activation import swiglu
from linalg.matmul.gpu.sm100_structured.fused_swiglu.config import (
    swiglu_matmul_config,
)
from linalg.matmul.gpu.sm100_structured.fused_swiglu.dispatch import (
    matmul_swiglu_dispatch_sm100,
    matmul_swiglu_dispatch_sm100_bf16,
)


def test_strided[has_bias: Bool](ctx: DeviceContext) raises:
    comptime dtype = DType.bfloat16
    comptime m = 5
    comptime k = 8
    comptime h = 8
    comptime stride = 9
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * stride)
    var w_host = ctx.enqueue_create_host_buffer[dtype](2 * h * stride)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](2 * h)
    var out_host = ctx.enqueue_create_host_buffer[dtype](m * stride)
    for i in range(m * stride):
        a_host[i] = Scalar[dtype](-16)
    for i in range(2 * h * stride):
        w_host[i] = Scalar[dtype](-16)
    for row in range(m):
        for col in range(k):
            a_host[row * stride + col] = (Float32(row + 1) / Float32(k)).cast[
                dtype
            ]()
    for col in range(h):
        bias_host[2 * col] = Scalar[dtype](0.25)
        bias_host[2 * col + 1] = Scalar[dtype](0.5)
        for index in range(k):
            w_host[(2 * col) * stride + index] = (
                Float32(1) + Float32(col) / Float32(16)
            ).cast[dtype]()
            w_host[(2 * col + 1) * stride + index] = (
                Float32(2) + Float32(col) / Float32(16)
            ).cast[dtype]()

    var a_dev = ctx.enqueue_create_buffer[dtype](m * stride)
    var w_dev = ctx.enqueue_create_buffer[dtype](2 * h * stride)
    var bias_dev = ctx.enqueue_create_buffer[dtype](2 * h)
    var out_dev = ctx.enqueue_create_buffer[dtype](m * stride)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(w_dev, w_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_memset(out_dev, Scalar[dtype](-7))

    var a = TileTensor(
        a_dev, TileLayout((m, Idx[k]), (Idx[stride], Idx[1]))
    ).as_imm()
    var w = TileTensor(
        w_dev, TileLayout((Idx[2 * h], Idx[k]), (Idx[stride], Idx[1]))
    ).as_imm()
    var output = TileTensor(
        out_dev, TileLayout((m, Idx[h]), (Idx[stride], Idx[1]))
    )
    var bias_ptr = ImmPointer[BFloat16, ImmutAnyOrigin](
        unsafe_from_address=Int(bias_dev.unsafe_ptr())
    )
    matmul_swiglu_dispatch_sm100_bf16[has_bias=has_bias](
        output, a, w, ctx, OptionalReg(bias_ptr)
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    for row in range(m):
        for col in range(h):
            var gate = Float32(row + 1) * (
                Float32(1) + Float32(col) / Float32(16)
            )
            var up = Float32(row + 1) * (
                Float32(2) + Float32(col) / Float32(16)
            )
            comptime if has_bias:
                gate += Float32(0.25)
                up += Float32(0.5)
            var actual = out_host[row * stride + col].cast[.float32]()
            var expected = swiglu[0.0](gate, up).cast[dtype]().cast[.float32]()
            assert_true(abs(actual - expected) < max(abs(expected), 1.0) * 0.01)
        assert_equal(out_host[row * stride + h], Scalar[dtype](-7))


def test_dynamic_k(ctx: DeviceContext, k: Int) raises:
    comptime dtype = DType.bfloat16
    var a_dev = ctx.enqueue_create_buffer[dtype](k)
    var w_dev = ctx.enqueue_create_buffer[dtype](16 * k)
    var output = ctx.enqueue_create_buffer[dtype](8)
    var host = ctx.enqueue_create_host_buffer[dtype](8)
    if k > 0:
        ctx.enqueue_memset(a_dev, (Float32(1) / Float32(k)).cast[dtype]())
        ctx.enqueue_memset(w_dev, Scalar[dtype](1))
    ctx.enqueue_memset(output, Scalar[dtype](-7))
    var a = TileTensor(a_dev, row_major((1, k))).as_imm()
    var w = TileTensor(w_dev, row_major((Idx[16], k))).as_imm()
    var c = TileTensor(output, row_major((1, Idx[8])))
    matmul_swiglu_dispatch_sm100_bf16(c, a, w, ctx)
    ctx.enqueue_copy(host, output)
    ctx.synchronize()
    var value = Float32(1) if k > 0 else Float32(0)
    var expected = swiglu[0.0](value, value).cast[dtype]().cast[.float32]()
    for actual in host.as_span():
        assert_true(
            abs(actual.cast[.float32]() - expected)
            < max(abs(expected), 1.0) * 0.01
        )


def main() raises:
    comptime dtype = DType.bfloat16
    comptime config = swiglu_matmul_config[dtype, dtype, dtype, True](
        mma_shape=(64, 16, 16),
        cluster_shape=(1, 1, 1),
        cta_group=1,
        AB_swapped=False,
    )
    with DeviceContext() as ctx:
        var a = ctx.enqueue_create_buffer[dtype](8)
        var b = ctx.enqueue_create_buffer[dtype](16 * 8)
        var output = ctx.enqueue_create_buffer[dtype](8)
        var host = ctx.enqueue_create_host_buffer[dtype](8)
        ctx.enqueue_memset(output, Scalar[dtype](7))

        var a_empty = TileTensor(a, row_major((0, Idx[8]))).as_imm()
        var b_tensor = TileTensor(b, row_major((Idx[16], Idx[8]))).as_imm()
        var out_empty = TileTensor(output, row_major((0, Idx[8])))
        matmul_swiglu_dispatch_sm100[config](out_empty, a_empty, b_tensor, ctx)

        ctx.enqueue_copy(host, output)
        ctx.synchronize()
        for value in host.as_span():
            assert_equal(value, Scalar[dtype](7))
        test_strided[False](ctx)
        test_strided[True](ctx)
        test_dynamic_k(ctx, 0)
        test_dynamic_k(ctx, 8)
        test_dynamic_k(ctx, 16)
