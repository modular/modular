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

from math import ceildiv

from buffer import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    fill,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul

from utils.index import Index, IndexList


fn test_naive_blockwise_fp8_matmul[
    input_type: DType,
    block_scales_sizes: IndexList[3],
    transpose_b: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,) raises:
    alias BLOCK_SCALE_M = block_scales_sizes[0]
    alias BLOCK_SCALE_N = block_scales_sizes[1]
    alias BLOCK_SCALE_K = block_scales_sizes[2]

    var M = m.value
    var N = n.value
    var K = k.value

    print(
        "== test_naive_blockwise_fp8_matmul",
        input_type,
        "x",
        M,
        "x",
        N,
        "x",
        K,
        "BLOCK_SCALE_M",
        BLOCK_SCALE_M,
        "BLOCK_SCALE_N",
        BLOCK_SCALE_N,
        "BLOCK_SCALE_K",
        BLOCK_SCALE_K,
        "transpose_b",
        transpose_b,
    )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)

    alias static_a_scale_shape = DimList(
        ceildiv(k.dim, BLOCK_SCALE_K), ceildiv(m.dim, BLOCK_SCALE_M)
    )
    alias static_b_scale_shape = DimList(
        ceildiv(n.dim, BLOCK_SCALE_N), ceildiv(k.dim, BLOCK_SCALE_K)
    ) if transpose_b else DimList(
        ceildiv(k.dim, BLOCK_SCALE_K), ceildiv(n.dim, BLOCK_SCALE_N)
    )

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)
    var dynamic_a_scale_shape = DimList(
        ceildiv(k.value, BLOCK_SCALE_K), ceildiv(m.value, BLOCK_SCALE_M)
    )
    var dynamic_b_scale_shape = DimList(
        ceildiv(n.value, BLOCK_SCALE_N), ceildiv(k.value, BLOCK_SCALE_K)
    ) if transpose_b else DimList(
        ceildiv(k.value, BLOCK_SCALE_K), ceildiv(n.value, BLOCK_SCALE_N)
    )

    var a_host = HostNDBuffer[input_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[input_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[DType.float32, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[DType.float32, 2, static_c_shape](
        dynamic_c_shape
    )

    random(a_host.tensor)
    random(b_host.tensor)

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    var a_device = DeviceNDBuffer[input_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[input_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[DType.float32, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    var a_scale_host = HostNDBuffer[DType.float32, 2, static_a_scale_shape](
        dynamic_a_scale_shape
    )
    var b_scale_host = HostNDBuffer[DType.float32, 2, static_b_scale_shape](
        dynamic_b_scale_shape
    )

    random(a_scale_host.tensor)
    random(b_scale_host.tensor)

    var a_scale_device = DeviceNDBuffer[DType.float32, 2, static_a_scale_shape](
        dynamic_a_scale_shape, ctx=ctx
    )
    var b_scale_device = DeviceNDBuffer[DType.float32, 2, static_b_scale_shape](
        dynamic_b_scale_shape, ctx=ctx
    )

    # run blockwise CPU as the reference output
    for _m in range(M):
        for _n in range(N):
            var res: Float32 = 0.0
            for _k in range(K):
                var a_scale = a_scale_host.tensor[
                    _k // BLOCK_SCALE_K, _m // BLOCK_SCALE_M
                ]
                var b_scale = b_scale_host.tensor[
                    _n // BLOCK_SCALE_N, _k // BLOCK_SCALE_K
                ] if transpose_b else b_scale_host.tensor[
                    _k // BLOCK_SCALE_K, _n // BLOCK_SCALE_N
                ]
                var b_elem = b_host.tensor[
                    _n, _k
                ] if transpose_b else b_host.tensor[_k, _n]
                res += (
                    a_host.tensor[_m, _k].cast[DType.float32]()
                    * b_elem.cast[DType.float32]()
                    * a_scale
                    * b_scale
                )

            c_host_ref.tensor[_m, _n] = res

    ctx.enqueue_copy(a_scale_device.buffer, a_scale_host.tensor.data)
    ctx.enqueue_copy(b_scale_device.buffer, b_scale_host.tensor.data)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    if (
        M % BLOCK_SCALE_M != 0
        or N % BLOCK_SCALE_N != 0
        or K % BLOCK_SCALE_K != 0
    ):
        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
            scales_granularity_mnk = Index(
                BLOCK_SCALE_M, BLOCK_SCALE_N, BLOCK_SCALE_K
            ),
        ](
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            a_scale_device.tensor,
            b_scale_device.tensor,
            ctx,
        )
    else:
        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
        ](
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            a_scale_device.tensor,
            b_scale_device.tensor,
            ctx,
        )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=0.0001,
    )

    _ = a_device^
    _ = b_device^
    _ = c_device^

    _ = a_host^
    _ = b_host^
    _ = c_host^
    _ = c_host_ref^

    _ = a_scale_device^
    _ = b_scale_device^
    _ = a_scale_host^
    _ = b_scale_host^


def main():
    with DeviceContext() as ctx:

        @parameter
        for transpose_b in range(0, 2):
            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b=transpose_b,
            ](ctx, dynamic(128), static[128](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 64, 128),
                transpose_b=transpose_b,
            ](ctx, dynamic(128), static[256](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 64, 16),
                transpose_b=transpose_b,
            ](ctx, dynamic(128), static[128](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b=transpose_b,
            ](ctx, dynamic(120), static[128](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b=transpose_b,
            ](ctx, dynamic(120), static[129](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(32, 128, 64),
                transpose_b=transpose_b,
            ](ctx, dynamic(120), static[129](), static[129]())
