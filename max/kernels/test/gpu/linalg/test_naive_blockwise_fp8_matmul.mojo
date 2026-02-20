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

from math import ceildiv

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from internal_utils import assert_almost_equal
from random import rand
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul

from utils.index import Index, IndexList


fn test_naive_blockwise_fp8_matmul[
    input_type: DType,
    block_scales_sizes: IndexList[3],
    transpose_b: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,) raises:
    comptime BLOCK_SCALE_M = block_scales_sizes[0]
    comptime BLOCK_SCALE_N = block_scales_sizes[1]
    comptime BLOCK_SCALE_K = block_scales_sizes[2]

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

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    comptime static_c_shape = DimList(m.dim, n.dim)

    comptime static_a_scale_shape = DimList(
        ceildiv(k.dim, BLOCK_SCALE_K), ceildiv(m.dim, BLOCK_SCALE_M)
    )
    comptime static_b_scale_shape = DimList(
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

    var a_size = m.value * k.value
    var b_size = n.value * k.value
    var c_size = m.value * n.value
    var a_scale_size = ceildiv(k.value, BLOCK_SCALE_K) * ceildiv(
        m.value, BLOCK_SCALE_M
    )
    var b_scale_size = ceildiv(n.value, BLOCK_SCALE_N) * ceildiv(
        k.value, BLOCK_SCALE_K
    )

    var a_host_ptr = UnsafePointer[Scalar[input_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[input_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[DType.float32]].alloc(c_size)

    var a_host = NDBuffer[input_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host = NDBuffer[input_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    c_host.zero()
    c_host_ref.zero()

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_device = ctx.enqueue_create_buffer[DType.float32](c_size)

    var a_device_nd = NDBuffer[input_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device_nd = NDBuffer[input_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device_nd = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )

    var a_scale_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        a_scale_size
    )
    var b_scale_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        b_scale_size
    )

    var a_scale_host = NDBuffer[DType.float32, 2, _, static_a_scale_shape](
        a_scale_host_ptr, dynamic_a_scale_shape
    )
    var b_scale_host = NDBuffer[DType.float32, 2, _, static_b_scale_shape](
        b_scale_host_ptr, dynamic_b_scale_shape
    )

    rand(a_scale_host.data, a_scale_host.num_elements())
    rand(b_scale_host.data, b_scale_host.num_elements())

    var a_scale_device = ctx.enqueue_create_buffer[DType.float32](a_scale_size)
    var b_scale_device = ctx.enqueue_create_buffer[DType.float32](b_scale_size)

    var a_scale_device_nd = NDBuffer[DType.float32, 2, _, static_a_scale_shape](
        a_scale_device.unsafe_ptr(), dynamic_a_scale_shape
    )
    var b_scale_device_nd = NDBuffer[DType.float32, 2, _, static_b_scale_shape](
        b_scale_device.unsafe_ptr(), dynamic_b_scale_shape
    )

    # run blockwise CPU as the reference output
    for _m in range(M):
        for _n in range(N):
            var res: Float32 = 0.0
            for _k in range(K):
                var a_scale = a_scale_host[
                    _k // BLOCK_SCALE_K, _m // BLOCK_SCALE_M
                ]
                var b_scale = b_scale_host[
                    _n // BLOCK_SCALE_N, _k // BLOCK_SCALE_K
                ] if transpose_b else b_scale_host[
                    _k // BLOCK_SCALE_K, _n // BLOCK_SCALE_N
                ]
                var b_elem = b_host[_n, _k] if transpose_b else b_host[_k, _n]
                res += (
                    a_host[_m, _k].cast[DType.float32]()
                    * b_elem.cast[DType.float32]()
                    * a_scale
                    * b_scale
                )

            c_host_ref[_m, _n] = res

    ctx.enqueue_copy(a_scale_device, a_scale_host_ptr)
    ctx.enqueue_copy(b_scale_device, b_scale_host_ptr)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

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
            c_device_nd,
            a_device_nd,
            b_device_nd,
            a_scale_device_nd,
            b_scale_device_nd,
            ctx,
        )
    else:
        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
        ](
            c_device_nd,
            a_device_nd,
            b_device_nd,
            a_scale_device_nd,
            b_scale_device_nd,
            ctx,
        )

    ctx.enqueue_copy(c_host_ptr, c_device)

    ctx.synchronize()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=0.0001,
    )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scale_host_ptr.free()
    b_scale_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = a_scale_device^
    _ = b_scale_device^


def main():
    with DeviceContext() as ctx:
        comptime for transpose_b in range(0, 2):
            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b = Bool(transpose_b),
            ](ctx, dynamic(128), static[128](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 64, 128),
                transpose_b = Bool(transpose_b),
            ](ctx, dynamic(128), static[256](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 64, 16),
                transpose_b = Bool(transpose_b),
            ](ctx, dynamic(128), static[128](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b = Bool(transpose_b),
            ](ctx, dynamic(120), static[128](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b = Bool(transpose_b),
            ](ctx, dynamic(120), static[129](), static[128]())

            test_naive_blockwise_fp8_matmul[
                DType.float8_e4m3fn,
                Index(32, 128, 64),
                transpose_b = Bool(transpose_b),
            ](ctx, dynamic(120), static[129](), static[129]())
