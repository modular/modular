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

from collections import Optional
from hashlib import default_comp_time_hasher
from sys import align_of, size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

# Additional imports for testing
from internal_utils import assert_almost_equal
from random import rand
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    matmul_sm100_fallback,
)
from linalg.utils import elementwise_epilogue_type

from utils.index import Index, IndexList


def test_matmul_sm100_fallback[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    BK: Int = 64,
    use_epilogue: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,):
    var M = m.value
    var N = n.value
    var K = k.value

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_size = m.value * k.value
    var b_size = n.value * k.value
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)

    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    var c_tensor = c_device_nd

    print(
        "umma_shape",
        umma_shape,
        a_type,
        "x",
        b_type,
        "x",
        c_type,
        "transpose_b:",
        transpose_b,
        "use_epilogue:",
        use_epilogue,
        " : PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (",
        umma_shape[0],
        "x",
        umma_shape[1],
        "x",
        BK,
        ")",
    )

    @parameter
    @always_inline
    @__copy_capture(c_tensor)
    fn epilogue_fn[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor.store[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    # Initialize matmul operands
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())
    c_host.zero()
    c_host_ref.zero()

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    var a = from_ndbuffer_row_major(a_device_nd)
    var b = from_ndbuffer_row_major(b_device_nd)
    var c = from_ndbuffer_row_major(c_device_nd)

    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    matmul_sm100_fallback[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn = Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](c, a, b, ctx)

    ctx.synchronize()

    comptime assert a_type != DType.float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    vendor_blas.matmul(
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()
    comptime rtol = 1e-2
    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:

        @parameter
        for dtype in [DType.float8_e4m3fn, DType.bfloat16]:

            @parameter
            for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
                comptime MMA_K = 32 if dtype == DType.float8_e4m3fn else 16
                comptime BK = (swizzle.bytes() // size_of[dtype]())

                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    DType.bfloat16,
                    umma_shape = Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                ](
                    ctx,
                    dynamic(200),
                    static[128](),
                    static[128](),
                )
                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    DType.bfloat16,
                    umma_shape = Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                    use_epilogue=True,
                ](
                    ctx,
                    dynamic(128),
                    static[128](),
                    static[128](),
                )

                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    DType.bfloat16,
                    umma_shape = Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                ](
                    ctx,
                    dynamic(400),
                    static[128](),
                    static[128](),
                )

                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    DType.bfloat16,
                    umma_shape = Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                ](
                    ctx,
                    dynamic(1024),
                    static[2048](),
                    static[2048](),
                )

                comptime BK_list: List[Int] = [BK, BK * 2]

                @parameter
                for _BK in BK_list:
                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        umma_shape = Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        dynamic(1024),
                        static[2048](),
                        static[2048](),
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        umma_shape = Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        static[1024](),
                        static[2048](),
                        static[2048](),
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        umma_shape = Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        dynamic(100),
                        static[512](),
                        static[256](),
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        umma_shape = Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        dynamic(99),
                        static[1024](),
                        static[1024](),
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        umma_shape = Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        dynamic(201),
                        static[2048](),
                        static[256](),
                    )
