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

from collections import OptionalReg
from hashlib import default_comp_time_hasher
from buffer.dimlist import DimList
from linalg.matmul_sm100 import matmul_sm100_fallback
from sys import size_of
from gpu.host import DeviceContext
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg import vendor_blas
from gpu.host._nvidia_cuda import TensorMapSwizzle
from utils.index import Index, IndexList

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.utils import elementwise_epilogue_type
from sys import align_of


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

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    var c_tensor = c_device.tensor

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
    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)

    alias block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    matmul_sm100_fallback[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](c, a, b, ctx)

    ctx.synchronize()

    constrained[
        a_type != DType.float8_e4m3fn or transpose_b,
        (
            "Testing is only supported for transposed_b==True when"
            " a_type==float8_e4m3fn. Add the non-transposed case if needed."
        ),
    ]()

    vendor_blas.matmul(
        ctx,
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()
    alias rtol = 1e-2
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:

        @parameter
        for dtype in [DType.float8_e4m3fn, DType.bfloat16]:

            @parameter
            for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
                alias MMA_K = 32 if dtype == DType.float8_e4m3fn else 16
                alias BK = (swizzle.bytes() // size_of[dtype]())

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

                alias BK_list = List[Int](BK, BK * 2)

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
