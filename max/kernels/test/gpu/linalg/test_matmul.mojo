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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from collections.optional import OptionalReg
from math import ceildiv
from sys import alignof, bitwidthof, has_nvidia_gpu_accelerator, simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import barrier, block_dim, block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host import get_gpu_target
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    arange,
    fill,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg import vendor_blas
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig, MatmulKernels
from memory import memset_zero, stack_allocation
from memory.pointer import _GPUAddressSpace as GPUAddressSpace
from test_utils import ulp_distance
from testing import assert_almost_equal, assert_true

from utils import IndexList
from utils.index import Index
from utils.numerics import FPUtils

alias init_fn_type = fn (buff: NDBuffer[mut=True, *_]) capturing -> None

alias epilogue_func_type = fn[type: DType, width: Int, *, alignment: Int = 1] (
    IndexList[2], IndexList[2], SIMD[type, width]
) capturing -> SIMD[type, width]


@parameter
@always_inline
fn epilogue_test_fn[
    type: DType, width: Int, *, alignment: Int = 1
](
    idx: IndexList[2],
    dim_space: IndexList[2],
    val: SIMD[type, width],
) -> SIMD[
    type, width
]:
    var bias = SIMD[type, width](0)

    @parameter
    for i in range(width):
        bias[i] = (
            0.5
            + ((idx[0] + idx[1] + i) / (dim_space[0] + dim_space[1])).cast[
                type
            ]()
        )

    return val + bias


fn select_max_ulp_distance[
    lambda_fn: Optional[epilogue_func_type]
](max_ulp_distance: Optional[Int]) -> Int:
    if max_ulp_distance:
        return max_ulp_distance.value()
    else:

        @parameter
        if lambda_fn:
            return 4
        return 2


fn test[
    type: DType,
    /,
    *,
    transpose_b: Bool = False,
    init_a: Optional[init_fn_type] = None,
    init_b: Optional[init_fn_type] = None,
    lambda_fn: Optional[epilogue_func_type] = None,
    config: OptionalReg[MatmulConfig[type, type, type, transpose_b]] = None,
](
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    rtol: Float64 = 1e-3 if type is DType.float32 else 1e-2,
    max_ulp_distance: Optional[Int] = None,
) raises:
    constrained[
        Int(n.dim) > 0 and Int(k.dim) > 0,
        "This test currently requires static N and K.",
    ]()

    var M = m.value
    var N = n.value
    var K = k.value
    print(M, "x", N, "x", K)

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

    var a_host = HostNDBuffer[type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    @parameter
    if init_a:
        alias init_a_fn = init_a.value()
        init_a_fn(a_host.tensor)
    else:
        random(a_host.tensor)

    @parameter
    if init_b:
        alias init_b_fn = init_b.value()
        init_b_fn(b_host.tensor)
    else:
        random(b_host.tensor)

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    var c_tensor = c_device.tensor

    @parameter
    @always_inline
    @__copy_capture(c_tensor, M, N)
    fn epilogue_fn[
        _type: DType,
        width: Int,
        *,
        alignment: Int = alignof[SIMD[_type, width]](),
    ](idx: IndexList[2], val: SIMD[_type, width]) capturing -> None:
        var update_val: SIMD[_type, width] = val

        @parameter
        if lambda_fn:
            alias func = lambda_fn.value()
            update_val = func(idx, (M, N), update_val)
        c_tensor.store[alignment=alignment](
            idx, rebind[SIMD[type, width]](update_val)
        )

    @parameter
    if lambda_fn:
        _matmul_gpu[
            use_tensor_core=True,
            transpose_b=transpose_b,
            elementwise_lambda_fn=epilogue_fn,
            config=config,
        ](
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            ctx,
        )
    else:
        _matmul_gpu[
            use_tensor_core=True,
            transpose_b=transpose_b,
            config=config,
        ](
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            ctx,
        )

    ctx.synchronize()

    vendor_blas.matmul(
        ctx,
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    var c_ref_tensor = c_device_ref.tensor
    alias pack_size = simdwidthof[type, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_ref_tensor, M, N)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[2]](idx0)

        var val = c_ref_tensor.load[width=simd_width](idx)

        var update_val = val

        @parameter
        if lambda_fn:
            alias element_lambda = lambda_fn.value()
            update_val = element_lambda(idx, (M, N), val)

        c_ref_tensor.store(
            idx,
            update_val,
        )

    @parameter
    if lambda_fn:
        elementwise[func, pack_size, target="gpu"](
            IndexList[2](M, Int(N)),
            ctx,
        )
    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    c_host_tensor = c_host.tensor
    c_host_ref_tensor = c_host_ref.tensor
    var _max_ulp_distance = select_max_ulp_distance[lambda_fn](max_ulp_distance)
    for m in range(M):
        for n in range(N):
            var expect = c_host_ref_tensor[m, n]
            var actual = c_host_tensor[m, n]

            @parameter
            if bitwidthof[type]() <= 16:
                var ulp_dist = ulp_distance(actual, expect)
                if ulp_dist <= _max_ulp_distance:
                    continue

            assert_almost_equal(actual, expect, rtol=rtol)

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


def main():
    with DeviceContext() as ctx:
        print("===> tfloat32-float32 mma")
        test[
            DType.float32,
            init_a=arange,
            init_b=arange,
        ](ctx, dynamic(512), static[12288](), static[4096]())
        test[DType.float32, init_a=arange](
            ctx, dynamic(256), static[384](), static[128]()
        )
        test[DType.float32, init_b=arange](
            ctx, dynamic(128), static[4096](), static[4096]()
        )
        test[
            DType.float32,
            init_a=arange,
            init_b=arange,
        ](ctx, dynamic(512), static[12288](), static[4096]())
        test[DType.float32](ctx, dynamic(23), static[4096](), static[11008]())
        test[DType.float32](ctx, dynamic(67), static[4096](), static[12288]())
        test[DType.float32](ctx, dynamic(555), static[4096](), static[4096]())

        print("===> bfloat16-float32 mma")
        test[
            DType.bfloat16,
            init_a=arange,
            transpose_b=True,
            config = MatmulConfig[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                transpose_b=True,
            ](
                block_tile_shape=Index(64, 128, 64),
                warp_tile_shape=(16, 128, 64),
                num_pipeline_stages=3,
            ),
        ](ctx, dynamic(100), static[128](), static[128]())
        test[DType.bfloat16, init_b=arange](
            ctx, dynamic(1024), static[12288](), static[3072]()
        )
        test[
            DType.bfloat16,
            init_a=arange,
            init_b=arange,
        ](ctx, dynamic(1024), static[5120](), static[3072]())
        test[DType.bfloat16](
            ctx, dynamic(1024), static[3072](), static[32768]()
        )
        test[DType.bfloat16](ctx, dynamic(1024), static[3072](), static[3072]())

        @parameter
        if has_nvidia_gpu_accelerator():
            test[
                DType.bfloat16,
                transpose_b=True,
                config = MatmulConfig[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    transpose_b=True,
                ](
                    block_tile_shape=Index(16, 64, 64),
                    warp_tile_shape=Index(16, 64, 32),
                    num_pipeline_stages=3,
                    num_k_partitions=1,
                    num_warp_k_partitions=2,
                ),
            ](ctx, dynamic(32), static[4096](), static[4096]())
            test[
                DType.bfloat16,
                transpose_b=True,
                config = MatmulConfig[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    transpose_b=True,
                ](
                    block_tile_shape=Index(32, 64, 32),
                    warp_tile_shape=Index(16, 64, 32),
                    num_pipeline_stages=3,
                    num_k_partitions=1,
                    num_warp_k_partitions=4,
                ),
            ](ctx, dynamic(32), static[4096](), static[4096]())

        print("===> tfloat32-float32 mma with epilogue")
        test[
            DType.float32,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(999), static[3072](), static[3072]())
        test[
            DType.float32,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(777), static[12288](), static[2048]())

        print("===> bfloat16-float32 mma with epilogue")
        # Our default split-k reduction precision is output precision. For
        # bfloat16, we need a larger tolerance since the reference may reduce
        # in float32.
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(14), static[3072](), static[12288](), rtol=2e-2)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(33), static[12288](), static[3072]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(101), static[5120](), static[3072]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(400), static[3072](), static[32768](), rtol=2e-2)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(910), static[3072](), static[3072]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(50), static[6144](), static[4096]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(22), static[4096](), static[4096]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(88), static[28672](), static[4096]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(100), static[4096](), static[14336]())
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, dynamic(600), static[128256](), static[4096]())
