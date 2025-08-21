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
from buffer.dimlist import DimList, Dim
from buffer.buffer import NDBuffer
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_sm100_blockwise_scaled_fp8,
)
from linalg.matmul_sm100_blockwise_fp8 import matmul_sm100_blockwise_scaled_fp8
from sys import sizeof
from gpu.host import DeviceContext
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg import vendor_blas
from gpu.host._nvidia_cuda import TensorMapSwizzle
from utils.index import Index, IndexList
from linalg.fp8_quantization import (
    naive_blockwise_scaled_fp8_matmul,
    naive_blockwise_scaled_fp8_grouped_matmul,
)
from internal_utils._measure import relative_difference

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
    zero,
)
from testing import assert_almost_equal
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.utils import elementwise_epilogue_type
from sys import alignof


def test_grouped_matmul_sm100_blockwise_scaled_fp8[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    umma_shape: IndexList[3] = Index(64, 64, 32),
    use_epilogue: Bool = False,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
):
    alias BLOCK_SCALE_K = 128
    alias block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)
    alias transpose_b = True

    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]
    alias K = expert_shape[1]
    alias swizzle = TensorMapSwizzle.SWIZZLE_128B

    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        var M = num_tokens_by_expert[i]
        total_num_tokens += M
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, M)

    debug_assert(
        total_num_tokens * sizeof[DType.float32]() % 16 == 0,
        "TMA expects total_num_tokens to be divisible by 16 bytes",
    )

    # Create host A C buffers
    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_active_experts + 1
    )

    # Create host B buffers
    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_active_experts)

    # Setup  offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_active_experts):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + num_tokens_by_expert[i]
        )
        expert_ids_host.tensor[i] = expert_ids_list[i]

    print(
        "== test_grouped_sm100_blockwise_scaled_fp8_matmul",
        a_type,
        "problem shape: (",
        total_num_tokens,
        "x",
        N,
        "x",
        K,
        ")",
        "block_tile_shape: (",
        block_tile_shape[0],
        "x",
        block_tile_shape[1],
        "x",
        block_tile_shape[2],
        ")",
        "transpose_b:",
        transpose_b,
    )

    debug_assert(
        (K % BLOCK_SCALE_K == 0),
        "K must be divisible by BLOCK_SCALE_K",
    )

    alias static_a_scales_shape = DimList(K // BLOCK_SCALE_K, Dim())
    var dynamic_a_scales_shape = DimList(K // BLOCK_SCALE_K, total_num_tokens)
    alias static_b_scales_shape = DimList(
        num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_device = DeviceNDBuffer[DType.uint32, 1](
        num_active_experts + 1, ctx=ctx
    )
    var expert_ids_device = DeviceNDBuffer[DType.int32, 1](
        num_active_experts, ctx=ctx
    )

    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    var a_scales_host = HostNDBuffer[DType.float32, 2, static_a_scales_shape](
        dynamic_a_scales_shape
    )
    var b_scales_host = HostNDBuffer[DType.float32, 3, static_b_scales_shape](
        static_b_scales_shape
    )

    var a_scales_device = DeviceNDBuffer[
        DType.float32, 2, static_a_scales_shape
    ](dynamic_a_scales_shape, ctx=ctx)
    var b_scales_device = DeviceNDBuffer[
        DType.float32, 3, static_b_scales_shape
    ](static_b_scales_shape, ctx=ctx)

    ctx.enqueue_copy(a_offsets_device.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_device.buffer, expert_ids_host.tensor.data)

    var c_tensor = c_device.tensor

    @parameter
    @always_inline
    @__copy_capture(c_tensor)
    fn epilogue_fn[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = alignof[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor.store[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    random(a_scales_host.tensor)
    random(b_scales_host.tensor)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)
    var c_ref = from_ndbuffer_row_major(c_device_ref.tensor)
    var a_scales = from_ndbuffer_row_major(a_scales_device.tensor)
    var b_scales = from_ndbuffer_row_major(b_scales_device.tensor)
    var a_offsets = from_ndbuffer_row_major(a_offsets_device.tensor)
    var expert_ids = from_ndbuffer_row_major(expert_ids_device.tensor)

    # Reference first
    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
    ](
        c_ref,
        a,
        b,
        a_offsets,
        expert_ids,
        a_scales,
        b_scales,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    grouped_matmul_sm100_blockwise_scaled_fp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        a_scales,
        b_scales,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    var rtol = 1e-2
    var atol = 1e-2
    var c_buf = c_host.tensor
    var c_ref_buf = c_host_ref.tensor
    for mi in range(total_num_tokens):
        for ni in range(N):
            assert_almost_equal(
                c_buf[mi, ni],
                c_ref_buf[mi, ni],
                msg=String("m: ", mi, " n: ", ni),
                rtol=rtol,
                atol=atol,
            )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device
    _ = a_scales_host
    _ = b_scales_host
    _ = a_scales_device
    _ = b_scales_device
    _ = a_offsets_host
    _ = expert_ids_host
    _ = a_offsets_device
    _ = expert_ids_device

    _ = a
    _ = b
    _ = c
    _ = a_scales
    _ = b_scales
    _ = a_offsets
    _ = expert_ids


def main():
    with DeviceContext() as ctx:
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(256, 256),
            use_epilogue=True,
        ](1, List[Int](128), List[Int](0), ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(512, 1024),
        ](1, List[Int](256), List[Int](0), ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(768, 1024),
        ](2, List[Int](128, 256), List[Int](0, 2), ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 1024),
        ](4, List[Int](20, 1500, 300, 28), List[Int](0, 3, 2, 4), ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 1024),
            use_epilogue=True,
        ](4, List[Int](20, 1500, 300, 28), List[Int](0, 3, 2, 4), ctx)
