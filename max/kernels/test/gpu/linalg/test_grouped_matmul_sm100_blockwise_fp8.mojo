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
from sys import align_of, size_of

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from internal_utils._measure import relative_difference
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_sm100_blockwise_scaled_fp8,
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent,
)
from linalg.matmul.gpu.sm100.blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8,
)
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.utils import elementwise_epilogue_type
from memory import LegacyUnsafePointer as UnsafePointer
from testing import assert_almost_equal

from utils.index import Index, IndexList


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
    comptime BLOCK_SCALE_K = 128
    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)
    comptime transpose_b = True

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]
    comptime swizzle = TensorMapSwizzle.SWIZZLE_128B

    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        var M = num_tokens_by_expert[i]
        total_num_tokens += M
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, M)

    debug_assert(
        total_num_tokens * size_of[DType.float32]() % 16 == 0,
        "TMA expects total_num_tokens to be divisible by 16 bytes",
    )

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

    # Define shapes
    comptime static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = IndexList[2](total_num_tokens, K)
    var a_size = total_num_tokens * K

    comptime static_b_shape = DimList(num_experts, N, K)
    var b_size = num_experts * N * K

    comptime static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = IndexList[2](total_num_tokens, N)
    var c_size = total_num_tokens * N

    comptime static_a_scales_shape = DimList(K // BLOCK_SCALE_K, Dim())
    var dynamic_a_scales_shape = IndexList[2](
        K // BLOCK_SCALE_K, total_num_tokens
    )
    var a_scales_size = (K // BLOCK_SCALE_K) * total_num_tokens

    comptime static_b_scales_shape = DimList(
        num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )
    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, K)
    comptime b_layout = Layout.row_major(num_experts, N, K)
    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, N)
    comptime a_scales_layout = Layout.row_major(
        K // BLOCK_SCALE_K, UNKNOWN_VALUE
    )
    comptime b_scales_layout = Layout.row_major(
        num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var a_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts + 1
    )
    var expert_ids_host_ptr = UnsafePointer[Scalar[DType.int32]].alloc(
        num_active_experts
    )
    var a_scales_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        a_scales_size
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        b_scales_size
    )

    var a_host = LayoutTensor[a_type, a_layout](
        a_host_ptr,
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
    )
    var b_host = LayoutTensor[b_type, b_layout](
        b_host_ptr,
        RuntimeLayout[b_layout].row_major(IndexList[3](num_experts, N, K)),
    )
    var c_host = LayoutTensor[c_type, c_layout](
        c_host_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )
    var c_host_ref = LayoutTensor[c_type, c_layout](
        c_host_ref_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )
    var a_scales_host = LayoutTensor[DType.float32, a_scales_layout](
        a_scales_host_ptr,
        RuntimeLayout[a_scales_layout].row_major(dynamic_a_scales_shape),
    )
    var b_scales_host = LayoutTensor[DType.float32, b_scales_layout](
        b_scales_host_ptr,
        RuntimeLayout[b_scales_layout].row_major(
            IndexList[3](num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K)
        ),
    )

    # Setup offsets and expert ids
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = (
            a_offsets_host_ptr[i] + num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = expert_ids_list[i]

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )
    var a_scales_device_buffer = ctx.enqueue_create_buffer[DType.float32](
        a_scales_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[DType.float32](
        b_scales_size
    )

    var a_device = NDBuffer[a_type, 2, _, static_a_shape](
        a_device_buffer.unsafe_ptr(),
        DimList(total_num_tokens, K),
    )
    var b_device = NDBuffer[b_type, 3, _, static_b_shape](
        b_device_buffer.unsafe_ptr(),
        static_b_shape,
    )
    var c_device = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_buffer.unsafe_ptr(),
        DimList(total_num_tokens, N),
    )
    var c_device_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref_buffer.unsafe_ptr(),
        DimList(total_num_tokens, N),
    )
    var a_offsets_device = NDBuffer[DType.uint32, 1](
        a_offsets_device_buffer.unsafe_ptr(),
        num_active_experts + 1,
    )
    var expert_ids_device = NDBuffer[DType.int32, 1](
        expert_ids_device_buffer.unsafe_ptr(),
        num_active_experts,
    )
    var a_scales_device = NDBuffer[DType.float32, 2, _, static_a_scales_shape](
        a_scales_device_buffer.unsafe_ptr(),
        DimList(K // BLOCK_SCALE_K, total_num_tokens),
    )
    var b_scales_device = NDBuffer[DType.float32, 3, _, static_b_scales_shape](
        b_scales_device_buffer.unsafe_ptr(),
        static_b_scales_shape,
    )

    var c_tensor = c_device

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

    random(a_host)
    random(b_host)
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    random(a_scales_host)
    random(b_scales_host)

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device_buffer, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)
    var c_ref = from_ndbuffer_row_major(c_device_ref)
    var a_scales = from_ndbuffer_row_major(a_scales_device)
    var b_scales = from_ndbuffer_row_major(b_scales_device)
    var a_offsets = from_ndbuffer_row_major(a_offsets_device)
    var expert_ids = from_ndbuffer_row_major(expert_ids_device)

    # Reference first
    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
        scales_granularity_mnk = Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_ref,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(1, 1, 1),
        mma_shape=umma_shape,
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
    )

    # grouped_matmul_sm100_blockwise_scaled_fp8[
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
        config=config,
        elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var rtol = 1e-2
    var atol = 1e-2
    for mi in range(total_num_tokens):
        for ni in range(N):
            assert_almost_equal(
                c_host[mi, ni][0],
                c_host_ref[mi, ni][0],
                msg=String("m: ", mi, " n: ", ni),
                rtol=rtol,
                atol=atol,
            )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_offsets_host_ptr.free()
    expert_ids_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = c_device_ref_buffer^
    _ = a_offsets_device_buffer^
    _ = expert_ids_device_buffer^
    _ = a_scales_device_buffer^
    _ = b_scales_device_buffer^

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
        ](1, [128], [0], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(512, 1024),
        ](1, [256], [0], ctx)

        # Simple expert routing
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(512, 1024),
        ](1, [256], [2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(4096, 7168),
        ](2, [128, 256], [0, 2], ctx)

        # Unaligned grouped matmul
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(7168, 2048),
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 1024),
            use_epilogue=True,
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float32,
            num_experts=4,
            expert_shape = Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float32,
            num_experts=1,
            expert_shape = Index(512, 1024),
        ](1, [512], [0], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float32,
            num_experts=6,
            expert_shape = Index(7168, 2048),
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float32,
            num_experts=6,
            expert_shape = Index(1280, 1024),
            use_epilogue=True,
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(4096, 7168),
        ](2, [8, 64], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(7168, 2048),
        ](4, [20, 4, 4, 40], [0, 3, 2, 4], ctx)
