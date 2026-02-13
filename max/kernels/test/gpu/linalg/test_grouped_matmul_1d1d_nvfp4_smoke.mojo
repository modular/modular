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
"""Smoke test for grouped_matmul_1d1d_nvfp4.

Exercises the grouped 1D-1D NVFP4 matmul kernel directly to verify
enqueue_function type identity. This test catches the type mismatch
that caused the DeepSeek-R1-NVFP4 pipeline failure (PR #77347/#77359).
"""

from math import ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from internal_utils._utils import InitializationType, init_vector_launch
from layout._ndbuffer_stub import from_ndbuffer_row_major
from utils.index import Index

from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_1d1d_nvfp4,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
)


fn test_grouped_1d1d_nvfp4[
    num_experts: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext, num_active_experts: Int, tokens_per_expert: Int) raises:
    comptime a_type = DType.uint8
    comptime b_type = DType.uint8
    comptime c_type = DType.bfloat16
    comptime packed_K = K // 2

    print(
        "  experts=",
        num_active_experts,
        "/",
        num_experts,
        " tokens=",
        tokens_per_expert,
        " N=",
        N,
        " K=",
        K,
    )

    var total_tokens = num_active_experts * tokens_per_expert

    # Offsets and expert IDs
    var a_offsets_host = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts + 1
    )
    var a_scale_offsets_host = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts
    )
    var expert_ids_host = UnsafePointer[Scalar[DType.int32]].alloc(
        num_active_experts
    )

    var a_scale_dim0 = 0
    a_offsets_host[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_host[i] = UInt32(
            a_scale_dim0 - Int(a_offsets_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(tokens_per_expert)
        a_scale_dim0 += ceildiv(tokens_per_expert, SF_MN_GROUP_SIZE)
        expert_ids_host[i] = Int32(i)

    # Device buffers
    var a_buf = ctx.enqueue_create_buffer[a_type](total_tokens * packed_K)
    var b_buf = ctx.enqueue_create_buffer[b_type](num_experts * N * packed_K)
    var c_buf = ctx.enqueue_create_buffer[c_type](total_tokens * N)
    var a_off_buf = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var a_soff_buf = ctx.enqueue_create_buffer[DType.uint32](num_active_experts)
    var eid_buf = ctx.enqueue_create_buffer[DType.int32](num_active_experts)

    # NDBuffers
    var a_nd = NDBuffer[a_type, 2, _, DimList(Dim(), packed_K)](
        a_buf.unsafe_ptr(), DimList(total_tokens, packed_K)
    )
    var b_nd = NDBuffer[b_type, 3, _, DimList(num_experts, N, packed_K)](
        b_buf.unsafe_ptr(), DimList(num_experts, N, packed_K)
    )
    var c_nd = NDBuffer[c_type, 2, _, DimList(Dim(), N)](
        c_buf.unsafe_ptr(), DimList(total_tokens, N)
    )
    var a_off_nd = NDBuffer[DType.uint32, 1](
        a_off_buf.unsafe_ptr(), num_active_experts + 1
    )
    var a_soff_nd = NDBuffer[DType.uint32, 1](
        a_soff_buf.unsafe_ptr(), num_active_experts
    )
    var eid_nd = NDBuffer[DType.int32, 1](
        eid_buf.unsafe_ptr(), num_active_experts
    )

    # Init data
    init_vector_launch[a_type](
        a_buf,
        total_tokens * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )
    init_vector_launch[b_type](
        b_buf,
        num_experts * N * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )
    ctx.enqueue_copy(a_off_buf, a_offsets_host)
    ctx.enqueue_copy(a_soff_buf, a_scale_offsets_host)
    ctx.enqueue_copy(eid_buf, expert_ids_host)

    # Scale factors
    var a_sf_size = (
        a_scale_dim0
        * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_sf_size = (
        num_experts
        * ceildiv(N, SF_MN_GROUP_SIZE)
        * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var a_sf_buf = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](a_sf_size)
    var b_sf_buf = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](b_sf_size)

    comptime a_sf_shape = DimList(
        Dim(),
        ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    comptime b_sf_shape = DimList(
        num_experts,
        ceildiv(N, SF_MN_GROUP_SIZE),
        ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var a_sf_nd = NDBuffer[NVFP4_SF_DTYPE, 5, _, a_sf_shape](
        a_sf_buf.unsafe_ptr(),
        DimList(
            a_scale_dim0,
            ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
            SF_ATOM_M[0],
            SF_ATOM_M[1],
            SF_ATOM_K,
        ),
    )
    var b_sf_nd = NDBuffer[NVFP4_SF_DTYPE, 6, _, b_sf_shape](
        b_sf_buf.unsafe_ptr(), b_sf_shape
    )

    init_vector_launch[NVFP4_SF_DTYPE](
        a_sf_buf, a_sf_size, InitializationType.uniform_distribution, ctx
    )
    init_vector_launch[NVFP4_SF_DTYPE](
        b_sf_buf, b_sf_size, InitializationType.uniform_distribution, ctx
    )

    # Expert scales
    var es_buf = ctx.enqueue_create_buffer[DType.float32](num_experts)
    var es_host = UnsafePointer[Scalar[DType.float32]].alloc(num_experts)
    for i in range(num_experts):
        es_host[i] = 1.0
    ctx.enqueue_copy(es_buf, es_host)
    var es_nd = NDBuffer[DType.float32, 1](es_buf.unsafe_ptr(), num_experts)

    # LayoutTensors
    var a = from_ndbuffer_row_major(a_nd)
    var b = from_ndbuffer_row_major(b_nd)
    var c = from_ndbuffer_row_major(c_nd)
    var a_offsets = from_ndbuffer_row_major(a_off_nd)
    var a_scale_offsets = from_ndbuffer_row_major(a_soff_nd)
    var expert_ids = from_ndbuffer_row_major(eid_nd)
    var a_scales = from_ndbuffer_row_major(a_sf_nd)
    var b_scales = from_ndbuffer_row_major(b_sf_nd)
    var expert_scales = from_ndbuffer_row_major(es_nd)

    # Launch kernel
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, NVFP4_SF_DTYPE, NVFP4_SF_DTYPE, True
    ](
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        cluster_shape=Index(1, 1, 1),
        mma_shape=Index(128, 128, 32),
        block_swizzle_size=0,
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
        num_accum_pipeline_stages=2,
    )

    grouped_matmul_1d1d_nvfp4[transpose_b=True, config=config](
        c,
        a,
        a_offsets,
        a_scale_offsets,
        b,
        expert_ids,
        a_scales,
        b_scales,
        expert_scales,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()
    print("    PASSED")

    a_offsets_host.free()
    a_scale_offsets_host.free()
    expert_ids_host.free()
    es_host.free()


def main():
    var ctx = DeviceContext()
    print("=== Grouped 1D1D NVFP4 Smoke Tests ===")
    test_grouped_1d1d_nvfp4[4, 128, 256](ctx, 4, 64)
    test_grouped_1d1d_nvfp4[8, 128, 256](ctx, 4, 64)
    test_grouped_1d1d_nvfp4[4, 1024, 1024](ctx, 2, 128)
    print("=== ALL TESTS PASSED ===")
