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
"""MXFP4 grouped matmul on AMD CDNA GPUs via dequant-to-FP8 + FP8 grouped GEMM.

Dequantizes MXFP4 expert weights to FP8, casts BF16 activations to FP8, then
dispatches to the AMD FP8 grouped GEMM via grouped_matmul.

The grouped matmul computes:
  for i in range(num_active_experts):
    C[offsets[i]:offsets[i+1], :] =
        A[offsets[i]:offsets[i+1], :] @ B[expert_ids[i], :, :].T

where B weights are stored as packed MXFP4 (uint8) with E8M0 scales, and
A activations are BF16. Both are dequantized/cast to FP8 before the GEMM.
"""

from std.math import ceildiv
from std.gpu.host import DeviceContext

from layout import Coord, Idx, TileTensor, row_major

from linalg.mxfp4_dequant import dequant_mxfp4, _cast_bf16_to_fp8
from linalg.grouped_matmul import grouped_matmul


def mxfp4_dequant_grouped_matmul_amd(
    c: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    a: TileTensor[address_space=AddressSpace.GENERIC, ...],
    b_packed: TileTensor[address_space=AddressSpace.GENERIC, ...],
    b_scales: TileTensor[address_space=AddressSpace.GENERIC, ...],
    a_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, DType.int32, address_space=AddressSpace.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    total_num_tokens: Int,
    ctx: DeviceContext,
) raises:
    """MXFP4 grouped matmul: dequant B weights, cast A to FP8, FP8 grouped GEMM.

    Args:
        c: Output [total_tokens, N] in bfloat16.
        a: Activations [total_tokens, K] in bfloat16.
        b_packed: Expert weights [num_experts, N, K//2] in uint8 (packed MXFP4).
        b_scales: Weight scales [num_experts, N, K//32] in float8_e8m0fnu.
        a_offsets: Token offsets [num_active_experts+1] in uint32.
        expert_ids: Expert indices [num_active_experts] in int32.
        max_num_tokens_per_expert: Maximum tokens per expert (for grid dim).
        num_active_experts: Number of active experts.
        total_num_tokens: Total number of tokens across all experts.
        ctx: Device context.
    """
    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b_packed.dtype
    comptime b_scales_type = b_scales.dtype

    comptime assert c_type == DType.bfloat16, "output must be bfloat16"
    comptime assert a_type == DType.bfloat16, "activations must be bfloat16"
    comptime assert b_type == DType.uint8, "weights must be uint8 (packed FP4)"
    comptime assert (
        b_scales_type == DType.float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"

    comptime assert b_packed.flat_rank == 3, "b_packed must be rank 3"
    comptime assert b_scales.flat_rank == 3, "b_scales must be rank 3"

    comptime num_experts = b_packed.static_shape[0]
    comptime static_N = b_packed.static_shape[1]
    comptime packed_K = b_packed.static_shape[2]
    comptime static_K = packed_K * 2
    comptime fp8_type = DType.float8_e4m3fn

    # Step 1: Dequantize all expert weights from MXFP4 to FP8.
    # B_packed is [num_experts, N, K//2], B_scales is [num_experts, N, K//32].
    # We dequant into a flat [num_experts*N, K] FP8 buffer, then reshape to
    # 3D [num_experts, N, K] for grouped_matmul.
    var b_fp8_buf = ctx.enqueue_create_buffer[fp8_type](
        num_experts * static_N * static_K
    )

    comptime scale_K = ceildiv(static_K, 32)
    for e in range(num_experts):
        var b_packed_expert = TileTensor(
            b_packed.ptr + e * static_N * packed_K,
            row_major[static_N, packed_K](),
        )
        var b_scales_expert = TileTensor(
            b_scales.ptr + e * static_N * scale_K,
            row_major[static_N, scale_K](),
        )
        var b_fp8_expert = TileTensor(
            b_fp8_buf.unsafe_ptr() + e * static_N * static_K,
            row_major((Idx[static_N](), Idx[static_K]())),
        )
        dequant_mxfp4(
            ctx,
            b_fp8_expert,
            b_packed_expert,
            b_scales_expert,
            num_rows=static_N,
            num_cols=static_K,
        )

    # Step 2: Cast all BF16 activations to FP8.
    var a_fp8_buf = ctx.enqueue_create_buffer[fp8_type](
        total_num_tokens * static_K
    )
    var a_fp8_tt = TileTensor(
        a_fp8_buf, row_major((Idx(total_num_tokens), Idx[static_K]()))
    )
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a, total_num_tokens, static_K)

    # Step 3: FP8 grouped GEMM.
    var b_fp8_tt = TileTensor(
        b_fp8_buf,
        row_major[num_experts, static_N, static_K](),
    )

    grouped_matmul(
        c,
        a_fp8_tt,
        b_fp8_tt,
        a_offsets,
        expert_ids,
        max_num_tokens_per_expert,
        num_active_experts,
        ctx,
    )

    # Keep temp buffers alive through async GEMM enqueue.
    _ = b_fp8_buf^
    _ = a_fp8_buf^
