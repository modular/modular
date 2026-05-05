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
"""Native MXFP4 grouped matmul on AMD CDNA4 via block-scaled MFMA.

Grouped matmul for Mixture of Experts (MoE):
  for i in range(num_active_experts):
    C[offsets[i]:offsets[i+1], :] =
        A[offsets[i]:offsets[i+1], :] @ B[expert_ids[i], :, :].T

Uses block_idx.z for expert dispatch and MXFP4MatmulAMD.run per-expert.

Entry point: mxfp4_grouped_matmul_amd()
"""

from std.math import ceildiv
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
)
from std.gpu.host import DeviceContext

from layout import Coord, Idx, TensorLayout, TileTensor
from layout.tile_layout import row_major

from std.utils import StaticTuple

from .mxfp4_matmul_amd import MXFP4MatmulAMD as _MXFP4MatmulAMD

# ===----------------------------------------------------------------------=== #
# Device kernel
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            _MXFP4MatmulAMD[
                BM=BM, BN=BN, BK_ELEMS=BK_ELEMS, WM=WM, WN=WN
            ].num_threads
        )
    )
)
def mxfp4_grouped_matmul_amd_kernel[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WM: Int,
    WN: Int,
    out_dtype: DType,
    LayoutC: TensorLayout,
    LayoutA: TensorLayout,
    LayoutB: TensorLayout,
    LayoutSFA: TensorLayout,
    LayoutSFB: TensorLayout,
    AOffsetsLayout: TensorLayout,
    ExpertIdsLayout: TensorLayout,
](
    c_tensor: TileTensor[mut=True, out_dtype, LayoutC, MutAnyOrigin],
    a_tensor: TileTensor[DType.uint8, LayoutA, ImmutAnyOrigin],
    b_tensor: TileTensor[DType.uint8, LayoutB, ImmutAnyOrigin],
    sfa_tensor: TileTensor[DType.float8_e8m0fnu, LayoutSFA, ImmutAnyOrigin],
    sfb_tensor: TileTensor[DType.float8_e8m0fnu, LayoutSFB, ImmutAnyOrigin],
    a_offsets: TileTensor[
        mut=False, DType.uint32, AOffsetsLayout, ImmutAnyOrigin
    ],
    expert_ids: TileTensor[
        mut=False, DType.int32, ExpertIdsLayout, ImmutAnyOrigin
    ],
    num_active_experts: Int,
):
    """MXFP4 grouped matmul kernel with expert dispatch via block_idx.z.

    b_tensor and sfb_tensor are flattened from 3D to 2D:
      b: [num_experts*N, K//2], sfb: [num_experts*N, K//32]
    """
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

    comptime Kernel = _MXFP4MatmulAMD[
        BM=BM, BN=BN, BK_ELEMS=BK_ELEMS, WM=WM, WN=WN
    ]
    comptime N = c_tensor.static_shape[1]
    comptime K_BYTES = b_tensor.static_shape[1]  # K//2
    comptime K_SCALES = sfa_tensor.static_shape[1]  # K//32

    var M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
    if M == 0 or N == 0:
        return
    var expert_id = expert_ids[block_idx.z]
    var a_start_row = a_offsets[block_idx.z]

    if expert_id == -1:
        return

    # Grid Y is ceildiv(max_tokens_per_expert, BM); skip blocks past this
    # expert's M rows (other experts may be shorter than max_tokens).
    if block_idx.y >= ceildiv(Int(M), BM):
        return

    var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)
    var a_ptr = a_tensor.ptr + a_start_row * UInt32(K_BYTES)
    var b_ptr = b_tensor.ptr + expert_id * Int32(N) * Int32(K_BYTES)
    var sfa_ptr = sfa_tensor.ptr + a_start_row * UInt32(K_SCALES)
    var sfb_ptr = sfb_tensor.ptr + expert_id * Int32(N) * Int32(K_SCALES)

    var c_tile = TileTensor(c_ptr, row_major(Coord(Idx(Int(M)), Idx[N]())))
    var a_tile = TileTensor(
        a_ptr, row_major(Coord(Idx(Int(M)), Idx[K_BYTES]()))
    )
    var b_tile = TileTensor(b_ptr, row_major[N, K_BYTES]())
    var sfa_tile = TileTensor(
        sfa_ptr, row_major(Coord(Idx(Int(M)), Idx[K_SCALES]()))
    )
    var sfb_tile = TileTensor(sfb_ptr, row_major[N, K_SCALES]())

    Kernel.run[
        out_dtype,
        type_of(c_tile).LayoutType,
        type_of(a_tile).LayoutType,
        type_of(b_tile).LayoutType,
        type_of(sfa_tile).LayoutType,
        type_of(sfb_tile).LayoutType,
    ](c_tile, a_tile, b_tile, sfa_tile, sfb_tile)


# ===----------------------------------------------------------------------=== #
# Public entry point
# ===----------------------------------------------------------------------=== #


def mxfp4_grouped_matmul_amd(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.uint8, ...],
    b: TileTensor[DType.uint8, ...],
    a_scales: TileTensor[DType.float8_e8m0fnu, ...],
    b_scales: TileTensor[DType.float8_e8m0fnu, ...],
    a_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, DType.int32, address_space=AddressSpace.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launch native MXFP4 grouped matmul on AMD CDNA4.

    Grouped matmul for MoE: dispatches one expert per block_idx.z,
    using MXFP4MatmulAMD.run per expert slice.

    Args:
        c: Output [total_tokens, N].
        a: Packed activations [total_tokens, K//2] uint8.
        b: Expert weights [num_experts, N, K//2] uint8.
        a_scales: Activation scales [total_tokens, K//32] float8_e8m0fnu.
        b_scales: Weight scales [num_experts, N, K//32] float8_e8m0fnu.
        a_offsets: Token offsets [num_active_experts+1] uint32.
        expert_ids: Expert indices [num_active_experts] int32.
        max_num_tokens_per_expert: Maximum token count for any active expert.
        num_active_experts: Number of active experts.
        ctx: Device context.
    """
    comptime assert b.flat_rank == 3, "b must be rank 3"
    comptime assert b_scales.flat_rank == 3, "b_scales must be rank 3"
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

    comptime K_BYTES = b.static_shape[2]  # K//2
    comptime can_use_bk_512 = K_BYTES >= 256 and K_BYTES % 256 == 0

    if max_num_tokens_per_expert <= 64:
        comptime if can_use_bk_512:
            _launch_mxfp4_grouped[BM=64, BN=128, BK_ELEMS=512, WM=64, WN=64](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
            return

    _launch_mxfp4_grouped[BM=128, BN=128, BK_ELEMS=128, WM=64, WN=64](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        max_num_tokens_per_expert,
        num_active_experts,
        ctx,
    )


def _launch_mxfp4_grouped[
    BM: Int, BN: Int, BK_ELEMS: Int, WM: Int, WN: Int
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.uint8, ...],
    b: TileTensor[DType.uint8, ...],
    a_scales: TileTensor[DType.float8_e8m0fnu, ...],
    b_scales: TileTensor[DType.float8_e8m0fnu, ...],
    a_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, DType.int32, address_space=AddressSpace.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Instantiates and launches the grouped MXFP4 kernel."""
    comptime Kernel = _MXFP4MatmulAMD[
        BM=BM, BN=BN, BK_ELEMS=BK_ELEMS, WM=WM, WN=WN
    ]
    comptime num_experts = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime K_BYTES = b.static_shape[2]  # K//2

    comptime num_experts_sf = b_scales.static_shape[0]
    comptime N_sf = b_scales.static_shape[1]
    comptime K_SCALES = b_scales.static_shape[2]  # K//32

    var a_i = TileTensor(
        a.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
        a.layout,
    )
    var b_2d = TileTensor(
        b.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
        row_major[num_experts * N, K_BYTES](),
    )
    var a_scales_i = TileTensor(
        a_scales.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
        a_scales.layout,
    )
    var sfb_2d = TileTensor(
        b_scales.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
        row_major[num_experts_sf * N_sf, K_SCALES](),
    )
    var a_off_i = TileTensor(
        a_offsets.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
        a_offsets.layout,
    )
    var expert_ids_i = TileTensor(
        expert_ids.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
        expert_ids.layout,
    )

    if max_num_tokens_per_expert == 0:
        return

    comptime out_dtype = type_of(c).dtype
    comptime kernel = mxfp4_grouped_matmul_amd_kernel[
        BM,
        BN,
        BK_ELEMS,
        WM,
        WN,
        out_dtype,
        type_of(c).LayoutType,
        type_of(a_i).LayoutType,
        type_of(b_2d).LayoutType,
        type_of(a_scales_i).LayoutType,
        type_of(sfb_2d).LayoutType,
        type_of(a_off_i).LayoutType,
        type_of(expert_ids_i).LayoutType,
    ]

    ctx.enqueue_function[kernel](
        c,
        a_i,
        b_2d,
        a_scales_i,
        sfb_2d,
        a_off_i,
        expert_ids_i,
        num_active_experts,
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=Kernel.num_threads,
    )
