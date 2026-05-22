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

from std.math import ceildiv
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.host.info import MI355X

from layout import Coord, Idx, TensorLayout, TileTensor
from layout.tile_layout import row_major

from std.utils import StaticTuple

from .mxfp4_matmul_amd import MXFP4MatmulAMD as _MXFP4MatmulAMD
from .mxfp4_matmul_amd_preb import MXFP4MatmulAMD_PreB as _MXFP4MatmulAMD_PreB


struct PreShuffledBGroupedGEMM[
    cu_count: Int,
    wg_per_cu: Int = 2,
]:
    # This Grouped GEMM works when the weights B are shuffled into the layout
    # that allows coalsced reads from shared memory and direct MFMA usage

    comptime num_xcd = 8
    comptime total_wg = Self.cu_count * Self.wg_per_cu
    comptime wg_per_xcd = Self.total_wg // Self.num_xcd

    @always_inline
    @staticmethod
    def to_swizzled_idx(linear_idx: Int) -> Int:
        # If we have 10 blcoks and 8 xcd's the block scheduler assigns
        # a block to the xcd in this round robin fashion

        # XCD:         0 1 2 3 4 5 6 7

        # block_idx.x: 0 1 2 3 4 5 6 7
        # continued:   8 9

        # blocks get assigned in a round robin fashion to xcd's
        # to make sure that blocks in the same xcd work on cache
        # local blocks we remap the id's in this fashion

        # XCD:         0 1 2 3 4 5 6 7

        # block_idx.x: 0 2 4 5 6 7 8 9
        # continued:   1 3

        xcd_idx = linear_idx % Self.num_xcd
        xcd_linear_idx = linear_idx // Self.num_xcd
        return xcd_idx * Self.wg_per_xcd + xcd_linear_idx

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(
                _MXFP4MatmulAMD_PreB[
                    BM=BM,
                    BN=BN,
                    BK_ELEMS=BK_ELEMS,
                    WN=WN,
                    B_PREFETCH=True,
                ].num_threads
            )
        )
    )
    def persistent_kernel[
        BM: Int,
        BN: Int,
        BK_ELEMS: Int,
        WN: Int,
        out_dtype: DType,
        LayoutC: TensorLayout,
        LayoutA: TensorLayout,
        LayoutBPre: TensorLayout,
        LayoutSFA: TensorLayout,
        LayoutSFB: TensorLayout,
        AOffsetsLayout: TensorLayout,
        ExpertIdsLayout: TensorLayout,
        N: Int,
        K_BYTES: Int,
    ](
        c_tensor: TileTensor[mut=True, out_dtype, LayoutC, MutAnyOrigin],
        a_tensor: TileTensor[DType.uint8, LayoutA, ImmutAnyOrigin],
        b_pre_tensor: TileTensor[DType.uint8, LayoutBPre, ImmutAnyOrigin],
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
        comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
        comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

        comptime Kernel = _MXFP4MatmulAMD_PreB[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            B_PREFETCH=True,
        ]
        comptime K_SCALES = sfa_tensor.static_shape[1]
        comptime gx_n = ceildiv(N, BN)

        if N == 0 or num_active_experts == 0:
            return

        var linear_wg = Int(block_idx.x)
        var logical_wg = Self.to_swizzled_idx(linear_wg)

        # grid stride loop over available work tiles using the logical WG ID
        # essentially each WG processes one tile per grid stride iteration
        # grid stride is the total number of WGs (Self.total_wg)

        # the next tile we want to work on for this WG
        var target_tile = logical_wg

        # the current tile in the grid stride loop, all WGs start at tile 0
        var current_tile = 0

        for expert_slot in range(num_active_experts):
            var M = a_offsets[expert_slot + 1] - a_offsets[expert_slot]

            # No real work — skip this expert and don't update current_tile.
            # Skipped experts contribute zero tiles to the global counter,
            # so the invariant holds.
            if M == 0:
                continue

            var expert_id = expert_ids[expert_slot]

            # No real work — skip this expert and don't update current_tile.
            if expert_id == -1:
                continue

            # This expert has real work. Check if our target tile falls
            # within this expert's slice of the global counter.

            var m_count = ceildiv(Int(M), BM)
            var expert_work = m_count * gx_n
            var expert_end = current_tile + expert_work

            # Our target tile is beyond this expert's range — advance the
            # current_tile to the start of the next expert and continue.
            if target_tile >= expert_end:
                current_tile = expert_end
                continue

            # This expert has tile(s) for this WG. Hoist per-expert tile
            # descriptors out of the inner while loop.

            var a_start_row = a_offsets[expert_slot]

            var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)
            var a_ptr = a_tensor.ptr + a_start_row * UInt32(K_BYTES)
            var b_pre_ptr = b_pre_tensor.ptr + expert_id * Int32(N) * Int32(
                K_BYTES
            )
            var sfa_ptr = sfa_tensor.ptr + a_start_row * UInt32(K_SCALES)
            var sfb_ptr = sfb_tensor.ptr + expert_id * Int32(N) * Int32(
                K_SCALES
            )

            var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
            var a_tile = TileTensor(
                a_ptr, row_major(Coord(Int(M), Idx[K_BYTES]))
            )
            var b_pre_tile = TileTensor(
                b_pre_ptr, row_major(Coord(Idx[1], Idx[N * K_BYTES]))
            )
            var sfa_tile = TileTensor(
                sfa_ptr, row_major(Coord(Int(M), Idx[K_SCALES]))
            )
            var sfb_tile = TileTensor(sfb_ptr, row_major[N, K_SCALES]())

            # An expert may span multiple grid strides — iterate by stride
            # until we exit this expert's range.

            while target_tile < expert_end:
                var local = target_tile - current_tile
                # M-fast within (expert, n_tile): n is outer, m is inner.
                var n_tile = local // m_count
                var m_tile = local - n_tile * m_count

                Kernel.run[
                    out_dtype,
                    type_of(c_tile).LayoutType,
                    type_of(a_tile).LayoutType,
                    type_of(b_pre_tile).LayoutType,
                    type_of(sfa_tile).LayoutType,
                    type_of(sfb_tile).LayoutType,
                    N,
                    K_BYTES,
                ](
                    c_tile,
                    a_tile,
                    b_pre_tile,
                    sfa_tile,
                    sfb_tile,
                    Int(n_tile),
                    Int(m_tile),
                )

                # advance to this WG's next claim in the global counter
                target_tile += Self.total_wg

            current_tile = expert_end

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(
                _MXFP4MatmulAMD_PreB[
                    BM=BM,
                    BN=BN,
                    BK_ELEMS=BK_ELEMS,
                    WN=WN,
                    B_PREFETCH=True,
                ].num_threads
            )
        )
    )
    def kernel[
        BM: Int,
        BN: Int,
        BK_ELEMS: Int,
        WN: Int,
        out_dtype: DType,
        LayoutC: TensorLayout,
        LayoutA: TensorLayout,
        LayoutBPre: TensorLayout,
        LayoutSFA: TensorLayout,
        LayoutSFB: TensorLayout,
        AOffsetsLayout: TensorLayout,
        ExpertIdsLayout: TensorLayout,
        N: Int,
        K_BYTES: Int,
    ](
        c_tensor: TileTensor[mut=True, out_dtype, LayoutC, MutAnyOrigin],
        a_tensor: TileTensor[DType.uint8, LayoutA, ImmutAnyOrigin],
        b_pre_tensor: TileTensor[DType.uint8, LayoutBPre, ImmutAnyOrigin],
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
        comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
        comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

        comptime Kernel = _MXFP4MatmulAMD_PreB[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            B_PREFETCH=True,
        ]
        comptime K_SCALES = sfa_tensor.static_shape[1]

        var M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
        if M == 0 or N == 0:
            return
        var expert_id = expert_ids[block_idx.z]
        if expert_id == -1:
            return
        if block_idx.y >= ceildiv(Int(M), BM):
            return

        var a_start_row = a_offsets[block_idx.z]

        var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)
        var a_ptr = a_tensor.ptr + a_start_row * UInt32(K_BYTES)
        var b_pre_ptr = b_pre_tensor.ptr + expert_id * Int32(N) * Int32(K_BYTES)
        var sfa_ptr = sfa_tensor.ptr + a_start_row * UInt32(K_SCALES)
        var sfb_ptr = sfb_tensor.ptr + expert_id * Int32(N) * Int32(K_SCALES)

        var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
        var a_tile = TileTensor(a_ptr, row_major(Coord(Int(M), Idx[K_BYTES])))
        var b_pre_tile = TileTensor(
            b_pre_ptr, row_major(Coord(Idx[1], Idx[N * K_BYTES]))
        )
        var sfa_tile = TileTensor(
            sfa_ptr, row_major(Coord(Int(M), Idx[K_SCALES]))
        )
        var sfb_tile = TileTensor(sfb_ptr, row_major[N, K_SCALES]())

        Kernel.run[
            out_dtype,
            type_of(c_tile).LayoutType,
            type_of(a_tile).LayoutType,
            type_of(b_pre_tile).LayoutType,
            type_of(sfa_tile).LayoutType,
            type_of(sfb_tile).LayoutType,
            N,
            K_BYTES,
        ](
            c_tile,
            a_tile,
            b_pre_tile,
            sfa_tile,
            sfb_tile,
            Int(block_idx.x),
            Int(block_idx.y),
        )

    # --------------------------------------------------------------------- #
    # Launch helper — picks persistent vs direct dispatch via comptime flag.
    # --------------------------------------------------------------------- #

    @staticmethod
    def launch[
        BM: Int,
        BN: Int,
        BK_ELEMS: Int,
        WN: Int,
        persistent: Bool,
    ](
        c: TileTensor[mut=True, ...],
        a: TileTensor[DType.uint8, ...],
        b_pre: TileTensor[DType.uint8, ...],
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
        comptime MatmulDeviceFunctionType = _MXFP4MatmulAMD_PreB[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            B_PREFETCH=True,
        ]

        comptime N = c.static_shape[1]
        comptime K_BYTES = a.static_shape[1]

        var a_i = TileTensor(
            a.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
            a.layout,
        )
        var b_pre_i = TileTensor(
            b_pre.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
            b_pre.layout,
        )
        var a_scales_i = TileTensor(
            a_scales.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
            a_scales.layout,
        )
        var b_scales_i = TileTensor(
            b_scales.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
            b_scales.layout,
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

        comptime if persistent:
            comptime kernel = Self.persistent_kernel[
                BM,
                BN,
                BK_ELEMS,
                WN,
                out_dtype,
                type_of(c).LayoutType,
                type_of(a_i).LayoutType,
                type_of(b_pre_i).LayoutType,
                type_of(a_scales_i).LayoutType,
                type_of(b_scales_i).LayoutType,
                type_of(a_off_i).LayoutType,
                type_of(expert_ids_i).LayoutType,
                N,
                K_BYTES,
            ]
            ctx.enqueue_function[kernel](
                c,
                a_i,
                b_pre_i,
                a_scales_i,
                b_scales_i,
                a_off_i,
                expert_ids_i,
                num_active_experts,
                grid_dim=(Self.total_wg, 1, 1),
                block_dim=MatmulDeviceFunctionType.num_threads,
            )
        else:
            comptime kernel = Self.kernel[
                BM,
                BN,
                BK_ELEMS,
                WN,
                out_dtype,
                type_of(c).LayoutType,
                type_of(a_i).LayoutType,
                type_of(b_pre_i).LayoutType,
                type_of(a_scales_i).LayoutType,
                type_of(b_scales_i).LayoutType,
                type_of(a_off_i).LayoutType,
                type_of(expert_ids_i).LayoutType,
                N,
                K_BYTES,
            ]
            ctx.enqueue_function[kernel](
                c,
                a_i,
                b_pre_i,
                a_scales_i,
                b_scales_i,
                a_off_i,
                expert_ids_i,
                num_active_experts,
                grid_dim=(
                    ceildiv(N, BN),
                    ceildiv(max_num_tokens_per_expert, BM),
                    num_active_experts,
                ),
                block_dim=MatmulDeviceFunctionType.num_threads,
            )


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

    var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
    var a_tile = TileTensor(a_ptr, row_major(Coord(Int(M), Idx[K_BYTES])))
    var b_tile = TileTensor(b_ptr, row_major[N, K_BYTES]())
    var sfa_tile = TileTensor(sfa_ptr, row_major(Coord(Int(M), Idx[K_SCALES])))
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


def mxfp4_grouped_matmul_amd_preb(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.uint8, ...],
    b_pre: TileTensor[DType.uint8, ...],
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
    estimated_total_m: Int = 0,
) raises:
    # TODO temporary dispatcher refactor to use proper dispatch table in the
    # future

    comptime assert (
        b_pre.flat_rank == 2 or b_pre.flat_rank == 3
    ), "b_pre must be rank-2 (flat) or rank-3 ([E, N, K_BYTES])"
    comptime num_experts = b_pre.static_shape[0]
    comptime m_threshhold = 4096

    comptime b_per_expert_bytes = (
        b_pre.static_shape[1] if b_pre.flat_rank
        == 2 else b_pre.static_shape[1] * b_pre.static_shape[2]
    )

    comptime N = c.static_shape[1]
    comptime packed_K = a.static_shape[1]

    comptime assert (
        b_per_expert_bytes == N * packed_K
    ), "b_pre shape mismatch with c.N and a.K"

    # TODO: drop once we generalize the persistent grid + cu_count derivation
    # across other AMD CDNA parts.
    comptime assert (
        ctx.default_device_info == MI355X
    ), "preb path currently only supports MI355X"

    comptime PreBGrouppedGemmType = PreShuffledBGroupedGEMM[
        cu_count=ctx.default_device_info.sm_count
    ]

    comptime can_use_bk_512 = packed_K >= 256 and packed_K % 256 == 0
    var use_direct = estimated_total_m >= m_threshhold
    comptime if can_use_bk_512:
        if use_direct:
            PreBGrouppedGemmType.launch[
                BM=64, BN=128, BK_ELEMS=512, WN=64, persistent=False
            ](
                c,
                a,
                b_pre,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
        else:
            PreBGrouppedGemmType.launch[
                BM=64, BN=128, BK_ELEMS=512, WN=64, persistent=True
            ](
                c,
                a,
                b_pre,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
    else:
        if use_direct:
            PreBGrouppedGemmType.launch[
                BM=64, BN=128, BK_ELEMS=128, WN=64, persistent=False
            ](
                c,
                a,
                b_pre,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
        else:
            PreBGrouppedGemmType.launch[
                BM=64, BN=128, BK_ELEMS=128, WN=64, persistent=True
            ](
                c,
                a,
                b_pre,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
