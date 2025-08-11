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

from algorithm.functional import unswitch
from buffer import NDBuffer
from collections import OptionalReg
from gpu import thread_idx
from gpu.memory import AddressSpace
from gpu.sync import async_copy_arrive
import gpu.warp as warp
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    cp_async_k_major,
    cp_async_mn_major,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.tma_async import SharedMemBarrier
from math import ceildiv
from math.constants import log2e
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    SeqInfo,
)
from nn.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from tensor_internal import ManagedTensorSlice
from utils.index import Index, IndexList


@register_passable("trivial")
struct MHAPosition[
    BM: Int, BN: Int, depth: Int, num_heads: Int, group: Int, decoding: Bool
](Copyable, Movable):
    """
    Position of the MHA-kernel.
    When `decoding=False`, `q_head_stride == num_heads`.
    When `decoding=True`, `q_head_stride == 1`.
    """

    var q_out_offset: Int
    var num_keys: UInt32
    var start_pos: UInt32
    var seq_len: UInt32
    var head_idx: UInt32  # when decoding, kv_head_idx
    var prompt_offset: UInt32  # when decoding, this is the position_idx
    var prompt_idx: UInt32

    alias q_stride: Int = Self.depth if decoding else Self.depth * Self.num_heads
    alias q_output_gmem_layout = Layout(
        IntTuple(Self.BM, Self.depth), IntTuple(Self.q_stride, 1)
    )

    @always_inline
    fn __init__(
        out self,
        q_out_offset: Int,
        num_keys: UInt32,
        start_pos: UInt32,
        seq_info: SeqInfo,
    ):
        self.q_out_offset = q_out_offset
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.seq_len = seq_info.seq_len
        self.head_idx = seq_info.head_idx
        self.prompt_offset = seq_info.prompt_offset
        self.prompt_idx = seq_info.prompt_idx  # batch idx

    @always_inline
    fn q_head_idx(self) -> UInt32:
        @parameter
        if Self.decoding:
            return self.head_idx * Self.group
        else:
            return self.head_idx

    @always_inline
    fn kv_head_idx(self) -> UInt32:
        @parameter
        if Self.decoding:
            return self.head_idx
        else:
            return self.head_idx // Self.group

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "(",
            self.q_out_offset,
            ", ",
            self.seq_len,
            ", ",
            self.num_keys,
            ", ",
            self.start_pos,
            ", ",
            self.prompt_offset,
            ", ",
            self.head_idx,
            ", ",
            self.prompt_idx,
            ")",
        )

    @always_inline
    fn q_tile_num_rows(self) -> UInt32:
        @parameter
        if decoding:
            return Self.group
        else:
            return min(BM, self.seq_len - self.prompt_offset)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.q_out_offset == other.q_out_offset

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self.q_out_offset != other.q_out_offset

    @always_inline
    fn q_out_gmem_tensor[
        dtype: DType
    ](
        self,
        ptr: UnsafePointer[Scalar[dtype]],
        out gmem_block: LayoutTensor[
            dtype,
            Self.q_output_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        gmem_block = __type_of(gmem_block)(
            ptr + self.q_out_offset,
            __type_of(gmem_block.runtime_layout)(
                __type_of(gmem_block.runtime_layout.shape)(
                    Int(self.q_tile_num_rows()), depth
                ),
                __type_of(gmem_block.runtime_layout.stride)(Self.q_stride, 1),
            ),
        )

    @always_inline
    fn mask_status[
        mask_t: MHAMask
    ](self, mask: mask_t, kv_tile_start_row: UInt32) -> TileMaskStatus:
        @parameter
        if decoding:

            @parameter
            if mask_t.check_mask_during_decoding:
                # In context encoding, we have BM rows of Q
                # In decoding, we have `group` rows, but these
                # correspond to the same position w/ respect to the mask.
                return mask.status(
                    Index[dtype = DType.int32](
                        Int(self.num_keys - 1),
                        Int(kv_tile_start_row),
                    ),
                    Index[dtype = DType.int32](Int(1), Int(Self.BN)),
                )
            else:
                return TileMaskStatus.PARTIAL_MASK
        else:
            return mask.status(
                Index[dtype = DType.int32](
                    Int(self.prompt_offset + self.start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.int32](Int(Self.BM), Int(Self.BN)),
            )

    @always_inline
    fn exp_sum_qk_max_ptr[
        partition_t: MHAPartitionScheme
    ](
        self,
        partition: partition_t,
        batch_size: UInt32,
    ) -> Tuple[
        UnsafePointer[Scalar[partition_t.accum_dtype]],
        UnsafePointer[Scalar[partition_t.accum_dtype]],
    ]:
        exp_sum_offset = Self.num_heads * (
            self.prompt_idx + batch_size * self.prompt_offset
        )
        exp_sum_ptr = partition.get_exp_sum_qk_max_pointer().offset(
            exp_sum_offset
        )
        qk_max_ptr = exp_sum_ptr.offset(
            Self.num_heads * batch_size * partition.num_partitions()
        )
        return (exp_sum_ptr, qk_max_ptr)

    @always_inline
    fn get_start_and_end_for_partitions[
        partition_t: MHAPartitionScheme, //, BN: Int
    ](self, partition: partition_t) -> Tuple[UInt32, UInt32]:
        @parameter
        if partition_t.do_partition:
            start, end = get_start_and_end_for_partitions[BN](
                Int(self.num_keys),
                Int(partition.num_partitions()),
                Int(self.prompt_offset),
            )
            return (UInt32(start), UInt32(end))
        else:
            return (UInt32(0), self.num_keys)


@always_inline
fn _get_position[
    k_t: MHAOperand,
    max_seq_len_t: OptionallyStaticInt, //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    out ret: MHAPosition[
        config.block_m(),
        config.block_n(),
        config.depth,
        config.num_heads,
        group,
        _is_decoding[max_seq_len_t](),
    ],
    seq_info: SeqInfo,
    k: k_t,
    max_seq_len: max_seq_len_t,
    num_keys_arg: UInt32,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
):
    alias depth = config.depth
    alias num_heads = config.num_heads

    var batch_idx: UInt32 = seq_info.prompt_idx
    # mha inputs
    var seq_len: UInt32 = seq_info.seq_len
    var num_keys: UInt32
    var start_pos: UInt32
    var q_offset: Int

    @parameter
    if ragged:
        cache_len = k.cache_length(Int(batch_idx))

        @parameter
        if not _is_cache_length_accurate:
            start_pos = cache_len
        else:
            start_pos = 0

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[Int(batch_idx)])
            kv_seq_end = Int(kv_row_offsets[Int(batch_idx) + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + cache_len
        else:
            num_keys = seq_len + cache_len
        q_offset = Int(seq_info.start_of_seq) * Int(num_heads)

    # NDBuffer inputs, homogeneous batching.
    else:
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len
        q_offset = Int(num_heads * batch_idx) * Int(max_seq_len)

    var kv_head_idx: UInt32

    @parameter
    if _is_decoding[max_seq_len_t]():
        q_offset += Int(seq_info.head_idx) * group
    else:  # head_idx is for q_heads
        q_offset += Int(seq_info.head_idx) + Int(seq_info.prompt_offset) * Int(
            num_heads
        )
    ret = __type_of(ret)(
        Int(depth) * q_offset,
        num_keys,
        start_pos,
        seq_info,
    )


@always_inline("nodebug")
fn _produce[
    smem_layout: Layout,
    kv_t: MHAOperand,
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool, //,
    kv_num_heads: Int,
    *,
    axis: Int,
    wait: Bool,
](
    write_idx: UInt32,
    write_phase: UInt32,
    kv_tile_start_row: UInt32,
    position: MHAPosition[BM, BN, depth, num_heads, group, decoding],
    consumed_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    produced_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    kv: kv_t,
    smem_tile: LayoutTensor[
        kv_t.dtype,
        smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
    ],
):
    kv_tile_num_rows = min(
        Int(BN), Int(position.num_keys) - Int(kv_tile_start_row)
    )
    alias kv_gmem_layout = Layout(
        IntTuple(Int(BN), Int(depth)),
        IntTuple(Int(kv_num_heads * depth), 1),
    )
    alias kv_runtime_layout_t = RuntimeLayout[
        kv_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    kv_runtime_layout = kv_runtime_layout_t(
        RuntimeTuple[
            kv_gmem_layout.shape,
            element_type = kv_runtime_layout_t.element_type,
        ](kv_tile_num_rows, depth),
        RuntimeTuple[
            kv_gmem_layout.stride,
            element_type = kv_runtime_layout_t.linear_idx_type,
        ](kv_num_heads * depth, 1),
    )

    # Int(batch_idx), # prompt_idx
    # Int(start_tok_idx), # Int(kv_tile_start_row)
    # Int(head_idx),   # Int(position.kv_head_idx())
    # Int(head_dim_idx), # 0
    gmem_ptr = kv.block_paged_ptr[BN](
        position.prompt_idx,
        Int(kv_tile_start_row),
        Int(position.kv_head_idx()),
        0,
    )

    @parameter
    if wait:
        consumed_mbar[write_idx].wait(write_phase)

    @parameter
    @always_inline("nodebug")
    fn copy_gmem_to_smem[masked: Bool]():
        gmem_block = LayoutTensor[
            kv_t.dtype,
            kv_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=masked,
        ](gmem_ptr, kv_runtime_layout)

        @parameter
        if axis == 0:
            cp_async_mn_major(smem_tile, gmem_block)
        else:
            cp_async_k_major(smem_tile, gmem_block)

    # if kv_tile_num_rows >= BN:
    #     copy_gmem_to_smem[False]()
    # else:
    #     copy_gmem_to_smem[True]()
    unswitch[copy_gmem_to_smem](kv_tile_num_rows < BN)

    p_mbar = produced_mbar + write_idx
    async_copy_arrive(p_mbar)
    _ = p_mbar[].arrive()


@always_inline
fn _apply_mask[
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    accum_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    reg_tile_layout: Layout,
    element_layout: Layout, //,
    # last_iter: Bool,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    use_score_mod: Bool,
](
    mask_warp_row_arg: UInt32,
    position: MHAPosition[BM, BN, depth, num_heads, group, decoding],
    lane: UInt32,
    max_seq_len: UInt32,
    scale_log2e: Scalar[accum_type],
    kv_tile_start_row: UInt32,
    mask: mask_t,
    mask_status: TileMaskStatus,
    score_mod: score_mod_t,
    p_reg_tile: LayoutTensor[
        accum_type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=element_layout,
    ],
):
    alias num_groups_per_thread = min(2, ceildiv(group, 8)) if decoding else 2
    var batch_cache_valid_length: UInt32

    @parameter
    if decoding:
        if warp.broadcast((thread_idx.x - 128) // 32) > ((group - 1) // 16):
            return
        if lane >= 4 * group:
            return
        batch_cache_valid_length = position.num_keys - 1
    else:
        batch_cache_valid_length = 0

    alias p_frag_simdwidth = element_layout.size()
    # Vectorize by 2.
    var fragment_row: UInt32 = lane // 4
    var fragment_col: UInt32 = (lane * p_frag_simdwidth % WN) % 8
    # Offset to current thread's fragment
    var mask_warp_row: UInt32 = mask_warp_row_arg + fragment_row
    var mask_warp_col: UInt32 = kv_tile_start_row + fragment_col

    @parameter
    @always_inline
    fn _apply_mask_capture[masked: Bool]():
        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                # Coordinates in mask for current mma tile.
                mask_frag_row = mask_warp_row + m_mma * WM
                mask_frag_col = mask_warp_col + n_mma * WN

                @parameter
                for i in range(num_groups_per_thread):
                    var q_head_idx: UInt32 = position.head_idx

                    @parameter
                    if decoding:
                        group_idx = i * 8 + fragment_row
                        q_head_idx = group * q_head_idx + group_idx
                    # The row in score matrix of shape seq_len x num_keys.
                    # Mask col is score col since we don't partition in col.
                    var score_row: UInt32
                    var score_row_with_start_pos: UInt32

                    @parameter
                    if decoding:
                        score_row = batch_cache_valid_length
                        score_row_with_start_pos = score_row
                    else:
                        score_row = (
                            position.prompt_offset + mask_frag_row + i * WM // 2
                        )
                        score_row_with_start_pos = (
                            score_row + position.start_pos
                        )

                    @parameter
                    for j in range(WN // 8):
                        score_col = mask_frag_col + j * 8
                        p = p_reg_tile[i, m_mma, j, n_mma]

                        @parameter
                        if masked:
                            p = mask.mask(
                                IndexList[4, element_type = DType.uint32](
                                    Int(position.prompt_idx),
                                    Int(q_head_idx),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p * scale_log2e,
                            )
                        else:
                            p *= scale_log2e

                        @parameter
                        if use_score_mod:
                            p = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32](
                                        Int(position.prompt_idx),
                                        Int(q_head_idx),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p,
                                    Int(max_seq_len),
                                )
                                * log2e
                            )
                        elif mask_t.apply_log2e_after_mask:
                            p *= log2e

                        var bound: IndexList[2, element_type = DType.uint32]

                        @parameter
                        if decoding:
                            bound = IndexList[2, element_type = DType.uint32](
                                Int(position.num_keys),
                                Int(
                                    min(
                                        BN + kv_tile_start_row,
                                        position.num_keys,
                                    )
                                ),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        elif masked:
                            bound = IndexList[2, element_type = DType.uint32](
                                Int(position.seq_len),
                                Int(position.num_keys),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        p_reg_tile[i, m_mma, j, n_mma] = p

    @parameter
    if decoding:
        _apply_mask_capture[True]()
    else:
        unswitch[_apply_mask_capture](
            (mask_status == TileMaskStatus.PARTIAL_MASK)
            # NOTE: mask_status should be either PARTIAL_MASK or NO_MASK at
            # this point.
            # In the NO_MASK case, we still need to mask out the scores for the
            # last tile, which goes beyond num_keys (for num_keys % 128 != 0).
            or (BN + kv_tile_start_row > position.num_keys)
        )


# TODO: Remove this when we're no longer using NDBuffers.
@always_inline
fn valid_length_managed_tensor_slice_to_ndbuffer(
    tensor: ManagedTensorSlice[dtype = DType.uint32, rank=1]
) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
    var ptr = tensor._ptr.address_space_cast[AddressSpace.GENERIC]()
    return NDBuffer[DType.uint32, 1, MutableAnyOrigin](
        ptr, tensor.shape(), tensor._runtime_strides
    )
