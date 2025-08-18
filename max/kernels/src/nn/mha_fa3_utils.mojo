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
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from gpu.sync import async_copy_arrive
import gpu.warp as warp
from layout.int_tuple import IntTuple
from layout.layout import Layout, UNKNOWN_VALUE
from layout.layout_tensor import (
    LayoutTensor,
    cp_async_k_major,
    cp_async_mn_major,
)
from gpu.host._nvidia_cuda import TensorMapSwizzle
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMANestedTensorTile,
    create_nested_tma_tile,
)
from layout.tensor_core_async import tile_layout_k_major
from math import ceildiv
from math.constants import log2e
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
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
from sys import sizeof


@register_passable("trivial")
struct MHAPosition[
    BM: Int, BN: Int, depth: Int, q_num_heads: Int, group: Int, decoding: Bool
](Copyable, Movable):
    """
    Position of the MHA-kernel.
    When `decoding=False`, `q_head_stride == q_num_heads`.
    When `decoding=True`, `q_head_stride == 1`.
    """

    var q_row: UInt32
    var q_col: UInt32
    var q_out_offset: Int
    var num_keys: UInt32
    var start_pos: UInt32
    var seq_len: UInt32
    var head_idx: UInt32  # when decoding, kv_head_idx
    var prompt_offset: UInt32  # when decoding, this is the position_idx
    var prompt_idx: UInt32

    alias q_stride: Int = Self.depth if decoding else Self.depth * Self.q_num_heads
    alias q_output_gmem_layout = Layout(
        IntTuple(Self.BM, Self.depth), IntTuple(Self.q_stride, 1)
    )

    @always_inline
    fn __init__(
        out self,
        q_row: UInt32,
        q_col: UInt32,
        q_out_offset: Int,
        num_keys: UInt32,
        start_pos: UInt32,
        seq_info: SeqInfo,
    ):
        self.q_row = q_row
        self.q_col = q_col
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
        exp_sum_offset = Self.q_num_heads * (
            self.prompt_idx + batch_size * self.prompt_offset
        )
        exp_sum_ptr = partition.get_exp_sum_qk_max_pointer().offset(
            exp_sum_offset
        )
        qk_max_ptr = exp_sum_ptr.offset(
            Self.q_num_heads * batch_size * partition.num_partitions()
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
    KVLUTType: MHAOperand,
    MaxSeqLenType: OptionallyStaticInt, //,
    BM: Int,
    BN: Int,
    depth: Int,
    q_num_heads: Int,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    out ret: MHAPosition[
        BM,
        BN,
        depth,
        q_num_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ],
    seq_info: SeqInfo,
    kv_lut: KVLUTType,
    max_seq_len: MaxSeqLenType,
    num_keys_arg: UInt32,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
):
    var batch_idx: UInt32 = seq_info.prompt_idx
    # mha inputs
    var seq_len: UInt32 = seq_info.seq_len
    var num_keys: UInt32
    var start_pos: UInt32
    var q_row: UInt32

    @parameter
    if ragged:
        cache_len = kv_lut.cache_length(Int(batch_idx))

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
        q_row = seq_info.start_of_seq

    # NDBuffer inputs, homogeneous batching.
    else:
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len
        q_row = batch_idx * max_seq_len.as_uint32()

    var q_offset: Int
    var q_col: UInt32

    @parameter
    if _is_decoding[MaxSeqLenType]():
        # q matrix view is rows x depth
        q_row = q_row * q_num_heads + seq_info.head_idx * group
        q_col = 0
        q_offset = Int(depth) * Int(q_row)
    else:  # head_idx is for q_heads
        # q matrix view is rows x (depth*q_num_heads)
        q_row += seq_info.prompt_offset
        q_col = seq_info.head_idx * depth
        q_offset = Int(depth * q_num_heads) * Int(q_row) + Int(q_col)
    ret = __type_of(ret)(
        q_row,
        q_col,
        q_offset,
        num_keys,
        start_pos,
        seq_info,
    )


@always_inline
fn q_out_tma[
    dtype: DType, //,
    BM: Int,
    depth: Int,
    swizzle_mode: TensorMapSwizzle,
    *,
    q_num_heads: Int,
    decoding: Bool,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype]],
    rows: Int,
    out res: TMANestedTensorTile[
        dtype,
        max(8, BM),
        64 if decoding else depth,
        swizzle_mode,
        is_k_major=True,
    ],
) raises:
    alias tile_cols: Int = 64 if decoding else depth
    alias matrix_cols: Int = depth if decoding else depth * q_num_heads

    alias layout = Layout.row_major(UNKNOWN_VALUE, matrix_cols)
    rt_layout = RuntimeLayout[layout].row_major(IndexList[2](rows, matrix_cols))
    var tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](ptr, rt_layout)

    return create_nested_tma_tile[
        max(8, BM), tile_cols, swizzle_mode, is_k_major=True
    ](ctx, tensor)


@always_inline("nodebug")
fn _produce[
    smem_layout: Layout,
    kv_t: MHAOperand,
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    swizzle_mode: TensorMapSwizzle,
    is_k_major: Bool, //,
    kv_num_heads: Int,
    *,
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
    tma_tile: TMANestedTensorTile[
        kv_t.dtype, BN, depth, swizzle_mode, is_k_major=is_k_major
    ],
    smem_tile: LayoutTensor[
        kv_t.dtype,
        smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
    ],
):
    src_col = kv.col_idx(position.kv_head_idx())
    src_row = kv.row_idx(position.prompt_idx, kv_tile_start_row)

    @parameter
    if wait:
        consumed_mbar[write_idx].wait(write_phase)

    p_mbar = produced_mbar + write_idx
    p_mbar[0].expect_bytes(BN * depth * sizeof[kv_t.dtype]())
    tma_tile.async_copy(smem_tile, p_mbar[0], (UInt(src_col), UInt(src_row)))


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


@always_inline
fn produce[
    qkv_type: DType,
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int,
    group: Int,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
    q_tma_rows: Int,
    q_tma_cols: Int,
    MaxSeqLenType: OptionallyStaticInt,
    SchedulerType: MHATileScheduler,
    KVLUTType: MHAOperand,
    MaskType: MHAMask, //,
    *,
    pipeline_stages: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    q_tma_op: TMANestedTensorTile[
        qkv_type,
        q_tma_rows,
        q_tma_cols,
        swizzle_mode,
        is_k_major=True,
    ],
    k_tma_op: TMANestedTensorTile[
        qkv_type,
        BN,
        depth,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        qkv_type,
        BN,
        depth,
        swizzle_mode,
        is_k_major=False,
    ],
    q_smem: UnsafePointer[
        Scalar[qkv_type], address_space = AddressSpace.SHARED, alignment=128
    ],
    kv_smem: UnsafePointer[
        Scalar[qkv_type], address_space = AddressSpace.SHARED, alignment=128
    ],
    produced_mbar_kv: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
    ],
    consumed_mbar_kv: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
    ],
    produced_mbar_q: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
    ],
    consumed_mbar_q: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=8
    ],
    kv_lut: KVLUTType,
    initial_position: MHAPosition[
        BM, BN, depth, num_heads, group, _is_decoding[MaxSeqLenType]()
    ],
    partition: PartitionType,
    scheduler: SchedulerType,
    mask: MaskType,
    tile_summary: MHATileSummary,
    tile_state_arg: MHATileState,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
):
    alias decoding: Bool = _is_decoding[MaxSeqLenType]()
    alias PositionType = MHAPosition[BM, BN, depth, num_heads, group, decoding]
    alias persistent = SchedulerType.may_advance

    alias q_smem_layout_producer = q_tma_op.layout
    alias q_smem_layout_consumer = tile_layout_k_major[
        DType.bfloat16, BM, depth, swizzle_mode=swizzle_mode
    ]()
    alias k_smem_layout = k_tma_op.layout
    alias v_smem_layout = v_tma_op.layout

    alias q_size = q_smem_layout_consumer.size()
    alias q_smem_size = (2 * q_size if persistent else q_size)

    alias q_copy_rows = max(group, 8) if decoding else Int(BM)
    alias qk_bytes = (q_copy_rows + BN) * depth * sizeof[qkv_type]()

    tile_state = tile_state_arg
    position = initial_position

    @parameter
    @always_inline("nodebug")
    fn q_producer[
        depth_idx: Int
    ](
        q_idx: UInt32,
    ) -> LayoutTensor[
        qkv_type,
        q_smem_layout_producer,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]:
        # alias stride = q_smem_layout_consumer.stride[1][1].value()
        # alias depth_offset = depth_idx * stride
        alias depth_offset = q_smem_layout_consumer(
            IntTuple(0, 64 * depth_idx)
        ) if decoding else 0
        return {q_smem + depth_offset + q_size * q_idx}

    @parameter
    @always_inline
    fn k_tile(
        idx: UInt32,
        out k_smem: LayoutTensor[
            qkv_type,
            k_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            alignment=128,
        ],
    ):
        alias sz = BN * depth
        k_smem = __type_of(k_smem)(kv_smem + sz * idx)

    @parameter
    @always_inline
    fn v_tile(
        idx: UInt32,
        out v_smem: LayoutTensor[
            qkv_type,
            v_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            alignment=128,
        ],
    ):
        alias sz = BN * depth
        v_smem = __type_of(v_smem)(kv_smem + sz * idx)

    @parameter
    @always_inline("nodebug")
    fn produce_k[
        wait: Bool
    ](mut state: PipelineState[pipeline_stages], row: UInt32, col: UInt32):
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]
        k_sub = k_tile(write_idx)

        @parameter
        if wait:
            consumed_mbar_kv[write_idx].wait(write_phase)
            alias bytes = BN * depth * sizeof[qkv_type]()
            p_mbar.expect_bytes(bytes)
        k_tma_op.async_copy(k_sub, p_mbar, (UInt(col), UInt(row)))
        state.step()

    @parameter
    @always_inline("nodebug")
    fn produce_v(
        mut state: PipelineState[pipeline_stages], row: UInt32, col: UInt32
    ):
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]
        v_sub = v_tile(write_idx)
        consumed_mbar_kv[write_idx].wait(write_phase)
        alias bytes = BN * depth * sizeof[qkv_type]()
        p_mbar.expect_bytes(bytes)
        v_tma_op.async_copy(v_sub, p_mbar, (UInt(col), UInt(row)))
        state.step()

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM, BN, depth, num_heads, group, ragged, _is_cache_length_accurate
        ](
            seq_info,
            kv_lut,
            max_seq_len,
            num_keys_arg,
            kv_input_row_offsets,
        )

    write_pipeline_states = PipelineState[pipeline_stages]()
    q_pipeline_state = PipelineState[2 if persistent else 1]()

    @parameter
    if PartitionType.do_partition:
        startend = position.get_start_and_end_for_partitions[BN=BN](partition)
        start = startend[0]
        end = startend[1]
        if start >= end:
            return
    else:
        # delay partitioning until after we've begun copying `q`
        start = 0
        end = 0

    produced_mbar_kv[0].expect_bytes(qk_bytes)

    @parameter
    for d in range((depth // 64) if decoding else 1):
        q_tma_op.async_copy(
            q_producer[d](q_pipeline_state.index()),
            produced_mbar_kv[0],
            (UInt(position.q_col + 64 * d), UInt(position.q_row)),
        )

    @parameter
    if not PartitionType.do_partition:
        startend = position.get_start_and_end_for_partitions[BN=BN](partition)
        start = startend[0]
        end = startend[1]
    var kv_tile_start_row: UInt32 = start
    var kv_col: UInt32 = kv_lut.col_idx(position.kv_head_idx())

    while (
        position.mask_status(mask, kv_tile_start_row)
        == TileMaskStatus.FULL_MASK
    ):
        kv_tile_start_row += BN

    var kv_row: UInt32 = kv_lut.row_idx(position.prompt_idx, kv_tile_start_row)

    produce_k[False](write_pipeline_states, kv_row, kv_col)

    var kv_row_prev: UInt32 = kv_row
    var kv_col_prev: UInt32 = kv_col

    # wait to flip phase, but only bother after producing
    # there isn't any memory we can throttle
    # the order of the consumer's arrivals determines the
    # order of the producer's waits.
    # few_keys = num_keys <= BN

    # Process work with the tile size until there's not enough remaining work
    # to fit in a tile.
    # Production order:
    # Preheader: Q0, K0
    # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
    # Exit: V{-1}
    while True:
        # this loops over num_keys
        kv_tile_start_row += BN
        if kv_tile_start_row >= end:

            @parameter
            if persistent:
                kv_tile_start_row = 0
                var q_idx_old: UInt32 = q_pipeline_state.index()
                var q_phase_old: UInt32 = q_pipeline_state.phase()
                q_pipeline_state.step()
                consumed_mbar_q[q_idx_old].wait(q_phase_old)
                # we must wait before advancing, as this mbar
                # is for both `q_smem` and `sidx_ptr`
                var q_idx: UInt32 = q_pipeline_state.index()
                docontinue = scheduler.advance[
                    ragged, True, MHASchedulerSynchronization.DEFAULT
                ](tile_summary, tile_state, q_idx_old)
                # FIXME: persistent kernel that uses a counter
                # must signal somehow
                if not docontinue:
                    break
                ref pq_mbar = produced_mbar_q[q_idx_old]
                position = get_position(docontinue.value())
                pq_mbar.expect_bytes(q_copy_rows * depth * sizeof[qkv_type]())

                @parameter
                for d in range((depth // 64) if decoding else 1):
                    q_tma_op.async_copy(
                        q_producer[d](q_idx),
                        pq_mbar,
                        (
                            UInt(position.q_col + 64 * d),
                            UInt(position.q_row),
                        ),
                    )
                kv_col = kv_lut.col_idx(position.kv_head_idx())
                start, new_end = position.get_start_and_end_for_partitions[
                    BN=BN
                ](partition)
                kv_tile_start_row = start
                end = new_end
            else:
                break

        if (
            position.mask_status(mask, kv_tile_start_row)
            == TileMaskStatus.FULL_MASK
        ):
            continue
        kv_row = kv_lut.row_idx(position.prompt_idx, kv_tile_start_row)
        produce_k[True](write_pipeline_states, kv_row, kv_col)
        produce_v(write_pipeline_states, kv_row_prev, kv_col_prev)
        kv_row_prev = kv_row
        kv_col_prev = kv_col

    produce_v(write_pipeline_states, kv_row_prev, kv_col_prev)
