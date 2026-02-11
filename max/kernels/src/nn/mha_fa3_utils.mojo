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

from collections import OptionalReg
from math import ceildiv, align_up
from math.constants import log2e
from memory import (
    bitcast,
)

from sys import size_of

import gpu.primitives.warp as warp
from algorithm.functional import unswitch
from gpu import block_idx, thread_idx
from gpu.globals import WARPGROUP_SIZE
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.compute.mma import st_matrix
from gpu.sync import async_copy_arrive
from layout.int_tuple import IntTuple
from layout.layout import UNKNOWN_VALUE, Layout
from layout._layout import Layout as InternalLayout, row_major
from layout._tile_tensor import TileTensor
from layout.layout_tensor import (
    LayoutTensor,
    copy_local_to_shared,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle
from layout.tensor_core_async import st_matrix_n_layout, tile_layout_k_major
from layout.tma_async import (
    create_split_tma,
    PipelineState,
    RaggedTMA3DTile,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    _split_last_layout,
    TMATensorTile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from nn.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from builtin.device_passable import DevicePassable
from utils import StaticTuple


comptime _LocalTT[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types = layout.shape_types,
        stride_types = layout.stride_types,
    ],
    MutAnyOrigin,
    address_space = AddressSpace.LOCAL,
]
comptime _SharedMemTT[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types = layout.shape_types,
        stride_types = layout.stride_types,
    ],
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
]


trait OptionalPointer(Copyable, TrivialRegisterPassable):
    comptime dtype: DType
    comptime is_null: Bool

    @always_inline
    fn value(self) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        ...


struct NonNullPointer[dtype_: DType](OptionalPointer):
    comptime dtype: DType = Self.dtype_
    comptime is_null: Bool = False

    var ptr: UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]

    @always_inline
    fn __init__(
        out self, ptr: UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]
    ):
        self.ptr = ptr

    @always_inline
    fn __init__(out self, ptr: DeviceBuffer[Self.dtype]):
        self.ptr = ptr.unsafe_ptr()

    @always_inline
    fn value(self) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        debug_assert(
            Bool(self.ptr),
            (
                "NonNullPointer is supposed to provide a compile-time guarantee"
                " of being non-null"
            ),
        )
        return self.ptr


struct NullPointer[dtype_: DType](OptionalPointer):
    comptime dtype: DType = Self.dtype_
    comptime is_null: Bool = True

    @always_inline
    fn __init__(out self):
        pass

    @always_inline
    fn value(self) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        return {}


struct Pack[
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](Copyable, DevicePassable, TrivialRegisterPassable):
    var mask: Self.MaskType
    var score_mod: Self.ScoreModType
    var scheduler: Self.SchedulerType
    var valid_length: Self.ValidLengthType
    var sink_weights: Self.SinkType
    var kv_input_row_offsets: Self.KVRowOffsetsType
    var max_seq_len: Self.MaxSeqLenType
    var partition: Self.PartitionType

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "Pack"

    @always_inline
    fn __init__(
        out self,
        mask: Self.MaskType,
        score_mod: Self.ScoreModType,
        scheduler: Self.SchedulerType,
        valid_length: Self.ValidLengthType,
        sink_weights: Self.SinkType,
        kv_input_row_offsets: Self.KVRowOffsetsType,
        max_seq_len: Self.MaxSeqLenType,
        partition: Self.PartitionType,
    ):
        self.mask = mask
        self.score_mod = score_mod
        self.scheduler = scheduler
        self.valid_length = valid_length
        self.sink_weights = sink_weights
        self.kv_input_row_offsets = kv_input_row_offsets
        self.max_seq_len = max_seq_len
        self.partition = partition


struct MHAPosition[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    q_num_heads: Int,
    group: Int,
    decoding: Bool,
](TrivialRegisterPassable):
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

    comptime q_stride: Int = Self.depth if Self.decoding else Self.depth * Self.q_num_heads
    comptime q_output_gmem_layout = Layout(
        IntTuple(Self.BM, Self.depth), IntTuple(Self.q_stride, 1)
    )
    comptime split_gmem_layout = Layout(
        IntTuple(Self.BM // 2, Self.depth), IntTuple(Self.q_stride, 1)
    )
    comptime num_q_heads_per_thread: Int = min(
        2, ceildiv(Self.group, 8)
    ) if Self.decoding else 1

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
            return self.head_idx * UInt32(Self.group)
        else:
            return self.head_idx

    @always_inline
    fn kv_head_idx(self) -> UInt32:
        @parameter
        if Self.decoding:
            return self.head_idx
        else:
            return self.head_idx // UInt32(Self.group)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
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
        if Self.decoding:
            return UInt32(Self.group)
        else:
            return min(self.seq_len - self.prompt_offset, UInt32(Self.BM))

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
            MutAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        gmem_block = {
            ptr + self.q_out_offset,
            type_of(gmem_block.runtime_layout)(
                type_of(gmem_block.runtime_layout.shape)(
                    Int(self.q_tile_num_rows()), Self.depth
                ),
                type_of(gmem_block.runtime_layout.stride)(Self.q_stride, 1),
            ),
        }

    @always_inline
    fn mask_status[
        MaskType: MHAMask
    ](self, mask: MaskType, kv_tile_start_row: UInt32) -> TileMaskStatus:
        @parameter
        if Self.decoding:

            @parameter
            if MaskType.check_mask_during_decoding:
                # In context encoding, we have BM rows of Q
                # In decoding, we have `group` rows, but these
                # correspond to the same position w/ respect to the mask.
                return mask.status(
                    Index[dtype = DType.int32](
                        Int(self.num_keys - 1),
                        Int(kv_tile_start_row),
                    ),
                    Index[dtype = DType.int32](Int(1), Self.BN),
                )
            else:
                return TileMaskStatus.PARTIAL_MASK
        else:
            return mask.status(
                Index[dtype = DType.int32](
                    Int(self.prompt_offset + self.start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.int32](Self.BM, Self.BN),
            )

    @always_inline
    fn get_score_row(self) -> UInt32:
        @parameter
        if Self.decoding:
            return self.num_keys - 1
        else:
            return self.prompt_offset + self.start_pos

    @always_inline
    fn exp_sum_qk_max_ptr[
        partition_t: MHAPartitionScheme
    ](
        self,
        partition: partition_t,
        batch_size: UInt32,
    ) -> Tuple[
        UnsafePointer[Scalar[partition_t.accum_dtype], MutAnyOrigin],
        UnsafePointer[Scalar[partition_t.accum_dtype], MutAnyOrigin],
    ]:
        exp_sum_offset = UInt32(Self.q_num_heads) * (
            self.prompt_idx + batch_size * self.prompt_offset
        )
        exp_sum_ptr = partition.get_exp_sum_qk_max_pointer() + exp_sum_offset
        qk_max_ptr = exp_sum_ptr + (
            UInt32(Self.q_num_heads) * batch_size * partition.num_partitions()
        )
        return (exp_sum_ptr, qk_max_ptr)

    @always_inline
    fn get_start_and_end_for_partitions[
        PartitionType: MHAPartitionScheme, MaskType: MHAMask, //, page_size: Int
    ](self, partition: PartitionType, mask: MaskType) -> Tuple[UInt32, UInt32]:
        var start_col: UInt32 = mask.start_column[Self.BM, Self.BN, page_size](
            self.get_score_row()
        )

        @parameter
        if PartitionType.do_partition:
            start, end = get_start_and_end_for_partitions[Self.BN](
                Int(self.num_keys - start_col),
                Int(partition.num_partitions()),
                Int(self.prompt_offset),
            )
            return (UInt32(start) + start_col, UInt32(end) + start_col)
        else:
            return (start_col, self.num_keys)

    @staticmethod
    @always_inline
    fn get_q_gmem_row[
        MaxSeqLenType: OptionallyStaticInt, //, ragged: Bool
    ](seq_info: SeqInfo, max_seq_len: MaxSeqLenType) -> UInt32:
        var q_row: UInt32

        @parameter
        if ragged:
            q_row = seq_info.start_of_seq

        # NDBuffer inputs, homogeneous batching.
        else:
            # When cache length (num_keys) is greater, we assume it has
            # prefix preceding the input seq_len.
            q_row = seq_info.prompt_idx * max_seq_len.as_uint32()

        @parameter
        if _is_decoding[MaxSeqLenType]():
            # q matrix view is rows x depth
            return q_row * UInt32(
                Self.q_num_heads
            ) + seq_info.head_idx * UInt32(Self.group)
        else:  # head_idx is for q_heads
            # q matrix view is rows x (depth*q_num_heads)
            return q_row + seq_info.prompt_offset

    @staticmethod
    @always_inline
    fn get_q_gmem_row[
        ragged: Bool
    ](seq_info: SeqInfo, max_seq_len: UInt32) -> UInt32:
        var q_row: UInt32

        @parameter
        if ragged:
            q_row = seq_info.start_of_seq

        # NDBuffer inputs, homogeneous batching.
        else:
            # When cache length (num_keys) is greater, we assume it has
            # prefix preceding the input seq_len.
            q_row = seq_info.prompt_idx * max_seq_len

        # q matrix view is rows x (depth*q_num_heads)
        return q_row + seq_info.prompt_offset


@always_inline
fn get_seq_info[
    MaxSeqLenType: OptionallyStaticInt,
    ValidLengthType: OptionalPointer,
    PartitionType: MHAPartitionScheme,
    //,
    BM: Int,
    num_heads: Int,
](
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,
    valid_length: ValidLengthType,
    partition: PartitionType,
) -> SeqInfo:
    var tile_summary = MHATileSummary[ValidLengthType](
        batch_size,
        ceildiv(max_seq_len.as_uint32(), UInt32(BM))
        * partition.num_partitions(),
        valid_length,
        max_seq_len.as_uint32(),
    )
    scheduler = TransientScheduler[UInt32(BM), UInt32(num_heads)]()
    var state: MHATileState = scheduler.initial_state(
        UnsafePointer[
            UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
        ](),
        tile_summary,
    )
    return scheduler.unsafe_seq_info(tile_summary, state)


struct PositionSummary(TrivialRegisterPassable):
    var num_keys: UInt32
    var score_row: UInt32

    @always_inline
    fn __init__(out self, num_keys: UInt32, score_row: UInt32):
        self.num_keys = num_keys
        self.score_row = score_row

    @staticmethod
    @always_inline
    fn get_start_pos[
        KVLUTType: MHAOperand,
        //,
        ragged: Bool,
        _is_cache_length_accurate: Bool,
    ](kv_lut: KVLUTType, seq_info: SeqInfo, num_keys_arg: UInt32) -> UInt32:
        @parameter
        if not ragged:
            return num_keys_arg - seq_info.seq_len
        elif _is_cache_length_accurate:
            return 0
        else:
            return UInt32(
                warp.broadcast(kv_lut.cache_length(Int(seq_info.prompt_idx)))
            )

    @staticmethod
    @always_inline
    fn get_num_keys[
        MaxSeqLenType: OptionallyStaticInt,
        KVInputRowOffsetsType: OptionalPointer,
        //,
        ragged: Bool,
        _is_cache_length_accurate: Bool,
    ](
        kv_input_row_offsets: KVInputRowOffsetsType,
        seq_info: SeqInfo,
        max_seq_len: MaxSeqLenType,
        num_keys_arg: UInt32,
        start_pos: UInt32,
    ) -> UInt32:
        @parameter
        if not ragged:
            return num_keys_arg
        else:
            var batch_idx: UInt32 = seq_info.prompt_idx

            @parameter
            if KVInputRowOffsetsType.is_null:
                return seq_info.seq_len + start_pos
            else:
                var kv_row_offsets = kv_input_row_offsets.value()
                kv_seq_start = warp.broadcast(
                    UInt32(kv_row_offsets[Int(batch_idx)])
                )
                kv_seq_end = warp.broadcast(
                    UInt32(kv_row_offsets[Int(batch_idx) + 1])
                )
                cur_kv_len = kv_seq_end - kv_seq_start
                return cur_kv_len + start_pos

    @staticmethod
    @always_inline
    fn get_score_row[
        *, ragged: Bool, _is_cache_length_accurate: Bool, decoding: Bool
    ](seq_info: SeqInfo, num_keys: UInt32, start_pos: UInt32) -> UInt32:
        @parameter
        if decoding:
            return num_keys - 1
        elif ragged and _is_cache_length_accurate:
            return seq_info.prompt_offset
        else:
            return seq_info.prompt_offset + start_pos

    @staticmethod
    @always_inline
    fn create[
        KVLUTType: MHAOperand,
        KVRowOffsetsType: OptionalPointer,
        MaxSeqLenType: OptionallyStaticInt,
        //,
        ragged: Bool,
        _is_cache_length_accurate: Bool,
    ](
        kv_lut: KVLUTType,
        seq_info: SeqInfo,
        num_keys_arg: UInt32,
        kv_input_row_offsets: KVRowOffsetsType,
        max_seq_len: MaxSeqLenType,
    ) -> PositionSummary:
        start_pos = Self.get_start_pos[
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](kv_lut, seq_info, num_keys_arg)
        num_keys = Self.get_num_keys[
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](
            kv_input_row_offsets,
            seq_info,
            max_seq_len,
            num_keys_arg,
            start_pos,
        )
        score_row = Self.get_score_row[
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding = _is_decoding[MaxSeqLenType](),
        ](seq_info, num_keys, start_pos)
        return {num_keys, score_row}


@always_inline
fn _get_position[
    KVLUTType: MHAOperand,
    MaxSeqLenType: OptionallyStaticInt,
    KVInputRowOffsetsType: OptionalPointer,
    //,
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    q_num_heads: Int,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    out ret: MHAPosition[
        BM,
        BN,
        depth,
        padded_depth,
        q_num_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ],
    seq_info: SeqInfo,
    kv_lut: KVLUTType,
    max_seq_len: MaxSeqLenType,
    num_keys_arg: UInt32,
    kv_input_row_offsets: KVInputRowOffsetsType,
):
    var batch_idx: UInt32 = seq_info.prompt_idx
    # mha inputs
    var seq_len: UInt32 = seq_info.seq_len
    var num_keys: UInt32
    var start_pos: UInt32
    var q_row: UInt32

    @parameter
    if ragged:

        @parameter
        if not _is_cache_length_accurate:
            start_pos = UInt32(
                warp.broadcast(kv_lut.cache_length(Int(batch_idx)))
            )
        else:
            start_pos = 0

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        @parameter
        if KVInputRowOffsetsType.is_null:
            num_keys = seq_len + UInt32(Int(start_pos))
        else:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[Int(batch_idx)])
            kv_seq_end = Int(kv_row_offsets[Int(batch_idx) + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = UInt32(cur_kv_len + Int(start_pos))
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
        q_col = 0
        q_offset = depth * Int(
            q_row * UInt32(q_num_heads) + seq_info.head_idx * UInt32(group)
        )
    else:  # head_idx is for q_heads
        # q matrix view is rows x (depth*q_num_heads)
        q_row += seq_info.prompt_offset
        q_col = seq_info.head_idx * UInt32(depth)
        q_offset = depth * q_num_heads * Int(q_row) + Int(q_col)
    ret = {q_row, q_col, q_offset, num_keys, start_pos, seq_info}


fn q_smem_shape[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    group: Int,
    depth: Int,
    decoding: Bool,
    num_qk_stages: Int = 1,
](out res: IndexList[4 if decoding else 3]):
    comptime L = res.size
    comptime assert L in (3, 4)
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()

    @parameter
    if L == 3:  # prefill

        @parameter
        if num_qk_stages == 1:
            return {BM, 1, depth}
        else:
            return {
                BM,
                1,
                align_up(depth, swizzle_granularity) // num_qk_stages,
            }
    else:
        return {1, 1, max(group, 8), swizzle_granularity}


fn q_gmem_shape[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    group: Int,
    q_num_heads: Int,
    depth: Int,
    decoding: Bool,
](out res: IndexList[4 if decoding else 3]):
    comptime L = res.size
    comptime assert L in (3, 4)

    @parameter
    if L == 3:  # prefill
        return {UNKNOWN_VALUE, q_num_heads, depth}
    else:
        return {UNKNOWN_VALUE, q_num_heads // group, group, depth}


comptime QTMATile[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    depth: Int,
    group: Int,
    decoding: Bool,
    num_qk_stages: Int = 1,
] = SplitLastDimTMATensorTile[
    dtype,
    q_smem_shape[
        dtype,
        swizzle_mode,
        BM=BM,
        group=group,
        depth=depth,
        decoding=decoding,
        num_qk_stages=num_qk_stages,
    ](),
    swizzle_mode,
]

comptime KVTMATile[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BN: Int,
    BK: Int,
] = SplitLastDimTMATensorTile[
    dtype,
    IndexList[3](BN, 1, BK),
    swizzle_mode,
]


@always_inline
fn q_tma[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    depth: Int,
    q_num_heads: Int,
    group: Int,
    decoding: Bool,
    num_qk_stages: Int = 1,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype]],
    rows: Int,
) raises -> QTMATile[
    dtype,
    swizzle_mode,
    BM=BM,
    depth=depth,
    group=group,
    decoding=decoding,
    num_qk_stages=num_qk_stages,
]:
    comptime smem_dim = q_smem_shape[
        dtype,
        swizzle_mode,
        BM=BM,
        group=group,
        depth=depth,
        decoding=decoding,
        num_qk_stages=num_qk_stages,
    ]()
    comptime gmem_dim = q_gmem_shape[
        dtype,
        swizzle_mode,
        group=group,
        q_num_heads=q_num_heads,
        depth=depth,
        decoding=decoding,
    ]()
    return create_split_tma[smem_dim, gmem_dim, swizzle_mode](ctx, ptr, rows)


@always_inline
fn get_q_head_idx[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    //,
](
    position: MHAPosition[
        BM, BN, depth, padded_depth, num_heads, group, decoding
    ],
    lane: UInt32,
    out indices: StaticTuple[UInt32, type_of(position).num_q_heads_per_thread],
):
    @parameter
    if decoding:
        var q_head_idx_0: UInt32 = UInt32(group) * position.head_idx + lane // 4

        indices = {}
        indices[0] = q_head_idx_0

        @parameter
        for i in range(1, position.num_q_heads_per_thread):
            indices[i] = q_head_idx_0 + UInt32(8 * i)

    else:
        indices = {position.head_idx}


@always_inline
fn _apply_mask[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    accum_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    reg_tile_layout: Layout,
    element_layout: Layout,
    //,
    # last_iter: Bool,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    use_score_mod: Bool,
](
    mask_warp_row_arg: UInt32,
    position: MHAPosition[
        BM, BN, depth, padded_depth, num_heads, group, decoding
    ],
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
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=element_layout,
    ],
):
    comptime num_groups_per_thread = min(
        2, ceildiv(group, 8)
    ) if decoding else 2
    var batch_cache_valid_length: UInt32

    @parameter
    if decoding:
        if warp.broadcast((thread_idx.x - 128) // 32) > UInt((group - 1) // 16):
            return
        if lane >= UInt32(4 * group):
            return
        batch_cache_valid_length = position.num_keys - 1
    else:
        batch_cache_valid_length = 0

    comptime p_frag_simdwidth = element_layout.size()
    # Vectorize by 2.
    var fragment_row: UInt32 = lane // 4
    var fragment_col: UInt32 = (
        lane * UInt32(p_frag_simdwidth) % UInt32(WN)
    ) % 8
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
                mask_frag_row = mask_warp_row + UInt32(m_mma * WM)
                mask_frag_col = mask_warp_col + UInt32(n_mma * WN)

                @parameter
                for i in range(num_groups_per_thread):
                    var q_head_idx: UInt32 = position.head_idx

                    @parameter
                    if decoding:
                        group_idx = UInt32(i * 8) + fragment_row
                        q_head_idx = UInt32(group) * q_head_idx + group_idx
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
                            position.prompt_offset
                            + mask_frag_row
                            + UInt32(i * WM // 2)
                        )
                        score_row_with_start_pos = (
                            score_row + position.start_pos
                        )

                    @parameter
                    for j in range(WN // 8):
                        score_col = mask_frag_col + UInt32(j * 8)
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
                                Int(position.num_keys),
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
            or (UInt32(BN) + kv_tile_start_row > position.num_keys)
        )


@always_inline
fn q_coord[
    *,
    depth: Int,
    swizzle_granularity: Int,
    decoding: Bool,
](
    row: UInt32,
    head_idx: UInt32,
    out res: StaticTuple[UInt32, (4 if decoding else 3)],
):
    """
    Returns the coordinates for a tma load on the `Q` matrix.
    This load can be 3D, 4D, or 5D.

    Arguments:
        row: the row to load from.
        head_idx: q_head_idx if prefill, kv_head_idx if decoding.
    """
    comptime rank: Int = res.size
    comptime assert rank in (3, 4)

    res = {}

    @parameter
    for i in range(rank - 2):
        res[i] = 0

    res[rank - 2] = head_idx
    res[rank - 1] = row


@always_inline
fn kv_coord[
    *, depth: Int, swizzle_granularity: Int
](row: UInt32, head_idx: UInt32) -> StaticTuple[UInt32, 3]:
    return {0, head_idx, row}


@always_inline
fn produce[
    qkv_type: DType,
    BM: Int,
    BN: Int,
    q_smem_layout: Layout,
    q_desc_layout: Layout,
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    PartitionType: MHAPartitionScheme,
    MaxSeqLenType: OptionallyStaticInt,
    SchedulerType: MHATileScheduler,
    KVLUTType: MHAOperand,
    MaskType: MHAMask,
    KVInputRowOffsetsType: OptionalPointer,
    ValidLengthType: OptionalPointer,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    pipeline_stages: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    q_tma_op: TMATensorTile[
        qkv_type,
        q_smem_layout,
        q_desc_layout,
    ],
    k_tma_op: KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=BN,
        BK=padded_depth,
    ],
    v_tma_op: KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=BN,
        BK=padded_depth,
    ],
    q_smem: UnsafePointer[
        Scalar[qkv_type], MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    kv_smem: UnsafePointer[
        Scalar[qkv_type], MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    produced_mbar_kv: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    consumed_mbar_kv: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    produced_mbar_q: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    consumed_mbar_q: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    kv_lut: KVLUTType,
    initial_position: MHAPosition[
        BM,
        BN,
        depth,
        padded_depth,
        num_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ],
    partition: PartitionType,
    scheduler: SchedulerType,
    mask: MaskType,
    tile_summary: MHATileSummary[ValidLengthType],
    tile_state_arg: MHATileState,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    kv_input_row_offsets: KVInputRowOffsetsType,
):
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[qkv_type]()

    comptime decoding: Bool = _is_decoding[MaxSeqLenType]()
    comptime PositionType = MHAPosition[
        BM, BN, depth, padded_depth, num_heads, group, decoding
    ]
    comptime persistent = SchedulerType.may_advance

    comptime q_smem_layout_consumer = tile_layout_k_major[
        DType.bfloat16, BM, padded_depth, swizzle_mode=swizzle_mode
    ]()

    comptime q_size = q_smem_layout_consumer.size()
    comptime q_smem_size = (2 * q_size if persistent else q_size)

    comptime q_copy_rows = max(group, 8) if decoding else BM
    comptime qk_bytes = (q_copy_rows + BN) * padded_depth * size_of[qkv_type]()

    tile_state = tile_state_arg
    position = initial_position

    @parameter
    @always_inline("nodebug")
    fn q_producer(
        q_idx: UInt32, offset: UInt32 = 0
    ) -> LayoutTensor[
        qkv_type,
        q_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]:
        return {q_smem + UInt32(q_size) * q_idx + offset}

    comptime k_smem_layout = tile_layout_k_major[
        qkv_type, BN, padded_depth, swizzle_mode
    ]()

    @parameter
    @always_inline
    fn kv_tile(
        idx: UInt32,
        out tile: LayoutTensor[
            qkv_type,
            k_smem_layout,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            alignment=128,
        ],
    ):
        comptime sz = BN * padded_depth
        tile = {kv_smem + UInt32(sz) * idx}

    @parameter
    @always_inline("nodebug")
    fn produce_k[
        wait: Bool
    ](
        mut state: PipelineState[pipeline_stages],
        row: UInt32,
        kv_head_idx: UInt32,
    ):
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]
        k_sub = kv_tile(write_idx)

        @parameter
        if wait:
            consumed_mbar_kv[write_idx].wait(write_phase)
            comptime bytes = BN * padded_depth * size_of[qkv_type]()
            p_mbar.expect_bytes(Int32(bytes))
        k_tma_op.async_copy(
            k_sub,
            p_mbar,
            kv_coord[depth=depth, swizzle_granularity=swizzle_granularity](
                row, kv_head_idx
            ),
        )
        state.step()

    @parameter
    @always_inline("nodebug")
    fn produce_v(
        mut state: PipelineState[pipeline_stages],
        row: UInt32,
        kv_head_idx: UInt32,
        key_start: UInt32,
        key_end: UInt32,
    ):
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]
        v_sub = kv_tile(write_idx)
        consumed_mbar_kv[write_idx].wait(write_phase)
        comptime bytes = BN * padded_depth * size_of[qkv_type]()
        p_mbar.expect_bytes(Int32(bytes))
        v_tma_op.async_copy(
            v_sub,
            p_mbar,
            kv_coord[depth=depth, swizzle_granularity=swizzle_granularity](
                row, kv_head_idx
            ),
        )
        state.step()

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM,
            BN,
            depth,
            padded_depth,
            num_heads,
            group,
            ragged,
            _is_cache_length_accurate,
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
        startend = position.get_start_and_end_for_partitions[
            page_size = KVLUTType.page_size
        ](partition, mask)
        start = startend[0]
        end = startend[1]
        if start >= end:
            return
    else:
        # delay partitioning until after we've begun copying `q`
        start = 0
        end = 0

    produced_mbar_kv[0].expect_bytes(Int32(qk_bytes))

    @parameter
    if decoding:
        ref q_mbar = produced_mbar_kv[0]
        var q_idx: UInt32 = q_pipeline_state.index()

        @parameter
        for d_idx in range(ceildiv(depth, swizzle_granularity)):
            comptime d: Int = d_idx * swizzle_granularity
            comptime smem_offset = q_smem_layout_consumer(IntTuple(0, d))

            q_tma_op.async_copy_4d(
                q_producer(q_idx, UInt32(smem_offset)),
                q_mbar,
                (
                    d,
                    0,
                    Int(position.head_idx),
                    Int(position.q_row),
                ),
            )
    else:
        q_tma_op.async_copy(
            q_producer(q_pipeline_state.index()),
            produced_mbar_kv[0],
            q_coord[
                depth=depth,
                swizzle_granularity=swizzle_granularity,
                decoding=decoding,
            ](position.q_row, position.head_idx),
        )

    @parameter
    if not PartitionType.do_partition:
        startend = position.get_start_and_end_for_partitions[
            page_size = KVLUTType.page_size
        ](partition, mask)
        start = startend[0]
        end = startend[1]
    var kv_tile_start_row: UInt32 = start

    while (
        position.mask_status(mask, kv_tile_start_row)
        == TileMaskStatus.FULL_MASK
    ):
        kv_tile_start_row += UInt32(BN)

    var kv_row: UInt32 = kv_lut.row_idx(position.prompt_idx, kv_tile_start_row)
    var kv_head_idx: UInt32 = position.kv_head_idx()

    produce_k[False](write_pipeline_states, kv_row, kv_head_idx)

    var kv_row_prev: UInt32 = kv_row
    var kv_head_idx_prev: UInt32 = kv_head_idx
    var kv_tile_start_row_prev: UInt32 = kv_tile_start_row
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
        kv_tile_start_row += UInt32(BN)
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
                    producer=True, sync = MHASchedulerSynchronization.DEFAULT
                ](tile_summary, tile_state, q_idx_old)
                # FIXME: persistent kernel that uses a counter
                # must signal somehow
                if not docontinue:
                    break
                ref pq_mbar = produced_mbar_q[q_idx_old]
                position = get_position(docontinue.value())
                pq_mbar.expect_bytes(
                    Int32(q_copy_rows * padded_depth * size_of[qkv_type]())
                )

                @parameter
                if not decoding:
                    q_tma_op.async_copy(
                        q_producer(q_idx),
                        pq_mbar,
                        q_coord[
                            depth=depth,
                            swizzle_granularity=swizzle_granularity,
                            decoding=decoding,
                        ](position.q_row, position.head_idx),
                    )

                else:

                    @parameter
                    for d_idx in range(depth // 64):
                        comptime d: UInt = UInt(d_idx * 64)
                        q_tma_op.async_copy_4d(
                            q_producer(q_idx),
                            pq_mbar,
                            (
                                Int(d),
                                0,
                                Int(position.head_idx),
                                Int(position.q_row),
                            ),
                        )

                kv_head_idx = position.kv_head_idx()
                start, new_end = position.get_start_and_end_for_partitions[
                    page_size = KVLUTType.page_size
                ](partition, mask)
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
        produce_k[True](write_pipeline_states, kv_row, kv_head_idx)
        produce_v(
            write_pipeline_states,
            kv_row_prev,
            kv_head_idx_prev,
            kv_tile_start_row_prev,
            end,
        )
        kv_row_prev = kv_row
        kv_head_idx_prev = kv_head_idx
        kv_tile_start_row_prev = kv_tile_start_row

    produce_v(
        write_pipeline_states,
        kv_row_prev,
        kv_head_idx_prev,
        kv_tile_start_row_prev,
        end,
    )


@always_inline
fn output_reg_to_smem_st_matrix[
    output_type: DType,
    accum_type: DType,
    num_m_mmas: Int,
    padded_depth: Int,
    o_frag_size: Int,
    //,
    BM: Int,
    swizzle: Swizzle,
    num_consumer: Int,
](
    warp_group_thread_idx: UInt32,
    local_warp_group_idx: UInt32,
    output_reg_tile: _LocalTT[accum_type, row_major[num_m_mmas, o_frag_size]()],
    accum_smem_tile: _SharedMemTT[output_type, row_major[BM, padded_depth]()],
):
    comptime st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[
            output_type, padded_depth, num_m_mmas, num_consumer
        ](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for i in range(padded_depth // 16):
            var st_matrix_args = RuntimeTuple[
                IntTuple(UNKNOWN_VALUE, IntTuple(i, m_mma, UNKNOWN_VALUE))
            ](
                Int(warp_group_thread_idx),
                i,
                m_mma,
                Int(local_warp_group_idx),
            )
            var accum_smem_idx = swizzle(st_matrix_rt_layout(st_matrix_args))
            var offset = accum_smem_tile.ptr + accum_smem_idx
            var output_frag = output_reg_tile.ptr.load[width=8](
                m_mma * o_frag_size + i * 8
            ).cast[output_type]()
            var output_frag_f32_packed = bitcast[DType.float32, 4](output_frag)
            st_matrix[simd_width=4](offset, output_frag_f32_packed)


@always_inline
fn output_reg_to_smem[
    output_type: DType,
    accum_type: DType,
    num_m_mmas: Int,
    o_frag_size: Int,
    //,
    BM: Int,
    BN: Int,
    padded_depth: Int,
    swizzle: Swizzle,
    num_consumer: Int,
](
    tid: UInt32,
    local_warp_group_idx: UInt32,
    warp_y: UInt32,
    q_smem: UnsafePointer[
        Scalar[output_type], MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    output_reg_tile: LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas, o_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ],
) -> LayoutTensor[
    output_type,
    Layout.row_major(BM, padded_depth),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
]:
    accum_smem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, padded_depth),
        address_space = AddressSpace.SHARED,
    ](q_smem)
    comptime use_stmatrix = accum_type == DType.float32 and padded_depth % 16 == 0 and size_of[
        output_type
    ]() == 2 and o_frag_size % 8 == 0

    @parameter
    if use_stmatrix:
        var warp_group_thread_idx = tid % UInt32(WARPGROUP_SIZE)
        comptime reg_layout = row_major[num_m_mmas, o_frag_size]()
        comptime smem_layout = row_major[BM, padded_depth]()
        output_reg_to_smem_st_matrix[BM, swizzle, num_consumer](
            warp_group_thread_idx,
            local_warp_group_idx,
            _LocalTT[accum_type, reg_layout](output_reg_tile.ptr, reg_layout),
            _SharedMemTT[output_type, smem_layout](
                accum_smem_tile.ptr, smem_layout
            ),
        )
    else:
        comptime mma_thread_layout = Layout.row_major(8, 4)
        accum_smem_warp_tile = accum_smem_tile.tile[16, BN](Int(warp_y), Int(0))
        copy_local_to_shared[thread_layout=mma_thread_layout, swizzle=swizzle](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
    return accum_smem_tile
