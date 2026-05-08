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

from std.collections import OptionalReg
from std.math import ceildiv, align_up
from std.math.uutils import ufloordiv
from std.math.constants import log2e
from std.memory import (
    bitcast,
)

from std.sys import size_of

import std.gpu.primitives.warp as warp
from std.algorithm.functional import unswitch
from std.gpu import thread_idx
from std.gpu.globals import WARPGROUP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.compute.mma import st_matrix
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    LTToTTLayout,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
    row_major,
)
from layout.tile_layout import (
    Layout as InternalLayout,
    row_major as tt_row_major,
)
from layout.layout_tensor import copy_local_to_shared
from layout.swizzle import Swizzle
from layout.tensor_core_async import st_matrix_n_layout, tile_layout_k_major
from layout.tma_async import (
    create_split_tma,
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    TMATensorTile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import (
    MHAOperand,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
)
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from std.builtin.device_passable import DevicePassable
from std.utils import StaticTuple


comptime _LocalTT[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutAnyOrigin,
    address_space=AddressSpace.LOCAL,
]
comptime _SharedMemTT[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
]

# TileTensor type alias for 1D row-major tensors with dynamic size, used for
# kv_input_row_offsets and sink_weights dispatch params.
comptime _1d_row_major_tt_layout = LTToTTLayout[Layout.row_major(UNKNOWN_VALUE)]
comptime ImmutTileTensor1D[dtype: DType] = TileTensor[
    dtype,
    _1d_row_major_tt_layout,
    ImmutAnyOrigin,
]


@always_inline
def _optional_lt_to_tt[
    dtype: DType,
](
    opt: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
) -> OptionalReg[ImmutTileTensor1D[dtype]]:
    """Convert an OptionalReg[LayoutTensor] to OptionalReg[TileTensor]."""
    if opt:
        return lt_to_tt(opt.value())
    return None


trait OptionalPointer(Copyable, TrivialRegisterPassable):
    comptime dtype: DType
    comptime is_null: Bool
    comptime address_space: AddressSpace

    @always_inline
    def value(
        self,
    ) -> UnsafePointer[
        Scalar[Self.dtype], ImmutAnyOrigin, address_space=Self.address_space
    ]:
        ...


struct NonNullPointer[
    dtype_: DType, address_space_: AddressSpace = AddressSpace.GENERIC
](OptionalPointer):
    comptime dtype: DType = Self.dtype_
    comptime is_null: Bool = False
    comptime address_space: AddressSpace = Self.address_space_
    comptime PtrType = UnsafePointer[
        Scalar[Self.dtype], ImmutAnyOrigin, address_space=Self.address_space
    ]

    var ptr: Self.PtrType

    @always_inline
    def __init__(out self, ptr: Self.PtrType):
        self.ptr = ptr

    @always_inline
    def __init__(out self, ptr: DeviceBuffer[Self.dtype]):
        comptime assert Self.address_space == AddressSpace.GENERIC
        self.ptr = rebind[Self.PtrType](ptr.unsafe_ptr())

    @always_inline
    def value(self) -> Self.PtrType:
        assert Int(self.ptr) != 0, (
            "NonNullPointer is supposed to provide a compile-time guarantee"
            " of being non-null"
        )
        return self.ptr


struct NullPointer[
    dtype_: DType, address_space_: AddressSpace = AddressSpace.GENERIC
](OptionalPointer):
    comptime dtype: DType = Self.dtype_
    comptime is_null: Bool = True
    comptime address_space: AddressSpace = Self.address_space_
    comptime PtrType = UnsafePointer[
        Scalar[Self.dtype], ImmutAnyOrigin, address_space=Self.address_space
    ]

    @always_inline
    def __init__(out self):
        pass

    @always_inline
    def value(self) -> Self.PtrType:
        # NullPointer.value() should never be called at runtime — it exists
        # only for trait conformance. Return dangling as a safe sentinel.
        return Self.PtrType.unsafe_dangling()


struct Pack[
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](Copyable, DevicePassable, TrivialRegisterPassable):
    var mask: Self.MaskType
    var scheduler: Self.SchedulerType
    var valid_length: Self.ValidLengthType
    var sink_weights: Self.SinkType
    var kv_input_row_offsets: Self.KVRowOffsetsType
    var max_seq_len: Self.MaxSeqLenType
    var partition: Self.PartitionType

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "Pack"

    @always_inline
    def __init__(
        out self,
        mask: Self.MaskType,
        scheduler: Self.SchedulerType,
        valid_length: Self.ValidLengthType,
        sink_weights: Self.SinkType,
        kv_input_row_offsets: Self.KVRowOffsetsType,
        max_seq_len: Self.MaxSeqLenType,
        partition: Self.PartitionType,
    ):
        self.mask = mask
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
    def __init__(
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
    def q_head_idx(self) -> UInt32:
        comptime if Self.decoding:
            return self.head_idx * UInt32(Self.group)
        else:
            return self.head_idx

    @always_inline
    def kv_head_idx(self) -> UInt32:
        comptime if Self.decoding:
            return self.head_idx
        else:
            return self.head_idx // UInt32(Self.group)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
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
    def q_tile_num_rows(self) -> UInt32:
        comptime if Self.decoding:
            return UInt32(Self.group)
        else:
            return min(self.seq_len - self.prompt_offset, UInt32(Self.BM))

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.q_out_offset == other.q_out_offset

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.q_out_offset != other.q_out_offset

    @always_inline
    def q_out_gmem_tensor[
        dtype: DType
    ](
        self,
        ptr: UnsafePointer[mut=True, Scalar[dtype], _],
        out gmem_block: LayoutTensor[
            dtype,
            Self.q_output_gmem_layout,
            MutAnyOrigin,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
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
    def mask_status[
        MaskType: MHAMask
    ](self, mask: MaskType, kv_tile_start_row: UInt32) -> TileMaskStatus:
        comptime if Self.decoding:
            comptime if MaskType.check_mask_during_decoding:
                # In context encoding, we have BM rows of Q
                # In decoding, we have `group` rows, but these
                # correspond to the same position w/ respect to the mask.
                return mask.status(
                    Index[dtype=DType.int32](
                        Int(self.num_keys - 1),
                        Int(kv_tile_start_row),
                    ),
                    Index[dtype=DType.int32](Int(1), Self.BN),
                )
            else:
                return TileMaskStatus.PARTIAL_MASK
        else:
            return mask.status(
                Index[dtype=DType.int32](
                    Int(self.prompt_offset + self.start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[dtype=DType.int32](Self.BM, Self.BN),
            )

    @always_inline
    def get_score_row(self) -> UInt32:
        comptime if Self.decoding:
            return self.num_keys - 1
        else:
            return self.prompt_offset + self.start_pos

    @always_inline
    def exp_sum_qk_max_ptr[
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
    def get_start_and_end_for_partitions[
        PartitionType: MHAPartitionScheme, MaskType: MHAMask, //, page_size: Int
    ](self, partition: PartitionType, mask: MaskType) -> Tuple[UInt32, UInt32]:
        var start_col: UInt32 = mask.start_column[Self.BM, Self.BN, page_size](
            self.get_score_row()
        )

        comptime if PartitionType.do_partition:
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
    def get_q_gmem_row[
        MaxSeqLenType: OptionallyStaticInt, //, ragged: Bool
    ](seq_info: SeqInfo, max_seq_len: MaxSeqLenType) -> UInt32:
        var q_row: UInt32

        comptime if ragged:
            q_row = seq_info.start_of_seq

        # Homogeneous batching.
        else:
            # When cache length (num_keys) is greater, we assume it has
            # prefix preceding the input seq_len.
            q_row = seq_info.prompt_idx * max_seq_len.as_uint32()

        comptime if _is_decoding[MaxSeqLenType]():
            # q matrix view is rows x depth
            return q_row * UInt32(
                Self.q_num_heads
            ) + seq_info.head_idx * UInt32(Self.group)
        else:  # head_idx is for q_heads
            # q matrix view is rows x (depth*q_num_heads)
            return q_row + seq_info.prompt_offset

    @staticmethod
    @always_inline
    def get_q_gmem_row[
        ragged: Bool
    ](seq_info: SeqInfo, max_seq_len: UInt32) -> UInt32:
        var q_row: UInt32

        comptime if ragged:
            q_row = seq_info.start_of_seq

        # Homogeneous batching.
        else:
            # When cache length (num_keys) is greater, we assume it has
            # prefix preceding the input seq_len.
            q_row = seq_info.prompt_idx * max_seq_len

        # q matrix view is rows x (depth*q_num_heads)
        return q_row + seq_info.prompt_offset


@always_inline
def get_seq_info[
    MaxSeqLenType: OptionallyStaticInt,
    ValidLengthType: OptionalPointer,
    PartitionType: MHAPartitionScheme,
    //,
    BM: Int,
    num_heads: Int,
    flip_prompt_idx: Bool,
    pair_cta: Bool = False,
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
    scheduler = TransientScheduler[
        UInt32(BM),
        UInt32(num_heads),
        flip_prompt_idx=flip_prompt_idx,
        pair_cta=pair_cta,
    ]()
    # SAFETY: Stored in MHATileState.sidx_ptr but never dereferenced.
    var state: MHATileState = scheduler.initial_state(
        UnsafePointer[
            UInt32, MutAnyOrigin, address_space=AddressSpace.SHARED
        ].unsafe_dangling(),
        tile_summary,
    )
    return scheduler.unsafe_seq_info(tile_summary, state)


struct PositionSummary(TrivialRegisterPassable):
    var num_keys: UInt32
    var score_row: UInt32

    @always_inline
    def __init__(out self, num_keys: UInt32, score_row: UInt32):
        self.num_keys = num_keys
        self.score_row = score_row

    @staticmethod
    @always_inline
    def get_start_pos[
        KVLUTType: MHAOperand,
        //,
        ragged: Bool,
        _is_cache_length_accurate: Bool,
    ](kv_lut: KVLUTType, seq_info: SeqInfo, num_keys_arg: UInt32) -> UInt32:
        comptime if not ragged:
            return num_keys_arg - seq_info.seq_len
        elif _is_cache_length_accurate:
            return 0
        else:
            return UInt32(
                warp.broadcast(kv_lut.cache_length(Int(seq_info.prompt_idx)))
            )

    @staticmethod
    @always_inline
    def get_num_keys[
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
        comptime if not ragged:
            return num_keys_arg
        else:
            var batch_idx: UInt32 = seq_info.prompt_idx

            comptime if KVInputRowOffsetsType.is_null:
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
    def get_score_row[
        *, ragged: Bool, _is_cache_length_accurate: Bool, decoding: Bool
    ](seq_info: SeqInfo, num_keys: UInt32, start_pos: UInt32) -> UInt32:
        comptime if decoding:
            return num_keys - 1
        elif ragged and _is_cache_length_accurate:
            return seq_info.prompt_offset
        else:
            return seq_info.prompt_offset + start_pos

    @staticmethod
    @always_inline
    def create[
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
            decoding=_is_decoding[MaxSeqLenType](),
        ](seq_info, num_keys, start_pos)
        return {num_keys, score_row}


@always_inline
def _get_position[
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

    comptime if ragged:
        comptime if not _is_cache_length_accurate:
            start_pos = UInt32(
                warp.broadcast(kv_lut.cache_length(Int(batch_idx)))
            )
        else:
            start_pos = 0

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        comptime if KVInputRowOffsetsType.is_null:
            num_keys = seq_len + UInt32(Int(start_pos))
        else:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[Int(batch_idx)])
            kv_seq_end = Int(kv_row_offsets[Int(batch_idx) + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = UInt32(cur_kv_len + Int(start_pos))
        q_row = seq_info.start_of_seq

    # Homogeneous batching.
    else:
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len
        q_row = batch_idx * max_seq_len.as_uint32()

    var q_offset: Int
    var q_col: UInt32

    comptime if _is_decoding[MaxSeqLenType]():
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


def q_smem_shape[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    group: Int,
    depth: Int,
    decoding: Bool,
    fuse_gqa: Bool = False,
    num_qk_stages: Int = 1,
](out res: IndexList[4 if (decoding or fuse_gqa) else 3]):
    comptime L = res.size
    comptime assert L in (3, 4)
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()

    comptime if decoding:
        return {1, 1, max(group, 8), swizzle_granularity}
    elif fuse_gqa:
        comptime if num_qk_stages == 1:
            return {BM // group, 1, group, depth}
        else:
            return {
                BM // group,
                1,
                group,
                align_up(depth, swizzle_granularity) // num_qk_stages,
            }
    else:
        comptime if num_qk_stages == 1:
            return {BM, 1, depth}
        else:
            return {
                BM,
                1,
                align_up(depth, swizzle_granularity) // num_qk_stages,
            }


def q_gmem_shape[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    group: Int,
    q_num_heads: Int,
    depth: Int,
    decoding: Bool,
    fuse_gqa: Bool = False,
](out res: IndexList[4 if (decoding or fuse_gqa) else 3]):
    comptime L = res.size
    comptime assert L in (3, 4)

    comptime if L == 3:  # prefill, no fusion
        return {UNKNOWN_VALUE, q_num_heads, depth}
    else:  # decoding or fuse_gqa prefill
        return {UNKNOWN_VALUE, q_num_heads // group, group, depth}


comptime QTMATile[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    depth: Int,
    group: Int,
    decoding: Bool,
    fuse_gqa: Bool = False,
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
        fuse_gqa=fuse_gqa,
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
def q_tma[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    depth: Int,
    q_num_heads: Int,
    group: Int,
    decoding: Bool,
    fuse_gqa: Bool = False,
    num_qk_stages: Int = 1,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype], _],
    rows: Int,
) raises -> QTMATile[
    dtype,
    swizzle_mode,
    BM=BM,
    depth=depth,
    group=group,
    decoding=decoding,
    fuse_gqa=fuse_gqa,
    num_qk_stages=num_qk_stages,
]:
    comptime smem_dim = q_smem_shape[
        dtype,
        swizzle_mode,
        BM=BM,
        group=group,
        depth=depth,
        decoding=decoding,
        fuse_gqa=fuse_gqa,
        num_qk_stages=num_qk_stages,
    ]()
    comptime gmem_dim = q_gmem_shape[
        dtype,
        swizzle_mode,
        group=group,
        q_num_heads=q_num_heads,
        depth=depth,
        decoding=decoding,
        fuse_gqa=fuse_gqa,
    ]()
    return create_split_tma[smem_dim, gmem_dim, swizzle_mode](ctx, ptr, rows)


@always_inline
def get_q_head_idx[
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
    comptime if decoding:
        var q_head_idx_0: UInt32 = UInt32(group) * position.head_idx + lane // 4

        indices = {}
        indices[0] = q_head_idx_0

        comptime for i in range(1, position.num_q_heads_per_thread):
            indices[i] = q_head_idx_0 + UInt32(8 * i)

    else:
        indices = {position.head_idx}


@always_inline
def _apply_mask[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    accum_type: DType,
    mask_t: MHAMask,
    reg_tile_layout: Layout,
    element_layout: Layout,
    //,
    # last_iter: Bool,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
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
    p_reg_tile: LayoutTensor[
        accum_type,
        reg_tile_layout,
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
        element_layout=element_layout,
    ],
):
    comptime num_groups_per_thread = min(
        2, ceildiv(group, 8)
    ) if decoding else 2
    var batch_cache_valid_length: UInt32

    comptime if decoding:
        if warp.broadcast(ufloordiv(thread_idx.x - 128, 32)) > (
            (group - 1) // 16
        ):
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
    def _apply_mask_capture[masked: Bool]():
        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                # Coordinates in mask for current mma tile.
                mask_frag_row = mask_warp_row + UInt32(m_mma * WM)
                mask_frag_col = mask_warp_col + UInt32(n_mma * WN)

                comptime for i in range(num_groups_per_thread):
                    var q_head_idx: UInt32 = position.head_idx

                    comptime if decoding:
                        group_idx = UInt32(i * 8) + fragment_row
                        q_head_idx = UInt32(group) * q_head_idx + group_idx
                    # The row in score matrix of shape seq_len x num_keys.
                    # Mask col is score col since we don't partition in col.
                    var score_row: UInt32
                    var score_row_with_start_pos: UInt32

                    comptime if decoding:
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

                    comptime for j in range(WN // 8):
                        score_col = mask_frag_col + UInt32(j * 8)
                        p = p_reg_tile[i, m_mma, j, n_mma]

                        comptime if masked:
                            p = mask.mask(
                                IndexList[4, element_type=DType.uint32](
                                    Int(position.prompt_idx),
                                    Int(q_head_idx),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p * scale_log2e,
                            )
                        else:
                            p *= scale_log2e

                        comptime if mask_t.apply_log2e_after_mask:
                            p *= log2e

                        var bound: IndexList[2, element_type=DType.uint32]

                        comptime if decoding:
                            bound = IndexList[2, element_type=DType.uint32](
                                Int(position.num_keys),
                                Int(position.num_keys),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type=DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        elif masked:
                            bound = IndexList[2, element_type=DType.uint32](
                                Int(position.seq_len),
                                Int(position.num_keys),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type=DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        p_reg_tile[i, m_mma, j, n_mma] = p

    comptime if decoding:
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
def q_coord[
    *,
    depth: Int,
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

    comptime for i in range(rank - 2):
        res[i] = 0

    res[rank - 2] = head_idx
    res[rank - 1] = row


@always_inline
def kv_coord[
    *, depth: Int
](row: UInt32, head_idx: UInt32) -> StaticTuple[UInt32, 3]:
    return {0, head_idx, row}


@always_inline
def produce[
    qkv_type: DType,
    BM: Int,
    BN: Int,
    q_rank: Int,
    q_tile_shape: IndexList[q_rank],
    q_desc_shape: IndexList[q_rank],
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
        q_rank,
        q_tile_shape,
        q_desc_shape,
    ],
    k_tma_op: KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=kv_sub_tile_rows(BN, KVLUTType.page_size),
        BK=padded_depth,
    ],
    v_tma_op: KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=kv_sub_tile_rows(BN, KVLUTType.page_size),
        BK=padded_depth,
    ],
    q_smem: UnsafePointer[
        Scalar[qkv_type], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    kv_smem: UnsafePointer[
        Scalar[qkv_type], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    produced_mbar_kv: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    consumed_mbar_kv: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    produced_mbar_q: Optional[
        UnsafePointer[
            SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
        ]
    ],
    consumed_mbar_q: Optional[
        UnsafePointer[
            SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
        ]
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
    def q_producer(
        q_idx: UInt32, offset: UInt32 = 0
    ) -> LayoutTensor[
        qkv_type,
        Layout.row_major(q_tile_shape),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]:
        return {q_smem + UInt32(q_size) * q_idx + offset}

    comptime k_smem_layout = tile_layout_k_major[
        qkv_type, BN, padded_depth, swizzle_mode
    ]()
    comptime assert pipeline_stages >= 2

    @parameter
    @always_inline
    def kv_tile(
        idx: UInt32,
        out tile: LayoutTensor[
            qkv_type,
            k_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            alignment=128,
        ],
    ):
        comptime sz = BN * padded_depth
        tile = {kv_smem + UInt32(sz) * idx}

    comptime kv_sub_BN = kv_sub_tile_rows(BN, KVLUTType.page_size)

    comptime KVTMA = KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=kv_sub_BN,
        BK=padded_depth,
    ]
    comptime KVPagedRows = PagedRowIndices[BN, KVLUTType.page_size]
    comptime page_size = KVLUTType.page_size
    comptime needs_partial = page_size > 0 and page_size < BN
    comptime kv_bytes_pp = padded_depth * KVPagedRows.eff_page * size_of[
        qkv_type
    ]()
    # Alignment of `kv_tile_row` produced by mask-driven iteration.
    comptime base_alignment: Int = MaskType.start_column_alignment[
        BM, BN, page_size
    ]()

    @parameter
    @always_inline("nodebug")
    def _num_valid_pages(end_row: UInt32, current_kv_row: UInt32) -> UInt32:
        return min(
            UInt32(ceildiv(Int(end_row - current_kv_row), page_size)),
            UInt32(KVPagedRows.num_pages),
        )

    @parameter
    @always_inline("nodebug")
    def produce_kv[
        is_k_side: Bool,
        wait: Bool,
    ](
        tma_op: KVTMA,
        mut state: PipelineState[pipeline_stages],
        prompt_idx: UInt32,
        kv_tile_row: UInt32,
        kv_head_idx: UInt32,
    ):
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]

        comptime if wait:
            consumed_mbar_kv[write_idx].wait(write_phase)
            comptime bytes = BN * padded_depth * size_of[qkv_type]()
            p_mbar.expect_bytes(Int32(bytes))

        comptime stage_sz = BN * padded_depth
        var paged_rows = kv_lut.populate[BN, base_alignment](
            prompt_idx, kv_tile_row
        )
        # SM90's producer runs only on thread 0 (gated by `if thread_idx.x == 0:`
        # in `sm90/mha.mojo`), so we hard-code `elect=1` — the lone producer
        # thread is always "elected" and must issue the TMA.
        comptime if is_k_side:
            paged_rows.tma_copy_k[needs_partial=False](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=Int32(1),
            )
        else:
            paged_rows.tma_copy_v[needs_partial=False](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=Int32(1),
            )
        state.step()

    @parameter
    @always_inline("nodebug")
    def produce_kv_partial[
        is_k_side: Bool,
        wait: Bool,
    ](
        tma_op: KVTMA,
        mut state: PipelineState[pipeline_stages],
        prompt_idx: UInt32,
        kv_tile_row: UInt32,
        kv_head_idx: UInt32,
        num_valid_pages: UInt32,
    ):
        """Like `produce_kv` but with runtime-bounded page loops.

        Uses `populate` + `tma_copy_{k,v}[needs_partial=True]` so the
        row-index lookup (one SIMD LUT load for the paged case) is
        amortized across all TMA copies issued by this call.
        """
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]

        comptime if wait:
            consumed_mbar_kv[write_idx].wait(write_phase)
            p_mbar.expect_bytes(Int32(kv_bytes_pp * Int(num_valid_pages)))

        comptime stage_sz = BN * padded_depth
        # See `produce_kv`: SM90 producer is single-threaded, so `elect=1`.
        var paged_rows = kv_lut.populate[BN, base_alignment](
            prompt_idx, kv_tile_row
        )
        comptime if is_k_side:
            paged_rows.tma_copy_k[needs_partial=True](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=Int32(1),
                k_num_valid_pages=num_valid_pages,
            )
        else:
            paged_rows.tma_copy_v[needs_partial=True](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=Int32(1),
                num_valid_pages=num_valid_pages,
            )
        state.step()

    @parameter
    @always_inline
    def get_position(seq_info: SeqInfo) -> PositionType:
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

    comptime if PartitionType.do_partition:
        startend = position.get_start_and_end_for_partitions[
            page_size=KVLUTType.page_size
        ](partition, mask)
        start = startend[0]
        end = startend[1]
        if start >= end:
            return
    else:
        # delay partitioning until after we've begun copying `q`
        start = 0
        end = 0

    comptime q_bytes = q_copy_rows * padded_depth * size_of[qkv_type]()
    comptime if not needs_partial:
        produced_mbar_kv[0].expect_bytes(Int32(qk_bytes))

    comptime if decoding:
        ref q_mbar = produced_mbar_kv[0]
        var q_idx: UInt32 = q_pipeline_state.index()

        comptime for d_idx in range(ceildiv(depth, swizzle_granularity)):
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
                decoding=decoding,
            ](position.q_row, position.head_idx),
        )

    comptime if not PartitionType.do_partition:
        startend = position.get_start_and_end_for_partitions[
            page_size=KVLUTType.page_size
        ](partition, mask)
        start = startend[0]
        end = startend[1]
    var kv_tile_start_row: UInt32 = start

    while (
        position.mask_status(mask, kv_tile_start_row)
        == TileMaskStatus.FULL_MASK
    ):
        kv_tile_start_row += UInt32(BN)

    var kv_head_idx: UInt32 = position.kv_head_idx()

    # When needs_partial, nvp0 tracks the valid page count for K0.
    # Initialize to num_pages (full) and overwrite inside the comptime if.
    var nvp0: UInt32 = UInt32(KVPagedRows.num_pages)
    comptime if needs_partial:
        nvp0 = _num_valid_pages(end, kv_tile_start_row)
        produced_mbar_kv[0].expect_bytes(
            Int32(q_bytes + kv_bytes_pp * Int(nvp0))
        )
        if nvp0 < UInt32(KVPagedRows.num_pages):
            produce_kv_partial[is_k_side=True, wait=False](
                k_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row,
                kv_head_idx,
                nvp0,
            )
        else:
            produce_kv[is_k_side=True, wait=False](
                k_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row,
                kv_head_idx,
            )
    else:
        produce_kv[is_k_side=True, wait=False](
            k_tma_op,
            write_pipeline_states,
            position.prompt_idx,
            kv_tile_start_row,
            kv_head_idx,
        )

    var kv_head_idx_prev: UInt32 = kv_head_idx
    var kv_tile_start_row_prev: UInt32 = kv_tile_start_row
    var nvp_prev: UInt32 = nvp0
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
            comptime if persistent:
                kv_tile_start_row = 0
                var q_idx_old: UInt32 = q_pipeline_state.index()
                var q_phase_old: UInt32 = q_pipeline_state.phase()
                q_pipeline_state.step()
                consumed_mbar_q.unsafe_value()[q_idx_old].wait(q_phase_old)
                # we must wait before advancing, as this mbar
                # is for both `q_smem` and `sidx_ptr`
                var q_idx: UInt32 = q_pipeline_state.index()
                docontinue = scheduler.advance[
                    producer=True, sync=MHASchedulerSynchronization.DEFAULT
                ](tile_summary, tile_state, q_idx_old)
                # FIXME: persistent kernel that uses a counter
                # must signal somehow
                if not docontinue:
                    break
                ref pq_mbar = produced_mbar_q.unsafe_value()[q_idx_old]
                position = get_position(docontinue.value())
                pq_mbar.expect_bytes(
                    Int32(q_copy_rows * padded_depth * size_of[qkv_type]())
                )

                comptime if not decoding:
                    q_tma_op.async_copy(
                        q_producer(q_idx),
                        pq_mbar,
                        q_coord[
                            depth=depth,
                            decoding=decoding,
                        ](position.q_row, position.head_idx),
                    )

                else:
                    comptime for d_idx in range(depth // 64):
                        comptime d: Int = d_idx * 64
                        q_tma_op.async_copy_4d(
                            q_producer(q_idx),
                            pq_mbar,
                            (
                                d,
                                0,
                                Int(position.head_idx),
                                Int(position.q_row),
                            ),
                        )

                kv_head_idx = position.kv_head_idx()
                start, new_end = position.get_start_and_end_for_partitions[
                    page_size=KVLUTType.page_size
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

        comptime if needs_partial:
            # Compute valid page count for current K tile.
            var nvp_cur: UInt32
            if kv_tile_start_row + UInt32(BN) > end:
                nvp_cur = _num_valid_pages(end, kv_tile_start_row)
            else:
                nvp_cur = UInt32(KVPagedRows.num_pages)

            # K: use partial if this tile doesn't fill all pages.
            if nvp_cur < UInt32(KVPagedRows.num_pages):
                produce_kv_partial[is_k_side=True, wait=True](
                    k_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row,
                    kv_head_idx,
                    nvp_cur,
                )
            else:
                produce_kv[is_k_side=True, wait=True](
                    k_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row,
                    kv_head_idx,
                )

            # V: for previous K's tile. Use partial if that K was partial.
            if nvp_prev < UInt32(KVPagedRows.num_pages):
                produce_kv_partial[is_k_side=False, wait=True](
                    v_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row_prev,
                    kv_head_idx_prev,
                    nvp_prev,
                )
            else:
                produce_kv[is_k_side=False, wait=True](
                    v_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row_prev,
                    kv_head_idx_prev,
                )

            nvp_prev = nvp_cur
        else:
            produce_kv[is_k_side=True, wait=True](
                k_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row,
                kv_head_idx,
            )
            produce_kv[is_k_side=False, wait=True](
                v_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row_prev,
                kv_head_idx_prev,
            )
        kv_head_idx_prev = kv_head_idx
        kv_tile_start_row_prev = kv_tile_start_row

    # Exit: V for the last K tile.
    comptime if needs_partial:
        if nvp_prev < UInt32(KVPagedRows.num_pages):
            produce_kv_partial[is_k_side=False, wait=True](
                v_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row_prev,
                kv_head_idx_prev,
                nvp_prev,
            )
        else:
            produce_kv[is_k_side=False, wait=True](
                v_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row_prev,
                kv_head_idx_prev,
            )
    else:
        produce_kv[is_k_side=False, wait=True](
            v_tma_op,
            write_pipeline_states,
            position.prompt_idx,
            kv_tile_start_row_prev,
            kv_head_idx_prev,
        )


@always_inline
def output_reg_to_smem_st_matrix[
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
        element_type=DType.int32,
        linear_idx_type=DType.int32,
    ]()

    comptime for m_mma in range(num_m_mmas):
        comptime for i in range(padded_depth // 16):
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
            var output_frag = output_reg_tile.raw_load[width=8](
                m_mma * o_frag_size + i * 8
            ).cast[output_type]()
            var output_frag_f32_packed = bitcast[DType.float32, 4](output_frag)
            st_matrix[simd_width=4](offset, output_frag_f32_packed)


@always_inline
def output_reg_to_smem[
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
        Scalar[output_type], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    output_reg_tile: LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas, o_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ],
) -> LayoutTensor[
    output_type,
    Layout.row_major(BM, padded_depth),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
]:
    accum_smem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, padded_depth),
        address_space=AddressSpace.SHARED,
    ](q_smem)
    comptime use_stmatrix = accum_type == DType.float32 and padded_depth % 16 == 0 and size_of[
        output_type
    ]() == 2 and o_frag_size % 8 == 0

    comptime if use_stmatrix:
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
