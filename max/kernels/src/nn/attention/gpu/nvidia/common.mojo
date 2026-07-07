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
"""Shared NVIDIA GPU attention primitives used by both SM90 and SM100 kernels.

This module hosts helpers that are not architecture-specific so that neither
`sm90/` nor `sm100/` has to import from the other. It currently provides:

- `elect()`: single-lane election via the `elect.sync` PTX instruction.
"""

from std.math import ceildiv, align_up
from std.memory import (
    bitcast,
)
from std.sys import size_of
from std.sys._assembly import inlined_assembly

import std.gpu.primitives.warp as warp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.compute.mma import st_matrix
from layout import (
    ComptimeInt,
    Coord,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.tile_layout import (
    Layout as InternalLayout,
)
from layout.swizzle import Swizzle
from layout.tensor_core_async import st_matrix_n_layout
from layout.tma_async import (
    create_split_tma,
    SplitLastDimTMATensorTile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import (
    MHAOperand,
)
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from nn.attention.mha_utils import (
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    get_start_and_end_for_partitions,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


@always_inline
def elect() -> Int32:
    # CAUTION: This function cannot be used to guard a `print`, else it will
    # introduce a deadlock!
    return inlined_assembly[
        """{
            .reg .b32 %re;
            .reg .pred %pa;
            mov.s32 $0, 0;
            elect.sync %re|%pa, $1;
            @%pa mov.s32 $0, 1;
        }""",
        Int32,
        constraints="=r,r",
    ](-1)


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
comptime _1d_row_major_tt_layout = InternalLayout[
    shape_types=Coord[Int64].element_types,
    stride_types=Coord[ComptimeInt[1]].element_types,
]
comptime ImmutTileTensor1D[dtype: DType] = TileTensor[
    dtype,
    _1d_row_major_tt_layout,
    ImmutAnyOrigin,
]


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

    @__allow_legacy_any_origin_fields
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

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

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
            type_of(ptr).origin,
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
                    self.prompt_idx,
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
                self.prompt_idx,
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
            self.prompt_idx, self.get_score_row()
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
    # The store packs 8 elements per lane through bitcast<f32x4>, which is
    # well-defined only when output_type is exactly bf16/f16.
    comptime assert (
        size_of[output_type]() == 2
    ), "output_reg_to_smem_st_matrix only supports bf16/f16 output_type"

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
