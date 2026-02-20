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

from os.atomic import Atomic

import gpu.primitives.warp as warp
from builtin.device_passable import DevicePassable
from gpu.host.info import H100
from gpu import block_idx, thread_idx
from gpu.sync import barrier, named_barrier
from nn.mha_fa3_utils import NullPointer, OptionalPointer

from builtin.device_passable import DevicePassable


@fieldwise_init
struct WorkInfo(Stringable, TrivialRegisterPassable, Writable):
    # (query_offset, head_idx, sequence idx in batch)
    var prompt_offset: UInt32
    var head_idx: UInt32
    var prompt_idx: UInt32
    # Currently each work tile travser entire cache length.
    # TODO: Add starting kv index in cache len dim
    # var kv_start: UInt32 = 0
    # var kv_end: UInt32 = 0
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "(",
            self.prompt_offset,
            ", ",
            self.head_idx,
            ", ",
            self.prompt_idx,
            ", ",
            self.is_valid_tile,
            ")",
        )


struct SeqInfo(TrivialRegisterPassable):
    var seq_len: UInt32
    var start_of_seq: UInt32
    var prompt_offset: UInt32
    var head_idx: UInt32
    var prompt_idx: UInt32

    @always_inline
    fn __init__(
        out self, seq_len: UInt32, start_of_seq: UInt32, work: WorkInfo
    ):
        self.seq_len = seq_len
        self.start_of_seq = start_of_seq
        self.prompt_offset = work.prompt_offset
        self.head_idx = work.head_idx
        self.prompt_idx = work.prompt_idx

    @always_inline
    fn is_valid(self) -> Bool:
        return self.seq_len > self.prompt_offset

    @staticmethod
    @always_inline
    fn create[
        ValidLengthType: OptionalPointer,
        //,
    ](
        work: WorkInfo,
        valid_length: ValidLengthType,
        max_seq_len: UInt32,
    ) -> SeqInfo:
        var batch_idx: UInt32 = work.prompt_idx

        comptime if not ValidLengthType.is_null:
            # treat valid_lengths as a input_row_offsets
            ptr = rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](
                valid_length.value()
            )
            seq = ptr.load[width=2](batch_idx)
            start_of_seq = warp.broadcast(seq[0])
            end_of_seq = warp.broadcast(seq[1])
            seq_len = end_of_seq - start_of_seq
            return SeqInfo(seq_len, start_of_seq, work)
        else:
            seq_len = max_seq_len
            return SeqInfo(seq_len, 0, work)


@fieldwise_init
struct MHASchedulerSynchronization(TrivialRegisterPassable):
    var _value: Int32

    comptime NONE = Self(0)  # use for TMA
    comptime PRODUCER = Self(1)  # use for copy-async
    comptime ALL = Self(2)  # use when all threads are synced
    comptime DEFAULT = Self.PRODUCER  # default is currently copy-async

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# This class is constructed within the fully inlined kernel,
# so unneeded fields can be optimized away.
struct MHATileState(TrivialRegisterPassable):
    # Linear work tile index i.e. idx-th work among all possible workload.
    var idx: UInt32
    var sidx_ptr: UnsafePointer[
        UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var max_idx: UInt32

    @always_inline
    fn __init__(
        out self,
        idx: UInt32,
        sidx_ptr: UnsafePointer[
            UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        max_idx: UInt32,
    ):
        self.idx = idx
        self.sidx_ptr = sidx_ptr
        self.max_idx = max_idx

    @always_inline
    fn is_valid(self, idx: UInt32) -> Bool:
        return idx < self.max_idx

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid(self.idx)


struct MHATileSummary[ValidLengthType: OptionalPointer](
    TrivialRegisterPassable
):
    # Number of sequences in batch.
    var batch_size: UInt32
    # Maximum num tiles.
    var max_num_prompt_tiles: UInt32
    var valid_length: Self.ValidLengthType
    var max_seq_len: UInt32

    @always_inline
    fn __init__(
        out self,
        batch_size: UInt32,
        max_num_prompt_tiles: UInt32,
        valid_length: Self.ValidLengthType,
        max_seq_len: UInt32,
    ):
        self.batch_size = batch_size
        self.max_num_prompt_tiles = max_num_prompt_tiles
        self.valid_length = valid_length
        self.max_seq_len = max_seq_len

    @always_inline
    fn _index_to_coords[
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        """Map the thread block's index to coordinates of work tile."""

        comptime if schedule == MHASchedule.PROMPT_ROTATE:
            return self._index_to_coords_prompt_rotate[num_heads](idx)

        return self._index_to_coords_default[num_heads](idx)

    @always_inline
    fn _index_to_coords_default[
        num_heads: UInt32
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        # First dim, offset in prompt length
        #
        # The goal is to keep kv in l2 cache.
        # As kv is constant with prompt_tile_idx,
        # this changes fastest.
        # kv changes for every `group` iterations of
        # head-idx, and for every iteration of `prompt-idx`.
        # Thus, we have head_idx vary fastest.
        #
        # self.idx's max-value = self.max_num_prompt_tiles*num_heads*batch_size
        quotient = idx // self.max_num_prompt_tiles
        prompt_tile_idx = idx % self.max_num_prompt_tiles
        # max value = num_heads-1
        # head index
        # changes kv whenever head_idx//group changes
        head_idx = quotient % num_heads
        # max value = batch_size-1
        # prompt index
        # changes kv
        prompt_idx = quotient // num_heads

        return (prompt_tile_idx, head_idx, prompt_idx)

    @always_inline
    fn _index_to_coords_prompt_rotate[
        num_heads: UInt32
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        # First dim, offset in prompt length
        quotient = idx // self.max_num_prompt_tiles
        prompt_tile_idx = idx % self.max_num_prompt_tiles
        # head index
        head_idx = quotient % num_heads
        # Switch the traverse direction in prompt for odd head.
        prompt_tile_idx = (
            prompt_tile_idx if head_idx % 2
            == 0 else self.max_num_prompt_tiles - 1 - prompt_tile_idx
        )
        # prompt index
        prompt_idx = quotient // num_heads

        return (prompt_tile_idx, head_idx, prompt_idx)

    @always_inline
    fn get_current_work_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> WorkInfo:
        prompt_tile_idx, head_idx, prompt_idx = self._index_to_coords[
            num_heads, schedule
        ](idx)
        is_valid = (
            prompt_tile_idx < self.max_num_prompt_tiles
            and head_idx < num_heads
            and prompt_idx < self.batch_size
        )

        return WorkInfo(
            prompt_tile_idx * tile_shape,
            head_idx,
            prompt_idx,
            is_valid,
        )

    @always_inline
    fn unsafe_get_current_work_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> WorkInfo:
        prompt_tile_idx, head_idx, prompt_idx = self._index_to_coords[
            num_heads, schedule
        ](idx)

        debug_assert(prompt_tile_idx < self.max_num_prompt_tiles)
        debug_assert(head_idx < num_heads)
        debug_assert(prompt_idx < self.batch_size)

        return WorkInfo(
            prompt_tile_idx * tile_shape,
            head_idx,
            prompt_idx,
            True,
        )

    @always_inline
    fn max_idx(self, num_heads: UInt32) -> UInt32:
        return self.max_num_prompt_tiles * self.batch_size * num_heads

    @always_inline
    fn get_current_work_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: MHATileState) -> WorkInfo:
        return self.get_current_work_info[tile_shape, num_heads, schedule](
            idx.idx
        )

    @staticmethod
    @always_inline
    fn grid_dim[
        num_heads: UInt32
    ](max_num_prompt_tiles: UInt32, batch_size: UInt32) -> Tuple[Int, Int, Int]:
        return (Int(max_num_prompt_tiles), Int(num_heads), Int(batch_size))

    @always_inline
    fn seq_info(self, work: WorkInfo) -> SeqInfo:
        return SeqInfo.create(work, self.valid_length, self.max_seq_len)

    @always_inline
    fn unsafe_seq_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> SeqInfo:
        work = self.unsafe_get_current_work_info[
            tile_shape, num_heads, schedule
        ](idx)
        return SeqInfo.create(work, self.valid_length, self.max_seq_len)

    @always_inline
    fn unsafe_seq_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, state: MHATileState) -> SeqInfo:
        return self.unsafe_seq_info[tile_shape, num_heads, schedule](state.idx)


trait MHATileScheduler(Copyable, DevicePassable, TrivialRegisterPassable):
    comptime may_advance: Bool
    comptime mha_schedule: MHASchedule

    """The MHATileScheduler trait describes a schedule for the persistent kernel.
    """

    fn get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        """Returns the current `WorkInfo`."""
        ...

    @always_inline
    fn advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        """Advance state to the next work item.
        `func` must return a `Bool` indicating whether there is more work.
        Returns `True` if there is more work."""
        ...

    @staticmethod
    @always_inline
    fn grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        """Return the grid_dim required for the kernel."""
        ...

    @always_inline
    fn initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[
            UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        """Create the initial state object."""
        ...

    @always_inline
    fn unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        ...


@fieldwise_init
struct MHASchedule(TrivialRegisterPassable):
    var _value: Int32

    comptime DEFAULT = Self(0)
    comptime PROMPT_ROTATE = Self(1)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


struct TransientScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
](Defaultable, MHATileScheduler, TrivialRegisterPassable):
    comptime may_advance: Bool = False
    comptime mha_schedule: MHASchedule = MHASchedule.DEFAULT

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return (
            "TransientScheduler[tile_shape = "
            + String(Self.tile_shape)
            + ", num_heads = "
            + String(Self.num_heads)
            + "]"
        )

    @always_inline
    fn __init__(out self):
        pass

    @always_inline
    fn get_current_work_info(self) -> WorkInfo:
        return WorkInfo(
            UInt32(block_idx.x) * Self.tile_shape,
            UInt32(block_idx.y),
            UInt32(block_idx.z),
            True,
        )

    @always_inline
    fn get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        return self.get_current_work_info()

    @always_inline
    fn advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        return None

    @staticmethod
    @always_inline
    fn grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        return (
            Int(max_num_prompt_tiles),
            Int(Self.num_heads),
            Int(batch_size),
        )

    @always_inline
    fn initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[
            UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        return MHATileState.__init__(0, ptr, 1)

    @always_inline
    fn unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        return SeqInfo.create(
            self.get_current_work_info(), ts.valid_length, ts.max_seq_len
        )


struct TileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    num_ctas: UInt32 = UInt32(H100.sm_count),
    schedule: MHASchedule = MHASchedule.DEFAULT,
](Defaultable, MHATileScheduler, TrivialRegisterPassable):
    comptime may_advance: Bool = True
    comptime mha_schedule: MHASchedule = Self.schedule

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return (
            "TileScheduler[tile_shape = "
            + String(Self.tile_shape)
            + ", num_heads = "
            + String(Self.num_heads)
            + ", num_ctas = "
            + String(Self.num_ctas)
            + ", schedule = "
            + String(Self.schedule._value)
            + "]"
        )

    @always_inline
    fn __init__(out self):
        pass

    @always_inline
    fn get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        return ts.get_current_work_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state)

    @always_inline
    fn fetch_next_work(
        self,
        ts: MHATileSummary,
        mut state: MHATileState,
    ) -> WorkInfo:
        state.idx += Self.num_ctas
        return ts.get_current_work_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)

    @always_inline
    fn advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        state.idx += Self.num_ctas
        if not state.is_valid(state.idx):
            return None
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)

    @staticmethod
    @always_inline
    fn grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # NOTE: mha_sm90 assumes `grid_dim` limits the grid
        # size for persistent kernels, so that it doesn't
        # need to check the first `work_info` for validity.
        bx, by, bz = MHATileSummary[NullPointer[DType.uint32]].grid_dim[
            Self.num_heads
        ](max_num_prompt_tiles, batch_size)
        size = min(Int(Self.num_ctas), bx * by * bz)
        return (size, 1, 1)

    @always_inline
    fn initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[
            UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        return MHATileState(
            UInt32(block_idx.x), ptr, tile_summary.max_idx(Self.num_heads)
        )

    @always_inline
    fn unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)


struct QueuedTileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    decoding: Bool,
    num_ctas: UInt32 = UInt32(H100.sm_count),
    schedule: MHASchedule = MHASchedule.DEFAULT,
](DevicePassable, MHATileScheduler, TrivialRegisterPassable):
    """
    If `decoding == False`, then `num_heads` is `q_num_heads`.
    If `decoding == True`, then `num_heads` is `kv_num_heads`.
    """

    # Linear work tile index i.e. idx-th work among all possible workload.
    var gidx_ptr: UnsafePointer[
        UInt32, MutAnyOrigin, address_space = AddressSpace.GLOBAL
    ]

    comptime may_advance: Bool = True
    comptime mha_schedule: MHASchedule = Self.schedule

    @always_inline
    fn __init__(
        out self,
        gidx_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    ):
        self.gidx_ptr = gidx_ptr.address_space_cast[AddressSpace.GLOBAL]()

    @always_inline
    fn get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        return ts.get_current_work_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state)

    @always_inline
    fn advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        """The parameter `func` must return a `Bool` indicating whether the `WorkInfo` arg is valid.
        This function returns whether the current idx corresponds to a valid `WorkInfo`.
        Note that if `MHASchedulerSynchronization` is `NONE`, then we assume it is only called by `thread_idx.x==0`.
        """

        comptime if producer:
            if thread_idx.x == 0:
                var idx: UInt32
                while True:
                    idx = Atomic.fetch_add(self.gidx_ptr, 1)
                    if not state.is_valid(idx):
                        comptime if sync == MHASchedulerSynchronization.NONE:
                            state.idx = idx
                            state.sidx_ptr.store(offset=pipeline_idx, val=idx)
                            return None

                        else:
                            break
                    var seq_info: SeqInfo = ts.unsafe_seq_info[
                        Self.tile_shape, Self.num_heads, Self.schedule
                    ](idx)

                    comptime if not Self.decoding:
                        if seq_info.is_valid():
                            comptime if sync == MHASchedulerSynchronization.NONE:
                                state.idx = idx
                                state.sidx_ptr.store(
                                    offset=pipeline_idx, val=idx
                                )
                                # tma with producer doesn't need to sync
                                return seq_info
                            else:
                                break

                state.sidx_ptr.store(offset=pipeline_idx, val=idx)

            # producer needs to sync before loading
            comptime if sync == MHASchedulerSynchronization.PRODUCER:
                named_barrier[128,](id=1)

        comptime if sync == MHASchedulerSynchronization.ALL:
            barrier()

        # when !ALL, consumers rely on `async_copy_arrive`
        state.idx = warp.broadcast(state.sidx_ptr.load(pipeline_idx))
        if not state.is_valid():
            return None
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state)

    @staticmethod
    @always_inline
    fn grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # NOTE: mha_sm90 assumes `grid_dim` limits the grid
        # size for persistent kernels, so that it doesn't
        # need to check the first `work_info` for validity.
        bx, by, bz = MHATileSummary[NullPointer[DType.uint32]].grid_dim[
            Self.num_heads
        ](max_num_prompt_tiles, batch_size)
        size = min(Int(Self.num_ctas), bx * by * bz)
        return (size, 1, 1)

    @always_inline
    fn initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[
            UInt32, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        state = MHATileState(
            UInt32(block_idx.x), ptr, tile_summary.max_idx(Self.num_heads)
        )

        if thread_idx.x == 0:
            state.sidx_ptr.store(state.idx)
        return state

    @always_inline
    fn unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)

    # `trait DevicePassable` implementation
    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @no_inline
    @staticmethod
    fn get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "QueuedTileScheduler[tile_shape = ",
            String(Self.tile_shape),
            ", num_heads = ",
            String(Self.num_heads),
            ", decoding = ",
            String(Self.decoding),
            ", num_ctas = ",
            String(Self.num_ctas),
            ", schedule = ",
            String(Self.schedule._value),
            "]",
        )
