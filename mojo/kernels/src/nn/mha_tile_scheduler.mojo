# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from collections import OptionalReg
from math import ceildiv
from gpu.id import block_idx
from gpu.host.info import H100
from gpu.id import block_idx, thread_idx
from gpu.memory import AddressSpace
from gpu.sync import named_barrier, barrier
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
import gpu.warp as warp
from gpu import WARP_SIZE
from memory import UnsafePointer
from nn.mha_mask import MHAMask, TileMaskStatus
from utils.index import Index, IndexList
from linalg.fast_div import FastDiv
from os.atomic import Atomic


@value
@register_passable("trivial")
struct WorkInfo(Stringable, Writable):
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
    fn write_to[W: Writer](self, mut writer: W):
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


@value
@register_passable("trivial")
struct SeqInfo:
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
        ragged: Bool
    ](
        work: WorkInfo,
        valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        max_seq_len: UInt32,
    ) -> SeqInfo:
        var batch_idx: UInt32 = work.prompt_idx

        @parameter
        if ragged:
            # treat valid_lengths as a input_row_offsets
            start_of_seq = valid_length[Int(batch_idx)]
            end_of_seq = valid_length[Int(batch_idx + 1)]
            seq_len = end_of_seq - start_of_seq
            return SeqInfo(seq_len, start_of_seq, work)
        else:
            seq_len = max_seq_len
            return SeqInfo(seq_len, 0, work)


@value
@register_passable("trivial")
struct MHASchedulerSynchronization:
    var _value: Int32

    alias NONE = Self(0)  # use for TMA
    alias PRODUCER = Self(1)  # use for copy-async
    alias ALL = Self(2)  # use when all threads are synced
    alias DEFAULT = Self.PRODUCER  # default is currently copy-async

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# This class is constructed within the fully inlined kernel,
# so unneeded fields can be optimized away.
@register_passable("trivial")
struct MHATileState:
    # Linear work tile index i.e. idx-th work among all possible workload.
    var idx: UInt32
    var sidx_ptr: UnsafePointer[UInt32, address_space = AddressSpace.SHARED]
    var max_idx: UInt32

    @always_inline
    fn __init__(
        out self,
        idx: UInt32,
        sidx_ptr: UnsafePointer[UInt32, address_space = AddressSpace.SHARED],
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


@value
@register_passable("trivial")
struct MHATileSummary:
    # Number of sequences in batch.
    var batch_size: UInt32
    # Maximum num tiles.
    var max_num_prompt_tiles: UInt32
    var valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var max_seq_len: UInt32

    @always_inline
    fn __init__(
        out self,
        batch_size: UInt32,
        max_num_prompt_tiles: UInt32,
        valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
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

        @parameter
        if schedule == MHASchedule.PROMPT_ROTATE:
            return self._index_to_coords_prompt_rotate[num_heads](idx)

        return self._index_to_coords_default[num_heads](idx)

    @always_inline
    fn _index_to_coords_default[
        num_heads: UInt32
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        # Frist dim, offset in prompt length
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
    fn seq_info[ragged: Bool](self, work: WorkInfo) -> SeqInfo:
        return SeqInfo.create[ragged](work, self.valid_length, self.max_seq_len)

    @always_inline
    fn unsafe_seq_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        ragged: Bool,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> SeqInfo:
        work = self.unsafe_get_current_work_info[
            tile_shape, num_heads, schedule
        ](idx)
        return SeqInfo.create[ragged](work, self.valid_length, self.max_seq_len)

    @always_inline
    fn unsafe_seq_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        ragged: Bool,
        schedule: MHASchedule,
    ](self, state: MHATileState) -> SeqInfo:
        return self.unsafe_seq_info[tile_shape, num_heads, ragged, schedule](
            state.idx
        )


@register_passable("trivial")
trait MHATileScheduler:
    alias may_advance: Bool
    alias mha_schedule: MHASchedule

    """The MHATileScheduler trait describes a schedule for the persistent kernel.
    """

    fn get_current_work_info(
        self, ts: MHATileSummary, state: MHATileState
    ) -> WorkInfo:
        """Returns the current `WorkInfo`."""
        ...

    @always_inline
    fn advance[
        ragged: Bool,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary,
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
    fn initial_state(
        self,
        ptr: UnsafePointer[UInt32, address_space = AddressSpace.SHARED],
        tile_summary: MHATileSummary,
    ) -> MHATileState:
        """Create the initial state object."""
        ...

    @always_inline
    fn unsafe_seq_info[
        ragged: Bool
    ](self, ts: MHATileSummary, state: MHATileState) -> SeqInfo:
        ...


@value
@register_passable("trivial")
struct MHASchedule:
    var _value: Int32

    alias DEFAULT = Self(0)
    alias PROMPT_ROTATE = Self(1)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct TransientScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
](MHATileScheduler):
    alias may_advance: Bool = False
    alias mha_schedule: MHASchedule = MHASchedule.DEFAULT

    @always_inline
    fn __init__(out self):
        pass

    @always_inline
    fn get_current_work_info(self) -> WorkInfo:
        return WorkInfo(
            block_idx.x * tile_shape,
            block_idx.y,
            block_idx.z,
            True,
        )

    @always_inline
    fn get_current_work_info(
        self, ts: MHATileSummary, state: MHATileState
    ) -> WorkInfo:
        return self.get_current_work_info()

    @always_inline
    fn advance[
        ragged: Bool,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary,
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
    fn initial_state(
        self,
        ptr: UnsafePointer[UInt32, address_space = AddressSpace.SHARED],
        tile_summary: MHATileSummary,
    ) -> MHATileState:
        return MHATileState.__init__(0, ptr, 1)

    @always_inline
    fn unsafe_seq_info[
        ragged: Bool,
    ](self, ts: MHATileSummary, state: MHATileState) -> SeqInfo:
        return SeqInfo.create[ragged](
            self.get_current_work_info(), ts.valid_length, ts.max_seq_len
        )


@value
@register_passable("trivial")
struct TileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    num_ctas: UInt32 = H100.sm_count,
    schedule: MHASchedule = MHASchedule.DEFAULT,
](MHATileScheduler):
    alias may_advance: Bool = True
    alias mha_schedule: MHASchedule = schedule

    @always_inline
    fn __init__(out self):
        pass

    @always_inline
    fn get_current_work_info(
        self, ts: MHATileSummary, state: MHATileState
    ) -> WorkInfo:
        return ts.get_current_work_info[tile_shape, num_heads, schedule](state)

    @always_inline
    fn fetch_next_work(
        self,
        ts: MHATileSummary,
        mut state: MHATileState,
    ) -> WorkInfo:
        state.idx += num_ctas
        return ts.get_current_work_info[tile_shape, num_heads, schedule](
            state.idx
        )

    @always_inline
    fn advance[
        ragged: Bool,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary,
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        state.idx += num_ctas
        if not state.is_valid(state.idx):
            return None
        return ts.unsafe_seq_info[tile_shape, num_heads, ragged, schedule](
            state.idx
        )

    @staticmethod
    @always_inline
    fn grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # NOTE: mha_sm90 assumes `grid_dim` limits the grid
        # size for persistent kernels, so that it doesn't
        # need to check the first `work_info` for validity.
        bx, by, bz = MHATileSummary.grid_dim[num_heads](
            max_num_prompt_tiles, batch_size
        )
        size = min(Int(num_ctas), bx * by * bz)
        return (size, 1, 1)

    @always_inline
    fn initial_state(
        self,
        ptr: UnsafePointer[UInt32, address_space = AddressSpace.SHARED],
        tile_summary: MHATileSummary,
    ) -> MHATileState:
        return MHATileState(
            block_idx.x, ptr, tile_summary.max_idx(Self.num_heads)
        )

    @always_inline
    fn unsafe_seq_info[
        ragged: Bool,
    ](self, ts: MHATileSummary, state: MHATileState) -> SeqInfo:
        return ts.unsafe_seq_info[tile_shape, num_heads, ragged, Self.schedule](
            state.idx
        )


@value
@register_passable("trivial")
struct QueuedTileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    decoding: Bool,
    num_ctas: UInt32 = H100.sm_count,
    schedule: MHASchedule = MHASchedule.DEFAULT,
](MHATileScheduler):
    """
    If `decoding == False`, then `num_heads` is `q_num_heads`.
    If `decoding == True`, then `num_heads` is `kv_num_heads`.
    """

    # Linear work tile index i.e. idx-th work among all possible workload.
    var gidx_ptr: UnsafePointer[UInt32, address_space = AddressSpace.GLOBAL]

    alias may_advance: Bool = True
    alias mha_schedule: MHASchedule = schedule

    @always_inline
    fn __init__(
        out self,
        gidx_ptr: UnsafePointer[UInt32],
    ):
        self.gidx_ptr = gidx_ptr.address_space_cast[
            address_space = AddressSpace.GLOBAL
        ]()

    @always_inline
    fn get_current_work_info(
        self, ts: MHATileSummary, state: MHATileState
    ) -> WorkInfo:
        return ts.get_current_work_info[tile_shape, num_heads, schedule](state)

    @always_inline
    fn advance[
        ragged: Bool,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary,
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        """The parameter `func` must return a `Bool` indicating whether the `WorkInfo` arg is valid.
        This function returns whether the current idx corresponds to a valid `WorkInfo`.
        Note that if `MHASchedulerSynchronization` is `NONE`, then we assume it is only called by `thread_idx.x==0`.
        """

        @parameter
        if producer:
            if thread_idx.x == 0:
                var idx: UInt32
                while True:
                    idx = Atomic._fetch_add(self.gidx_ptr, 1)
                    if not state.is_valid(idx):

                        @parameter
                        if sync == MHASchedulerSynchronization.NONE:
                            state.idx = idx
                            state.sidx_ptr.store(offset=pipeline_idx, val=idx)
                            return None

                        else:
                            break
                    var seq_info: SeqInfo = ts.unsafe_seq_info[
                        tile_shape, num_heads, ragged, schedule
                    ](idx)

                    @parameter
                    if not decoding:
                        if seq_info.is_valid():

                            @parameter
                            if sync == MHASchedulerSynchronization.NONE:
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
            @parameter
            if sync == MHASchedulerSynchronization.PRODUCER:
                named_barrier[128, id=1]()

        @parameter
        if sync == MHASchedulerSynchronization.ALL:
            barrier()

        # when !ALL, consumers rely on `async_copy_arrive`
        state.idx = warp.broadcast(state.sidx_ptr.load(pipeline_idx))
        if not state.is_valid():
            return None
        return ts.unsafe_seq_info[tile_shape, num_heads, ragged, schedule](
            state
        )

    @staticmethod
    @always_inline
    fn grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # NOTE: mha_sm90 assumes `grid_dim` limits the grid
        # size for persistent kernels, so that it doesn't
        # need to check the first `work_info` for validity.
        bx, by, bz = MHATileSummary.grid_dim[num_heads](
            max_num_prompt_tiles, batch_size
        )
        size = min(Int(num_ctas), bx * by * bz)
        return (size, 1, 1)

    @always_inline
    fn initial_state(
        self,
        ptr: UnsafePointer[UInt32, address_space = AddressSpace.SHARED],
        tile_summary: MHATileSummary,
    ) -> MHATileState:
        state = MHATileState(
            block_idx.x, ptr, tile_summary.max_idx(Self.num_heads)
        )

        if thread_idx.x == 0:
            state.sidx_ptr.store(state.idx)
        return state

    @always_inline
    fn unsafe_seq_info[
        ragged: Bool,
    ](self, ts: MHATileSummary, state: MHATileState) -> SeqInfo:
        return ts.unsafe_seq_info[tile_shape, num_heads, ragged, Self.schedule](
            state.idx
        )
