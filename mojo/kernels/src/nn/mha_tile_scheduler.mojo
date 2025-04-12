# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv

from gpu.host.info import H100
from gpu.id import block_idx, grid_dim
from linalg.fast_div import FastDiv

from utils.index import Index, IndexList


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


@register_passable("trivial")
struct TileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    num_ctas: UInt32 = H100.sm_count,
    num_heads_per_cta: UInt32 = 1,
    schedule: MHASchedule = MHASchedule.DEFAULT,
]:
    # Linear work tile indix i.e. idx-th work among all possible workload.
    var idx: UInt32
    # Number of sequences in batch.
    var batch_size: UInt32
    # Maximum prompt length in batch.
    var max_prompt_len: UInt32
    var max_num_prompt_tiles: UInt32

    @always_inline
    fn __init__(out self, batch_size: UInt32, max_prompt_len: UInt32):
        self.idx = block_idx.x
        self.batch_size = batch_size
        self.max_prompt_len = max_prompt_len
        self.max_num_prompt_tiles = ceildiv(self.max_prompt_len, tile_shape)

    @always_inline
    fn get_current_work_info(self) -> WorkInfo:
        prompt_offset, head_idx, prompt_idx = self._index_to_coords()
        is_valid = (
            prompt_offset < self.max_prompt_len
            and head_idx < num_heads
            and prompt_idx < self.batch_size
        )

        return WorkInfo(
            prompt_offset,
            head_idx,
            prompt_idx,
            is_valid,
        )

    @always_inline
    fn advance(mut self):
        self.idx += num_ctas

    @always_inline
    fn fetch_next_work(mut self) -> WorkInfo:
        self.advance()
        return self.get_current_work_info()

    @always_inline
    fn _index_to_coords(self) -> Tuple[UInt32, UInt32, UInt32]:
        """Map the thread block's index to coordinates of work tile."""

        @parameter
        if schedule == MHASchedule.DEFAULT:
            return self._index_to_coords_default()

        return self._index_to_coords_prompt_rotate()

    @always_inline
    fn _index_to_coords_default(self) -> Tuple[UInt32, UInt32, UInt32]:
        # Frist dim, offset in prompt length
        quotient = self.idx // self.max_num_prompt_tiles
        prompt_offset = (self.idx % self.max_num_prompt_tiles) * tile_shape
        # head index
        head_idx = quotient % num_heads
        # prompt index
        prompt_idx = quotient // num_heads

        return (prompt_offset, head_idx, prompt_idx)

    @always_inline
    fn _index_to_coords_prompt_rotate(self) -> Tuple[UInt32, UInt32, UInt32]:
        # First dim, offset in prompt length
        quotient = self.idx // self.max_num_prompt_tiles
        prompt_tile_idx = self.idx % self.max_num_prompt_tiles
        # head index
        head_idx = quotient % num_heads
        # Switch the traverse direction in prompt for odd head.
        prompt_tile_idx = (
            prompt_tile_idx if head_idx % 2
            == 0 else self.max_num_prompt_tiles - 1 - prompt_tile_idx
        )
        prompt_offset = prompt_tile_idx * tile_shape
        # prompt index
        prompt_idx = quotient // num_heads

        return (prompt_offset, head_idx, prompt_idx)
