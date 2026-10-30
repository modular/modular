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
"""Defines the `TensorEngine` that views Blackwell Tensor Memory (TMEM).

TMEM is the SM100 accumulator memory: a grid of `TMEM_NUM_LANES` lanes by
`TMEM_NUM_COLS` columns per CTA, each cell 32 bits. A tile over `TMemEngine`
places its elements on that grid, so its layout has a lane stride of `1` and
a column stride of `TMEM_NUM_LANES`: the whole accumulator is
`(128, 512):(1, 128)`, an `N`-column allocation is `(128, N):(1, 128)`, and
nested shapes express the lane placement of other MMA configurations. Lanes
are the stride-1 dimension because every allocation has 128 of them, so the
conversion between a flat index and a cell is a shift and a mask that no
column count enters. The hardware names a cell by a `UInt32` whose upper half
is the lane and whose lower half is the column; that encoding stays inside
the engine, whose `offset` and `distance` translate between it and the
layout's flat indices.

TMEM has no pointer representation, so a tile over this engine has no element
loads or stores; `unsafe_ptr` rejects them at compile time. Data moves through
`copy_from`. Copying a register or shared-memory tile into a TMEM tile issues
`tcgen05.st`, copying a TMEM tile into one issues `tcgen05.ld`. Each
instruction names at most 64 registers, and a copy moves a row in slices of
64 columns with one wait per slice, so a wide row never holds more than 64
staging registers live. The `_async` variants issue every instruction of a
copy without waiting, so several copies can share one wait issued through
`wait_store` or `wait_load`. An async copy stages its whole row, so its row is
capped at 64 columns; a wider row is tiled into slices with one copy each.

Access is warp-collective: `tcgen05.ld` and `tcgen05.st` take a collective
base in the address and place each thread's registers relative to it, so a
TMEM operand is always a base, never `base + lane`, and its layout says which
cells this thread's registers hold. Two views exist. The row view of a warp's
`(32, N)` tile is its row 0 (`tile.tile[1, N](coord[0, 0])`): the
`32x32b` shape hands thread `t` lane `base + t`, so the copy means "these
columns of the lane this thread owns". The `16x256b` view of a `(16, N)` half
of that tile, `tmem_16x256b(block)`, has layout `TMem16x256bLayout[N]`, a
column pair by two lanes 8 apart, repeated every 8 columns: the `16x256b`
shape hands thread `t` the cells of that layout from lane `base + t // 4`,
column `2 * (t % 4)`, in the layout's flat order, which is the MMA accumulator
fragment order. `tmem_cell` turns a view coordinate back into the lane and
column this thread's element occupies.

The design is written up in `Mojo/docs/stdlib/internal/gpua/tmem_engine.md`.
"""

from std.math import ceildiv
from std.sys import align_of, bit_width_of, size_of
from std.sys.info import is_gpu

from max.gpu.compute.arch.tcgen05 import (
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_st,
    tcgen05_store_wait,
)

from layout import Coord, CoordLike, Idx, TensorLayout, coord
from layout.tile_layout import Layout as TileLayout
from max.gpu import lane_id
from layout.tile_tensor import TileTensor
from layout.tensor_engine import (
    TensorEngine,
    _copy_widen_factor,
    _layout_row_major,
)


comptime TMEM_NUM_LANES = 128
"""The number of TMEM lanes (rows) per CTA.

Also the column stride of a layout over `TMemEngine`, whose flat indices walk
the lane-by-column grid lane-first. Every allocation has this many lanes,
whatever its column count, so the stride is the same for every tile.
"""

comptime TMEM_NUM_COLS = 512
"""The number of 32-bit TMEM columns per lane."""

comptime _LANE_SHIFT = 16
"""The bit position of the lane in an encoded TMEM address."""

comptime _LANE_BITS = 7
"""Log2 of `TMEM_NUM_LANES`: the bits a flat grid index spends on the
lane."""


@inline(.always)
def _grid_index(addr: UInt32) -> UInt32:
    """Returns the cell `addr` encodes as a flat lane-first index on the
    grid, `col * TMEM_NUM_LANES + lane`.
    """
    var lane = addr >> _LANE_SHIFT
    var col = addr & ((1 << _LANE_SHIFT) - 1)
    return (col << _LANE_BITS) | lane


@inline(.always)
def _encode(grid_index: UInt32) -> UInt32:
    """Returns the address of the cell at flat grid index `grid_index`."""
    var lane = grid_index & (TMEM_NUM_LANES - 1)
    var col = grid_index >> _LANE_BITS
    return (lane << _LANE_SHIFT) | col


def _is_lane_run[L: TensorLayout]() -> Bool:
    """Returns True if `L` views consecutive columns of one lane.

    Every flat dimension of extent above 1 must advance by whole columns
    without gaps, so flat index `i` is column `i` of the lane: the trailing
    stride is `TMEM_NUM_LANES` and each earlier one is the product of the
    trailing shapes times that. Extent-1 dimensions are skipped, so a `(1, N)`
    row view qualifies whatever its lane stride.
    """
    comptime if not L.all_dims_known:
        return False
    var expected = TMEM_NUM_LANES
    comptime for i in range(L.flat_rank - 1, -1, -1):
        comptime if L.static_shape[i] != 1:
            if L.static_stride[i] != expected:
                return False
            expected *= L.static_shape[i]
    return True


comptime _FRAGMENT_COLS_PER_REPEAT = 8
"""Columns of one lane that one `16x256b` repeat covers: 256 bits of 32-bit
cells."""
comptime _FRAGMENT_LANE_PAIR_STRIDE = 8
"""Lanes between the two rows of a thread's `16x256b` fragment."""

comptime TMem16x256bLayout[num_cols: Int] = TileLayout(
    shape=Coord(coord[2, 2], Idx[num_cols // _FRAGMENT_COLS_PER_REPEAT]),
    stride=Coord(
        coord[TMEM_NUM_LANES, _FRAGMENT_LANE_PAIR_STRIDE],
        Idx[_FRAGMENT_COLS_PER_REPEAT * TMEM_NUM_LANES],
    ),
)
"""This thread's `16x256b` fragment of a `(16, num_cols)` lane block.

The first mode is a 2 by 2 quad of cells, a column pair (stride one column)
by two lanes 8 apart; the second mode repeats the quad every 8 columns. The
offsets are relative to the block base plus the lane and column the hardware
adds for this thread, and the flat order, column within the pair fastest,
then lane, then repeat, is the order `tcgen05.ld.16x256b` fills registers in.

Parameters:
    num_cols: The columns of the block, a multiple of 8.
"""


comptime TMem16x256bRegisters[num_cols: Int] = TileLayout(
    shape=Coord(coord[2, 2], Idx[num_cols // _FRAGMENT_COLS_PER_REPEAT]),
    stride=Coord(coord[1, 2], Idx[4]),
)
"""The register tile layout for a `TMem16x256bLayout[num_cols]` view.

Same shape as the view, packed in the view's flat order, so copying a
fragment into a tile with this layout lets a register be addressed by the
coordinate of its cell, and `copy_from` moves it without reordering.

Parameters:
    num_cols: The columns of the block, a multiple of 8.
"""


@fieldwise_init
struct TMemAccessShape(EnumLike, Equatable, TrivialRegisterPassable):
    """Names a supported `tcgen05.ld` or `tcgen05.st` data movement shape."""

    var _value: Int32

    comptime ROW_32X32B = Self(0)
    """Selects the 32x32b row access shape."""

    comptime BLOCK_16X256B = Self(1)
    """Selects the 16x256b block access shape."""

    comptime _enum_case_names = ParameterList.of[
        "ROW_32X32B".value,
        "BLOCK_16X256B".value,
    ].values
    comptime _enum_case_types = TypeList.splat[
        ParameterList[Self._enum_case_names].length, NoneType
    ].values

    @inline(.always)
    def _get_enum_discriminant(self) -> Int:
        return Int(self._value)

    @inline(.always)
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        comptime assert False, "TMemAccessShape has no payload"


def _access_shape_for_layout[L: TensorLayout]() -> TMemAccessShape:
    """Returns the instruction shape selected by this thread's TMEM view."""
    comptime if _is_lane_run[L]():
        return TMemAccessShape.ROW_32X32B
    else:
        comptime assert L.all_dims_known and L.flat_rank == 3, (
            "TMemEngine copies need a (1, N) lane row or a"
            " `tmem_16x256b` view of a (16, N) block"
        )
        comptime assert (
            L.static_shape[0] == 2
            and L.static_stride[0] == TMEM_NUM_LANES
            and L.static_shape[1] == 2
            and L.static_stride[1] == _FRAGMENT_LANE_PAIR_STRIDE
            and L.static_stride[2] == _FRAGMENT_COLS_PER_REPEAT * TMEM_NUM_LANES
        ), "TMemEngine: unsupported TMEM data movement layout"
        return TMemAccessShape.BLOCK_16X256B


@inline(.always)
def _gather[
    ptr_dtype: DType,
    mut: Bool,
    origin: Origin[mut=mut],
    address_space: AddressSpace,
    L: TensorLayout,
    buf_dtype: DType,
    num_cols: Int,
    //,
    widen: Int,
    start: Int,
](
    ptr: Pointer[Scalar[ptr_dtype], origin, address_space=address_space],
    layout: L,
    mut buf: Array[Scalar[buf_dtype], num_cols],
):
    """Loads elements `start` to `start + num_cols` of the tile at `ptr` into
    `buf`, cast to `buf_dtype`.

    With `widen > 1` the source is a contiguous run and is read in
    `widen`-wide vectors at raw offsets; otherwise each element's offset goes
    through `layout`.
    """
    comptime alignment = align_of[SIMD[ptr_dtype, widen]]() if is_gpu() else 1
    comptime if widen > 1:
        comptime for i in range(num_cols // widen):
            Pointer(to=buf[i * widen]).unsafe_bitcast[
                SIMD[buf_dtype, widen]
            ]()[] = ptr.unsafe_load[width=widen, alignment=alignment](
                start + i * widen
            ).cast[
                buf_dtype
            ]()
    else:
        comptime for i in range(num_cols):
            buf[i] = ptr.unsafe_load[width=1, alignment=alignment](
                layout(Idx[start + i])
            ).cast[buf_dtype]()


@inline(.always)
def _scatter[
    ptr_dtype: DType,
    origin: MutOrigin,
    address_space: AddressSpace,
    L: TensorLayout,
    buf_dtype: DType,
    num_cols: Int,
    //,
    widen: Int,
    start: Int,
](
    ptr: Pointer[Scalar[ptr_dtype], origin, address_space=address_space],
    layout: L,
    buf: Array[Scalar[buf_dtype], num_cols],
):
    """Stores `buf` into elements `start` to `start + num_cols` of the tile at
    `ptr`, cast to `ptr_dtype`.

    With `widen > 1` the destination is a contiguous run and is written in
    `widen`-wide vectors at raw offsets; otherwise each element's offset goes
    through `layout`.
    """
    comptime alignment = align_of[SIMD[ptr_dtype, widen]]() if is_gpu() else 1
    comptime if widen > 1:
        comptime for i in range(num_cols // widen):
            ptr.unsafe_store[alignment=alignment](
                start + i * widen,
                Pointer(to=buf[i * widen])
                .unsafe_bitcast[SIMD[buf_dtype, widen]]()[]
                .cast[ptr_dtype](),
            )
    else:
        comptime for i in range(num_cols):
            ptr.unsafe_store[alignment=alignment](
                layout(Idx[start + i]), buf[i].cast[ptr_dtype]()
            )


struct TMemStorage[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](TrivialRegisterPassable):
    """Encoded TMEM address: lane in the upper 16 bits, column in the lower.

    The hardware address is untyped; `dtype` and `origin` type the view the
    way `Pointer`'s parameters do. TMEM is not one of Mojo's `AddressSpace`
    values, so the handle carries none.

    Parameters:
        mut: The mutability of the viewed storage, inferred from `origin`.
        dtype: The element data type the tile views the cells as.
        origin: The origin tracking the lifetime of the allocation.
    """

    var addr: UInt32
    """The encoded TMEM address of the first element."""

    @inline(.always)
    def __init__(out self, addr: UInt32):
        """Wraps a raw TMEM address as produced by `tcgen05_alloc`.

        Args:
            addr: The encoded TMEM address.
        """
        self.addr = addr


struct TMemEngine(TensorEngine):
    """Implements `TensorEngine` over Tensor Memory via `tcgen05.ld`/`st`.

    Element offsets are positions on the lane-by-column grid, so layouts over
    this engine use a lane stride of `1` and a column stride of
    `TMEM_NUM_LANES`; the engine encodes them into hardware addresses. It has
    no `unsafe_ptr`, so its only data path is `copy_from` in either direction,
    which moves consecutive columns of the lane the calling thread owns. A
    copy issues one `tcgen05` instruction per power-of-two chunk of columns,
    largest first, none wider than 64 registers, and waits once per 64-column
    slice; the `_async` variants issue everything and leave the wait to the
    caller.

    The `tcgen05` access shape is not stored on the engine; the TMEM operand's
    layout selects a `TMemAccessShape` case. A `(1, N)` lane row moves in
    `32x32b`, one thread per lane, and a `TMem16x256bLayout` view, from
    `tmem_16x256b`, moves in `16x256b`. Only 4-byte element types are
    supported; two-byte types would need the `pack::16b` path.
    """

    comptime element_size = 1
    """One scalar per logical element."""

    comptime _BASE_TYPE_NAME: StaticString = "TMemEngine"
    """The unparameterized name of this engine."""

    comptime _ACCESS_SHAPES = ((32, 32), (16, 256))
    """Data movement shapes indexed by `TMemAccessShape`."""

    comptime _access_shape[shape: TMemAccessShape]: Tuple[Int, Int] = (
        Self._ACCESS_SHAPES[Int(shape._value)]
    )
    """The `tcgen05` shape a copy issues, lanes by bits per lane.

    Parameters:
        shape: The data movement shape selected by the TMEM view.
    """
    comptime _regs_per_unit[shape: TMemAccessShape] = (
        Self._access_shape[shape][0] * Self._access_shape[shape][1]
    ) // (32 * 32)
    """Registers one thread receives per unit of the instruction's repeat
    count.

    Parameters:
        shape: The data movement shape selected by the TMEM view.
    """
    comptime _cols_per_unit[shape: TMemAccessShape] = Self._access_shape[shape][
        1
    ] // 32
    """Columns of one lane one unit of the repeat count covers.

    Parameters:
        shape: The data movement shape selected by the TMEM view.
    """
    comptime _MAX_LOG2_WIDTH = 6
    """Log2 of the most registers one instruction names per thread.

    The ISA allows 128, but a 128-register operand list needs 128 consecutive
    registers, which `ptxas` rejects with C7602 in any kernel that cannot
    spare them. 64 is the cap the SM100 attention kernels settled on.
    """

    comptime _MAX_WIDTH = 1 << Self._MAX_LOG2_WIDTH
    """The most registers one instruction names per thread, and the slice a
    waiting copy stages at a time."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = TMemStorage[dtype, origin]
    """The encoded TMEM address handle.

    The `address_space` is part of the `TensorEngine` interface but unused:
    TMEM has no pointer, so no `AddressSpace` describes it and the handle
    does not carry one.

    Parameters:
        mut: The mutability of the viewed storage, inferred from `origin`.
        dtype: The element data type of the viewed storage.
        origin: The origin tracking the lifetime of the allocation.
        address_space: Unused.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Writes the engine type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        writer.write("TMemEngine")

    @doc_hidden
    @staticmethod
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) -> Pointer[
        Scalar[dtype], origin, address_space=address_space
    ]:
        """Fails to compile: TMEM cells are not addressable through a pointer.

        Parameters:
            mut: The mutability of the storage, inferred from `origin`.
            dtype: The element data type of the storage.
            origin: The origin tracking the lifetime of the storage.
            address_space: The address space parameter of the storage.

        Args:
            storage: The storage handle.

        Returns:
            Never returns; any instantiation is rejected at compile time.
        """
        comptime assert (
            False
        ), "TMemEngine: tensor memory has no pointer representation"

    @staticmethod
    @inline(.always)
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](storage: Self.StorageType[...]) -> Self.StorageType[
        to_dtype, to_origin, to_address_space
    ]:
        """Reinterprets the handle with new type parameters.

        The address is untyped, so this only re-labels the handle.

        Parameters:
            to_mut: The mutability of the new origin.
            to_dtype: The element data type to view the cells as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space parameter to carry.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle with the same address and the new parameters.
        """
        return Self.StorageType[to_dtype, to_origin, to_address_space](
            storage.addr
        )

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorEngine = Self
    """Offsetting never changes the engine, so this is `Self`.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @inline(.always)
    def offset[
        offset_mut: Bool,
        offset_types: TypeList[Trait=CoordLike, ...],
        //,
        offset_dtype: DType,
        offset_origin: Origin[mut=offset_mut],
        offset_address_space: AddressSpace,
    ](
        var storage: Self.StorageType[
            offset_dtype, offset_origin, offset_address_space
        ],
        var offset_coord: Coord[*offset_types],
    ) -> Self.OffsetResultType[offset_types].StorageType[
        offset_dtype, offset_origin, offset_address_space
    ]:
        """Returns a handle advanced by a number of grid positions.

        The offset is a flat index on the lane-by-column grid, as a layout
        with column stride `TMEM_NUM_LANES` produces; the engine re-encodes
        the resulting cell as a hardware address.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space parameter of the storage.

        Args:
            storage: The storage to offset from.
            offset_coord: A flat coordinate whose components sum to the
                grid offset.

        Returns:
            A handle at the cell `offset` positions after `storage`'s.
        """
        # Encoding is additive while no lane sum reaches `TMEM_NUM_LANES`,
        # which holds for non-negative components of a cell on the grid, so
        # each component is encoded on its own: a folded `ComptimeInt` part
        # becomes a constant, and the address is never decoded.
        var addr = storage.addr
        comptime for i in range(offset_coord.flat_rank):
            addr += _encode(UInt32(Int(offset_coord[i])))
        return type_of(storage)(addr)

    @staticmethod
    def _check_copy[
        TMemLayoutType: TensorLayout,
        PtrLayoutType: TensorLayout,
        tmem_dtype: DType,
        PtrEngine: TensorEngine,
    ]():
        """Rejects at compile time a copy the hardware or the engine cannot
        express."""
        comptime assert (
            PtrEngine._BASE_TYPE_NAME != Self._BASE_TYPE_NAME
        ), "TMemEngine: TMEM to TMEM copies must stage through a register tile"
        comptime assert (
            PtrEngine.element_size == 1
        ), "TMemEngine copies require a scalar logical element on both sides"
        comptime assert (
            size_of[tmem_dtype]() == 4
        ), "TMemEngine supports 4-byte element types only"
        comptime assert (
            TMemLayoutType.shape_known and PtrLayoutType.shape_known
        ), "TMemEngine copies require statically known shapes"
        comptime assert (
            TMemLayoutType.static_product == PtrLayoutType.static_product
        ), "TMemEngine copies require matching total element count"
        comptime access_shape = _access_shape_for_layout[TMemLayoutType]()
        comptime num_cols = (
            TMemLayoutType.static_product * Self._cols_per_unit[access_shape]
        ) // Self._regs_per_unit[access_shape]
        comptime assert (
            num_cols <= TMEM_NUM_COLS
        ), "TMemEngine copies must fit in the columns of one lane"

    @staticmethod
    def _check_async_fits[num_regs: Int, *, wait: Bool]():
        """Rejects at compile time an async copy wider than one slice.

        Without a wait between slices the store side could not reuse its
        staging registers, so an async copy stages its whole operand; capping
        it at `_MAX_WIDTH` registers keeps that footprint within the engine's
        bound.
        """
        comptime assert wait or num_regs <= Self._MAX_WIDTH, (
            "TMemEngine: an async copy stages its whole operand and holds"
            " its registers live, so it must fit in 64 registers; tile a"
            " wider view and issue one copy per slice"
        )

    @staticmethod
    @inline(.always)
    def _move_chunk[
        dtype: DType,
        num_regs: Int,
        //,
        access_shape: TMemAccessShape,
        units: Int,
        *,
        store: Bool,
    ](addr: UInt32, mut buf: Array[Scalar[dtype], num_regs], unit_off: Int):
        """Issues one `tcgen05.st` from, or `tcgen05.ld` into, the `units`
        repeats of `buf` that start at repeat `unit_off`, at the matching
        column of `addr`, without waiting."""
        comptime width = units * Self._regs_per_unit[access_shape]
        var col = unit_off * Self._cols_per_unit[access_shape]
        ref chunk = Pointer(
            to=buf[unit_off * Self._regs_per_unit[access_shape]]
        ).unsafe_bitcast[Array[Scalar[dtype], width]]()[]
        comptime if store:
            tcgen05_st[
                datapaths=Self._access_shape[access_shape][0],
                bits=Self._access_shape[access_shape][1],
                repeat=units,
                pack=False,
            ](addr + UInt32(col), chunk)
        else:
            chunk = tcgen05_ld[
                datapaths=Self._access_shape[access_shape][0],
                bits=Self._access_shape[access_shape][1],
                repeat=units,
                dtype=dtype,
                pack=False,
                width=width,
            ](addr + UInt32(col))

    @staticmethod
    @inline(.always)
    def _move_columns[
        dtype: DType,
        num_regs: Int,
        //,
        access_shape: TMemAccessShape,
        *,
        store: Bool,
    ](addr: UInt32, mut buf: Array[Scalar[dtype], num_regs]):
        """Moves this thread's `num_regs` registers between `addr` and `buf`,
        one instruction per chunk and no wait.

        The ISA accepts power-of-two repeat counts, and the engine caps an
        instruction at `_MAX_WIDTH` registers, so the repeats are covered by
        full-width chunks first and then by one chunk per set bit of the
        remainder, largest first.
        """
        comptime units = num_regs // Self._regs_per_unit[access_shape]
        comptime max_units = Self._MAX_WIDTH // Self._regs_per_unit[
            access_shape
        ]
        comptime full = units // max_units
        comptime for i in range(full):
            Self._move_chunk[access_shape, max_units, store=store](
                addr, buf, i * max_units
            )
        comptime rem = units - full * max_units
        comptime for b in range(Self._MAX_LOG2_WIDTH - 1, -1, -1):
            comptime chunk_units = 1 << b
            comptime if rem & chunk_units != 0:
                # Larger set bits precede this chunk, so it starts where
                # the remainder's lower bits (this one included) begin.
                comptime unit_off = units - (rem & (2 * chunk_units - 1))
                Self._move_chunk[access_shape, chunk_units, store=store](
                    addr, buf, unit_off
                )

    @staticmethod
    @inline(.always)
    def wait_store():
        """Waits for every `tcgen05.st` this thread has issued.

        Call it after one or more `copy_from_async` before the source tiles
        are modified or the stored columns are read back.
        """
        tcgen05_store_wait()

    @staticmethod
    @inline(.always)
    def wait_load():
        """Waits for every `tcgen05.ld` this thread has issued.

        Call it after one or more `copy_to_async` before the destination
        tiles are read.
        """
        tcgen05_load_wait()

    @staticmethod
    @inline(.always)
    def _copy_in[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        //,
        dst_dtype: DType,
        src_dtype: DType,
        self_address_space: AddressSpace,
        other_address_space: AddressSpace,
        OtherEngine: TensorEngine,
        *,
        wait: Bool,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Gathers `other` through its pointer and stores it into this thread's
        lane.

        With `wait`, the row moves in `_MAX_WIDTH`-column slices and each
        slice waits for its stores, so a slice's staging registers are dead
        before the next slice is gathered and a wide row never holds more
        than `_MAX_WIDTH` of them live. Without it, the whole row is gathered
        and stored with no wait, since `tcgen05.st` reads its registers
        asynchronously and a later gather could not reuse them safely.
        """
        Self._check_copy[
            SelfLayoutType, OtherLayoutType, dst_dtype, OtherEngine
        ]()
        comptime access_shape = _access_shape_for_layout[SelfLayoutType]()
        comptime num_regs = SelfLayoutType.static_product
        Self._check_async_fits[num_regs, wait=wait]()
        # The TMEM side is walked by the instruction in the view's flat
        # order, so only the pointer side decides whether the raw walk can
        # widen.
        comptime widen = _copy_widen_factor[
            dst_dtype=dst_dtype,
            src_dtype=src_dtype,
            element_size=1,
            dst_row_major=True,
            src_row_major=_layout_row_major[OtherLayoutType](),
            num_elements=num_regs,
        ]()
        var ptr = OtherEngine.unsafe_ptr(other[0])
        comptime if wait:
            comptime for s in range(ceildiv(num_regs, Self._MAX_WIDTH)):
                comptime start = s * Self._MAX_WIDTH
                comptime regs = min(Self._MAX_WIDTH, num_regs - start)
                comptime col = (
                    start // Self._regs_per_unit[access_shape]
                ) * Self._cols_per_unit[access_shape]
                var buf = Array[Scalar[dst_dtype], regs](uninitialized=True)
                _gather[widen=widen, start=start](ptr, other[1], buf)
                Self._move_columns[access_shape, store=True](
                    storage[0].addr + UInt32(col), buf
                )
                tcgen05_store_wait()
        else:
            var buf = Array[Scalar[dst_dtype], num_regs](uninitialized=True)
            _gather[widen=widen, start=0](ptr, other[1], buf)
            Self._move_columns[access_shape, store=True](storage[0].addr, buf)

    @staticmethod
    @inline(.always)
    def _copy_out[
        SelfLayoutType: TensorLayout,
        self_mut: Bool,
        self_origin: Origin[mut=self_mut],
        OtherLayoutType: TensorLayout,
        other_origin: MutOrigin,
        //,
        src_dtype: DType,
        dst_dtype: DType,
        self_address_space: AddressSpace,
        other_address_space: AddressSpace,
        OtherEngine: TensorEngine,
        *,
        wait: Bool,
    ](
        storage: Tuple[
            Self.StorageType[src_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                dst_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Loads this thread's lane and scatters it into `other` through its
        pointer.

        With `wait`, the row moves in `_MAX_WIDTH`-column slices, each loaded,
        waited for and scattered before the next, so a wide row never holds
        more than `_MAX_WIDTH` staging registers live. Without it, every load
        is issued and scattered with no wait, which is only meaningful when
        the destination is a register tile the caller reads after
        `wait_load`.
        """
        Self._check_copy[
            SelfLayoutType, OtherLayoutType, src_dtype, OtherEngine
        ]()
        comptime access_shape = _access_shape_for_layout[SelfLayoutType]()
        comptime num_regs = SelfLayoutType.static_product
        Self._check_async_fits[num_regs, wait=wait]()
        comptime widen = _copy_widen_factor[
            dst_dtype=dst_dtype,
            src_dtype=src_dtype,
            element_size=1,
            dst_row_major=_layout_row_major[OtherLayoutType](),
            src_row_major=True,
            num_elements=num_regs,
        ]()
        var ptr = OtherEngine.unsafe_ptr(other[0])
        comptime if wait:
            comptime for s in range(ceildiv(num_regs, Self._MAX_WIDTH)):
                comptime start = s * Self._MAX_WIDTH
                comptime regs = min(Self._MAX_WIDTH, num_regs - start)
                comptime col = (
                    start // Self._regs_per_unit[access_shape]
                ) * Self._cols_per_unit[access_shape]
                var buf = Array[Scalar[src_dtype], regs](uninitialized=True)
                Self._move_columns[access_shape, store=False](
                    storage[0].addr + UInt32(col), buf
                )
                tcgen05_load_wait()
                _scatter[widen=widen, start=start](ptr, other[1], buf)
        else:
            var buf = Array[Scalar[src_dtype], num_regs](uninitialized=True)
            Self._move_columns[access_shape, store=False](storage[0].addr, buf)
            _scatter[widen=widen, start=0](ptr, other[1], buf)

    @staticmethod
    @inline(.always)
    def copy_from[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies the elements of `other` into this thread's lane, in place.

        Reads the source through `OtherEngine.unsafe_ptr`, casting to
        `dst_dtype`, then issues one `tcgen05.st` per column chunk. A row of
        up to 64 columns is one slice with a single `tcgen05.wait::st`; a
        wider row moves in 64-column slices, each waited on, so at most 64
        staging registers are live at once. `storage` must be a warp-base
        handle whose layout is a `(1, N)` row of a warp's tile, such as its row
        0, or a `tmem_16x256b` view.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherEngine: The engine of the source. Must not be `TMemEngine`.

        Args:
            storage: A tuple of the destination storage and its layout.
            other: A tuple of the source storage and its layout.
        """
        Self._copy_in[
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            OtherEngine=OtherEngine,
            wait=True,
        ](storage, other)

    @staticmethod
    @inline(.always)
    def copy_from_async[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Issues the stores of `copy_from` without waiting for them.

        The caller must call `wait_store` before the source tile is modified
        or the stored columns are read back; until then the registers the
        stores read from are in flight. Several `copy_from_async` calls can
        share one wait. The whole row is staged at once, so a row of `N`
        columns keeps `N` registers live until the wait.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherEngine: The engine of the source. Must not be `TMemEngine`.

        Args:
            storage: A tuple of the destination storage and its layout.
            other: A tuple of the source storage and its layout.
        """
        Self._copy_in[
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            OtherEngine=OtherEngine,
            wait=False,
        ](storage, other)

    @staticmethod
    @inline(.always)
    def copy_to[
        SelfLayoutType: TensorLayout,
        self_mut: Bool,
        self_origin: Origin[mut=self_mut],
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_origin: MutOrigin,
        other_address_space: AddressSpace,
        //,
        src_dtype: DType,
        dst_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[src_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                dst_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies this thread's lane at `storage` into `other`, in place.

        Issues one `tcgen05.ld` per column chunk, then writes the columns
        through `OtherEngine.unsafe_ptr`, casting to `dst_dtype`. A row of up
        to 64 columns is one slice with a single `tcgen05.wait::ld`; a wider
        row moves in 64-column slices, each loaded, waited on and written
        before the next, so at most 64 staging registers are live at once.
        `storage` must be a warp-base handle whose layout is a `(1, N)` row of
        a warp's tile, such as its row 0, or a `tmem_16x256b` view.

        Parameters:
            SelfLayoutType: The layout type of the source storage.
            self_mut: The mutability of the source storage.
            self_origin: The origin of the source storage.
            self_address_space: The address space of the source storage.
            OtherLayoutType: The layout type of the destination storage.
            other_origin: The origin of the destination storage.
            other_address_space: The address space of the destination
                storage.
            src_dtype: The element data type of the source storage.
            dst_dtype: The element data type of the destination storage.
            OtherEngine: The engine of the destination. Must not be
                `TMemEngine`.

        Args:
            storage: A tuple of the source storage and its layout.
            other: A tuple of the destination storage and its layout.
        """
        Self._copy_out[
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            OtherEngine=OtherEngine,
            wait=True,
        ](storage, other)

    @staticmethod
    @inline(.always)
    def copy_to_async[
        SelfLayoutType: TensorLayout,
        self_mut: Bool,
        self_origin: Origin[mut=self_mut],
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_origin: MutOrigin,
        other_address_space: AddressSpace,
        //,
        src_dtype: DType,
        dst_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[src_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                dst_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Issues the loads of `copy_to` without waiting for them.

        The destination must be a register tile: the loaded registers are
        only valid after `wait_load`, so a destination that reaches memory
        before the wait receives stale data. Several `copy_to_async` calls
        can share one wait, and the destinations must not be read until it
        returns. The whole row is in flight at once, so a row of `N` columns
        keeps `N` registers live until the wait.

        Parameters:
            SelfLayoutType: The layout type of the source storage.
            self_mut: The mutability of the source storage.
            self_origin: The origin of the source storage.
            self_address_space: The address space of the source storage.
            OtherLayoutType: The layout type of the destination storage.
            other_origin: The origin of the destination storage.
            other_address_space: The address space of the destination
                storage.
            src_dtype: The element data type of the source storage.
            dst_dtype: The element data type of the destination storage.
            OtherEngine: The engine of the destination. Must not be
                `TMemEngine`.

        Args:
            storage: A tuple of the source storage and its layout.
            other: A tuple of the destination storage and its layout.
        """
        Self._copy_out[
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            OtherEngine=OtherEngine,
            wait=False,
        ](storage, other)


@inline(.always)
def tmem_16x256b(
    block: TileTensor[Engine=TMemEngine, ...],
) -> TileTensor[
    block.dtype,
    type_of(TMem16x256bLayout[type_of(block).static_shape[1]]),
    block.origin,
    Engine=TMemEngine,
    address_space=block.address_space,
]:
    """Returns this thread's `16x256b` fragment view of a `(16, N)` lane block.

    The view keeps the block's storage, the collective base the instruction
    takes, and lays `TMem16x256bLayout[N]` over it. Copying it into an
    `N / 2`-element register tile issues `tcgen05.ld.16x256b`, copying a
    register tile into it issues `tcgen05.st.16x256b`, and the registers
    follow the layout's flat order.

    Args:
        block: A `(16, N)` block of a warp's `(32, N)` TMEM tile, such as
            `warp_tile.tile[16, N](coord[half, 0])`.

    Returns:
        The fragment view over the block's storage.
    """
    comptime L = type_of(block).LayoutType
    comptime assert (
        L.all_dims_known
        and L.flat_rank == 2
        and L.static_shape[0] == 2 * _FRAGMENT_LANE_PAIR_STRIDE
        and L.static_stride[0] == 1
        and L.static_stride[1] == TMEM_NUM_LANES
    ), "tmem_16x256b takes a (16, N) lane block of a warp's TMEM tile"
    comptime assert (
        L.static_shape[1] % _FRAGMENT_COLS_PER_REPEAT == 0
    ), "a 16x256b fragment covers columns in groups of 8"
    return {block._storage, TMem16x256bLayout[L.static_shape[1]]}


@inline(.always)
def tmem_cell(
    view: TileTensor[Engine=TMemEngine, ...], coord: Coord
) -> Tuple[Int, Int]:
    """Returns the lane and column this thread's element at `coord` of a TMEM
    view occupies.

    The view's layout gives the cell relative to its base; the hardware adds
    this thread's placement, which depends on the access shape the layout
    selects: lane `t` for a `(1, N)` row of a warp tile, and lane `t // 4`,
    column `2 * (t % 4)` for a `tmem_16x256b` view. This is the one place
    that placement is written down.

    Args:
        view: A row view or fragment view with a warp-base storage.
        coord: A coordinate of the view.

    Returns:
        The `(lane, column)` pair on the TMEM grid.
    """
    comptime L = type_of(view).LayoutType
    comptime access_shape = _access_shape_for_layout[L]()
    var base = _grid_index(view._storage.addr) + UInt32(Int(view.layout(coord)))
    var lane = base & (TMEM_NUM_LANES - 1)
    var col = base >> _LANE_BITS
    var thread = UInt32(lane_id())
    comptime if access_shape == TMemAccessShape.BLOCK_16X256B:
        return (Int(lane + thread // 4), Int(col + 2 * (thread % 4)))
    else:
        return (Int(lane + thread), Int(col))


@inline(.always)
def tmem_copy_async(
    dst: TileTensor[mut=True, Engine=TMemEngine, ...], src: TileTensor
):
    """Issues the stores that copy `src` into the TMEM row `dst`, without
    waiting.

    The tile-level spelling of `TMemEngine.copy_from_async`. Pair it with
    `TMemEngine.wait_store` before `src` is modified or `dst` is read back.

    Args:
        dst: This thread's row view of a TMEM tile.
        src: The register or shared-memory tile to store.
    """
    # `TMemStorage` carries no address space, so the trait-shaped
    # `copy_from_async` cannot infer it from a concrete engine; pass the
    # tiles' address spaces explicitly.
    TMemEngine._copy_in[
        self_address_space=type_of(dst).address_space,
        other_address_space=type_of(src).address_space,
        OtherEngine=type_of(src).Engine,
        wait=False,
    ]((dst._storage, dst.layout), (src._storage, src.layout))


@inline(.always)
def tmem_copy_async(
    dst: TileTensor[mut=True, ...], src: TileTensor[Engine=TMemEngine, ...]
):
    """Issues the loads that copy the TMEM row `src` into `dst`, without
    waiting.

    The tile-level spelling of `TMemEngine.copy_to_async`. `dst` must be a
    register tile, and it holds valid data only after `TMemEngine.wait_load`.

    Args:
        dst: The register tile to load into.
        src: This thread's row view of a TMEM tile.
    """
    TMemEngine._copy_out[
        self_address_space=type_of(src).address_space,
        other_address_space=type_of(dst).address_space,
        OtherEngine=type_of(dst).Engine,
        wait=False,
    ]((src._storage, src.layout), (dst._storage, dst.layout))
