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

Access is warp-collective. `tcgen05.ld` and `tcgen05.st` take the warp's base
lane in the address and hand thread `t` lane `base + t`, so a thread can only
touch its own lane. A copy therefore means "these columns of the lane this
thread owns", and the TMEM operand must be a warp base, never `base + lane`.
The per-thread view of a warp's `(32, N)` tile is its row 0
(`tile.tile[1, N](Coord(Idx[0], Idx[0]))`), whose storage is the unchanged
warp base.

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

from layout import Coord, CoordLike, Idx, TensorLayout
from layout.tile_tensor import TileTensor
from layout.tensor_engine import (
    TensorEngine,
    _copy_widen_factor,
    _layout_row_major,
    _offset_elements,
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
def _grid_index(addr: UInt32) -> Int:
    """Returns the cell `addr` encodes as a flat lane-first index on the
    grid, `col * TMEM_NUM_LANES + lane`."""
    var lane = Int(addr >> _LANE_SHIFT)
    var col = Int(addr & ((1 << _LANE_SHIFT) - 1))
    return (col << _LANE_BITS) | lane


@inline(.always)
def _encode(grid_index: Int) -> UInt32:
    """Returns the encoded TMEM address of the cell at flat `grid_index`."""
    var lane = grid_index & (TMEM_NUM_LANES - 1)
    var col = grid_index >> _LANE_BITS
    return UInt32((lane << _LANE_SHIFT) + col)


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

    The `tcgen05` access shape is not stored on the engine: its lane count is
    always 32, one thread per lane, and its bits per lane are the element
    type's width, so a float32 tile moves in the `32x32b` shape. Only 4-byte
    element types are supported; two-byte types would need the `pack::16b`
    path.
    """

    comptime element_size = 1
    """One scalar per logical element."""

    comptime _BASE_TYPE_NAME: StaticString = "TMemEngine"
    """The unparameterized name of this engine."""

    comptime _DATAPATHS = 32
    """The lane count of the access shape: one thread per lane, which is what
    makes a copy mean "this thread's row"."""

    comptime _cols_per_repeat[dtype: DType] = (
        Self._DATAPATHS * bit_width_of[dtype]()
    ) // (32 * 32)
    """Columns one thread receives per unit of the instruction repeat count
    in the `32x{bits}b` shape whose bits are `dtype`'s width."""

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
        return type_of(storage)(
            _encode(_grid_index(storage.addr) + _offset_elements(offset_coord))
        )

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the number of grid positions from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' address space parameter.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The signed difference of the two cells' flat grid indices.
        """
        return _grid_index(storage.addr) - _grid_index(other.addr)

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
        comptime assert (
            TMemLayoutType.rank == 2
        ), "TMemEngine tiles are two-dimensional: lanes by columns"
        comptime assert _is_lane_run[TMemLayoutType](), (
            "TMemEngine copies address consecutive columns of one lane; tile"
            " the TMEM tensor down to this thread's (1, N) row view first"
        )
        comptime assert (
            TMemLayoutType.static_product <= TMEM_NUM_COLS
        ), "TMemEngine copies must fit in the columns of one lane"
        comptime assert (
            TMemLayoutType.static_product % Self._cols_per_repeat[tmem_dtype]
            == 0
        ), "TMemEngine: column count must be a multiple of the shape's width"

    @staticmethod
    def _check_async_fits[num_cols: Int, *, wait: Bool]():
        """Rejects at compile time an async copy wider than one slice.

        Without a wait between slices the store side could not reuse its
        staging registers, so an async copy stages its whole row; capping the
        row at `_MAX_WIDTH` keeps that footprint within the engine's bound.
        """
        comptime assert wait or num_cols <= Self._MAX_WIDTH, (
            "TMemEngine: an async copy stages its whole row and holds"
            " num_cols registers live, so the row must fit in 64 columns;"
            " tile a wider row and issue one copy per slice"
        )

    @staticmethod
    @inline(.always)
    def _move_chunk[
        dtype: DType, num_cols: Int, //, width: Int, *, store: Bool
    ](addr: UInt32, mut buf: Array[Scalar[dtype], num_cols], col: Int):
        """Issues one `tcgen05.st` from, or `tcgen05.ld` into, `buf[col:col +
        width]` at `addr + col`, without waiting."""
        comptime repeat = width // Self._cols_per_repeat[dtype]
        ref chunk = Pointer(to=buf[col]).unsafe_bitcast[
            Array[Scalar[dtype], width]
        ]()[]
        comptime if store:
            tcgen05_st[
                datapaths=Self._DATAPATHS,
                bits=bit_width_of[dtype](),
                repeat=repeat,
                pack=False,
            ](addr + UInt32(col), chunk)
        else:
            chunk = tcgen05_ld[
                datapaths=Self._DATAPATHS,
                bits=bit_width_of[dtype](),
                repeat=repeat,
                dtype=dtype,
                pack=False,
                width=width,
            ](addr + UInt32(col))

    @staticmethod
    @inline(.always)
    def _move_columns[
        dtype: DType, num_cols: Int, //, *, store: Bool
    ](addr: UInt32, mut buf: Array[Scalar[dtype], num_cols]):
        """Moves `num_cols` consecutive columns of this thread's lane between
        `addr` and `buf`, one instruction per chunk and no wait.

        The ISA accepts power-of-two repeat counts, and the engine caps an
        instruction at `_MAX_WIDTH` registers, so the columns are covered by
        full-width chunks first and then by one chunk per set bit of the
        remainder, largest first.
        """
        comptime cols_per_unit = Self._cols_per_repeat[dtype]
        comptime units = num_cols // cols_per_unit
        comptime max_units = Self._MAX_WIDTH // cols_per_unit
        comptime full = units // max_units
        comptime for i in range(full):
            Self._move_chunk[Self._MAX_WIDTH, store=store](
                addr, buf, i * Self._MAX_WIDTH
            )
        comptime rem = units - full * max_units
        comptime for b in range(Self._MAX_LOG2_WIDTH - 1, -1, -1):
            comptime chunk_units = 1 << b
            comptime if rem & chunk_units != 0:
                # Larger set bits precede this chunk, so it starts where
                # the remainder's lower bits (this one included) begin.
                comptime unit_off = units - (rem & (2 * chunk_units - 1))
                Self._move_chunk[chunk_units * cols_per_unit, store=store](
                    addr, buf, unit_off * cols_per_unit
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
        comptime num_cols = SelfLayoutType.static_product
        Self._check_async_fits[num_cols, wait=wait]()
        # The TMEM side is a run of columns by `_check_copy`, which the
        # instruction walks contiguously, so only the pointer side decides
        # whether the raw walk can widen.
        comptime widen = _copy_widen_factor[
            dst_dtype=dst_dtype,
            src_dtype=src_dtype,
            element_size=1,
            dst_row_major=True,
            src_row_major=_layout_row_major[OtherLayoutType](),
            num_elements=num_cols,
        ]()
        var ptr = OtherEngine.unsafe_ptr(other[0])
        comptime if wait:
            comptime for s in range(ceildiv(num_cols, Self._MAX_WIDTH)):
                comptime start = s * Self._MAX_WIDTH
                comptime cols = min(Self._MAX_WIDTH, num_cols - start)
                var buf = Array[Scalar[dst_dtype], cols](uninitialized=True)
                _gather[widen=widen, start=start](ptr, other[1], buf)
                Self._move_columns[store=True](
                    storage[0].addr + UInt32(start), buf
                )
                tcgen05_store_wait()
        else:
            var buf = Array[Scalar[dst_dtype], num_cols](uninitialized=True)
            _gather[widen=widen, start=0](ptr, other[1], buf)
            Self._move_columns[store=True](storage[0].addr, buf)

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
        comptime num_cols = SelfLayoutType.static_product
        Self._check_async_fits[num_cols, wait=wait]()
        comptime widen = _copy_widen_factor[
            dst_dtype=dst_dtype,
            src_dtype=src_dtype,
            element_size=1,
            dst_row_major=_layout_row_major[OtherLayoutType](),
            src_row_major=True,
            num_elements=num_cols,
        ]()
        var ptr = OtherEngine.unsafe_ptr(other[0])
        comptime if wait:
            comptime for s in range(ceildiv(num_cols, Self._MAX_WIDTH)):
                comptime start = s * Self._MAX_WIDTH
                comptime cols = min(Self._MAX_WIDTH, num_cols - start)
                var buf = Array[Scalar[src_dtype], cols](uninitialized=True)
                Self._move_columns[store=False](
                    storage[0].addr + UInt32(start), buf
                )
                tcgen05_load_wait()
                _scatter[widen=widen, start=start](ptr, other[1], buf)
        else:
            var buf = Array[Scalar[src_dtype], num_cols](uninitialized=True)
            Self._move_columns[store=False](storage[0].addr, buf)
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
        handle whose layout addresses consecutive columns of one lane, such as
        row 0 of a warp's `(32, N)` tile.

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
        `storage` must be a warp-base handle whose layout addresses
        consecutive columns of one lane, such as row 0 of a warp's `(32, N)`
        tile.

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
