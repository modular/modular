# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import min
from buffer.buffer import NDBuffer, partial_simd_load, partial_simd_store
from buffer.list import DimList
from utils.index import Index


@always_inline
fn _initialize_c_tile_default[
    a_row_size: Int,
    pack_inner_size: Int,
](c0_local: NDBuffer):
    """Utility function on the inner loop. Initializes a local c buffer with
    all zeros.

    """
    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type
    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)

    @always_inline
    @parameter
    fn outer_body[idx0: Int, idx1: Int]():
        c_local.store[
            width=simd_size,
            alignment = alignof[SIMD[c_type, simd_size]](),
        ](
            Index(idx0, idx1 * simd_size),
            SIMD[c_type, simd_size](0),
        )

    unroll[outer_body, a_row_size, pack_inner_size // simd_size]()


@always_inline
fn _load_c_tile_default[
    a_row_size: Int, pack_inner_size: Int, skip_boundary_check: Bool
](
    c_ptr: DTypePointer,
    c_stride: Int,
    c0_local: NDBuffer,
    tile_n_idx: Int,
    c_bound: StaticIntTuple[2],
):
    """Utility function on the inner loop. Loads a local c_buffer with the
    value stored in the output buffer space, given the indices within the
    tile being processed.

    Args:
    c_ptr: TODO.
    c_stride: TODO.
    c0_local: pre-allocated local buffer for c partial sums.
    tile_n_idx: n coordinate within the current processing tile.
    c_bound: Boundary of valid output space within the local tile, in (a_row_size, TileN).
    """
    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type
    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)
    var c_ptr_loc = rebind[DTypePointer[c_type]](c_ptr.offset(tile_n_idx))

    @always_inline
    @parameter
    fn body[idx0: Int, idx1: Int]():
        var c_data: SIMD[c_type, simd_size] = 0
        if skip_boundary_check or (
            idx1 * simd_size + simd_size <= c_bound[1] - tile_n_idx
        ):
            # Use simd load if all within bound
            c_data = c_ptr_loc.load[width=simd_size](idx1 * simd_size)
        elif idx1 * simd_size <= c_bound[1]:
            # Use partial load if row inbound but col not
            #  in simd bound.
            c_data = partial_simd_load[simd_size](
                c_ptr_loc.offset(idx1 * simd_size),
                0,
                c_bound[1] - tile_n_idx - idx1 * simd_size,
                0,
            )
        else:
            # Fill zero if row out of bound
            c_data = 0

            # Store data to local buffer.
            c_local.store(Index(idx0, idx1 * simd_size), c_data)

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr_loc = c_ptr_loc.offset(c_stride)

    unroll[body, a_row_size, pack_inner_size // simd_size]()


@always_inline
fn _store_c_tile_default[
    a_row_size: Int, pack_inner_size: Int, skip_boundary_check: Bool
](
    c_ptr: DTypePointer,
    c_stride: Int,
    c0_local: NDBuffer,
    tile_n_idx: Int,
    c_bound: StaticIntTuple[2],
):
    """Utility function on the inner loop. Stores the value of a local c
    buffer to the corresponding position in the output buffer space.

    Args:
    c_ptr: TODO.
    c_stride: TODO.
    c0_local: pre-allocated local buffer for c partial sums.
    tile_n_idx: n coordinate within the current processing tile.
    c_bound: Boundary of valid output space within the local tile, in (a_row_size, TileN).
    """
    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type
    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)
    var c_ptr_loc = rebind[DTypePointer[c_type]](c_ptr.offset(tile_n_idx))

    @always_inline
    @parameter
    fn body[idx0: Int, idx1: Int]():
        var c_data = c_local.load[width=simd_size](
            Index(idx0, idx1 * simd_size)
        )
        if skip_boundary_check or (
            idx1 * simd_size + simd_size <= c_bound[1] - tile_n_idx
        ):
            # Use simd store if all within bound
            c_ptr_loc.offset(idx1 * simd_size).store[width=simd_size](c_data)
        elif idx1 * simd_size <= c_bound[1]:
            # Use partial store if col not in simd bound.
            partial_simd_store(
                c_ptr_loc.offset(idx1 * simd_size),
                0,
                c_bound[1] - tile_n_idx - idx1 * simd_size,
                c_data,
            )

        @parameter
        if idx1 == pack_inner_size // simd_size - 1:
            c_ptr_loc = c_ptr_loc.offset(c_stride)

    unroll[body, a_row_size, pack_inner_size // simd_size]()


@always_inline
fn _load_c_tile_i8mm[
    a_row_size: Int,
    pack_inner_size: Int,
    skip_boundary_check: Bool,
    single_row_i8mm: Bool = False,
](
    c_ptr: DTypePointer,
    c_stride: Int,
    c0_local: NDBuffer,
    tile_n_idx: Int,
    c_bound: StaticIntTuple[2],
):
    """Utility function on the inner loop. Loads a local c_buffer with the
    value stored in the output buffer space, given the indices within the
    tile being processed.

    Args:
    c_ptr: TODO.
    c_stride: TODO.
    c0_local: pre-allocated local buffer for c partial sums.
    tile_n_idx: n coordinate within the current processing tile.
    c_bound: Boundary of valid output space within the local tile, in (a_row_size, TileN).
    """
    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type

    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)
    var c_ptr_loc = rebind[DTypePointer[c_type]](c_ptr.offset(tile_n_idx))

    @always_inline
    @parameter
    fn body[idx0: Int, idx1: Int]():
        var c_data: SIMD[c_type, simd_size] = 0
        if skip_boundary_check or (idx1 * 2 + 2 <= c_bound[1] - tile_n_idx):
            var t0 = c_ptr_loc.load[width=2](
                c_stride * (2 * idx0 + 0) + 2 * idx1
            )
            var t1 = c_ptr_loc.load[width=2](
                c_stride * (2 * idx0 + 1) + 2 * idx1
            ) if not single_row_i8mm else SIMD[c_type, 2](0)
            c_data = rebind[SIMD[c_type, simd_size]](t0.join(t1))
        elif idx1 * 2 <= c_bound[1]:
            var t0 = partial_simd_load[2](
                c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                0,
                c_bound[1] - tile_n_idx - idx1 * 2,
                0,
            )
            var t1 = partial_simd_load[2](
                c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1),
                0,
                c_bound[1] - tile_n_idx - idx1 * 2,
                0,
            ) if not single_row_i8mm else SIMD[c_type, 2](0)
            c_data = rebind[SIMD[c_type, simd_size]](t0.join(t1))

        c_local.store[width=simd_size](
            Index(idx0, idx1 * simd_size),
            rebind[SIMD[c_type, simd_size]](c_data),
        )

    unroll[body, a_row_size, pack_inner_size // simd_size]()


@always_inline
fn _store_c_tile_i8mm[
    a_row_size: Int,
    pack_inner_size: Int,
    skip_boundary_check: Bool,
    single_row_i8mm: Bool = False,
](
    c_ptr: DTypePointer,
    c_stride: Int,
    c0_local: NDBuffer,
    tile_n_idx: Int,
    c_bound: StaticIntTuple[2],
):
    """Utility function on the inner loop. Stores the value of a local c
    buffer to the corresponding position in the output buffer space.

    Args:
    c_ptr: TODO.
    c_stride: TODO.
    c0_local: pre-allocated local buffer for c partial sums.
    tile_n_idx: n coordinate within the current processing tile.
    c_bound: Boundary of valid output space within the local tile, in (a_row_size, TileN).
    """

    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type
    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)
    var c_ptr_loc = rebind[DTypePointer[c_type]](c_ptr.offset(tile_n_idx))

    @always_inline
    @parameter
    fn body[idx0: Int, idx1: Int]():
        var c_data = c_local.load[width=simd_size](
            Index(idx0, idx1 * simd_size)
        )
        if skip_boundary_check or (idx1 * 2 + 2 <= c_bound[1] - tile_n_idx):
            c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1).store[
                width=2
            ](c_data.slice[2]())

            @parameter
            if not single_row_i8mm:
                c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1).store[
                    width=2
                ](c_data.slice[2, offset=2]())
        elif idx1 * 2 <= c_bound[1]:
            partial_simd_store(
                c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                0,
                c_bound[1] - tile_n_idx - idx1 * 2,
                c_data.slice[2](),
            )

            @parameter
            if not single_row_i8mm:
                partial_simd_store(
                    c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1),
                    0,
                    c_bound[1] - tile_n_idx - idx1 * 2,
                    c_data.slice[2, offset=2](),
                )

    unroll[body, a_row_size, pack_inner_size // simd_size]()


@always_inline
fn _load_c_tile_neon[
    a_row_size: Int,
    pack_inner_size: Int,
](
    c_ptr: DTypePointer,
    c_stride: Int,
    c0_local: NDBuffer,
    tile_n_idx: Int,
    c_bound: StaticIntTuple[2],
):
    """Utility function on the inner loop. Loads a local c_buffer with the
    value stored in the output buffer space, given the indices within the
    tile being processed.

    Args:
        c_ptr: TODO.
        c_stride: TODO.
        c0_local: pre-allocated local buffer for c partial sums.
        tile_n_idx: n coordinate within the current processing tile.
        c_bound: TODO.
    """
    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type

    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)
    var c_ptr_loc = rebind[DTypePointer[c_type]](c_ptr.offset(tile_n_idx))

    _initialize_c_tile_default[
        a_row_size,
        pack_inner_size,
    ](c_local)
    return LoadStoreOutputTile[
        c_type, simd_size, a_row_size, pack_inner_size, True
    ].run(
        c_local,
        c_ptr_loc,
        c_stride,
        min(c_bound[1] - tile_n_idx, pack_inner_size),
    )


@always_inline
fn _store_c_tile_neon[
    a_row_size: Int,
    pack_inner_size: Int,
](
    c_ptr: DTypePointer,
    c_stride: Int,
    c0_local: NDBuffer,
    tile_n_idx: Int,
    c_bound: StaticIntTuple[2],
):
    """Utility function on the inner loop. Stores the value of a local c
    buffer to the corresponding position in the output buffer space.

    Args:
        c_ptr: TODO.
        c_stride: TODO.
        c0_local: pre-allocated local buffer for c partial sums.
        tile_n_idx: n coordinate within the current processing tile.
        c_bound: TODO.
    """
    alias simd_size = simdwidthof[c0_local.type]()
    alias c_type = c0_local.type
    var c_local = rebind[
        NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ]
    ](c0_local)
    var c_ptr_loc = rebind[DTypePointer[c_type]](c_ptr.offset(tile_n_idx))

    return LoadStoreOutputTile[
        c_type, simd_size, a_row_size, pack_inner_size, False
    ].run(
        c_local,
        c_ptr_loc,
        c_stride,
        min(c_bound[1] - tile_n_idx, pack_inner_size),
    )


struct LoadStoreOutputTile[
    type: DType,
    simd_size: Int,
    tile_rows: Int,
    tile_columns: Int,
    is_load: Bool,
]:
    var output_tile: NDBuffer[type, 2, DimList(tile_rows, tile_columns)]
    var row_ptrs: Pointer[DTypePointer[type]]
    var load_store_count: Int

    @always_inline
    fn __init__(
        inout self,
        output_tile: NDBuffer[type, 2, DimList(tile_rows, tile_columns)],
        row_ptrs: Pointer[DTypePointer[type]],
        load_store_count: Int,
    ):
        self.output_tile = output_tile
        self.row_ptrs = row_ptrs
        self.load_store_count = load_store_count

    @always_inline
    fn _load_store_columns[
        base_column: Int,
        column_count: Int,
    ](self):
        """Loads or stores one or more columns from the base column for each
        row of the tile."""

        @unroll
        for row in range(tile_rows):
            # Iterate twice for a pairwise load/store or once for any other access.
            alias column_step = min(column_count, simd_size)

            @unroll
            for col in range(
                base_column, base_column + column_count, column_step
            ):

                @parameter
                if is_load:
                    var data = self.row_ptrs[row].offset(col).load[
                        width=column_step
                    ]()
                    self.output_tile.store(Index(row, col), data)
                else:
                    var data = self.output_tile.load[width=column_step](
                        Index(row, col)
                    )
                    self.row_ptrs[row].offset(col).store(data)

    @always_inline
    fn _load_store_tail[
        base_column: Int,
        tail_size: Int,
    ](self):
        """Loads/stores the last elements of the tile that cannot be accessed
        pairwise."""

        if self.load_store_count & tail_size:
            self._load_store_columns[base_column, tail_size]()

            alias tile_columns_remaining = tile_columns - base_column - tail_size

            @parameter
            if tile_columns_remaining >= tail_size // 2 and tail_size > 1:
                self._load_store_tail[base_column + tail_size, tail_size // 2]()
            return

        @parameter
        if tail_size > 1:
            self._load_store_tail[base_column, tail_size // 2]()

    @always_inline
    fn _load_store_pairwise[
        base_column: Int,
    ](self):
        """Loads/stores all pairwise vectors of the tile and dispatches the
        remaining non-pairwise elements."""

        alias tile_columns_remaining = tile_columns - base_column

        # Support fusion of LDP/STP instructions by emitting pairs of load/store
        # vector instructions.
        @parameter
        if tile_columns_remaining >= 2 * simd_size:
            if self.load_store_count >= base_column + 2 * simd_size:
                self._load_store_columns[base_column, 2 * simd_size]()
                self._load_store_pairwise[base_column + 2 * simd_size]()
                return

        @parameter
        if tile_columns_remaining >= simd_size:
            self._load_store_tail[base_column, simd_size]()

    @staticmethod
    @always_inline
    fn run(
        output_tile: NDBuffer[type, 2, DimList(tile_rows, tile_columns)],
        ptr: DTypePointer[type],
        stride: Int,
        load_store_count: Int,
    ):
        """Interface function to run the load/store output tile.
        Args:
            output_tile(NDBuffer): output register tile buffer.
            ptr(DTypePointer): data buffer to use for transferring the tile
                buffer.
            stride(Int): stride to use when stepping through rows of the data
                buffer.
            load_store_count(Int): number of elements to load/store.
        """
        # Compute the pointers to each row of the memory buffer.
        # Note that the compiler produces better code if each pointer is calculated
        # relative to the previous pointer. Using (N * row) causes the compiler to
        # allocate locals to cache the intermediate results.
        var row_ptrs = stack_allocation[tile_rows, DTypePointer[type]]()

        @unroll
        for row in range(tile_rows):
            row_ptrs[row] = ptr if row == 0 else (row_ptrs[row - 1] + stride)

        var instance = Self(
            output_tile,
            row_ptrs,
            load_store_count,
        )
        instance._load_store_pairwise[0]()
