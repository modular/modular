# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer.buffer import partial_simd_load, partial_simd_store
from buffer.list import DimList
from utils.index import Index
from .accumulate import _Accumulator
from sys.info import has_avx2


struct LoadStore_default[
    type: DType,
    simd_size: Int,
    skip_boundary_check: Bool,
    tile_rows: Int,
    tile_columns: Int,
]:
    alias num_simd_cols = tile_columns // simd_size
    var output_tile: _Accumulator[
        type, tile_rows, Self.num_simd_cols, simd_size
    ]

    @always_inline
    fn __init__(inout self):
        self.output_tile = _Accumulator[
            type, tile_rows, Self.num_simd_cols, simd_size
        ]()

    @always_inline
    fn _initialize_c_tile(inout self):
        self.output_tile.init(0)

    @always_inline
    fn _load_c_tile(
        inout self,
        c_ptr: DTypePointer[type],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        LoadStoreOutputTile[
            is_load=True, skip_boundary_check=skip_boundary_check
        ].run(
            self.output_tile,
            c_ptr_loc,
            c_stride,
            min(c_bound[1] - tile_n_idx, tile_columns),
        )

    @always_inline
    fn _store_c_tile(
        inout self,
        c_ptr: DTypePointer[type],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        LoadStoreOutputTile[
            is_load=False, skip_boundary_check=skip_boundary_check
        ].run(
            self.output_tile,
            c_ptr_loc,
            c_stride,
            min(c_bound[1] - tile_n_idx, tile_columns),
        )


struct LoadStore_i8mm[
    type: DType,
    simd_size: Int,
    skip_boundary_check: Bool,
    single_row: Bool,
    tile_rows: Int,
    tile_columns: Int,
]:
    alias num_simd_cols = tile_columns // simd_size
    var output_tile: _Accumulator[
        type, tile_rows, Self.num_simd_cols, simd_size
    ]

    @always_inline
    fn __init__(inout self):
        self.output_tile = _Accumulator[
            type, tile_rows, Self.num_simd_cols, simd_size
        ]()

    @always_inline
    fn _initialize_c_tile(inout self):
        self.output_tile.init(0)

    @always_inline
    fn _load_c_tile(
        inout self,
        c_ptr: DTypePointer[type],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data: SIMD[type, simd_size] = 0
            if skip_boundary_check or (idx1 * 2 + 2 <= c_bound[1] - tile_n_idx):
                var t0 = c_ptr_loc.load[width=2](
                    c_stride * (2 * idx0 + 0) + 2 * idx1
                )
                var t1 = c_ptr_loc.load[width=2](
                    c_stride * (2 * idx0 + 1) + 2 * idx1
                ) if not single_row else SIMD[type, 2](0)
                c_data = rebind[SIMD[type, simd_size]](t0.join(t1))
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
                ) if not single_row else SIMD[type, 2](0)
                c_data = rebind[SIMD[type, simd_size]](t0.join(t1))

            self.output_tile[idx0, idx1] = c_data

        unroll[body, tile_rows, tile_columns // simd_size]()

    @always_inline
    fn _store_c_tile(
        inout self,
        c_ptr: DTypePointer[type],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data = self.output_tile[idx0, idx1]
            if skip_boundary_check or (idx1 * 2 + 2 <= c_bound[1] - tile_n_idx):
                c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1).store[
                    width=2
                ](c_data.slice[2]())

                @parameter
                if not single_row:
                    c_ptr_loc.offset(
                        c_stride * (2 * idx0 + 1) + 2 * idx1
                    ).store[width=2](c_data.slice[2, offset=2]())
            elif idx1 * 2 <= c_bound[1]:
                partial_simd_store(
                    c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                    0,
                    c_bound[1] - tile_n_idx - idx1 * 2,
                    c_data.slice[2](),
                )

                @parameter
                if not single_row:
                    partial_simd_store(
                        c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        c_data.slice[2, offset=2](),
                    )

        unroll[body, tile_rows, tile_columns // simd_size]()


struct LoadStore_neon[
    type: DType,
    simd_size: Int,
    skip_boundary_check: Bool,
    tile_rows: Int,
    tile_columns: Int,
]:
    alias num_simd_cols = tile_columns // simd_size
    var output_tile: _Accumulator[
        type, tile_rows, Self.num_simd_cols, simd_size
    ]

    @always_inline
    fn __init__(inout self):
        self.output_tile = _Accumulator[
            type, tile_rows, Self.num_simd_cols, simd_size
        ]()

    @always_inline
    fn _initialize_c_tile(inout self):
        self.output_tile.init(0)

    @always_inline
    fn _load_c_tile(
        inout self,
        c_ptr: DTypePointer[type],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        LoadStoreOutputTile[is_load=True].run(
            self.output_tile,
            c_ptr_loc,
            c_stride,
            min(c_bound[1] - tile_n_idx, tile_columns),
        )

    @always_inline
    fn _store_c_tile(
        inout self,
        c_ptr: DTypePointer[type],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        LoadStoreOutputTile[is_load=False].run(
            self.output_tile,
            c_ptr_loc,
            c_stride,
            min(c_bound[1] - tile_n_idx, tile_columns),
        )


struct LoadStoreOutputTile[
    type: DType,
    simd_size: Int,
    tile_rows: Int,
    tile_columns: Int,
    is_load: Bool,
    skip_boundary_check: Bool = False,
]:
    alias num_simd_cols = tile_columns // simd_size
    var output_tile: _Accumulator[
        type, tile_rows, Self.num_simd_cols, simd_size
    ]
    var row_ptrs: Pointer[DTypePointer[type]]
    var load_store_count: Int
    var stride: Int

    @always_inline
    fn __init__(
        inout self,
        inout output_tile: _Accumulator[
            type, tile_rows, Self.num_simd_cols, simd_size
        ],
        row_ptrs: Pointer[DTypePointer[type]],
        load_store_count: Int,
        stride: Int,
    ):
        # NOTE: This is NOT a deepcopy; self.output_tile uses the same storage as output_tile.
        self.output_tile = output_tile
        self.row_ptrs = row_ptrs
        self.load_store_count = load_store_count
        self.stride = stride

    @always_inline
    fn _load_store_columns[
        base_column: Int,
        column_count: Int,
    ](inout self):
        """Loads or stores one or more columns from the base column for each
        row of the tile."""

        alias column_step = min(column_count, simd_size)
        var row_ptr = self.row_ptrs[0]

        @unroll
        for row in range(tile_rows):
            # Iterate twice for a pairwise load/store or once for any other access.

            @unroll
            for col in range(
                base_column, base_column + column_count, column_step
            ):

                @parameter
                if is_load:

                    @parameter
                    if has_avx2():
                        var data = row_ptr.load[width=column_step](
                            self.stride * row + col
                        )
                        self.output_tile._partial_set(
                            row * tile_columns + col, data
                        )
                    else:
                        var data = self.row_ptrs[row].load[width=column_step](
                            col
                        )
                        self.output_tile._partial_set(
                            row * tile_columns + col, data
                        )
                else:
                    var data = self.output_tile._partial_get[column_step](
                        row * tile_columns + col
                    )

                    @parameter
                    if has_avx2():
                        row_ptr.store[width=column_step](
                            self.stride * row + col, data
                        )
                    else:
                        self.row_ptrs[row].store[width=column_step](col, data)

    @always_inline
    fn _load_store_tail[
        base_column: Int,
        tail_size: Int,
    ](inout self):
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
    fn _load_store_tail_mask[
        base_column: Int,
    ](inout self):
        var tail_size = self.load_store_count - base_column
        var row_ptr = self.row_ptrs[0].offset(base_column)

        @unroll
        for row in range(tile_rows):
            var col = base_column // simd_size

            @parameter
            if is_load:
                var data = partial_simd_load[simd_size](
                    row_ptr.offset(self.stride * row),
                    0,
                    tail_size,
                    0,
                )
                self.output_tile[row, col] = data
            else:
                var data = self.output_tile[row, col]
                partial_simd_store(
                    row_ptr.offset(self.stride * row),
                    0,
                    tail_size,
                    data,
                )

    @always_inline
    fn _load_store_loop[base_column: Int](inout self):
        """Loads/stores all pairwise vectors of the tile and dispatches the
        remaining non-pairwise elements."""

        alias tile_columns_remaining = tile_columns - base_column
        # Support fusion of LDP/STP instructions by emitting pairs of load/store with neon
        alias column_groups = 1 if has_avx2() else 2

        @parameter
        if skip_boundary_check:

            @parameter
            if is_load:
                self.output_tile.load(self.row_ptrs[0], self.stride)
            else:
                self.output_tile.store(self.row_ptrs[0], self.stride)
            return

        # vector instructions.
        @parameter
        if tile_columns_remaining >= column_groups * simd_size:
            if self.load_store_count >= base_column + column_groups * simd_size:
                self._load_store_columns[
                    base_column, column_groups * simd_size
                ]()
                self._load_store_loop[base_column + column_groups * simd_size]()
                return

        @parameter
        if tile_columns_remaining >= simd_size:

            @parameter
            if has_avx2():
                self._load_store_tail_mask[base_column]()
            else:
                self._load_store_tail[base_column, simd_size]()

    @staticmethod
    @always_inline
    fn run(
        inout output_tile: _Accumulator[
            type, tile_rows, Self.num_simd_cols, simd_size
        ],
        ptr: DTypePointer[type],
        stride: Int,
        load_store_count: Int,
    ):
        """Interface function to run the load/store output tile.
        Args:
            output_tile(_Accumulator): output register tile buffer.
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
            stride,
        )
        instance._load_store_loop[0]()
