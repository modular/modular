# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file is a placeholder for rewriting Matmul.
# See:
# https://www.notion.so/modularai/Ingredients-for-Matmul-Rewrite-on-CPUs-8f631d688c6049e3a267ec0d9c66634c

from math import align_down, align_up, ceildiv, fma, min
from sys.info import (
    alignof,
    has_avx2,
    has_avx512f,
    has_neon,
    has_neon_int8_dotprod,
    simdwidthof,
)
from sys.intrinsics import PrefetchOptions

from algorithm import sync_parallelize, tile, unswitch, vectorize
from algorithm.functional import tile_and_unswitch
from buffer.list import Dim, DimList
from Gemv import gemv
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier, lane_id
from gpu.host import Function, Stream
from gpu.host.memory import _memset_async
from gpu.memory import AddressSpace
from gpu.shuffle import shuffle_down, shuffle_idx, warp_reduce
from MatmulUtils import (
    GemmShape,
    MatmulConfig,
    MatmulDataType,
    MatmulOperandLayout,
    PartitionHeuristic,
    SubMatmulConfig,
    _get_tile_n_k,
    calculate_tile_n_k,
    dispatch_get_kernel_type,
    elementwise_epilogue_type,
    get_kernel_type,
    get_matmul_arch_factor,
    get_min_task_size,
    get_packB_unroll_factor,
    get_partitioned_matmul,
    packA_i8mm,
    search_mm_config,
    use_i8mm_fn,
    use_vnni_fn,
)
from memory import memset_zero, stack_allocation
from memory.buffer import (
    Buffer,
    DynamicRankBuffer,
    NDBuffer,
    partial_simd_load,
    partial_simd_store,
)
from memory.unsafe import DTypePointer, bitcast
from Neon import _neon_dotprod, _neon_matmul
from runtime.llcl import Runtime
from Transpose import transpose_inplace
from VNNI import dot_i8_to_i32_saturated_x86, dot_i8_to_i32_x86

from collections import OptionalReg as Optional
from utils.index import Index, StaticIntTuple
from utils.loop import unroll
from utils.static_tuple import StaticTuple


fn elementwise_epilogue_c_tile[
    simd_width: Int,
    type: DType,
    c_shape: DimList,
    func: fn[type: DType, width: Int] (
        StaticIntTuple[2], SIMD[type, width]
    ) capturing -> None,
](offset: GemmShape, tile_len: GemmShape, c: NDBuffer[type, 2, c_shape]):
    @always_inline
    @parameter
    fn activation_on_col_chunk[col_chunk_size: Int](idx_n: Int):
        var n_coord = idx_n + offset.N
        for idx_m in range(tile_len.M):
            var m_coord = idx_m + offset.M
            var c_coord = (m_coord, n_coord)
            var c_val = c.load[width=col_chunk_size](c_coord)
            func[type, col_chunk_size](c_coord, c_val)

    vectorize[activation_on_col_chunk, simd_width](tile_len.N)


# ===----------------------------------------------------------------------=== #
# Packing routines.
# ===----------------------------------------------------------------------=== #


@value
struct PackMatrixRows[
    # original matrix shape list
    original_shape: DimList,
    # packed matrix shape list
    packed_shape: DimList,
    type: DType,
    simd_size: Int,
    row_inner_size: Int,
]:
    """Pack rows from a matrix into the mlas packed layout and
    extract inner vectors of rows into the packed inner dimension,
    e.g. extract tile [X, Y] and pack into [Xo][Y][Xi].
    """

    # packed matrix
    var packed_matrix: NDBuffer[type, 3, packed_shape]
    # original matrix:
    var original_matrix: NDBuffer[type, 2, original_shape]
    # offsets in original matrix
    var global_offset: StaticIntTuple[2]
    # number of Row and Col to pack.
    #  in [Row, Col]
    var pack_tile_dim: StaticIntTuple[2]
    # valid data bound within the tile.
    var valid_data_dim: StaticIntTuple[2]
    # valid multiple-of-simd data bound within the tile.
    var valid_simd_dim: StaticIntTuple[2]

    # Interface method:
    #  run the packing and store to the given buffer.
    @staticmethod
    fn run(
        packed_matrix: NDBuffer[type, 3, packed_shape],
        original_matrix: NDBuffer[type, 2, original_shape],
        global_offset: StaticIntTuple[2],
        pack_tile_dim: StaticIntTuple[2],
        valid_data_dim: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            original_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            pack_tile_dim(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile.
            valid_data_dim(StaticIntTuple): 2D dimension tuple describing the
                amount of valid data on the global buffer starting from the
                offset.
        """
        constrained[row_inner_size % simd_size == 0]()

        var instance = Self(
            packed_matrix,
            original_matrix,
            global_offset,
            pack_tile_dim,
            valid_data_dim,
            (
                align_down(
                    min(
                        valid_data_dim[0],
                        pack_tile_dim[0],
                    ),
                    simd_size,
                ),
                align_down(
                    min(
                        valid_data_dim[1],
                        pack_tile_dim[1],
                    ),
                    simd_size,
                ),
            ),
        )

        instance._pack()

    fn _transpose_pack_helper[
        skip_row_bound: Bool,
        skip_col_bound: Bool,
    ](
        self,
        transpose_buffer: NDBuffer[
            type,
            2,
            DimList(simd_size, simd_size),
        ],
        local_off_set: StaticIntTuple[2],
    ):
        """Helper function: transpose packs a [simd_size, simd_size] subtile of
        matrix, with bound checking and zero-filling. Bound checking can be
        statically skipped, based on the parameters.
           Args:
               skip_row_bound(Bool): boundary check on x dimension will be
                   skipped if true.
               skip_col_bound(Bool): boundary check on y dimension will be
                   skpped if true.
               transpose_buffer(NDBuffer): pre-allocated work space to hold
                   transposed temporary data.
               local_offset(StaticIntTuple): offset of the subtile to work on
                   within the whole tile of data to pack.
        """
        # Calculate the remaining bound from the local offset.
        # Boundaries for readable data.
        var read_bound = self.valid_data_dim - local_off_set
        # Boundaries for writeable space.
        var write_bound = self.pack_tile_dim - local_off_set

        # Global index the packing is starting from.
        var start_idx_global = local_off_set + self.global_offset

        # Fill the simd_size x simd_size transpose buffer
        #  with un-transposed data.
        @always_inline
        @__copy_capture(read_bound, start_idx_global)
        @parameter
        fn body[idx: Int]():
            alias inner_row_idx = idx
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound[0]):
                var row_global_index = (
                    start_idx_global[0] + inner_row_idx,
                    start_idx_global[1],
                )
                var row_data: SIMD[type, simd_size]
                if skip_col_bound:
                    # This is fastest path where both row and col bounds
                    #  are skipped so the code path is simd-in and simd-out
                    #  without any predicate.
                    row_data = self.original_matrix.load[width=simd_size](
                        row_global_index
                    )
                else:
                    # Not skipping col bound, need to to a partial fill of
                    #  the transpose buffer row.
                    row_data = partial_simd_load[simd_size](
                        self.original_matrix._offset(row_global_index),
                        0,  # no left bound.
                        read_bound[1],
                        0,
                    )

                transpose_buffer.store[width=simd_size](
                    (inner_row_idx, 0), row_data
                )
            else:
                # Row out of defined bound, fill the transpose buffer with zero
                transpose_buffer.store[width=simd_size](
                    (inner_row_idx, 0), SIMD[type, simd_size](0)
                )

        unroll[body, simd_size]()

        # Transpose the buffered data
        transpose_inplace[simd_size, simd_size, type](transpose_buffer)

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        @always_inline
        @__copy_capture(write_bound)
        @parameter
        fn transposed_inner_row_body[idx: Int]():
            var transposed_data = transpose_buffer.load[width=simd_size](
                (idx, 0)
            )
            # compute the packed index
            var _row_outer = local_off_set[0] // row_inner_size
            var _row_inner = local_off_set[0] % row_inner_size

            if skip_col_bound or (idx < write_bound[1]):
                self.packed_matrix.store[width=simd_size](
                    (
                        _row_outer,
                        local_off_set[1] + idx,
                        _row_inner,
                    ),
                    transposed_data,
                )
            # Out of bound columns are discarded as there's no allocation for them
            #  in the packed buffer.

        unroll[transposed_inner_row_body, simd_size]()

    fn _pack(self):
        """Helper function: Allocates transpose workspace and launch the
        transpose helper function until all required data has been packed.
        """

        var transpose_buffer = NDBuffer[
            type,
            2,
            DimList(simd_size, simd_size),
        ].aligned_stack_allocation[alignof[SIMD[type, simd_size]]()]()

        var valid_tile_simd_dim = Index(
            min(
                self.valid_simd_dim[0],
                self.pack_tile_dim[0],
            ),
            min(
                self.valid_simd_dim[1],
                self.pack_tile_dim[1],
            ),
        )

        # fill rows with valid data

        var row_idx: Int = 0
        var col_idx: Int = 0

        # An unswitch-able unit function that transpose packs a small tile.
        @always_inline
        @__copy_capture(transpose_buffer)
        @parameter
        fn transpose_pack_unit[static_switch0: Bool, static_switch1: Bool]():
            self._transpose_pack_helper[
                # skip_row_bound, skip_col_bound
                static_switch0,
                static_switch1,
            ](
                transpose_buffer,
                # local offset
                (row_idx, col_idx),
            )

        # Pack the whole matrices with the unit helper.
        while row_idx < self.pack_tile_dim[0]:
            col_idx = 0
            while col_idx < self.pack_tile_dim[1]:
                unswitch[transpose_pack_unit](
                    row_idx + simd_size <= valid_tile_simd_dim[0],
                    col_idx + simd_size <= valid_tile_simd_dim[1],
                )
                col_idx += simd_size
            row_idx += simd_size


@value
struct PackMatrixCols[
    # original matrix shape list
    original_shape: DimList,
    # packed matrix shape list
    packed_shape: DimList,
    type: DType,
    simd_size: Int,
    column_inner_size: Int,
    use_vnni: Bool,
    use_i8mm: Bool,
]:
    """Pack columns from a matrix into the mlas packed layout and
    extract inner vectors of columns into the packed inner dimension,
    e.g. extracts [X, Y] and packs as [Yo][X][Yi].
    """

    # packed matrix
    var packed_matrix: NDBuffer[type, 3, packed_shape]
    # original matrix:
    var original_matrix: NDBuffer[type, 2, original_shape]
    # offsets in original matrix:
    var global_offset: StaticIntTuple[2]
    # number of Row and Col to pack.
    #  in [Row, Col]
    var pack_tile_dim: StaticIntTuple[2]
    # valid data bound within the tile.
    var valid_data_dim: StaticIntTuple[2]

    # Interface function:
    @staticmethod
    fn run(
        packed_matrix: NDBuffer[type, 3, packed_shape],
        original_matrix: NDBuffer[type, 2, original_shape],
        global_offset: StaticIntTuple[2],
        pack_tile_dim: StaticIntTuple[2],
        valid_data_dim: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            original_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            pack_tile_dim(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile.
            valid_data_dim(StaticIntTuple): 2D dimension tuple describing the
                amount of valid data on the global buffer starting from the
                offset.
        """
        constrained[column_inner_size % simd_size == 0]()
        debug_assert(
            pack_tile_dim[1] % column_inner_size == 0,
            "Unimplemented tile pattern.",
        )

        var instance = Self(
            packed_matrix,
            original_matrix,
            global_offset,
            pack_tile_dim,
            valid_data_dim,
        )

        instance._pack()

    @always_inline
    fn _pack_helper[
        skip_row_bound: Bool, skip_col_bound: Bool
    ](self, row_start: Int, valid_row_count: Int, col_start: Int):
        """Helper function: copy several simd vectors on the column from the
        original matrix to the packed buffer. The copies are unrolled and
        prefetched (if not with neon).
            Args:
                skip_row_bound: Boundary check on row dimension will be skipped
                    if true.
                skip_col_bound: Boundary check on column dimension will be skipped
                    if true.
        """

        alias unroll_factor = get_packB_unroll_factor()

        @always_inline
        @parameter
        fn pack_vector(row_idx: Int, col_idx: Int):
            var global_idx = self.global_offset + Index(row_idx, col_idx)
            var data = SIMD[type, simd_size](0)
            if skip_col_bound or (
                col_idx + simd_size <= self.valid_data_dim[1]
            ):
                data = self.original_matrix.load[width=simd_size](global_idx)
            elif col_idx < self.valid_data_dim[1]:
                # Starting point within bound but cannot load a whole
                #  vector. Do a partial load.
                data = partial_simd_load[simd_size](
                    self.original_matrix._offset(global_idx),
                    0,
                    self.valid_data_dim[1] - col_idx,
                    0,
                )

            # map to packed index
            var col_idx_outer = col_idx // column_inner_size
            var col_idx_inner = col_idx % column_inner_size
            self.packed_matrix.store[width=simd_size](
                (col_idx_outer, row_idx, col_idx_inner),
                data,
            )

        @always_inline
        @parameter
        fn pack_body[idx: Int]():
            pack_vector(row_start + idx, col_start)

        @always_inline
        @parameter
        fn prefetch_body[idx: Int]():
            var global_row_idx = (
                self.global_offset[0] + row_start + unroll_factor + idx
            )
            var global_col_idx = self.global_offset[1] + col_start
            self.original_matrix.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](global_row_idx, global_col_idx)

        @parameter
        if skip_row_bound:
            if not has_neon():
                unroll[prefetch_body, unroll_factor]()
            unroll[pack_body, unroll_factor]()
        else:
            for row_idx in range(row_start, valid_row_count):
                pack_vector(row_idx, col_start)

    fn _pack_vnni(self):
        """Copy the B tile from the original matrix to the packed buffer for VNNI.
        """
        constrained[use_vnni]()
        var kc = self.valid_data_dim[0]
        var nc = self.valid_data_dim[1]
        var nr = column_inner_size
        for i in range(0, align_up(kc, 4), 4):
            for j in range(self.pack_tile_dim[1] // nr):
                for p in range(nr):

                    @unroll
                    for l in range(4):
                        var local_idx = Index(i + l, p + nr * j)
                        var val = 0 if local_idx[0] >= kc or local_idx[
                            1
                        ] >= nc else self.original_matrix[
                            self.global_offset + local_idx
                        ]
                        self.packed_matrix.store[width=1](
                            (j, i // 4, 4 * p + l),
                            val,
                        )

    fn _pack_i8mm(self):
        constrained[use_i8mm]()
        var kc = self.valid_data_dim[0]
        var nc = self.valid_data_dim[1]
        alias column_inner_size2 = column_inner_size // 2
        for i in range(0, align_up(kc, 8), 8):
            for j in range(ceildiv(nc, column_inner_size2)):
                for p in range(0, column_inner_size2, 2):
                    for i2 in range(8):
                        for p2 in range(2):
                            var local_idx = Index(
                                i + i2, column_inner_size2 * j + p + p2
                            )
                            var val = 0 if local_idx[0] >= kc or local_idx[
                                1
                            ] >= nc else self.original_matrix[
                                self.global_offset + local_idx
                            ]
                            self.packed_matrix.store[width=1](
                                Index(j, i // 8, 8 * p + 8 * p2 + i2),
                                val,
                            )

    fn _pack_default(self):
        """Copy the B tile from the original matrix to the packed buffer.
        Each iteration copies a block of shape (unroll_factor, simd_size)."""
        constrained[not use_vnni and not use_i8mm]()
        var valid_row_count = min(self.valid_data_dim[0], self.pack_tile_dim[0])
        alias unroll_factor = get_packB_unroll_factor()

        var row_idx: Int = 0
        var col_idx: Int = 0

        @always_inline
        @__copy_capture(valid_row_count)
        @parameter
        fn pack_unit[skip_row_bound: Bool, skip_col_bound: Bool]():
            self._pack_helper[skip_row_bound, skip_col_bound](
                row_idx, valid_row_count, col_idx
            )

        while row_idx < valid_row_count:
            col_idx = 0
            while col_idx < self.pack_tile_dim[1]:
                unswitch[pack_unit](
                    row_idx + unroll_factor < valid_row_count,
                    col_idx + simd_size < self.valid_data_dim[1],
                )
                col_idx += simd_size
            row_idx += unroll_factor

    fn _pack(self):
        @parameter
        if use_vnni:
            self._pack_vnni()
        elif use_i8mm:
            self._pack_i8mm()
        else:
            self._pack_default()


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
                        column_step
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


struct MatmulInnerLoopBPacked[
    a_shape: DimList,
    c_shape: DimList,
    packed_shape: DimList,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
    prefetch_b_distance: Int,
    saturated_vnni: Bool,
    single_row_i8mm: Bool = False,
]:
    """Inner loop implementation for mlas-style tiled matmul. Accumulates a
    tile of input defined by (a_row_size, TileN, TileK).
    """

    # Parameters for global reference.
    var c_stride: Int
    var c_ptr: DTypePointer[c_type]
    var a: NDBuffer[a_type, 2, a_shape]
    var b_packed: NDBuffer[b_type, 3, packed_shape]
    # 3D global offset within the whole matmul problem space.
    var global_offset: GemmShape
    # Dynamic tiling parameter for this inner loop
    #  in (TileN, TileK).
    var tile_n_k: StaticIntTuple[2]
    # Boundary of valid output space within the
    #  local tile, in (a_row_size, TileN).
    var c_bound: StaticIntTuple[2]

    alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    fn __init__(
        inout self,
        c: NDBuffer[c_type, 2, c_shape],
        a: NDBuffer[a_type, 2, a_shape],
        b_packed: NDBuffer[b_type, 3, packed_shape],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
        c_bound: StaticIntTuple[2],
    ):
        self.c_stride = c.dim[1]()
        self.c_ptr = c.data.offset(
            global_offset.M * self.c_stride + global_offset.N
        )
        self.a = a
        self.b_packed = b_packed
        self.global_offset = global_offset
        self.tile_n_k = tile_n_k
        self.c_bound = c_bound

    @staticmethod
    fn run(
        c: NDBuffer[c_type, 2, c_shape],
        a: NDBuffer[a_type, 2, a_shape],
        b_packed: NDBuffer[b_type, 3, packed_shape],
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.

        Args:
            c: Pre-allocated buffer space for packed result.
            a: Data buffer operand A.
            b_packed: Data buffer operand B in packed layout.
            global_offset: Offset to use when indexing the original matrix.
            global_bound: Tile upper boundary of the current tile function call.
            tile_n_k: 2D dimension tuple describing the size of the packed tile
                of B.
        """
        var instance = Self(
            c,
            a,
            b_packed,
            global_offset,
            tile_n_k,
            Index(global_bound.M, global_bound.N)
            - Index(global_offset.M, global_offset.N),
        )
        instance._run_inner_loop()

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
    ):
        """Utility function on the inner loop. Initializes a local c buffer with
        all zeros.

        Args:
            c_local: pre-allocated local buffer for c partial sums.
        """

        @always_inline
        @parameter
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.store[
                width=simd_size, alignment = alignof[SIMD[c_type, simd_size]]()
            ](
                Index(idx0, idx1 * simd_size),
                SIMD[c_type, simd_size](0),
            )

        unroll[outer_body, a_row_size, pack_inner_size // simd_size]()

    @always_inline
    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_idx: Int,
    ):
        """Utility function on the inner loop. Loads a local c_buffer with the
        value stored in the output buffer space, given the indices within the
        tile being processed.

        Args:
            c_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
        """

        var c_ptr = self.c_ptr.offset(tile_n_idx)

        @parameter
        if self.use_i8mm:

            @always_inline
            @parameter
            fn body_i8mm[idx0: Int, idx1: Int]():
                var c_data: SIMD[c_type, simd_size] = 0
                if skip_boundary_check or (
                    idx1 * 2 + 2 <= self.c_bound[1] - tile_n_idx
                ):
                    var t0 = c_ptr.load[width=2](
                        self.c_stride * (2 * idx0 + 0) + 2 * idx1
                    )
                    var t1 = c_ptr.load[width=2](
                        self.c_stride * (2 * idx0 + 1) + 2 * idx1
                    ) if not single_row_i8mm else SIMD[c_type, 2](0)
                    c_data = rebind[SIMD[c_type, simd_size]](t0.join(t1))
                elif idx1 * 2 <= self.c_bound[1]:
                    var t0 = partial_simd_load[2](
                        c_ptr.offset(self.c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        self.c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    )
                    var t1 = partial_simd_load[2](
                        c_ptr.offset(self.c_stride * (2 * idx0 + 1) + 2 * idx1),
                        0,
                        self.c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    ) if not single_row_i8mm else SIMD[c_type, 2](0)
                    c_data = rebind[SIMD[c_type, simd_size]](t0.join(t1))

                # Store data to local buffer.
                c_local.store[width=simd_size](
                    Index(idx0, idx1 * simd_size),
                    rebind[SIMD[c_type, simd_size]](c_data),
                )

            unroll[body_i8mm, a_row_size, pack_inner_size // simd_size]()
            return

        @parameter
        if has_neon():
            self._initialize_c_tile(c_local)
            return LoadStoreOutputTile[
                c_type, simd_size, a_row_size, pack_inner_size, True
            ].run(
                c_local,
                c_ptr,
                self.c_stride,
                min(self.c_bound[1] - tile_n_idx, pack_inner_size),
            )

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data: SIMD[c_type, simd_size] = 0
            if skip_boundary_check or (
                idx1 * simd_size + simd_size <= self.c_bound[1] - tile_n_idx
            ):
                # Use simd load if all within bound
                c_data = c_ptr.load[width=simd_size](idx1 * simd_size)
            elif idx1 * simd_size <= self.c_bound[1]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[simd_size](
                    c_ptr.offset(idx1 * simd_size),
                    0,
                    self.c_bound[1] - tile_n_idx - idx1 * simd_size,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = 0

            # Store data to local buffer.
            c_local.store(Index(idx0, idx1 * simd_size), c_data)

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr = c_ptr.offset(self.c_stride)

        unroll[body, a_row_size, pack_inner_size // simd_size]()

    @always_inline
    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_idx: Int,
    ):
        """Utility function on the inner loop. Stores the value of a local c
        buffer to the corresponding position in the output buffer space.

        Args:
            c_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
        """
        var c_ptr = self.c_ptr.offset(tile_n_idx)

        @parameter
        if self.use_i8mm:

            @always_inline
            @parameter
            fn body_i8mm[idx0: Int, idx1: Int]():
                var c_data = c_local.load[width=simd_size](
                    Index(idx0, idx1 * simd_size)
                )
                if skip_boundary_check or (
                    idx1 * 2 + 2 <= self.c_bound[1] - tile_n_idx
                ):
                    c_ptr.offset(
                        self.c_stride * (2 * idx0 + 0) + 2 * idx1
                    ).store[width=2](c_data.slice[2](0))

                    @parameter
                    if not single_row_i8mm:
                        c_ptr.offset(
                            self.c_stride * (2 * idx0 + 1) + 2 * idx1
                        ).store[width=2](c_data.slice[2](2))
                elif idx1 * 2 <= self.c_bound[1]:
                    partial_simd_store(
                        c_ptr.offset(self.c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        self.c_bound[1] - tile_n_idx - idx1 * 2,
                        c_data.slice[2](0),
                    )

                    @parameter
                    if not single_row_i8mm:
                        partial_simd_store(
                            c_ptr.offset(
                                self.c_stride * (2 * idx0 + 1) + 2 * idx1
                            ),
                            0,
                            self.c_bound[1] - tile_n_idx - idx1 * 2,
                            c_data.slice[2](2),
                        )

            unroll[body_i8mm, a_row_size, pack_inner_size // simd_size]()
            return

        @parameter
        if has_neon():
            return LoadStoreOutputTile[
                c_type, simd_size, a_row_size, pack_inner_size, False
            ].run(
                c_local,
                c_ptr,
                self.c_stride,
                min(self.c_bound[1] - tile_n_idx, pack_inner_size),
            )

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data = c_local.load[width=simd_size](
                Index(idx0, idx1 * simd_size)
            )
            if skip_boundary_check or (
                idx1 * simd_size + simd_size <= self.c_bound[1] - tile_n_idx
            ):
                # Use simd store if all within bound
                c_ptr.offset(idx1 * simd_size).store[width=simd_size](c_data)
            elif idx1 * simd_size <= self.c_bound[1]:
                # Use partial store if col not in simd bound.
                partial_simd_store(
                    c_ptr.offset(idx1 * simd_size),
                    0,
                    self.c_bound[1] - tile_n_idx - idx1 * simd_size,
                    c_data,
                )

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr = c_ptr.offset(self.c_stride)

        unroll[body, a_row_size, pack_inner_size // simd_size]()

    fn _accumulate[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
        constrained[a_col_size == 1]()
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx[1]

        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, tile_n_k_idx[1], 0)
        )

        # Prefetch B matrix.
        @parameter
        if prefetch_b_distance > 0:
            alias prefetch_offset = prefetch_b_distance * pack_inner_size

            @unroll
            for idx in range(pack_inner_size // simd_size):
                b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        # This inner kernels works with non-transposed A.
        var K = self.a.dim[1]()
        var a_ptr = self.a.data.offset(self.global_offset.M * K + global_k)

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                var c_idx = Index(idx0, idx1 * simd_size)
                var a_val = a_ptr[idx0 * K].cast[c_type]()
                alias alignment = alignof[SIMD[c_type, simd_size]]()
                var c_val = c_local.load[width=simd_size, alignment=alignment](
                    c_idx
                )
                var b_val = b_ptr.load[width=simd_size, alignment=alignment](
                    idx1 * simd_size
                ).cast[c_type]()
                c_val = fma[c_type, simd_size](a_val, b_val, c_val)
                c_local.store[width=simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn _accumulate_vnni[
        is_tail: Bool
    ](
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // pack_inner_size
        var kl = tile_n_k_idx[1]

        # Global K index.
        var global_k = self.global_offset.K + kl
        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, kl // 4, 0)
        ).bitcast[c_type]()

        @parameter
        if not is_tail:
            # Prefetch B matrix.
            @parameter
            if prefetch_b_distance > 0:
                alias prefetch_offset = prefetch_b_distance * pack_inner_size

                @unroll
                for idx in range(pack_inner_size // simd_size):
                    b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ]()

        # This inner kernels works with non-transposed A.
        var K = self.a.dim(1)

        var a_local = Buffer[a_type, 4 * a_row_size].stack_allocation()
        var a_base_ptr = self.a.data.offset(self.global_offset.M * K + global_k)
        var a_ptr = a_local.data if (
            is_tail and not has_avx512f()
        ) else a_base_ptr
        var a_ptr_stride = 4 if (is_tail and not has_avx512f()) else K

        var tail_length = self.tile_n_k[1] - kl

        # pack A if (tile_n_k_idx[1] - kl) is 1, 2, or 3
        @parameter
        if is_tail and not has_avx512f():
            for idx0 in range(a_row_size):
                for idx_k in range(tail_length):
                    a_local[4 * idx0 + idx_k] = a_base_ptr.offset(
                        idx0 * K + idx_k
                    ).load()

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                # width K bytes or K/4 ints, a_ptr is pointer to ints
                var a_val = bitcast[c_type, 1](
                    partial_simd_load[4](
                        a_ptr.offset(idx0 * a_ptr_stride), 0, tail_length, 0
                    )
                ) if (is_tail and has_avx512f()) else a_ptr.offset(
                    idx0 * a_ptr_stride
                ).bitcast[
                    c_type
                ]().load()

                alias alignment = alignof[SIMD[c_type, simd_size]]()
                var c_idx = Index(idx0, idx1 * simd_size)
                var c_val = c_local.load[width=simd_size, alignment=alignment](
                    c_idx
                )

                var b_val = b_ptr.offset(idx1 * simd_size).load[
                    width=simd_size, alignment=alignment
                ]()

                @parameter
                if has_neon_int8_dotprod():
                    var a_val2 = SIMD[c_type, simd_size].splat(a_val)
                    c_val = _neon_dotprod[a_type, b_type, c_type, simd_size](
                        c_val,
                        bitcast[a_type, 16](a_val2),
                        bitcast[b_type, 16](b_val),
                    )
                elif saturated_vnni:
                    c_val = dot_i8_to_i32_saturated_x86[simd_size](
                        c_val, a_val, b_val
                    )
                else:
                    c_val = dot_i8_to_i32_x86[simd_size](c_val, a_val, b_val)
                c_local.store[width=simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn _run_inner_loop_vnni(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[Self.use_vnni]()
        debug_assert(
            self.tile_n_k[1] % 0 == 0, "K dimension must be a multipel of 4"
        )

        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[alignof[SIMD[c_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            var kl = align_down(self.tile_n_k[1], 4)
            for idx_k in range(0, kl, 4):
                # accumulate data for this (n, k) index
                self._accumulate_vnni[False](c_local, Index(idx_n, idx_k))
            if kl != self.tile_n_k[1]:
                self._accumulate_vnni[True](c_local, Index(idx_n, kl))
            self._store_c_tile(c_local, idx_n)

    fn _run_inner_loop_default(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[not Self.use_vnni and not has_neon()]()
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[alignof[SIMD[c_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            for idx_k in range(self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate[1](c_local, Index(idx_n, idx_k))

            self._store_c_tile(c_local, idx_n)

    fn _accumulate_lane[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing `a_col_size` columns of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k)
                coordinates within the current processing tile to index the
                packed B matrix.
        """
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx[1]

        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, tile_n_k_idx[1], 0)
        )

        var a_vals = stack_allocation[a_row_size, SIMD[c_type, a_col_size]]()

        @unroll
        for row in range(a_row_size):
            var global_m = self.global_offset.M + row
            var a_val = self.a.load[width=a_col_size](global_m, global_k).cast[
                c_type
            ]()
            a_vals[row] = a_val

        @unroll
        for lane in range(a_col_size):

            @unroll
            for col in range(pack_inner_size // simd_size):
                var b_val = b_ptr.offset(col * simd_size).load[
                    width=simd_size
                ]().cast[c_type]()

                @unroll
                for row in range(a_row_size):
                    var a_val = a_vals[row]
                    var c_idx = Index(row, col * simd_size)
                    var c_val = c_local.load[width=simd_size](c_idx)
                    c_val = fma[c_type, simd_size](a_val[lane], b_val, c_val)
                    c_local.store[width=simd_size](c_idx, c_val)

            b_ptr = b_ptr.offset(pack_inner_size)

    fn _run_inner_loop_neon(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[has_neon() and not Self.use_vnni and not Self.use_i8mm]()
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[alignof[SIMD[c_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)

            var partition_end = simd_size * (self.tile_n_k[1] // simd_size)
            for idx_k0 in range(0, partition_end, simd_size):
                self._accumulate_lane[simd_size](c_local, Index(idx_n, idx_k0))

            for idx_k1 in range(partition_end, self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_lane[1](c_local, Index(idx_n, idx_k1))

            self._store_c_tile(c_local, idx_n)

    fn _accumulate_i8mm(
        self,
        c_local: NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
        var n_outer_idx = tile_n_k_idx[0] // (pack_inner_size // 2)
        var kl = tile_n_k_idx[1]
        var b_ptr = self.b_packed._offset(Index(n_outer_idx, kl // 8, 0))

        # This inner kernels works with non-transposed A.
        var K = self.a.dim(1)
        var a_ptr = self.a.data.offset(
            self.global_offset.M * K + self.global_offset.K + 2 * kl
        )

        # Prefetch B matrix.
        @parameter
        if prefetch_b_distance > 0:
            alias prefetch_offset = prefetch_b_distance * pack_inner_size

            @unroll
            for idx in range(pack_inner_size // simd_size):
                b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                alias alignment = alignof[SIMD[c_type, simd_size]]()
                var a_val = a_ptr.load[width=16](2 * idx0 * K)
                var b_val = b_ptr.offset(16 * idx1).load[
                    width=16, alignment=alignment
                ]()
                var c_idx = Index(idx0, 4 * idx1)
                var c_val = c_local.load[width=simd_size, alignment=alignment](
                    c_idx
                )
                c_val = _neon_matmul(c_val, a_val, b_val)
                c_local.store[width=simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn _run_inner_loop_i8mm(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[Self.use_i8mm]()

        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[alignof[SIMD[c_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size // 2):
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)
            var kl = align_up(self.tile_n_k[1], 8)
            for idx_k in range(0, kl, 8):
                self._accumulate_i8mm(c_local, Index(idx_n, idx_k))
            self._store_c_tile(c_local, idx_n)

    fn _run_inner_loop(self):
        @parameter
        if Self.use_i8mm:
            self._run_inner_loop_i8mm()
        elif has_neon() and not Self.use_vnni and not Self.use_i8mm:
            self._run_inner_loop_neon()
        elif not Self.use_vnni and not has_neon():
            self._run_inner_loop_default()
        elif Self.use_vnni:
            self._run_inner_loop_vnni()
        else:
            constrained[False, "no _run_inner_loop implementation"]()


# Tiled Matmul Implementation.
# TODO: not yet supporting transpose_a
@value
struct TiledMatmul[
    config: MatmulConfig,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    rowwise_epilogue_enabled: Bool,
]:
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    TODO: not yet supporting transpose_a.
    TODO: add tag based implementation dispatch.
    TODO: add fusion hooks.
    """

    var c: NDBuffer[c_type, 2, config.c_shape]
    var a: NDBuffer[a_type, 2, config.a_shape]
    var b: NDBuffer[b_type, 2, config.b_shape]
    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]

    # Tile starting points on the (M,N,K) coordinates.
    var global_tile_offset: GemmShape

    # Tile sizes this routine will process on the (M,N,K) coordinates.
    var global_tile_shape: GemmShape

    var b_tile_generator: BTileGenerator[config, b_type, transpose_b, b_packed]

    var elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None

    var rowwise_epilogue_fn: fn (Int, Int) escaping -> None

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[c_type, 2, config.c_shape],
        a: NDBuffer[a_type, 2, config.a_shape],
        b: NDBuffer[b_type, 2, config.b_shape],
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape = GemmShape {M: 0, N: 0, K: 0},
    ):
        fn null_elementwise_epilogue(offset: GemmShape, tile_len: GemmShape):
            pass

        fn null_rowwise_epilogue(offset: Int, num_rows: Int):
            pass

        Self.run(
            c,
            a,
            b,
            null_elementwise_epilogue,
            null_rowwise_epilogue,
            global_tile_shape,
            global_tile_offset,
        )

    @staticmethod
    fn run(
        c: NDBuffer[c_type, 2, config.c_shape],
        a: NDBuffer[a_type, 2, config.a_shape],
        b: NDBuffer[b_type, 2, config.b_shape],
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None,
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape,
    ):
        fn null_rowwise_epilogue(offset: Int, num_rows: Int):
            pass

        Self.run(
            c,
            a,
            b,
            elementwise_epilogue_fn,
            null_rowwise_epilogue,
            global_tile_shape,
            global_tile_offset,
        )

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[c_type, 2, config.c_shape],
        a: NDBuffer[a_type, 2, config.a_shape],
        b: NDBuffer[b_type, 2, config.b_shape],
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None,
        rowwise_epilogue_fn: fn (Int, Int) escaping -> None,
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape,
    ):
        """Interface function to run tiled matmul on a given sub-tile.

        Args:
            c: Pre-allocated buffer space for result.
            a: Operand A of the matmul.
            b: Operand B of the mamtul.
            elementwise_epilogue_fn: The elementwise epilogue function.
            rowwise_epilogue_fn: The row-wise epilogue function.
            global_tile_shape: Tile shape this call will process.
            global_tile_offset: Tile offset on the original buffer.
        """
        alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
        alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
        alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

        var tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size, factor
        ](global_tile_shape)

        var matmul = TiledMatmul[
            config,
            a_type,
            b_type,
            c_type,
            transpose_a,
            transpose_b,
            b_packed,
            elementwise_epilogue_enabled,
            rowwise_epilogue_enabled,
        ](
            c,
            a,
            b,
            tile_n_k,
            global_tile_offset,
            global_tile_shape,
            BTileGenerator[config, b_type, transpose_b, b_packed].get(
                b, tile_n_k
            ),
            elementwise_epilogue_fn,
            rowwise_epilogue_fn,
        )

        matmul._outer_k_loop()

    fn _outer_m_loop[
        last_n_tile: Bool,
        last_k_tile: Bool,
        m_loop_pack_inner_size: Int,
    ](self, global_offset: GemmShape, sub_tile_n: Int, sub_tile_k: Int):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

        Parameters:
            last_n_tile: The last n tile.
            last_k_tile: The last k tile.
            m_loop_pack_inner_size: Inner dimension of the packed data layout.

        Args:
            global_offset: 3D global offset within the whole
                matmul problem space.
            sub_tile_n: Dynamic tile size to use on N dimension.
            sub_tile_k: Dynamic tile size to use on K dimension.
        """
        # valid distance in each dimension from the current offset to the end of the matrix
        var knm_bounds = (
            self.global_tile_shape + self.global_tile_offset - global_offset
        )

        @__copy_capture(knm_bounds)
        @parameter
        @always_inline
        fn unswitch_residual_n[skip_col_bound: Bool]():
            var b_packed_tile = self.b_tile_generator.get_tile[
                m_loop_pack_inner_size
            ](
                global_offset,
                Index(sub_tile_n, sub_tile_k),
                Index(knm_bounds.N, knm_bounds.K),
            )

            # Launch the MLoop
            # The upper bounds apply to runtime packing. For pre-packing, the
            # tile has been padded to fit (sub_tile_n, sub_tile_k).
            var sub_tile_n_k = Index(
                min(sub_tile_n, knm_bounds.N), min(sub_tile_k, knm_bounds.K)
            )

            @__copy_capture(sub_tile_n_k, b_packed_tile)
            @parameter
            @always_inline
            fn row_iteration[tile_size: Int](row_offset: Int):
                alias tile_size2 = 2 if tile_size == 1 else tile_size
                alias a_row_size = tile_size2 // 2 if config.use_i8mm else tile_size
                MatmulInnerLoopBPacked[
                    config.a_shape,
                    config.c_shape,
                    config.packed_shape,
                    a_type,
                    b_type,
                    c_type,
                    config.simd_size,
                    a_row_size,
                    m_loop_pack_inner_size,
                    skip_col_bound,
                    config.prefetch_b_distance_k,
                    config.saturated_vnni,
                    tile_size == 1,
                ].run(
                    self.c,
                    self.a,
                    b_packed_tile,
                    global_offset + GemmShape(row_offset, 0, 0),
                    self.global_tile_offset + self.global_tile_shape,
                    sub_tile_n_k,
                )

                @parameter
                if elementwise_epilogue_enabled and last_k_tile:
                    self.elementwise_epilogue_fn(
                        global_offset + GemmShape(row_offset, 0, 0),
                        GemmShape {
                            M: tile_size, N: sub_tile_n_k[0], K: sub_tile_n_k[1]
                        },
                    )

                @parameter
                if rowwise_epilogue_enabled and last_n_tile and last_k_tile:
                    self.rowwise_epilogue_fn(
                        global_offset.M + row_offset, tile_size
                    )

            @parameter
            if config.use_i8mm:
                tile[
                    row_iteration,
                    VariadicList[Int](2 * config.a_row_size, 8, 6, 4, 2, 1),
                ](
                    0,  # starting row offset
                    knm_bounds.M,  # row bound
                )
            else:
                tile[
                    row_iteration,
                    VariadicList[Int](config.a_row_size, 4, 3, 2, 1),
                ](
                    0,  # starting row offset
                    knm_bounds.M,  # row bound
                )

        @parameter
        if has_neon():
            # The performance of the skip_col_bound=True path is the same as
            # skip_col_bound=False, so reduce code size and emit only the
            # skip_col_bound=False path.
            unswitch_residual_n[False]()
        else:
            unswitch[unswitch_residual_n](knm_bounds[1] > sub_tile_n)

    # Iterate on the N dimension of the gemm space.
    fn _outer_n_loop[
        last_k_tile: Bool
    ](self, global_offset: GemmShape, sub_tile_k: Int):
        """Iterate on the N dimension of the whole problem space.

        Args:
            global_offset: 3D global offset within the whole matmul problem
                space.
            sub_tile_k: Dynamic tile size to use on K dimension.
        """
        var valid_col_count: Int = (
            self.global_tile_shape.N
            + self.global_tile_offset.N
            - global_offset.N
        )
        var tile_n: Int = self.tile_n_k[0]

        @parameter
        @always_inline
        fn m_loop[secondary_tile_size: Int](col_idx: Int, tile_size_n: Int):
            @parameter
            @always_inline
            fn m_loop_switch[last_n_tile: Bool]():
                self._outer_m_loop[
                    last_n_tile, last_k_tile, secondary_tile_size
                ](
                    global_offset + GemmShape(0, col_idx, 0),
                    tile_size_n,
                    sub_tile_k,
                )

            unswitch[m_loop_switch](
                self.global_tile_offset.N + col_idx + tile_size_n
                >= self.global_tile_shape.N
            )

        # if b is packed, the packing was performed offline using a single inner
        # size and tile_n.
        @parameter
        if not b_packed:
            alias secondary_tiles = VariadicList[Int](
                config.pack_inner_size, 2 * config.simd_size, config.simd_size
            )
            var primary_tiles = VariadicList[Int](
                tile_n, 2 * config.simd_size, config.simd_size
            )
            tile[secondary_tiles, config.simd_size, m_loop](
                0, valid_col_count, primary_tiles, config.simd_size
            )
        else:
            alias secondary_tiles_packed_b = VariadicList[Int](
                config.pack_inner_size
            )
            var primary_tiles_packed_b = VariadicList[Int](tile_n)
            tile[secondary_tiles_packed_b, config.pack_inner_size, m_loop](
                0, valid_col_count, primary_tiles_packed_b, tile_n
            )

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(
        self,
    ):
        """Iterate on the K dimension of the whole problem space."""

        # Each tiled iteration on the k dimension.
        @always_inline
        @parameter
        fn k_iteration(k_offset: Int, k_tile_size: Int):
            @always_inline
            @parameter
            fn outer_n_loop[last_k_tile: Bool]():
                self._outer_n_loop[last_k_tile](
                    GemmShape(0, 0, k_offset) + self.global_tile_offset,
                    k_tile_size,
                )

            unswitch[outer_n_loop](
                k_offset + k_tile_size + self.global_tile_offset.K
                == self.global_tile_shape.K
            )

        tile[k_iteration](
            0,  # k offset
            self.global_tile_shape.K,  # valid K count
            self.tile_n_k[1],  # max tile k size
        )

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed_ptr: DTypePointer[b_type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[b_type, 3, config.packed_shape]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

        Args:
            b_packed_ptr: B matrix in packed layout.
            tile_n: Dynamic tile size to use on N dimension.
            tile_k: Dynamic tile size to use on K dimension.
            n_inner_size: Inner dimension size to use for the packed data
                layout.
        """
        return NDBuffer[b_type, 3, config.packed_shape](
            b_packed_ptr.address,
            DimList(tile_n // n_inner_size, tile_k, n_inner_size),
        )


@always_inline
fn pack_matmul_b_shape_func_M[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_in_0: Bool,
    single_thread_blocking_override: Bool,
](
    b_input: NDBuffer[b_type, 2, b_shape], kernel_type_m: Int = 0
) -> StaticIntTuple[2]:
    """Sets in shape_ref the shape required by `pack_b`'s `b_packed_ref`
    argument.

    If transpose_b is True, this returns the un-transposed shape, since pack_b
    will un-transpose `b_ref` as part of the packing layout transformation."""

    var output = StaticIntTuple[2]()

    var n = b_input.dim(0) if transpose_in_0 else b_input.dim(1)
    var k = b_input.dim(1) if transpose_in_0 else b_input.dim(0)
    var tile_n_k = StaticIntTuple[2]()

    if get_kernel_type(kernel_type_m, n, k):
        alias config = search_mm_config[a_type, b_type, c_type, True, True]()
        tile_n_k = _get_tile_n_k[
            config,
            transpose_in_0,
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
        ](b_input)
    else:
        alias config2 = search_mm_config[a_type, b_type, c_type, True, False]()
        tile_n_k = _get_tile_n_k[
            config2,
            transpose_in_0,
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
        ](b_input)

    @parameter
    if transpose_in_0:
        output[0] = b_input.dim(1)
        output[1] = b_input.dim(0)
    else:
        output[0] = b_input.dim(0)
        output[1] = b_input.dim(1)

    output[0] = ceildiv(output[0], tile_n_k[1]) * tile_n_k[1]
    output[1] = ceildiv(output[1], tile_n_k[0]) * tile_n_k[0]

    return output


@always_inline
fn pack_matmul_b_shape_func[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_in_0: Bool,
    single_thread_blocking_override: Bool,
](b_input: NDBuffer[b_type, 2, b_shape]) -> StaticIntTuple[2]:
    # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()

    return pack_matmul_b_shape_func_M[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transpose_in_0,
        single_thread_blocking_override,
    ](b_input, kernel_type_m)


fn pack_b[
    transpose_b: Bool,
    simd_size: Int,
    inner_size: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    src_shape: DimList,
    dst_shape: DimList,
](
    dst: NDBuffer[b_type, 2, dst_shape],
    src: NDBuffer[b_type, 2, src_shape],
    tile_n: Int,
    tile_k: Int,
):
    """Utility function to pack the entire B matrix, such that each
    [tile_n // inner_size, tile_k, inner_size] tile of src is contiguous in dst.

    Tiles (not tile contents) are stored in row major order, so tile[i, j] is
    tile_n * tile_k bytes away from tile[i, j+1].
    """
    dst.zero()  # zero the padding to be safe, shouldn't be necessary
    var dst_flat = dst.flatten()
    var dst_offset: Int = 0

    alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
    alias inner_size2 = inner_size // 2 if use_i8mm else inner_size

    @parameter
    if not transpose_b:
        var k_in = src.dim[0]()
        var n_in = src.dim[1]()
        var k_out = dst.dim[0]()
        var n_out = dst.dim[1]()

        debug_assert(
            k_out % tile_k == 0,
            "K dimension of output must be padded to tile_k",
        )
        debug_assert(
            n_out % tile_n == 0,
            "N dimension of output must be padded to tile_n",
        )
        for idx_k in range(0, k_out, tile_k):
            for idx_n in range(0, n_out, tile_n):
                var packed_dst_view = NDBuffer[b_type, 3](
                    dst_flat.data.offset(dst_offset),
                    DimList(
                        tile_n // inner_size2,
                        tile_k // factor,
                        inner_size2 * factor,
                    ),
                )
                var valid_k = min(tile_k, k_in - idx_k)
                var valid_n = min(tile_n, n_in - idx_n)
                PackMatrixCols[
                    src_shape,
                    DimList.create_unknown[3](),
                    b_type,
                    simd_size,
                    inner_size,
                    use_vnni,
                    use_i8mm,
                ].run(
                    packed_dst_view,
                    src,
                    # Input is [K, N]:
                    # Starting global offset for packing.
                    Index(idx_k, idx_n),
                    Index(tile_k, tile_n),
                    # Valid amount of input from the starting offset.
                    Index(valid_k, valid_n),
                )
                dst_offset += tile_n * tile_k
    else:
        # _t = transpose, annoying WAR since variables can't have same name in if/else
        var k_in_t = src.dim[1]()
        var n_in_t = src.dim[0]()
        var k_out_t = dst.dim[0]()
        var n_out_t = dst.dim[1]()

        debug_assert(
            k_out_t % tile_k == 0,
            "K dimension of output must be padded to tile_k",
        )
        debug_assert(
            n_out_t % tile_n == 0,
            "N dimension of output must be padded to tile_n",
        )

        for idx_k_t in range(0, k_out_t, tile_k):
            for idx_n_t in range(0, n_out_t, tile_n):
                var packed_dst_view_t = NDBuffer[b_type, 3](
                    dst_flat.data.offset(dst_offset),
                    DimList(tile_n // inner_size, tile_k, inner_size),
                )
                var valid_k_t = min(tile_k, k_in_t - idx_k_t)
                var valid_n_t = min(tile_n, n_in_t - idx_n_t)
                PackMatrixRows[
                    src_shape,
                    DimList.create_unknown[3](),
                    b_type,
                    simd_size,
                    inner_size,
                ].run(
                    packed_dst_view_t,
                    src,
                    # Input is [N, K]:
                    # Starting global offset for packing.
                    Index(idx_n_t, idx_k_t),
                    Index(tile_n, tile_k),
                    # Valid amount of input from the starting offset.
                    Index(valid_n_t, valid_k_t),
                )
                dst_offset += tile_n * tile_k


@always_inline
fn _pack_b_ndbuffer_impl[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transposed: Bool,
](
    b_input: NDBuffer[b_type, 2, b_shape],
    output_buffer: NDBuffer[b_type, 2],
    kernel_type_m: Int,
):
    """Performs the layout transformation on `b_input` expected by
    `matmul_dynamic_tile` when `b_packed` is True and stores the result in
    `output_buffer`.

    When transpose_b is True, this also un-transposes b_input as part of the layout
    transformation."""

    # Matrix by vector pattern -> use gemv
    if b_input.dim(1) == 1:
        # For gemv no packing is necessary
        memcpy(output_buffer.data, b_input.data, b_input.dim(0))

    else:
        var n = b_input.dim(0) if transposed else b_input.dim(1)
        var k = b_input.dim(1) if transposed else b_input.dim(0)

        # The config (in particular inner size and tile_k) needs to EXACTLY match the
        # values used in the matmul algorithm consuming this packed b matrix

        if get_kernel_type(kernel_type_m, n, k):
            alias config = search_mm_config[
                a_type, b_type, c_type, True, True
            ]()
            var tile_n_k = _get_tile_n_k[
                config,
                transposed,
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
            ](b_input)
            pack_b[
                transposed,
                config.simd_size,
                config.pack_inner_size,
                a_type,
                b_type,
                c_type,
                src_shape=b_shape,
                dst_shape = DimList.create_unknown[2](),
            ](output_buffer, b_input, tile_n_k[0], tile_n_k[1])
        else:
            alias config2 = search_mm_config[
                a_type, b_type, c_type, True, False
            ]()
            var tile_n_k = _get_tile_n_k[
                config2,
                transposed,
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
            ](b_input)
            pack_b[
                transposed,
                config2.simd_size,
                config2.pack_inner_size,
                a_type,
                b_type,
                c_type,
                src_shape=b_shape,
                dst_shape = DimList.create_unknown[2](),
            ](output_buffer, b_input, tile_n_k[0], tile_n_k[1])


@always_inline
fn pack_b_ndbuffer_M[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
](
    b_input: NDBuffer[b_type, 2, b_shape],
    output_buffer: NDBuffer[b_type, 2],
    kernel_type_m: Int,
):
    """
    Perform matmul weight packing on the given input.

    Performs the layout transformation on `b_input` expected by
    `matmul_dynamic_tile` when `b_packed` is True and stores the result in
    `output_buffer`.

    Parameters:
        a_type: The data type of elements inside a.
        a_shape: The shape of the A matrix.
        b_type: The data type of elements inside b.
        b_shape: The shape of the B matrix.
        c_type: The data type of elements inside c.
        c_shape: The shape of the C matrix.

    Args:
        b_input: Input buffer that contains the weight to be packed.
        output_buffer: Output buffer to store the packed weight.
        kernel_type_m: The M value of the a_shape (MxN).
    """
    _pack_b_ndbuffer_impl[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transposed=False,
    ](b_input, output_buffer, kernel_type_m)


fn pack_b_ndbuffer[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
](b_input: NDBuffer[b_type, 2, b_shape], output_buffer: NDBuffer[b_type, 2],):
    # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()
    return pack_b_ndbuffer_M[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
    ](b_input, output_buffer, kernel_type_m)


@always_inline
fn pack_transposed_b_ndbuffer_M[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
](
    b_input: NDBuffer[b_type, 2, b_shape],
    output_buffer: NDBuffer[b_type, 2],
    kernel_type_m: Int,
):
    """
    Perform matmul weight packing on a transposed input.

    Performs the layout transformation on `b_input` expected by
    `matmul_dynamic_tile` when `b_packed` is True and stores the result in
    `output_buffer`. This also un-transposes `b_input`.

    Parameters:
        a_type: The data type of elements inside a.
        a_shape: The shape of the A matrix.
        b_type: The data type of elements inside b.
        b_shape: The shape of the B matrix.
        c_type: The data type of elements inside c.
        c_shape: The shape of the C matrix.

    Args:
        b_input: Input buffer that contains the transposed weight to be packed.
        output_buffer: Output buffer to store the packed weight.
        kernel_type_m: The M value of the a_shape (MxN).
    """
    _pack_b_ndbuffer_impl[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transposed=True,
    ](b_input, output_buffer, kernel_type_m)


fn pack_transposed_b_ndbuffer[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
](b_input: NDBuffer[b_type, 2, b_shape], output_buffer: NDBuffer[b_type, 2],):
    # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()
    return pack_transposed_b_ndbuffer_M[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
    ](b_input, output_buffer, kernel_type_m)


@value
struct BTileGenerator[
    config: MatmulConfig,
    type: DType,
    transpose_b: Bool,
    b_packed: Bool,
]:
    """Struct to encapsulate a tile of B that supports prepacking.

    If b_packed is true, calls to get_tile will return a buffer view from B.
    Otherwise, calls to get_tile will copy a tile from B into a stack allocated
    scratch buffer and return a view of that."""

    var b: NDBuffer[
        type, 2, config.b_shape
    ]  # packed layout if b_packed is True
    var b_tile_stack_ptr: DTypePointer[type]
    var tile_n_k: StaticIntTuple[2]

    # needs to be always_inline so b_tile_stack_ptr gets allocated on caller's stack
    @always_inline
    @staticmethod
    fn get(
        b: NDBuffer[type, 2, config.b_shape], tile_n_k: StaticIntTuple[2]
    ) -> BTileGenerator[config, type, transpose_b, b_packed]:
        var b_tile_stack_ptr = DTypePointer[type].get_null()

        debug_assert(
            not (transpose_b and b_packed),
            "b cannot be both transposed and pre-packed.",
        )

        @parameter
        if not b_packed:
            b_tile_stack_ptr = stack_allocation[
                config.pack_data_size,
                type,
                alignof[SIMD[type, simdwidthof[type]()]](),
            ]()

        return BTileGenerator[config, type, transpose_b, b_packed](
            b, b_tile_stack_ptr, tile_n_k
        )

    fn get_tile[
        inner_size: Int
    ](
        self,
        global_offset: GemmShape,
        tile_dim_nk: StaticIntTuple[2],
        valid_data_dim_nk: StaticIntTuple[2],
    ) -> NDBuffer[type, 3, config.packed_shape]:
        """Get a packed matrix (B) tile.

        Args:
            global_offset: Offset in the global M, N, K dimensions.
            tile_dim_nk: Tile shape based on cache size and matrix dimensions.
            valid_data_dim_nk: The upper bounds for N and K dimensions.

        valid_data_tile_nk is ignored for pre-packing, where the tile is padded
        to have shape of tile_dim_nk.

        Returns:
            A view of the packed tile.

        """

        alias factor = get_matmul_arch_factor[
            config.use_vnni, config.use_i8mm
        ]()
        alias inner_size2 = inner_size // 2 if config.use_i8mm else inner_size

        var k = align_up(tile_dim_nk[1], factor)
        var tile_shape_nopack = DimList(
            tile_dim_nk[0] // inner_size2,
            k // factor,
            factor * inner_size2,
        )

        var packed_b = NDBuffer[type, 3, config.packed_shape](
            self.b_tile_stack_ptr, tile_shape_nopack
        )

        @parameter
        if transpose_b and not b_packed:
            PackMatrixRows[
                config.b_shape,
                config.packed_shape,
                type,
                config.simd_size,
                inner_size,
            ].run(
                packed_b,
                self.b,
                # Input is [N, K]:
                # Starting global offset for packing.
                Index(global_offset.N, global_offset.K),
                Index(tile_dim_nk[0], tile_dim_nk[1]),
                # Valid amount of input from the starting offset.
                Index(valid_data_dim_nk[0], valid_data_dim_nk[1]),
            )
            return packed_b
        elif (not transpose_b) and (not b_packed):
            PackMatrixCols[
                config.b_shape,
                config.packed_shape,
                type,
                config.simd_size,
                inner_size,
                config.use_vnni,
                config.use_i8mm,
            ].run(
                packed_b,
                self.b,
                # Input is [K, N]:
                # Starting global offset for packing.
                Index(global_offset.K, global_offset.N),
                Index(tile_dim_nk[1], tile_dim_nk[0]),
                # Valid amount of input from the starting offset.
                Index(valid_data_dim_nk[1], valid_data_dim_nk[0]),
            )
        elif b_packed and not transpose_b:
            # Need to use tile_k that generator was initialized with.
            # When packing is done online, tile_dim_nk can vary in each call to
            # get_tile (if handling a residual K tile), but packing assumes that
            # tile_k is constant.

            var factor = get_matmul_arch_factor[
                config.use_vnni, config.use_i8mm
            ]()
            alias inner_size2 = inner_size // 2 if config.use_i8mm else inner_size

            var tile_k = align_up(self.tile_n_k[1], factor)

            var tile_shape_pack = DimList(
                self.tile_n_k[0] // inner_size2,
                tile_k // factor,
                inner_size2 * factor,
            )
            var tile_k_idx = global_offset.K // tile_k
            var b_flat = self.b.flatten()
            var n_padded = self.b.dim[1]()
            var b_tile_view = NDBuffer[type, 3, config.packed_shape](
                # tiles are ordered in row-major order
                # a bit of trickieness going on here, this works because:
                #   1. tile_k is the same for every thread (tile_n is not) since threads
                #       don't currently partition on the K dimension
                #   2. the n dimension of each thread's tile is gauranteed to be an
                #       exact multiple of the inner size
                #   3. each tile has dims [tile_n/inner, tile_k, inner]
                b_flat.data.offset(
                    tile_k_idx * tile_k * n_padded + global_offset.N * tile_k
                ),
                tile_shape_pack,
            )
            return b_tile_view

        else:
            debug_assert(
                False, "unreachable, b_packed not supported with transpose_b"
            )

        return packed_b


@always_inline
fn _small_matmul[
    type: DType,
    a_shape: DimList,
    b_shape: DimList,
    c_shape: DimList,
    transpose_b: Bool,
    epilogue_wrapper: Optional[elementwise_epilogue_type],
](
    a: NDBuffer[type, 2, a_shape],
    b: NDBuffer[type, 2, b_shape],
    c: NDBuffer[type, 2, c_shape],
):
    alias simd_width = simdwidthof[type]()

    var M = a.dim[0]()
    var N = b.dim[0]() if transpose_b else b.dim[1]()
    var K = a.dim[1]()

    @parameter
    if transpose_b:
        for m in range(M):
            for n in range(N):
                var acc_vector = SIMD[type, simd_width]()
                var acc_scalar = Scalar[type]()

                @always_inline
                @parameter
                fn compute_fn[width: Int](k: Int):
                    @parameter
                    if width == 1:
                        acc_scalar += a[m, k] * b[n, k]
                    else:
                        acc_vector += a.load[width=simd_width](m, k) * b.load[
                            width=simd_width
                        ](n, k)

                vectorize[compute_fn, simd_width, unroll_factor=2](K)

                var val = acc_vector.reduce_add() + acc_scalar

                @parameter
                if epilogue_wrapper:
                    alias func = epilogue_wrapper.value()
                    func[type, 1](Index(m, n), val)
                else:
                    c[Index(m, n)] = val
    else:

        @parameter
        @always_inline
        fn normal_update[
            inner_type: DType, width: Int
        ](coords: StaticIntTuple[2], val: SIMD[inner_type, width]):
            c.store[width=width](
                Index(coords[0], coords[1]), rebind[SIMD[type, width]](val)
            )

        @parameter
        @always_inline
        fn last_update[
            _type: DType, width: Int
        ](coords: StaticIntTuple[2], val: SIMD[_type, width]):
            @parameter
            if epilogue_wrapper:
                alias func = epilogue_wrapper.value()
                func[_type, width](coords, val)
            else:
                c.store[width=width](coords, rebind[SIMD[type, width]](val))

        @always_inline
        @__copy_capture(N)
        @parameter
        fn accum_out_row[
            output_func: fn[type: DType, width: Int] (
                StaticIntTuple[2], SIMD[type, width]
            ) capturing -> None,
        ](m: Int, k: Int):
            var a_val = a[m, k]

            @always_inline
            @__copy_capture(a_val)
            @parameter
            fn _wrapper[simd_width: Int](n: Int):
                output_func[type, simd_width](
                    Index(m, n),
                    c.load[width=simd_width](m, n)
                    + a_val * b.load[width=simd_width](k, n),
                )

            vectorize[_wrapper, simd_width, unroll_factor=2](N)

        for m in range(M):
            memset_zero(c.data + m * N, N)
            for k in range(K - 1):
                accum_out_row[normal_update](m, k)
            accum_out_row[last_update](m, K - 1)


@always_inline
fn __nvvm_ldg_f4[type: DType](x: DTypePointer[type]) -> SIMD[type, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[type, 4]]())

    @parameter
    if type == DType.float32:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[DType.float32](), alignment)
        )
    elif type == DType.bfloat16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4bf16.p0v4bf16",
                SIMD[DType.bfloat16, 4],
            ](x.bitcast[DType.bfloat16](), alignment)
        )
    else:
        constrained[False, "Unhandled DType"]()
        return 0


# BM: The threadblock size for M dimension SMEM caching.
# BN: The threadblock size for N dimension SMEM caching.
# BK: The threadblock size for K dimension SMEM caching.
# WM: M dim of continuous tile computed by each warp.
# WN: N dim of continuous tile computed by each warp.
# WMITER: The number of subwarp tiling steps in M dimension.
# WNITER: The number of subwarp tiling steps in N dimension.
# TM: The per-thread tile size for M dimension.
# TN: The per-thread tile size for N dimension.
@__llvm_metadata(
    `nvvm.maxntid`=StaticTuple[Int32, 1](NUM_THREADS.cast[DType.int32]())
)
fn sgemm_warp_tiling_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    indexing_integral_dtype: DType,
    BM: Scalar[indexing_integral_dtype],
    BN: Scalar[indexing_integral_dtype],
    BK: Scalar[indexing_integral_dtype],
    WM: Scalar[indexing_integral_dtype],
    WN: Scalar[indexing_integral_dtype],
    WMITER: Scalar[indexing_integral_dtype],
    WNITER: Scalar[indexing_integral_dtype],
    TM: Scalar[indexing_integral_dtype],
    TN: Scalar[indexing_integral_dtype],
    NUM_THREADS: Scalar[indexing_integral_dtype],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    mat_c: NDBuffer[c_type, 2, c_shape],
    mat_a: NDBuffer[a_type, 2, a_shape],
    mat_b: NDBuffer[b_type, 2, b_shape],
    alpha: Scalar[c_type],
    beta: Scalar[c_type],
):
    var M: Scalar[indexing_integral_dtype] = mat_c.dim(0)
    var K: Scalar[indexing_integral_dtype] = mat_a.dim(1)
    var N: Scalar[indexing_integral_dtype] = mat_c.dim(1)

    var c_row: Scalar[indexing_integral_dtype] = BlockIdx.y()
    var c_col: Scalar[indexing_integral_dtype] = BlockIdx.x()

    # Placement of the warp in the threadblock tile.
    var warp_idx = Scalar[indexing_integral_dtype](
        ThreadIdx.x()
    ) // WARP_SIZE  # the warp this thread is in
    var warp_col = warp_idx % (BN // WN)
    var warp_row = warp_idx // (BN // WN)

    # Size of the warp subtile.
    alias w_sub_m = WM // WMITER  # 64/2=32
    alias w_sub_n = WN // WNITER  # 32/2=16

    # Placement of the thread in the warp subtile.
    var thread_Idx_In_warp = Scalar[indexing_integral_dtype](
        ThreadIdx.x()
    ) % WARP_SIZE  # [0, 31]
    var thread_col_in_warp = thread_Idx_In_warp % (w_sub_n // TN)  # i%(16/4)
    var thread_row_in_warp = thread_Idx_In_warp // (w_sub_n // TN)  # i/4

    # Allocate space for the current blocktile in SMEM.
    # Pad the A tile in share memory to avoid bank conflicts.
    # Use 4 to comply with f4 alignment used in accumulation.
    alias sram_bank_padding_size = 4
    alias BM_padded = BM + sram_bank_padding_size
    var a_sram = NDBuffer[
        a_type,
        1,
        DimList(int(BK * BM_padded)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var b_sram = NDBuffer[
        b_type,
        1,
        DimList(int(BK * BN)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Move blocktile to beginning of A's row and B's column.
    var aa_ptr = mat_a._offset(Index(c_row * BM, 0))
    var bb_ptr = mat_b._offset(Index(0, c_col * BN))
    # Move C_ptr to warp's output tile
    var cc_ptr = mat_c._offset(
        Index(c_row * BM + warp_row * WM, c_col * BN + warp_col * WN)
    )

    # Calculate the indices that this thread will load into SMEM.
    # We load 128bit / 32bit = 4 elements per thread at each step.
    var inner_row_a = Scalar[indexing_integral_dtype](ThreadIdx.x()) // (
        BK // 4
    )
    var inner_col_a = Scalar[indexing_integral_dtype](ThreadIdx.x()) % (BK // 4)
    alias row_stride_a = (NUM_THREADS * 4) // BK
    var inner_row_b = Scalar[indexing_integral_dtype](ThreadIdx.x()) // (
        BN // 4
    )
    var inner_co_ib = Scalar[indexing_integral_dtype](ThreadIdx.x()) % (BN // 4)
    alias row_stride_b = NUM_THREADS // (BN // 4)

    # TODO: We want these to be register-allocated!
    # Allocate thread-local cache for results in register file.
    var thread_results = NDBuffer[
        c_type,
        4,
        DimList(int(WMITER), int(WNITER), int(TM), int(TN)),
    ]().stack_allocation()
    thread_results.zero()

    # We cache into registers on the warptile level.
    var reg_m = NDBuffer[
        a_type, 2, DimList(int(WMITER), int(TM))
    ]().stack_allocation()
    reg_m.zero()

    var reg_n = NDBuffer[
        b_type, 2, DimList(int(WNITER), int(TN))
    ]().stack_allocation()
    reg_n.zero()

    # Outer-most loop over block tiles.
    for bk_idx in range(0, int(K), int(BK)):
        for offset in range(0, int(BM - row_stride_a + 1), int(row_stride_a)):
            # Load 4 elements at a time and store to shared memory.
            var tmp = __nvvm_ldg_f4[a_type](
                aa_ptr.offset(int((inner_row_a + offset) * K + inner_col_a * 4))
            )

            @unroll
            for i in range(4):
                a_sram[
                    int(
                        (inner_col_a * 4 + i) * BM_padded + inner_row_a + offset
                    )
                ] = tmp[i]

        for offset in range(0, int(BK - row_stride_b + 1), int(row_stride_b)):
            # Load 4 elements at a time and store to shared memory.
            var tmp = __nvvm_ldg_f4[b_type](
                bb_ptr.offset(int((inner_row_b + offset) * N + inner_co_ib * 4))
            )
            b_sram.store[width=4, alignment=16](
                Index((inner_row_b + offset) * BN + inner_co_ib * 4),
                tmp,
            )

        barrier()

        for dot_idx in range(BK):
            # Populate registers for whole warptile.
            @unroll
            for w_sub_row_idx in range(WMITER):

                @unroll
                for i in range(0, int(TM), 4):
                    var vec = a_sram.load[width=4, alignment=16](
                        int(
                            (dot_idx * BM_padded)
                            + warp_row * WM
                            + w_sub_row_idx * w_sub_m
                            + thread_row_in_warp * TM
                            + i
                        )
                    )
                    reg_m.store(Index(w_sub_row_idx, i), vec)

            @unroll
            for w_sub_col_idx in range(WNITER):

                @unroll
                for i in range(0, int(TN), 4):
                    var vec = b_sram.load[width=4, alignment=16](
                        int(
                            (dot_idx * BN)
                            + warp_col * WN
                            + w_sub_col_idx * w_sub_n
                            + thread_col_in_warp * TN
                        )
                    )
                    reg_n.store(Index(w_sub_col_idx, i), vec)

            # Execute warptile matmul.
            @unroll
            for w_sub_row_idx in range(WMITER):

                @unroll
                for w_sub_col_idx in range(WNITER):
                    # Calculate per-thread results.
                    @unroll
                    for res_idx_m in range(TM):

                        @unroll
                        for res_idx_n in range(TN):
                            thread_results[
                                Index(
                                    w_sub_row_idx,
                                    w_sub_col_idx,
                                    res_idx_m,
                                    res_idx_n,
                                )
                            ] += (
                                reg_m[w_sub_row_idx, res_idx_m].cast[c_type]()
                                * reg_n[w_sub_col_idx, res_idx_n].cast[c_type]()
                            )
        aa_ptr = aa_ptr.offset(int(BK))  # move BK columns to right
        bb_ptr = bb_ptr.offset(int(BK * N))  # move BK rows down
        barrier()

    # Write out the results.
    @unroll
    for w_sub_row_idx in range(WMITER):

        @unroll
        for w_sub_col_idx in range(WNITER):
            # Move C pointer to current warp subtile.
            var C_interim = cc_ptr.offset(
                int((w_sub_row_idx * w_sub_m) * N + w_sub_col_idx * w_sub_n)
            )

            @unroll
            for res_idx_m in range(TM):

                @unroll
                for res_idx_n in range(0, int(TN), 4):
                    var c_idx = (
                        thread_row_in_warp * TM + res_idx_m
                    ) * N + thread_col_in_warp * TN + res_idx_n
                    var result_vec = thread_results.load[width=4](
                        Index(
                            w_sub_row_idx,
                            w_sub_col_idx,
                            res_idx_m,
                            res_idx_n,
                        )
                    )

                    var vec = alpha * result_vec + beta * C_interim.load[
                        width=4, alignment=16
                    ](int(c_idx))

                    @parameter
                    if elementwise_lambda_fn:
                        alias elementwise_lambda = elementwise_lambda_fn.value()
                        elementwise_lambda[c_type, 4](
                            Index(
                                int(
                                    thread_row_in_warp * TM
                                    + res_idx_m
                                    + w_sub_row_idx * w_sub_m
                                ),
                                int(
                                    thread_col_in_warp * TN
                                    + res_idx_n
                                    + w_sub_col_idx * w_sub_n
                                ),
                            ),
                            vec,
                        )
                    else:
                        C_interim.store[width=4, alignment=16](int(c_idx), vec)


# Matrix-Column Vector Multiplication
fn gemv_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: DTypePointer[c_type],
    a: DTypePointer[a_type],
    b: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var warpId = x // WARP_SIZE
    var accum = SIMD[c_type, 1]()

    if warpId < m:
        # Every warp processes a single row of the resultant vector
        for i in range(ceildiv(k, WARP_SIZE)):
            var idx = i * WARP_SIZE + int(lane_id())
            var val = SIMD[c_type, 1]()
            if idx < k:
                val = (
                    a.load(warpId * k + idx).cast[c_type]()
                    * b.load(idx).cast[c_type]()
                )

            @parameter
            fn reduce_add[
                type: DType,
                width: Int,
            ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
                return x + y

            val = warp_reduce[shuffle_down, reduce_add](val)
            if lane_id() == 0:
                accum += val

        if lane_id() == 0:

            @parameter
            if elementwise_lambda_fn:
                alias elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda[c_type, 1](Index(warpId, 0), accum)
            else:
                c[warpId] = accum


# Row Vector-Matrix multiplication
fn gevm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: DTypePointer[c_type],
    a: DTypePointer[a_type],
    b: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var warpsPerBlock = BlockDim.x() // WARP_SIZE
    var warpId = ThreadIdx.x() // WARP_SIZE
    var accum = SIMD[c_type, 1]()
    var col = BlockIdx.x() * WARP_SIZE + int(lane_id())
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var globalWarpId = x // WARP_SIZE

    var x_shared = stack_allocation[
        tile_size,
        c_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Every block computes warp size length of output values
    for i in range(ceildiv(k, warpsPerBlock)):
        var val = SIMD[c_type, 1]()
        var row = i * warpsPerBlock + warpId
        if lane_id() == 0:
            val = a.load(row).cast[c_type]()
        val = shuffle_idx(val, 0)
        accum += val * b.load(row * n + col).cast[c_type]()

    x_shared[int(lane_id()) * WARP_SIZE + warpId] = accum
    barrier()

    @parameter
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    var total = SIMD[c_type, 1]()
    total = x_shared.load(ThreadIdx.x()).cast[c_type]()
    total = warp_reduce[shuffle_down, reduce_add](total)

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](Index(globalWarpId, 0), total)
        else:
            c[globalWarpId] = total


fn matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_ptr: DTypePointer[c_type],
    a_ptr: DTypePointer[a_type],
    b_ptr: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    """Matrix Multiplication using shared memory.
    This version loads blocks of size tile_size x tile_size from A and B
    and updates a tile_size x tile_size in C.

    The thread block should have shape (tile_size, tile_size, 1). Each
    thread is mapped one element in C. The grid should have shape
    (N/tile_size, M/tile_size, 1). N is the first dimension for coalesced
    access.
    """

    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    # Allocate A, B tile in shared memory.
    var a_shared = stack_allocation[
        tile_size * tile_size,
        a_type,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        b_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Global index in C.
    # These are the same indices in A and B when loading to SRAM.
    # Map thread x to column for coalesced access in B.
    var col = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var row = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    # Local index in the c sub-matrix updated by current block.
    var localCol = ThreadIdx.x()
    var localRow = ThreadIdx.y()

    # Result of current thread in C.
    var result = SIMD[c_type, 1](0.0)

    var K_roundbytile = align_down(k, tile_size)
    # Can't use 0 as tile size so set to 1 when the remainder is 0.
    var K_remainder = k - K_roundbytile if k - K_roundbytile > 0 else 1

    @parameter
    @__copy_capture(row, localCol, a, b, localRow, col, a_shared, b_shared)
    @always_inline
    fn update_tile[full_tile: Bool](offset: Int, end: Int, tile_size: Int):
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        var a_val: SIMD[a_type, 1]

        @parameter
        if not full_tile:
            a_val = a[row, offset + localCol] if (
                row < m and offset + localCol < k
            ) else 0.0
        else:
            a_val = a[row, offset + localCol] if row < m else 0.0
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: SIMD[b_type, 1]

        @parameter
        if not full_tile:
            b_val = b[offset + localRow, col] if (
                col < n and offset + localRow < k
            ) else 0.0
        else:
            b_val = b[offset + localRow, col] if col < n else 0.0
        b_shared[localRow * tile_size + localCol] = b_val

        barrier()

        for kk in range(tile_size):
            result += (
                a_shared[localRow * tile_size + kk].cast[c_type]()
                * b_shared[kk * tile_size + localCol].cast[c_type]()
            )

        barrier()

    tile_and_unswitch[update_tile](
        0, k, VariadicList[Int](tile_size, K_remainder)
    )

    if row < m and col < n:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](Index(row, col), result)
        else:
            c[Index(row, col)] = result


fn matmul_kernel_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_DIM: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_ptr: DTypePointer[c_type],
    a_ptr: DTypePointer[a_type],
    b_ptr: DTypePointer[b_type],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    var accum = SIMD[c_type, 1]()
    for i in range(k):
        accum = a[x, i].cast[c_type]() * b[i, y].cast[c_type]() + accum

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum)
    else:
        c[Index(x, y)] = accum


@always_inline
fn _matmul_gpu[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    saturated_vnni: Bool,
    single_thread_blocking_override: Bool = False,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    # HACK HACK HACK https://github.com/modularml/modular/issues/22959
    # single_thread_blocking_override should not be allowed, but the graph
    # compiler has a special case that does not insert the
    # on the GPU
    # constrained[
    #     not single_thread_blocking_override,
    #     "single_thread_blocking_override not applicable",
    # ]()
    constrained[transpose_a == False, "only NN matmul is supported"]()
    constrained[transpose_b == False, "only NN matmul is supported"]()
    constrained[not b_packed, "pre-packing not yet supported"]()
    constrained[not saturated_vnni, "saturated_vnni_flag not applicable"]()
    constrained[
        a_type == DType.float32 or a_type == DType.bfloat16,
        "only Float32/BFloat16 types are supported",
    ]()
    constrained[
        b_type == DType.float32 or b_type == DType.bfloat16,
        "only Float32/BFloat16 types are supported",
    ]()
    constrained[
        c_type == DType.float32 or c_type == DType.bfloat16,
        "only Float32/BFloat16 types are supported",
    ]()

    var shape = GemmShape.get[False, False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    # TODO: #25898, use max_finite
    alias max_uint32 = Int(0xFFFFFFFF)
    var use_32bit_indexing = m * n < max_uint32 and m * k < max_uint32 and n * k < max_uint32

    @parameter
    if elementwise_lambda_fn:
        if use_32bit_indexing:
            _matmul_gpu_dispatch[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
                indexing_integral_dtype = DType.uint32,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b)
        else:
            _matmul_gpu_dispatch[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
                indexing_integral_dtype = DType.uint64,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b)

    else:
        if use_32bit_indexing:
            _matmul_gpu_dispatch[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
                indexing_integral_dtype = DType.uint32,
            ](c, a, b)
        else:
            _matmul_gpu_dispatch[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
                indexing_integral_dtype = DType.uint64,
            ](c, a, b)


@always_inline
fn _matmul_gpu_dispatch[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    indexing_integral_dtype: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    var shape = GemmShape.get[False, False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    try:
        var stream = Stream.get_current_stream()

        # TODO implement optimized matmul for half types #33364
        @parameter
        if (
            a_type == DType.bfloat16
            or b_type == DType.bfloat16
            or c_type == DType.bfloat16
        ):
            alias BLOCK_DIM = 16
            var gpu_func = Function[
                fn (
                    DTypePointer[a_type],
                    DTypePointer[b_type],
                    DTypePointer[c_type],
                    Int,
                    Int,
                    Int,
                ) capturing -> None, matmul_kernel_naive[
                    a_type,
                    b_type,
                    c_type,
                    BLOCK_DIM,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
            ]()
            gpu_func(
                c.data,
                a.data,
                b.data,
                m,
                n,
                k,
                grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
                stream=stream,
            )
            return

        constrained[
            a_type == DType.float32,
            "Only Float32 types have optimized implementations",
        ]()
        constrained[
            b_type == DType.float32,
            "Only Float32 types have optimized implementations",
        ]()
        constrained[
            c_type == DType.float32,
            "Only Float32 types have optimized implementations",
        ]()

        # Currently sgemm_warp_tiling_kernel is supportred only for float32 and
        # no elementwise_epilogue, fallback to generic matmul_kernel.
        var warp_tiled_matmul_suppoered_shape = (
            m % 128 == 0 and n % 128 == 0 and k % 128 == 0
        )
        var warp_tiled_matmul_supported_format = (
            a_type == DType.float32
            and b_type == DType.float32
            and c_type == DType.float32
        )
        if (
            warp_tiled_matmul_suppoered_shape
            and warp_tiled_matmul_supported_format
        ):
            # TODO: Auto tune these for A100.
            # TODO: NUM_THREADS need to vary as M, N varies.
            alias NUM_THREADS = 128
            alias BN = 128
            alias BM = 128
            alias BK = 16
            alias WN = 64
            alias WM = 64
            alias WNITER = 4
            alias TN = 4
            alias TM = 8
            alias WMITER = (WM * WN) // (WARP_SIZE * TM * TN * WNITER)
            alias NUM_WARPS = NUM_THREADS / WARP_SIZE
            alias mm = sgemm_warp_tiling_kernel[
                c_type,
                c_shape,
                a_type,
                a_shape,
                b_type,
                b_shape,
                indexing_integral_dtype=indexing_integral_dtype,
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                WMITER=WMITER,
                WNITER=WNITER,
                TM=TM,
                TN=TN,
                NUM_THREADS=NUM_THREADS,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
            var gpu_func = Function[__type_of(mm), mm](
                threads_per_block=NUM_THREADS
            )
            gpu_func(
                c,
                a,
                b,
                1,
                0,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(NUM_THREADS),
                stream=stream,
            )
        elif n == 1:
            alias WARPS_PER_BLOCK = 32
            var gpu_func = Function[
                fn (
                    DTypePointer[c_type],
                    DTypePointer[a_type],
                    DTypePointer[b_type],
                    Int,
                    Int,
                    Int,
                ) capturing -> None, gemv_kernel[
                    c_type,
                    a_type,
                    b_type,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
            ]()
            gpu_func(
                c.data,
                a.data,
                b.data,
                m,
                n,
                k,
                grid_dim=ceildiv(m, WARPS_PER_BLOCK),
                block_dim=WARP_SIZE * WARPS_PER_BLOCK,
                stream=stream,
            )
        elif m == 1 and n % WARP_SIZE == 0 and k % 32 == 0:
            # k should be a multiple of warps per block
            alias WARPS_PER_BLOCK = 32
            var gpu_func = Function[
                fn (
                    DTypePointer[c_type],
                    DTypePointer[a_type],
                    DTypePointer[b_type],
                    Int,
                    Int,
                    Int,
                ) capturing -> None, gevm_kernel[
                    c_type,
                    a_type,
                    b_type,
                    WARP_SIZE * WARPS_PER_BLOCK,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
            ]()
            gpu_func(
                c.data,
                a.data,
                b.data,
                m,
                n,
                k,
                grid_dim=ceildiv(n, WARPS_PER_BLOCK),
                block_dim=WARP_SIZE * WARPS_PER_BLOCK,
                stream=stream,
            )
        else:
            # Tile size for tiling in shared memory.
            # Thread block would have shape (tile_size, tile_size, 1)
            # If k < tile_size use naive version.
            alias tile_size = 16
            if k >= tile_size:
                var gpu_func = Function[
                    fn (
                        DTypePointer[c_type],
                        DTypePointer[a_type],
                        DTypePointer[b_type],
                        Int,
                        Int,
                        Int,
                    ) capturing -> None, matmul_kernel[
                        c_type,
                        a_type,
                        b_type,
                        tile_size,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ]
                ]()
                gpu_func(
                    c.data,
                    a.data,
                    b.data,
                    m,
                    n,
                    k,
                    grid_dim=(ceildiv(n, tile_size), ceildiv(m, tile_size)),
                    block_dim=(tile_size, tile_size),
                    stream=stream,
                )
            else:
                alias BLOCK_DIM = 16
                var gpu_func = Function[
                    fn (
                        DTypePointer[a_type],
                        DTypePointer[b_type],
                        DTypePointer[c_type],
                        Int,
                        Int,
                        Int,
                    ) capturing -> None, matmul_kernel_naive[
                        a_type,
                        b_type,
                        c_type,
                        BLOCK_DIM,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ]
                ]()
                gpu_func(
                    c.data,
                    a.data,
                    b.data,
                    m,
                    n,
                    k,
                    grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_DIM),
                    stream=stream,
                )
    except e:
        abort(e)


@always_inline
fn _matmul_cpu[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    saturated_vnni: Bool,
    single_thread_blocking_override: Bool = False,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    @parameter
    if (
        single_thread_blocking_override
        and not transpose_a
        and not b_packed
        and a_type == b_type
        and b_type == c_type
    ):
        return _small_matmul[
            a_type,
            a_shape,
            b_shape,
            c_shape,
            transpose_b,
            elementwise_lambda_fn,
        ](
            a,
            rebind[NDBuffer[a_type, 2, b_shape]](b),
            rebind[NDBuffer[a_type, 2, c_shape]](c),
        )
    constrained[not transpose_a, "transpose_a not yet supported"]()

    var shape = GemmShape.get[False, transpose_b](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    # Matrix by vector pattern -> use gemv
    if n == 1:
        var out = Buffer[c_type](c.data, c.dim[0]())
        var lhs = rebind[NDBuffer[a_type, 2, a_shape]](a)
        var rhs = Buffer[b_type](b.data, b.dim[0]())
        gemv[
            parallelize=True,
            c_size = Dim(),
            c_type=c_type,
            a_shape=a_shape,
            a_type=a_type,
            b_size = Dim(),
            b_type=b_type,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](out, lhs, rhs)
    else:
        var complexity = m * n * k
        var num_tasks = min(
            ceildiv(complexity, get_min_task_size()),
            num_threads if num_threads > 0 else Runtime().parallelism_level(),
        )

        alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
        alias simd_size = simdwidthof[c_type]()
        alias alignment = alignof[SIMD[c_type, simd_size]]()
        var kh = align_up(k, 8)
        var mh = align_up(m, 2)
        var a_packed_ptr = DTypePointer[a_type]()
        if use_i8mm:
            a_packed_ptr = DTypePointer[a_type].alloc(
                mh * kh, alignment=alignment
            )
        var a_packed = NDBuffer[a_type, 2, a_shape](a_packed_ptr, (mh, kh))

        @always_inline
        @__copy_capture(m, k, num_tasks)
        @parameter
        fn pack_task_func(task_id: Int):
            var sub_matmul_config = get_partitioned_matmul[
                a_type, b_type, c_type, PartitionHeuristic.MOJO
            ](m, 1, k, task_id, num_tasks, kernel_type_m)
            var t0 = sub_matmul_config.offset[0]
            var t1 = t0 + sub_matmul_config.shape[0]
            packA_i8mm[a_type](t0, t1, k, a.data, a_packed_ptr)

        @always_inline
        @__copy_capture(m, k, num_tasks, n, a_packed)
        @parameter
        fn task_func(task_id: Int):
            var sub_matmul_config = get_partitioned_matmul[
                a_type, b_type, c_type, PartitionHeuristic.MOJO
            ](m, n, k, task_id, num_tasks, kernel_type_m)

            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            _submatmul_sequential_sync[
                a_type,
                a_shape,
                b_type,
                b_shape,
                c_type,
                c_shape,
                transpose_a,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                saturated_vnni,
            ](
                c,
                a_packed if use_i8mm else a,
                b,
                sub_matmul_config.shape,
                sub_matmul_config.offset,
                kernel_type_m,
            )

        # i8mm partition needs to be optimized as a function of m, n and k
        # Also parallelize currently is slower than asyn_parallelize which is depreciated now.
        # See issue 27734
        if use_i8mm:
            sync_parallelize[pack_task_func](num_tasks)

        # TODO (#12624): Closure captures some state on the stack so this needs
        # to be synchronous in order to keep that state alive
        sync_parallelize[task_func](num_tasks)
        a_packed_ptr.free()


@always_inline
fn matmul_M[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    constrained[target == "cpu" or target == "cuda", "unsupported target"]()
    alias func = _matmul_cpu if target == "cpu" else _matmul_gpu

    func[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transpose_a,
        transpose_b,
        b_packed,
        elementwise_lambda_fn,
        saturated_vnni,
        single_thread_blocking_override,
    ](c, a, b, kernel_type_m, num_threads)


@always_inline
fn matmul[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    num_threads: Int = -1,
):
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()

    matmul_M[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transpose_a,
        transpose_b,
        b_packed,
        elementwise_lambda_fn,
        saturated_vnni,
        single_thread_blocking_override,
        target,
    ](c, a, b, kernel_type_m, num_threads)


fn _submatmul_sequential_sync[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    rowwise_epilogue_enabled: Bool,
    saturated_vnni: Bool,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
    rowwise_epilogue_fn: fn (Int, Int) escaping -> None,
    kernel_type_m: Int = 0,
):
    constrained[not transpose_a, "transpose_a not yet supported"]()

    @parameter
    fn dispatch_on_kernel_type[kernel_type: Bool]():
        alias mm_config = search_mm_config[
            a_type, b_type, c_type, b_packed, kernel_type, saturated_vnni
        ]()

        fn elementwise_closure(offset: GemmShape, shape: GemmShape):
            @parameter
            if elementwise_lambda_fn:
                elementwise_epilogue_c_tile[
                    mm_config.simd_size,
                    c_type,
                    mm_config.c_shape,
                    elementwise_lambda_fn.value(),
                ](
                    offset,
                    shape,
                    rebind[NDBuffer[c_type, 2, mm_config.c_shape]](c),
                )
            else:
                pass

        TiledMatmul[
            mm_config,
            a_type,
            b_type,
            c_type,
            # transpose_a
            False,
            transpose_b,
            b_packed,
            elementwise_lambda_fn.__bool__(),
            rowwise_epilogue_enabled,
        ].run(
            rebind[NDBuffer[c_type, 2, mm_config.c_shape]](c),
            rebind[NDBuffer[a_type, 2, mm_config.a_shape]](a),
            rebind[NDBuffer[b_type, 2, mm_config.b_shape]](b),
            elementwise_closure,
            rowwise_epilogue_fn,
            sub_matrix_shape,
            sub_matrix_offset,
        )

    var shape = GemmShape.get[False, transpose_b](c, a, b)
    var n = shape.N
    var k = shape.K
    dispatch_get_kernel_type[dispatch_on_kernel_type](kernel_type_m, n, k)


fn _submatmul_sequential_sync[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    saturated_vnni: Bool,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
    kernel_type_m: Int = 0,
):
    fn null_rowwise_epilogue(offset: Int, num_rows: Int):
        pass

    _submatmul_sequential_sync[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transpose_a,
        transpose_b,
        b_packed,
        elementwise_lambda_fn,
        False,
        saturated_vnni,
    ](
        c,
        a,
        b,
        sub_matrix_shape,
        sub_matrix_offset,
        null_rowwise_epilogue,
        kernel_type_m,
    )
