# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Activations import ActivationType
from Assert import assert_param, debug_assert
from DType import DType
from Buffer import (
    NDBuffer,
    Buffer,
    DynamicRankBuffer,
    partial_simd_load,
    partial_simd_store,
    _raw_stack_allocation,
)
from Functional import (
    tile,
    unswitch,
    unroll,
    unroll,
    vectorize,
    async_parallelize,
)
from Index import Index, StaticIntTuple
from List import Dim, DimList, VariadicList
from LLCL import OutputChainPtr
from Math import min, fma, div_ceil, align_down
from MatmulUtils import (
    get_packB_unroll_factor,
    MatmulConfig,
    SubMatmulConfig,
    MatmulDataType,
    MatmulOperandLayout,
    GemmShape,
    calculate_tile_n_k,
    get_min_task_size,
    get_partitioned_matmul,
    PartitionHeuristic,
    search_mm_config,
    elementwise_lambda_fn_sig_type,
    dispatch_is_critical_stride,
    is_critical_stride,
)
from Matrix import Matrix
from Pointer import DTypePointer
from Range import range
from SIMD import SIMD
from TargetInfo import has_neon, alignof, dtype_simd_width
from Transpose import transpose_inplace
from TypeUtilities import rebind
from Intrinsics import PrefetchOptions, external_call
from IO import print


@closure
fn null_elementwise_epilogue(offset: GemmShape, tile_len: GemmShape):
    pass


@closure
fn null_rowwise_epilogue(offset: Int, num_rows: Int):
    pass


fn elementwise_epilogue_c_tile[
    simd_width: Int,
    type: DType,
    shape_c: DimList,
    func: fn[type: DType, width: Int] (
        StaticIntTuple[2], SIMD[type, width]
    ) capturing -> None,
](offset: GemmShape, tile_len: GemmShape, c: NDBuffer[2, shape_c, type],):
    @always_inline
    @parameter
    fn activation_on_col_chunk[col_chunk_size: Int](idx_n: Int):
        let n_coord = idx_n + offset.N
        for idx_m in range(tile_len.M):
            let m_coord = idx_m + offset.M
            let c_coord = Index(m_coord, n_coord)
            let c_val = c.simd_load[col_chunk_size](c_coord)
            func[type, col_chunk_size](c_coord, c_val)

    vectorize[simd_width, activation_on_col_chunk](tile_len.N)


@always_inline
fn naive_matmul[
    shape_a: DimList,
    shape_b: DimList,
    shape_c: DimList,
    accum_type: DType,
    value_type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    epilogue_elemwise_func: fn[type: DType] (Int, Int, SIMD[type, 1]) -> SIMD[
        type, 1
    ],
    epilogue_rowise_func: fn[type: DType] (Int, Buffer[Dim(), type]) -> None,
](
    c: NDBuffer[2, shape_c, accum_type],
    a: NDBuffer[2, shape_a, value_type],
    b: NDBuffer[2, shape_b, value_type],
):
    """Computes matrix multiplication with a naive algorithm.

    Args:
        c: Buffer with allocated output space.
        a: Buffer containing matrix operand A.
        b: Buffer containing matrix operand B.
        transpose_a: indicates if a is transposed.
        transpose_b: indicates if b is transposed.
    """
    let gemm_shape = GemmShape.get[
        transpose_a,
        transpose_b,
    ](c, a, b)
    let matrix_a = Matrix[shape_a, value_type, transpose_a](a)
    let matrix_b = Matrix[shape_b, value_type, transpose_b](b)
    let matrix_c = Matrix[shape_c, accum_type, False](c)

    for m in range(gemm_shape.M):
        var n: Int = 0
        while n < gemm_shape.N:
            var c_val: SIMD[accum_type, 1] = 0
            for k in range(gemm_shape.K):
                let a_val = matrix_a[m, k].cast[accum_type]()
                let b_val = matrix_b[k, n].cast[accum_type]()
                c_val += a_val * b_val
            c_val = epilogue_elemwise_func[accum_type](m, n, c_val)
            matrix_c[m, n] = c_val
            n += 1
        let row = Buffer[Dim(), accum_type](c.data.offset(m * gemm_shape.N), n)
        epilogue_rowise_func[accum_type](m, row)


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
    extract inner vectors of rows into the packed inner dimension.
    e.g. extract tile [X, Y] and pack into [Xo][Y][Xi]
    """

    # packed matrix
    var packed_matrix: NDBuffer[3, packed_shape, type]
    # original matrix:
    var original_matrix: NDBuffer[2, original_shape, type]
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
        packed_matrix: NDBuffer[3, packed_shape, type],
        original_matrix: NDBuffer[2, original_shape, type],
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
        assert_param[row_inner_size % simd_size == 0]()

        let instance = Self(
            packed_matrix,
            original_matrix,
            global_offset,
            pack_tile_dim,
            valid_data_dim,
            Index(
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
            2,
            DimList(simd_size, simd_size),
            type,
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
        let read_bound = self.valid_data_dim - local_off_set
        # Boundaries for writeable space.
        let write_bound = self.pack_tile_dim - local_off_set

        # Global index the packing is starting from.
        let start_idx_global = local_off_set + self.global_offset

        # Fill the simd_size x simd_size transpose buffer
        #  with un-transposed data.
        @always_inline
        @parameter
        fn body[idx: Int]():
            alias inner_row_idx = idx
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound[0]):
                let row_global_index = Index(
                    start_idx_global[0] + inner_row_idx,
                    start_idx_global[1],
                )
                let row_data: SIMD[type, simd_size]
                if skip_col_bound:
                    # This is fastest path where both row and col bounds
                    #  are skipped so the code path is simd-in and simd-out
                    #  without any predicate.
                    row_data = self.original_matrix.simd_load[simd_size](
                        row_global_index
                    )
                else:
                    # Not skipping col bound, need to to a partial fill of
                    #  the transpose buffer row.
                    row_data = partial_simd_load[type, simd_size](
                        self.original_matrix._offset(row_global_index),
                        0,  # no left bound.
                        read_bound[1],
                        0,
                    )

                transpose_buffer.simd_store[simd_size](
                    Index(inner_row_idx, 0), row_data
                )
            else:
                # Row out of defined bound, fill the transpose buffer with zero
                transpose_buffer.simd_store[simd_size](
                    Index(inner_row_idx, 0), SIMD[type, simd_size](0)
                )

        unroll[simd_size, body]()

        # Transpose the buffered data
        transpose_inplace[simd_size, simd_size, type](transpose_buffer)

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        @always_inline
        @parameter
        fn transposed_inner_row_body[idx: Int]():
            let transposed_data = transpose_buffer.simd_load[simd_size](
                Index(idx, 0)
            )
            # compute the packed index
            let _row_outer = local_off_set[0] // row_inner_size
            let _row_inner = local_off_set[0] % row_inner_size

            if skip_col_bound or (idx < write_bound[1]):
                self.packed_matrix.simd_store[simd_size](
                    Index(
                        _row_outer,
                        local_off_set[1] + idx,
                        _row_inner,
                    ),
                    transposed_data,
                )
            # Out of bound columns are discarded as there's no allocation for them
            #  in the packed buffer.

        unroll[simd_size, transposed_inner_row_body]()

    fn _pack(self):
        """Helper function: Allocates transpose workspace and launch the
        transpose helper function until all required data has been packed.
        """

        let transpose_buffer = NDBuffer[
            2,
            DimList(simd_size, simd_size),
            type,
        ].aligned_stack_allocation[alignof[SIMD[type, simd_size]]()]()

        let valid_tile_simd_dim = Index(
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
        @parameter
        fn transpose_pack_unit[static_switch0: Bool, static_switch1: Bool]():
            self._transpose_pack_helper[
                # skip_row_bound, skip_col_bound
                static_switch0,
                static_switch1,
            ](
                transpose_buffer,
                # local offset
                Index(row_idx, col_idx),
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
]:
    """Pack columns from a matrix into the mlas packed layout and
    extract inner vectors of columns into the packed inner dimension.
    e.g. extracts [X, Y] and packs as [Yo][X][Yi]
    """

    # packed matrix
    var packed_matrix: NDBuffer[3, packed_shape, type]
    # original matrix:
    var original_matrix: NDBuffer[2, original_shape, type]
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
        packed_matrix: NDBuffer[3, packed_shape, type],
        original_matrix: NDBuffer[2, original_shape, type],
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
        assert_param[column_inner_size % simd_size == 0]()
        debug_assert(
            pack_tile_dim[1] % column_inner_size == 0,
            "Unimplemented tile pattern.",
        )

        let instance = Self(
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
        alias alignment = alignof[SIMD[type, simd_size]]()
        alias is_row_aligned = original_shape.at[1]().is_multiple[alignment]()

        alias unroll_factor = get_packB_unroll_factor()

        @always_inline
        @parameter
        fn pack_vector(row_idx: Int, col_idx: Int):
            let global_idx = self.global_offset + Index(row_idx, col_idx)
            var data = SIMD[type, simd_size](0)
            if skip_col_bound or (
                col_idx + simd_size <= self.valid_data_dim[1]
            ):
                # Whole SIMD vector within bound.
                @parameter
                if is_row_aligned:
                    data = self.original_matrix.aligned_simd_load[
                        simd_size, alignment
                    ](global_idx)
                else:
                    data = self.original_matrix.simd_load[simd_size](global_idx)
            elif col_idx < self.valid_data_dim[1]:
                # Starting point within bound but cannot load a whole
                #  vector. Do a partial load.
                data = partial_simd_load[type, simd_size](
                    self.original_matrix._offset(global_idx),
                    0,
                    self.valid_data_dim[1] - col_idx,
                    SIMD[type, 1](0),
                )

            # map to packed index
            let col_idx_outer = col_idx // column_inner_size
            let col_idx_inner = col_idx % column_inner_size
            self.packed_matrix.aligned_simd_store[simd_size, alignment](
                Index(col_idx_outer, row_idx, col_idx_inner),
                data,
            )

        @always_inline
        @parameter
        fn pack_body[idx: Int]():
            pack_vector(row_start + idx, col_start)

        @always_inline
        @parameter
        fn prefetch_body[idx: Int]():
            let global_row_idx = (
                self.global_offset[0] + row_start + unroll_factor + idx
            )
            let global_col_idx = self.global_offset[1] + col_start
            self.original_matrix.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](global_row_idx, global_col_idx)

        @parameter
        if skip_row_bound:
            if not has_neon():
                unroll[unroll_factor, prefetch_body]()
            unroll[unroll_factor, pack_body]()
        else:
            for row_idx in range(row_start, valid_row_count):
                pack_vector(row_idx, col_start)

    fn _pack(self):
        """Copy the B tile from the original matrix to the packed buffer.
        Each iteration copies a block of shape (unroll_factor, simd_size)."""
        let valid_row_count = min(self.valid_data_dim[0], self.pack_tile_dim[0])

        alias unroll_factor = get_packB_unroll_factor()

        var row_idx: Int = 0
        var col_idx: Int = 0

        @always_inline
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


struct MatmulInnerLoopBPacked[
    shape_a: DimList,
    shape_c: DimList,
    packed_shape: DimList,
    accum_type: DType,
    value_type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
    prefetch_b_distance: Int,
    critical_stride: Bool,
]:
    """Inner loop implementation for mlas-style tiled matmul. Accumulates a
    tile of input defined by (a_row_size, TileN, TileK).
    """

    # Parameters for global reference.
    var c: NDBuffer[2, shape_c, accum_type]
    var a: NDBuffer[2, shape_a, value_type]
    var b_packed: NDBuffer[3, packed_shape, value_type]
    # 3D global offset within the whole matmul problem space.
    var global_offset: GemmShape
    # Dynamic tiling parameter for this inner loop
    #  in (TileN, TileK).
    var tile_n_k: StaticIntTuple[2]
    # Boundary of valid output space within the
    #  local tile, in (a_row_size, TileN).
    var c_bound: StaticIntTuple[2]

    fn __init__(
        inout self,
        c: NDBuffer[2, shape_c, accum_type],
        a: NDBuffer[2, shape_a, value_type],
        b_packed: NDBuffer[3, packed_shape, value_type],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
        c_bound: StaticIntTuple[2],
    ):
        self.c = c
        self.a = a
        self.b_packed = b_packed
        self.global_offset = global_offset
        self.tile_n_k = tile_n_k
        self.c_bound = c_bound

    @staticmethod
    fn run(
        c: NDBuffer[2, shape_c, accum_type],
        a: NDBuffer[2, shape_a, value_type],
        b_packed: NDBuffer[3, packed_shape, value_type],
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.
        Args:
            c(NDBuffer): pre-allocated buffer space for packed result.
            a(NDBuffer): data buffer operand A.
            b(NDBuffer): data buffer operand B in packed layout.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            global_bound(StaticIntTuple): Tile upper boundary of the current
            tile function call.
            tile_n_k(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile of B.
        """
        let instance = Self(
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
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ],
    ):
        """Utility funcion on the inner loop. Initializes a local c buffer with
        all zeros.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
        """

        @always_inline
        @parameter
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.aligned_simd_store[
                simd_size, alignof[SIMD[accum_type, simd_size]]()
            ](
                Index(idx0, idx1 * simd_size),
                SIMD[accum_type, simd_size](0),
            )

        unroll[a_row_size, pack_inner_size // simd_size, outer_body]()

    @always_inline
    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ],
        # indexing within tile, in (m,n)
        tile_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Loads a local c_buffer with the
        value stored in the output buffer space, given the indices within the
        tile being processed.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
                tile_idx(StaticIntTuple): index tuple with (m,n) coordinates
                    within the current processing tile.
        """
        alias alignment = alignof[SIMD[accum_type, simd_size]]()
        alias is_row_aligned = shape_c.at[1]().is_multiple[alignment]()
        let N = self.c.dim(1)
        var c_ptr = self.c.data.offset(
            (self.global_offset.M + tile_idx[0]) * N
            + self.global_offset.N
            + tile_idx[1]
        )

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data: SIMD[accum_type, simd_size] = 0
            if skip_boundary_check or (
                Index(idx0, idx1 * simd_size + simd_size)
                <= (self.c_bound - tile_idx)
            ):
                # Use simd load if all within bound
                @parameter
                if is_row_aligned:
                    c_data = c_ptr.offset(idx1 * simd_size).aligned_simd_load[
                        simd_size, alignment
                    ]()
                else:
                    c_data = c_ptr.offset(idx1 * simd_size).simd_load[
                        simd_size
                    ]()
            elif (idx0 + tile_idx[0]) < self.c_bound[
                0
            ] and idx1 * simd_size <= self.c_bound[1]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[accum_type, simd_size](
                    c_ptr.offset(idx1 * simd_size),
                    0,
                    self.c_bound[1] - tile_idx[1] - idx1 * simd_size,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = SIMD[accum_type, simd_size](0)

            # Store data to local buffer.
            c_local.aligned_simd_store[simd_size, alignment](
                Index(idx0, idx1 * simd_size), c_data
            )

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr = c_ptr.offset(N)

        unroll[a_row_size, pack_inner_size // simd_size, body]()

    @always_inline
    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ],
        tile_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Stores the value of a local c
        buffer to the corresponding position in the output buffer space.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
                tile_idx(StaticIntTuple): index tuple with (m,n) coordinates
                    within the current processing tile.
        """
        alias alignment = alignof[SIMD[accum_type, simd_size]]()
        alias is_row_aligned = shape_c.at[1]().is_multiple[alignment]()

        let N = self.c.dim(1)
        var c_ptr = self.c.data.offset(
            (self.global_offset.M + tile_idx[0]) * N
            + self.global_offset.N
            + tile_idx[1]
        )

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            let c_data = c_local.aligned_simd_load[simd_size, alignment](
                Index(idx0, idx1 * simd_size)
            )
            if skip_boundary_check or (
                Index(idx0, idx1 * simd_size + simd_size)
                <= (self.c_bound - tile_idx)
            ):
                # Use simd store if all within bound
                @parameter
                if is_row_aligned:
                    c_ptr.offset(idx1 * simd_size).aligned_simd_store[
                        simd_size, alignment
                    ](c_data)
                else:
                    c_ptr.offset(idx1 * simd_size).simd_store[simd_size](c_data)
            elif (
                idx0 < (self.c_bound[0] - tile_idx[0])
                and idx1 * simd_size <= self.c_bound[1]
            ):
                # Use partial store if row in bound but col not
                #  in simd bound.
                partial_simd_store(
                    c_ptr.offset(idx1 * simd_size),
                    0,
                    self.c_bound[1] - tile_idx[1] - idx1 * simd_size,
                    c_data,
                )

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr = c_ptr.offset(N)

        unroll[a_row_size, pack_inner_size // simd_size, body]()

    @adaptive
    fn _accumulate[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing `a_col_size` columns of A.

        Args:
            c_local(NDBuffer): pre-allocated local buffer for c partial
                sums.
            tile_n_k_idx(StaticIntTuple): index tuple with (n, k)
                coordinates within the current processing tile to index the
                packed B matrix.
        """
        assert_param[a_col_size > 1]()

        # Seek outer indices in packed layout.
        let n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        let global_k = self.global_offset.K + tile_n_k_idx[1]

        # Prefetch B matrix.
        @parameter
        if prefetch_b_distance > 0:

            @always_inline
            @parameter
            fn prefetch_body[idx: Int]():
                self.b_packed.prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ](
                    n_outer_idx,
                    tile_n_k_idx[1] + prefetch_b_distance,
                    idx * simd_size,
                )

            unroll[pack_inner_size // simd_size, prefetch_body]()

        # Loop over local accumulator tiles.
        @always_inline
        @parameter
        fn _do[idx: Int]():
            alias idx_outer = idx
            let global_m = self.global_offset.M + idx_outer
            let a_val = self.a.simd_load[a_col_size](global_m, global_k).cast[
                accum_type
            ]()

            @always_inline
            @parameter
            fn outer_body[idx0: Int, idx1: Int]():
                let b_val = self.b_packed.simd_load[simd_size](
                    n_outer_idx,
                    tile_n_k_idx[1] + idx0,
                    idx1 * simd_size,
                ).cast[accum_type]()

                let c_idx = Index(idx_outer, idx1 * simd_size)
                var c_val = c_local.simd_load[simd_size](c_idx)

                c_val = fma[accum_type, simd_size](a_val[idx0], b_val, c_val)
                c_local.simd_store[simd_size](c_idx, c_val)

            unroll[a_col_size, pack_inner_size // simd_size, outer_body]()

        unroll[a_row_size, _do]()

    @adaptive
    fn _accumulate[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local(NDBuffer): pre-allocated local buffer for c partial
                sums.
            tile_n_k_idx(StaticIntTuple): index tuple with (n, k)
                coordinates within the current processing tile to index the
                packed B matrix.
        """
        assert_param[a_col_size == 1]()
        # Seek outer indices in packed layout.
        let n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        let global_k = self.global_offset.K + tile_n_k_idx[1]

        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, tile_n_k_idx[1], 0)
        )

        # Prefetch B matrix.
        @parameter
        if prefetch_b_distance > 0:
            alias prefetch_offset = prefetch_b_distance * pack_inner_size

            @parameter
            @always_inline
            fn prefetch_body[idx: Int]():
                b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

            unroll[pack_inner_size // simd_size, prefetch_body]()

        # This inner kernels works with non-transposed A.
        let K = self.a.dim(1)
        var a_ptr = self.a.data.offset(self.global_offset.M * K + global_k)

        # Loop over local accumulator tiles.
        @parameter
        @always_inline
        fn body[idx0: Int, idx1: Int]():
            let a_val = a_ptr.offset(idx0 * K).simd_load[1]().cast[accum_type]()
            alias alignment = alignof[SIMD[accum_type, simd_size]]()
            let c_idx = Index(idx0, idx1 * simd_size)
            var c_val = c_local.aligned_simd_load[simd_size, alignment](c_idx)
            let b_val = b_ptr.offset(idx1 * simd_size).aligned_simd_load[
                simd_size, alignment
            ]().cast[accum_type]()
            c_val = fma[accum_type, simd_size](a_val, b_val, c_val)
            c_local.aligned_simd_store[simd_size, alignment](c_idx, c_val)

        unroll[a_row_size, pack_inner_size // simd_size, body]()

    @adaptive
    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        assert_param[not has_neon() or critical_stride]()
        # Allocate accumulation buffer.
        let c_local = NDBuffer[
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ].aligned_stack_allocation[alignof[SIMD[accum_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, Index(0, idx_n))

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            for idx_k in range(self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate[1](c_local, Index(idx_n, idx_k))

            self._store_c_tile(c_local, Index(0, idx_n))

    @adaptive
    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        assert_param[has_neon() and not critical_stride]()
        # Allocate accumulation buffer.
        let c_local = NDBuffer[
            2,
            DimList(a_row_size, pack_inner_size),
            accum_type,
        ].aligned_stack_allocation[alignof[SIMD[accum_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, Index(0, idx_n))

            let partition_end = simd_size * (self.tile_n_k[1] // simd_size)
            for idx_k0 in range(0, partition_end, simd_size):
                self._accumulate[simd_size](c_local, Index(idx_n, idx_k0))

            for idx_k1 in range(partition_end, self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate[1](c_local, Index(idx_n, idx_k1))

            self._store_c_tile(c_local, Index(0, idx_n))


# Tiled Matmul Implementation.
# TODO: not yet supporting transpose_a
@value
struct TiledMatmul[
    config: MatmulConfig,
    accum_type: DType,
    value_type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    rowwise_epilogue_enabled: Bool,
    critical_stride: Bool,
]:
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    TODO: not yet supporting transpose_a.
    TODO: add tag based implementation dispatch.
    TODO: add fusion hooks.
    """

    var c: NDBuffer[2, config.shape_c, accum_type]
    var a: NDBuffer[2, config.shape_a, value_type]
    var b: NDBuffer[2, config.shape_b, value_type]
    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]

    # Tile starting points on the (M,N,K) coordinates.
    var global_tile_offset: GemmShape

    # Tile sizes this routine will process on the (M,N,K) coordinates.
    var global_tile_shape: GemmShape

    var b_tile_generator: BTileGenerator[
        config, value_type, transpose_b, b_packed
    ]

    var elementwise_epilogue_fn: fn (GemmShape, GemmShape) capturing -> None

    var rowwise_epilogue_fn: fn (Int, Int) capturing -> None

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[2, config.shape_c, accum_type],
        a: NDBuffer[2, config.shape_a, value_type],
        b: NDBuffer[2, config.shape_b, value_type],
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape = GemmShape {M: 0, N: 0, K: 0},
    ):
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
        c: NDBuffer[2, config.shape_c, accum_type],
        a: NDBuffer[2, config.shape_a, value_type],
        b: NDBuffer[2, config.shape_b, value_type],
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) capturing -> None,
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape,
    ):
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
        c: NDBuffer[2, config.shape_c, accum_type],
        a: NDBuffer[2, config.shape_a, value_type],
        b: NDBuffer[2, config.shape_b, value_type],
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) capturing -> None,
        rowwise_epilogue_fn: fn (Int, Int) capturing -> None,
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape,
    ):
        """Interface function to run tiled matmul on a given sub-tile.

        Args:
            c(NDBuffer): Pre-allocated buffer space for result.
            a(NDBuffer): Operand A of the matmul.
            b(NDBuffer): Operand B of the mamtul.
            transpose_a: True if a is in transposed layout.
            transpose_b: True if b is in transposed layout.
            global_tile_offset(GemmShape): tile offset on the original buffer.
            global_tile_shape(GemmShape): tile shape this call will process.
        """

        let tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size
        ](global_tile_shape)

        let matmul = TiledMatmul[
            config,
            accum_type,
            value_type,
            transpose_a,
            transpose_b,
            b_packed,
            elementwise_epilogue_enabled,
            rowwise_epilogue_enabled,
            critical_stride,
        ](
            c,
            a,
            b,
            tile_n_k,
            global_tile_offset,
            global_tile_shape,
            BTileGenerator[config, value_type, transpose_b, b_packed].get(
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
    ](self, global_offset: GemmShape, sub_tile_n: Int, sub_tile_k: Int,):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

            Args:
                m_loop_pack_inner_size(index): Inner dimension of the packed data
                    layout.
                b_packed(NDBuffer): B matrix in packed layout.
                global_offset(GemmShape): 3D global offset within the whole
                    matmul problem space.
                sub_tile_n(Int): Dynamic tile size to use on N dimension.
                sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        # valid distance in each dimension from the current offset to the end of the matrix
        let knm_bounds = (
            self.global_tile_shape + self.global_tile_offset - global_offset
        )

        @parameter
        @always_inline
        fn unswitch_residual_n[skip_col_bound: Bool]():
            let b_packed_tile = self.b_tile_generator.get_tile[
                m_loop_pack_inner_size
            ](
                global_offset,
                Index(sub_tile_n, sub_tile_k),
                Index(knm_bounds.N, knm_bounds.K),
            )

            # Launch the MLoop
            # The upper bounds apply to runtime packing. For pre-packing, the
            # tile has been padded to fit (sub_tile_n, sub_tile_k).
            let sub_tile_n_k = Index(
                min(sub_tile_n, knm_bounds.N), min(sub_tile_k, knm_bounds.K)
            )

            @parameter
            @always_inline
            fn row_iteration[tile_size: Int](row_offset: Int):
                MatmulInnerLoopBPacked[
                    config.shape_a,
                    config.shape_c,
                    config.packed_shape,
                    accum_type,
                    value_type,
                    config.simd_size,
                    tile_size,
                    m_loop_pack_inner_size,
                    skip_col_bound,
                    config.prefetch_b_distance_k,
                    critical_stride,
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

            tile[
                row_iteration, VariadicList[Int](config.a_row_size, 4, 3, 2, 1)
            ](
                0,  # starting row offset
                knm_bounds.M,  # row bound
            )

        unswitch[unswitch_residual_n](knm_bounds[1] > sub_tile_n)

    # Iterate on the N dimension of the gemm space.
    fn _outer_n_loop[
        last_k_tile: Bool
    ](self, global_offset: GemmShape, sub_tile_k: Int,):
        """Iterate on the N dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        let valid_col_count: Int = (
            self.global_tile_shape.N
            + self.global_tile_offset.N
            - global_offset.N
        )
        let tile_n: Int = self.tile_n_k[0]

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
            let primary_tiles = VariadicList[Int](
                tile_n, 2 * config.simd_size, config.simd_size
            )
            tile[secondary_tiles, config.simd_size, m_loop](
                0, valid_col_count, primary_tiles, config.simd_size
            )
        else:
            alias secondary_tiles_packed_b = VariadicList[Int](
                config.pack_inner_size
            )
            let primary_tiles_packed_b = VariadicList[Int](tile_n)
            tile[secondary_tiles_packed_b, config.pack_inner_size, m_loop](
                0, valid_col_count, primary_tiles_packed_b, tile_n
            )

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(
        self,
    ):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
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
        b_packed_ptr: DTypePointer[value_type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[3, config.packed_shape, value_type]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

            Args:
                b_packed(NDBuffer): B matrix in packed layout.
                tile_n(Int): Dynamic tile size to use on N dimension.
                tile_k(Int): Dynamic tile size to use on K dimension.
                n_inner_size(Int): Inner dimension size to use for the packed
                    data layout.
        """
        return NDBuffer[3, config.packed_shape, value_type](
            b_packed_ptr.address,
            DimList(tile_n // n_inner_size, tile_k, n_inner_size),
            value_type,
        )


fn pack_b[
    transpose_b: Bool,
    simd_size: Int,
    inner_size: Int,
    type: DType,
    src_shape: DimList,
    dst_shape: DimList,
](
    dst: NDBuffer[2, dst_shape, type],
    src: NDBuffer[2, src_shape, type],
    tile_n: Int,
    tile_k: Int,
):
    """Utility function to pack the entire B matrix, such that each
    [tile_n // inner_size, tile_k, inner_size] tile of src is contiguous in dst.

    Tiles (not tile contents) are stored in row major order, so tile[i, j] is
    tile_n * tile_k bytes away from tile[i, j+1].
    """
    dst.zero()  # zero the padding to be safe, shouldn't be necessary
    let dst_flat = dst.flatten()
    var dst_offset: Int = 0

    @parameter
    if not transpose_b:
        let k_in = src.dim[0]()
        let n_in = src.dim[1]()
        let k_out = dst.dim[0]()
        let n_out = dst.dim[1]()

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
                let packed_dst_view = NDBuffer[
                    3, DimList.create_unknown[3](), type
                ](
                    dst_flat.data.offset(dst_offset),
                    DimList(tile_n // inner_size, tile_k, inner_size),
                    type,
                )
                let valid_k = min(tile_k, k_in - idx_k)
                let valid_n = min(tile_n, n_in - idx_n)
                PackMatrixCols[
                    src_shape,
                    DimList.create_unknown[3](),
                    type,
                    simd_size,
                    inner_size,
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
        let k_in_t = src.dim[1]()
        let n_in_t = src.dim[0]()
        let k_out_t = dst.dim[0]()
        let n_out_t = dst.dim[1]()

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
                let packed_dst_view_t = NDBuffer[
                    3, DimList.create_unknown[3](), type
                ](
                    dst_flat.data.offset(dst_offset),
                    DimList(tile_n // inner_size, tile_k, inner_size),
                    type,
                )
                let valid_k_t = min(tile_k, k_in_t - idx_k_t)
                let valid_n_t = min(tile_n, n_in_t - idx_n_t)
                PackMatrixRows[
                    src_shape,
                    DimList.create_unknown[3](),
                    type,
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
        2, config.shape_b, type
    ]  # packed layout if b_packed is True
    var b_tile_stack_ptr: DTypePointer[type]
    var tile_n_k: StaticIntTuple[2]

    # needs to be always_inline so b_tile_stack_ptr gets allocated on caller's stack
    @always_inline
    @staticmethod
    fn get(
        b: NDBuffer[2, config.shape_b, type], tile_n_k: StaticIntTuple[2]
    ) -> BTileGenerator[config, type, transpose_b, b_packed]:
        var b_tile_stack_ptr = DTypePointer[type].get_null()

        debug_assert(
            not (transpose_b and b_packed),
            "b cannot be both transposed and pre-packed.",
        )

        @parameter
        if not b_packed:
            b_tile_stack_ptr = _raw_stack_allocation[
                config.pack_data_size,
                type,
                alignof[SIMD[type, dtype_simd_width[type]()]](),
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
    ) -> NDBuffer[3, config.packed_shape, type]:
        """Get a packed matrix (B) tile.

        Args:
            global_offset: offset in the global M, N, K dimensions.
            tile_dim_nk: tile shape based on cache size and matrix dimensions.
            valid_data_dim_nk: the upper bounds for N and K dimensions.

        valid_data_tile_nk is ignored for pre-packing, where the tile is padded
        to have shape of tile_dim_nk.

        Returns:
            A view of the packed tile.

        """
        let tile_shape_nopack = DimList(
            tile_dim_nk[0] // inner_size, tile_dim_nk[1], inner_size
        )
        let packed_b = NDBuffer[3, config.packed_shape, type](
            self.b_tile_stack_ptr,
            tile_shape_nopack,
            type,
        )

        @parameter
        if transpose_b and not b_packed:
            PackMatrixRows[
                config.shape_b,
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
                config.shape_b,
                config.packed_shape,
                type,
                config.simd_size,
                inner_size,
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
            let tile_shape_pack = DimList(
                self.tile_n_k[0] // inner_size, self.tile_n_k[1], inner_size
            )
            let tile_k_idx = global_offset.K // self.tile_n_k[1]
            let b_flat = self.b.flatten()
            let n_padded = self.b.dim[1]()
            let b_tile_view = NDBuffer[3, config.packed_shape, type](
                # tiles are ordered in row-major order
                # a bit of trickieness going on here, this works because:
                #   1. tile_k is the same for every thread (tile_n is not) since threads
                #       don't currently partition on the K dimension
                #   2. the n dimension of each thread's tile is gauranteed to be an
                #       exact multiple of the inner size
                #   3. each tile has dims [tile_n/inner, tile_k, inner]
                b_flat.data.offset(
                    tile_k_idx * self.tile_n_k[1] * n_padded
                    + global_offset.N * self.tile_n_k[1]
                ),
                tile_shape_pack,
                type,
            )
            return b_tile_view

        else:
            debug_assert(
                False, "unreachable, b_packed not supported with transpose_b"
            )

        return packed_b


fn matmul_parallel_async[
    type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
](
    c: NDBuffer[2, DimList.create_unknown[2](), type],
    a: NDBuffer[2, DimList.create_unknown[2](), type],
    b: NDBuffer[2, DimList.create_unknown[2](), type],
    out_chain: OutputChainPtr,
    num_threads: Int = -1,
):
    @parameter
    fn null_lambda[
        val_type: DType, width: Int
    ](out_coords: StaticIntTuple[2], out_val: SIMD[val_type, width]):
        pass

    matmul_parallel_async[
        type,
        transpose_a,
        transpose_b,
        b_packed,
        False,
        null_lambda,
    ](c, a, b, out_chain, num_threads)


fn matmul_parallel_async[
    type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    elementwise_lambda_fn: elementwise_lambda_fn_sig_type,
](
    c: NDBuffer[2, DimList.create_unknown[2](), type],
    a: NDBuffer[2, DimList.create_unknown[2](), type],
    b: NDBuffer[2, DimList.create_unknown[2](), type],
    out_chain: OutputChainPtr,
    num_threads: Int = -1,
):
    assert_param[not transpose_a, "transpose_a not yet supported"]()

    let shape = GemmShape.get[False, transpose_b](c, a, b)
    let m = shape.M
    let n = shape.N
    let k = shape.K

    let complexity = m * n * k
    let num_tasks = min(
        div_ceil(complexity, get_min_task_size()),
        num_threads if num_threads
        > 0 else out_chain.get_runtime().parallelism_level(),
    )

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        let sub_matmul_config = get_partitioned_matmul[PartitionHeuristic.MOJO](
            m, n, k, task_id, num_tasks
        )

        if sub_matmul_config.shape[0] <= 0 or sub_matmul_config.shape[1] <= 0:
            return

        _submatmul_sequential_sync[
            type,
            transpose_a,
            transpose_b,
            b_packed,
            elementwise_epilogue_enabled,
            elementwise_lambda_fn,
        ](c, a, b, sub_matmul_config.shape, sub_matmul_config.offset)

    async_parallelize[task_func](out_chain, num_tasks)

    # TODO (#12624): Closure captures some state on the stack so this needs
    # to be synchronous in order to keep that state alive
    external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Await", NoneType](
        out_chain.ptr
    )


fn _submatmul_sequential_sync[
    type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    elementwise_lambda_fn: elementwise_lambda_fn_sig_type,
    rowwise_epilogue_enabled: Bool,
](
    c: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    a: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    b: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
    rowwise_epilogue_fn: fn (Int, Int) capturing -> None,
):
    assert_param[not transpose_a, "transpose_a not yet supported"]()

    let shape = GemmShape.get[False, transpose_b](c, a, b)
    let m = shape.M
    let n = shape.N
    let k = shape.K

    @parameter
    fn dispatch_on_critical_stride[critical_stride: Bool]():
        alias mm_config = search_mm_config[type, b_packed, critical_stride]()

        fn elementwise_closure(offset: GemmShape, shape: GemmShape):
            elementwise_epilogue_c_tile[
                mm_config.simd_size,
                type,
                mm_config.shape_c,
                elementwise_lambda_fn,
            ](
                offset,
                shape,
                rebind[NDBuffer[2, mm_config.shape_c, type]](c),
            )

        TiledMatmul[
            mm_config,
            # accum_type
            type,
            # value_type
            type,
            # transpose_a
            False,
            transpose_b,
            b_packed,
            elementwise_epilogue_enabled,
            rowwise_epilogue_enabled,
            critical_stride,
        ].run(
            rebind[NDBuffer[2, mm_config.shape_c, type]](c),
            rebind[NDBuffer[2, mm_config.shape_a, type]](a),
            rebind[NDBuffer[2, mm_config.shape_b, type]](b),
            elementwise_closure,
            rowwise_epilogue_fn,
            sub_matrix_shape,
            sub_matrix_offset,
        )

    dispatch_is_critical_stride[dispatch_on_critical_stride](k)


fn _submatmul_sequential_sync[
    type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
](
    c: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    a: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    b: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
):
    @parameter
    fn null_lambda[
        val_type: DType, width: Int
    ](out_coords: StaticIntTuple[2], out_val: SIMD[val_type, width]):
        pass

    _submatmul_sequential_sync[
        type,
        transpose_a,
        transpose_b,
        b_packed,
        # elementwise_epilogue_enabled,
        False,
        null_lambda,
    ](c, a, b, sub_matrix_shape, sub_matrix_offset)


fn _submatmul_sequential_sync[
    type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    elementwise_lambda_fn: elementwise_lambda_fn_sig_type,
](
    c: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    a: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    b: NDBuffer[
        2,
        DimList.create_unknown[2](),
        type,
    ],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
):
    @closure
    @always_inline
    fn null_rowwise_epilogue(
        start_row: Int,
        num_rows: Int,
    ):
        pass

    _submatmul_sequential_sync[
        type,
        transpose_a,
        transpose_b,
        b_packed,
        elementwise_epilogue_enabled,
        elementwise_lambda_fn,
        False,
    ](c, a, b, sub_matrix_shape, sub_matrix_offset, null_rowwise_epilogue)
