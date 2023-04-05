# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import (
    assert_param,
    assert_param_bool,
    debug_assert,
    assert_param_bool_msg,
)
from DType import DType
from Buffer import (
    NDBuffer,
    Buffer,
    DynamicRankBuffer,
    partial_simd_load,
    partial_simd_store,
    _raw_stack_allocation,
)
from Functional import tile, unswitch, unroll, unroll2
from Index import Index, StaticIntTuple
from Int import Int
from List import Dim, DimList, VariadicList, create_dim_list
from Math import min, fma, max, div_ceil
from Matrix import Matrix
from Memory import stack_allocation
from Pointer import DTypePointer
from Range import range
from SIMD import SIMD
from TargetInfo import (
    os_is_macos,
    has_neon,
    alignof,
    dtype_simd_width,
    dtype_sizeof,
)
from Transpose import transpose_inplace
from Intrinsics import PrefetchOptions
from IO import print


@register_passable("trivial")
struct MatmulConfig:
    """Static configuration of tiled matmul algorithms."""

    # Static shape info of Operand A.
    var shape_a: DimList[2]

    # Static shape info of Operand B.
    var shape_b: DimList[2]

    # Static shape info of Operand C.
    var shape_c: DimList[2]

    # Static packed shape info of the packed buffer.
    var packed_shape: DimList[3]

    # Static packed shape info of the bias vector.
    var shape_bias: DimList[1]

    # Static info on simd vector size.
    var simd_size: __mlir_type.index

    # Static loop unrolling size on M dimension.
    var a_row_size: __mlir_type.index

    # Static inner dimension of packed data layout.
    var pack_inner_size: __mlir_type.index

    # Static info on number of elements to pack in the packing routine.
    var pack_data_size: __mlir_type.index

    # Prefetch distance for packed b vectors in micro kernels.
    var prefetch_b_distance_k: Int


@register_passable("trivial")
struct MatmulDataType:
    """Record describing the data types of the matrices in a matmul"""

    # The data type of the result (matrix C), and the accumulator.
    var accum_type: DType

    # The data type of the operands (matrix A and B).
    var value_type: DType


@register_passable("trivial")
struct MatmulOperandLayout:
    """Record describing the data layouts of the matmul operands as well as
    intermediate matrices.
    """

    # Indicates if the input matrix A is transposed.
    var transpose_a: Bool

    # Indicates if the input matrix B is transposed.
    var transpose_b: Bool

    # Indicates if the input matrix A is pre-packed.
    var a_packed: Bool

    # Indicates if the input matrix B is pre-packed.
    var b_packed: Bool

    # The inner dimension size for packed A matrix if B is pre-packed.
    var pack_a_inner_size: Int

    # The inner dimension size for packed B matrix if B is pre-packed.
    var pack_b_inner_size: Int


@register_passable("trivial")
struct GemmShape:
    """Helper class to unpack gemm dimension and layout."""

    var M: Int
    var N: Int
    var K: Int

    # Construct from dynamic shaped input.
    @staticmethod
    fn get[
        transpose_a: Bool,
        transpose_b: Bool,
        shape_c: DimList[2],
        shape_a: DimList[2],
        shape_b: DimList[2],
        accum_type: DType,
        value_type: DType,
    ](
        c: NDBuffer[2, shape_c, accum_type],
        a: NDBuffer[2, shape_a, value_type],
        b: NDBuffer[2, shape_b, value_type],
    ) -> GemmShape:
        """Constructor of a gemm shape record from input buffers.

        M, N, and K are intentionally calculated using `a` and `c` ONLY. This
        is because `b` may be padded to a multiple of the tile size if it has
        been pre-packed.

        Args:
            c: Buffer with allocated output space.
            a: Buffer containing matrix operand A.
            b: Buffer containing matrix operand B.
        """
        var gemm_shape: GemmShape
        gemm_shape.M = c.dim[0]()
        gemm_shape.N = c.dim[1]()
        if transpose_a:
            gemm_shape.K = a.dim[0]()
        else:
            gemm_shape.K = a.dim[1]()
        return gemm_shape

    @staticmethod
    fn get[
        config: MatmulConfig,
        layout: MatmulOperandLayout,
        data_type: MatmulDataType,
    ](
        c: NDBuffer[2, config.shape_c, data_type.accum_type],
        a: NDBuffer[2, config.shape_a, data_type.value_type],
        b: NDBuffer[2, config.shape_b, data_type.value_type],
    ) -> GemmShape:
        """Constructor of a gemm shape record from input buffers.

        Args:
            c: Buffer with allocated output space.
            a: Buffer containing matrix operand A.
            b: Buffer containing matrix operand B.
        """
        var gemm_shape: GemmShape
        gemm_shape.M = c.dim[0]()
        gemm_shape.N = c.dim[1]()

        @parameter
        if layout.transpose_a:
            gemm_shape.K = a.dim[0]()
        else:
            gemm_shape.K = a.dim[1]()
        return gemm_shape

    @staticmethod
    fn get(
        c: DynamicRankBuffer,
        a: DynamicRankBuffer,
        b: DynamicRankBuffer,
        transpose_a: Bool,
        transpose_b: Bool,
    ) -> GemmShape:
        var gemm_shape: GemmShape
        gemm_shape.M = c.dim(0)
        gemm_shape.N = c.dim(1)

        if transpose_a:
            gemm_shape.K = a.dim(0)
        else:
            gemm_shape.K = a.dim(1)
        return gemm_shape

    # TODO: re-enable using StaticIntTuple.
    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        if idx == 0:
            return self.M
        if idx == 1:
            return self.N
        return self.K

    fn __setitem__(self&, idx: Int, value: Int):
        if idx == 0:
            self.M = value
            return
        if idx == 1:
            self.N = value
            return
        if idx == 2:
            self.K = value
            return

    fn __init__(m: Int, n: Int, k: Int) -> GemmShape:
        """Constructor of a gemm shape record by directly supplying the values.

        Args:
            m: M dimension of the gemm shape.
            n: N dimension of the gemm shape.
            k: K dimension of the gemm shape.

        Returns:
            The constructed shape record.
        """
        return GemmShape {M: m, N: n, K: k}

    fn __init__(index: StaticIntTuple[3]) -> GemmShape:
        """Constructor of a gemm shape record from a index tuple.

        Args:
            index (StaticIntTuple): The int tuple containing the index(m,n,k).

        Returns:
            The constructed shape record.
        """
        return GemmShape(
            index[0],
            index[1],
            index[2],
        )

    fn as_index(self) -> StaticIntTuple[3]:
        """Utility to convert the underlying data to an index tuple. So that the
        utilities such as elementwise add can be used.

        Returns:
            The constructed index tuple.
        """
        return Index(self.M, self.N, self.K)

    fn __add__(self, rhs: GemmShape) -> GemmShape:
        """Coordinate-wise addition of two gemm shape records.

        Args:
            rhs: Another gemm shape record to add with.
        """
        return self.as_index() + rhs.as_index()


@always_inline
fn naive_matmul[
    shape_a: DimList[2],
    shape_b: DimList[2],
    shape_c: DimList[2],
    accum_type: DType,
    value_type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    epilogue_elemwise_func: __mlir_type[
        `!kgen.signature<<`,
        DType,
        `>(`,
        Int,  # Row
        ` borrow,`,
        Int,  # Col
        ` borrow,`,
        SIMD[1, __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, DType]],
        ` borrow) -> `,
        SIMD[1, __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, DType]],
        `>`,
    ],
    epilogue_rowise_func: __mlir_type[
        `!kgen.signature<<`,
        DType,
        `>(`,
        Int,  # Row
        ` borrow,`,
        Buffer[
            Dim(), __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, DType]
        ],
        ` borrow) -> !lit.none>`,
    ],
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
            var c_val: SIMD[1, accum_type] = 0
            for k in range(gemm_shape.K):
                let a_val = matrix_a[m, k].cast[accum_type]()
                let b_val = matrix_b[k, n].cast[accum_type]()
                c_val += a_val * b_val
            c_val = epilogue_elemwise_func[accum_type](m, n, c_val)
            matrix_c[m, n] = c_val
            n += 1
        let row = Buffer[Dim(), accum_type](
            c.data.offset(m * gemm_shape.N).address, n
        )
        epilogue_rowise_func[accum_type](m, row)


# ===----------------------------------------------------------------------=== #
# Utilities.
# ===----------------------------------------------------------------------=== #

# Utility to compute inner block size that's divisible
#  by the block size, e.g. simd_size or TileSize.
fn round_down_to_block[block_size: Int](original_size: Int) -> Int:
    """Tile computation utility. Computes the largest multiple of block size
    below original_size.
        e.g. round_down_to_block[128](512+1) = 512

    Args:
        block_size (mlir_index): The block size to round down to.
        original_size (StaticIntTuple): The original size before rounding.

    Returns:
        The rounded data size.
    """
    let num_of_blocks = original_size // block_size
    return num_of_blocks * block_size


# ===----------------------------------------------------------------------=== #
# Packing routines.
# ===----------------------------------------------------------------------=== #


struct PackMatrixRows[
    # original matrix shape list
    original_shape: DimList[2],
    # packed matrix shape list
    packed_shape: DimList[3],
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

    fn __copy__(self) -> Self:
        return Self {
            packed_matrix: self.packed_matrix,
            original_matrix: self.original_matrix,
            global_offset: self.global_offset,
            pack_tile_dim: self.pack_tile_dim,
            valid_data_dim: self.valid_data_dim,
            valid_simd_dim: self.valid_simd_dim,
        }

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
        assert_param_bool[row_inner_size % simd_size == 0]()

        let instance = Self {
            packed_matrix: packed_matrix,
            original_matrix: original_matrix,
            global_offset: global_offset,
            pack_tile_dim: pack_tile_dim,
            valid_data_dim: valid_data_dim,
            valid_simd_dim: Index(
                round_down_to_block[simd_size](
                    min(
                        valid_data_dim[0],
                        pack_tile_dim[0],
                    )
                ),
                round_down_to_block[simd_size](
                    min(
                        valid_data_dim[1],
                        pack_tile_dim[1],
                    )
                ),
            ),
        }

        instance._pack()

    fn _transpose_pack_helper[
        skip_row_bound: Bool,
        skip_col_bound: Bool,
    ](
        self,
        transpose_buffer: NDBuffer[
            2,
            create_dim_list(simd_size, simd_size),
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
        fn body[idx: Int]():
            alias inner_row_idx = idx
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound[0]):
                let row_global_index = Index(
                    start_idx_global[0] + inner_row_idx,
                    start_idx_global[1],
                )
                var row_data: SIMD[simd_size, type]
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
                    row_data = partial_simd_load[simd_size, type](
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
                    Index(inner_row_idx, 0), SIMD[simd_size, type](0)
                )

        unroll[simd_size, body]()

        # Transpose the buffered data
        transpose_inplace[simd_size, simd_size, type](transpose_buffer)

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        @always_inline
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

        var transpose_buffer = NDBuffer[
            2,
            create_dim_list(simd_size, simd_size),
            type,
        ].aligned_stack_allocation[alignof[SIMD[simd_size, type]]()]()

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


struct PackMatrixCols[
    # original matrix shape list
    original_shape: DimList[2],
    # packed matrix shape list
    packed_shape: DimList[3],
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
        assert_param_bool[column_inner_size % simd_size == 0]()
        debug_assert(
            pack_tile_dim[1] % column_inner_size == 0,
            "Unimplemented tile pattern.",
        )

        let instance = Self {
            packed_matrix: packed_matrix,
            original_matrix: original_matrix,
            global_offset: global_offset,
            pack_tile_dim: pack_tile_dim,
            valid_data_dim: valid_data_dim,
        }

        instance._pack()

    fn __copy__(self) -> Self:
        return Self {
            packed_matrix: self.packed_matrix,
            original_matrix: self.original_matrix,
            global_offset: self.global_offset,
            pack_tile_dim: self.pack_tile_dim,
            valid_data_dim: self.valid_data_dim,
        }

    fn _pack_row_helper[
        # Skip column boundary checking in this row.
        skip_col_bound: Bool,
    ](self, tile_row_idx: Int):
        """Helper function:  Packs a tiled row of original matrix into the
        packed buffer, with boundary checking. Boundary checking can be
        statically skipped., based on the parameters.
        Args:
            skip_col_bound(Bool): boundary check on y dimension will be
                skipped if true.
            fill_zero(Bool): the given row will be filled all zero if true.
            tile_row_idx(Int): row index of the row to pack within the tile of
                data to pack.
        """
        alias alignment = alignof[SIMD[simd_size, type]]()
        alias is_row_aligned = original_shape.at[1]().is_multiple[alignment]()

        for col_idx in range(0, self.pack_tile_dim[1], simd_size):
            # Decl the data to fill in packed buffer.
            var data: SIMD[simd_size, type]

            # Calculate global coordinates.
            let global_idx_pair = self.global_offset + Index(
                tile_row_idx, col_idx
            )
            let global_idx = Index(
                global_idx_pair[0],
                global_idx_pair[1],
            )

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
            elif col_idx >= self.valid_data_dim[1]:
                # Starting point out of bound. Fill a zero vector.
                data = SIMD[simd_size, type](0)
            else:
                # Starting point within bound but cannot load a whole
                #  vector. Do a partial load.
                data = partial_simd_load[simd_size, type](
                    self.original_matrix._offset(global_idx),
                    0,
                    self.valid_data_dim[1] - col_idx,
                    SIMD[1, type](0),
                )

            # map to packed index
            let col_idx_outer = col_idx // column_inner_size
            let col_idx_inner = col_idx % column_inner_size
            self.packed_matrix.aligned_simd_store[simd_size, alignment](
                Index(col_idx_outer, tile_row_idx, col_idx_inner),
                data,
            )

    fn _pack_helper[skip_col_bound: Bool](self):
        """Helper function: packs all the rows within the tile of data to pack
        with statical option to skip boundary check.
            Args:
                skip_col_bound: Boundary check on column dimension will be skipped
                    if true.
        """
        let valid_row_count = min(
            self.valid_data_dim[0],
            self.pack_tile_dim[0],
        )

        for row_idx in range(valid_row_count):
            self._pack_row_helper[skip_col_bound](row_idx)

    fn _pack(self):
        """Helper function: packs all the rows within the tile of data to pack"""
        # TODO:
        #  This packing routine can be further peeled and vectorized
        #    but dynamical tiling could cover some of the sub-optimality
        #    here. In a follow up should extend the blocking scheme here.
        @always_inline
        fn pack_unit[static_switch: Bool]():
            self._pack_helper[
                # skip col bound check.
                static_switch
            ]()

        unswitch[pack_unit](self.pack_tile_dim[1] < self.valid_data_dim[1])


struct MatmulInnerLoopBPacked[
    shape_a: DimList[2],
    shape_c: DimList[2],
    packed_shape: DimList[3],
    accum_type: DType,
    value_type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
    prefetch_b_distance: Int,
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
        let instance = Self {
            c: c,
            a: a,
            b_packed: b_packed,
            global_offset: global_offset,
            tile_n_k: tile_n_k,
            c_bound: Index(global_bound.M, global_bound.N)
            - Index(global_offset.M, global_offset.N),
        }
        instance._run_inner_loop()

    fn __copy__(self) -> Self:
        return Self {
            c: self.c,
            a: self.a,
            b_packed: self.b_packed,
            global_offset: self.global_offset,
            tile_n_k: self.tile_n_k,
            c_bound: self.c_bound,
        }

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
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
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.aligned_simd_store[
                simd_size, alignof[SIMD[simd_size, accum_type]]()
            ](
                Index(idx0, idx1 * simd_size),
                SIMD[simd_size, accum_type](0),
            )

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
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
        alias alignment = alignof[SIMD[simd_size, accum_type]]()
        alias is_row_aligned = shape_c.at[1]().is_multiple[alignment]()

        @always_inline
        fn outer_body[idx0: Int, idx1: Int]():
            let global_idx_pair = (
                Index(self.global_offset.M, self.global_offset.N)
                + tile_idx
                + Index(idx0, idx1 * simd_size)
            )
            let global_idx = Index(
                global_idx_pair[0],
                global_idx_pair[1],
            )
            let local_idx = Index(idx0, idx1 * simd_size)

            # Load data from original matrix C.
            var c_data: SIMD[simd_size, accum_type] = 0
            if skip_boundary_check or (
                Index(idx0, idx1 * simd_size + simd_size)
                <= (self.c_bound - tile_idx)
            ):
                # Use simd load if all within bound
                @parameter
                if is_row_aligned:
                    c_data = self.c.aligned_simd_load[simd_size, alignment](
                        global_idx
                    )
                else:
                    c_data = self.c.simd_load[simd_size](global_idx)
            elif (idx0 + tile_idx[0]) < self.c_bound[
                0
            ] and idx1 * simd_size <= self.c_bound[1]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[simd_size, accum_type](
                    self.c._offset(global_idx),
                    0,
                    self.c_bound[1] - tile_idx[1] - idx1 * simd_size,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = SIMD[simd_size, accum_type](0)

            # Store data to local buffer.
            c_local.aligned_simd_store[simd_size, alignment](local_idx, c_data)

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
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
        alias alignment = alignof[SIMD[simd_size, accum_type]]()
        alias is_row_aligned = shape_c.at[1]().is_multiple[alignment]()

        @always_inline
        fn outer_body[idx0: Int, idx1: Int]():
            let global_idx_pair = (
                Index(self.global_offset.M, self.global_offset.N)
                + tile_idx
                + Index(idx0, idx1 * simd_size)
            )
            let global_idx = Index(
                global_idx_pair[0],
                global_idx_pair[1],
            )
            let local_idx = Index(idx0, idx1 * simd_size)

            # Load data from original matrix C.
            var c_data = c_local.aligned_simd_load[simd_size, alignment](
                local_idx
            )

            if skip_boundary_check or (
                Index(idx0, idx1 * simd_size + simd_size)
                <= (self.c_bound - tile_idx)
            ):
                # Use simd store if all within bound
                @parameter
                if is_row_aligned:
                    self.c.aligned_simd_store[simd_size, alignment](
                        global_idx, c_data
                    )
                else:
                    self.c.simd_store[simd_size](global_idx, c_data)
            elif (
                idx0 < (self.c_bound[0] - tile_idx[0])
                and idx1 * simd_size <= self.c_bound[1]
            ):
                # Use partial store if row in bound but col not
                #  in simd bound.
                partial_simd_store[simd_size, accum_type](
                    self.c._offset(global_idx),
                    0,
                    self.c_bound[1] - tile_idx[1] - idx1 * simd_size,
                    c_data,
                )

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

    @adaptive
    fn _accumulate[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
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
        assert_param_bool[a_col_size > 1]()

        # Seek outer indices in packed layout.
        let n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        let global_k = self.global_offset.K + tile_n_k_idx[1]

        # Prefetch B matrix.
        @parameter
        if prefetch_b_distance > 0:

            @always_inline
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
        fn _do[idx: Int]():
            alias idx_outer = idx
            let global_m = self.global_offset.M + idx_outer
            let a_val = self.a.simd_load[a_col_size](global_m, global_k).cast[
                accum_type
            ]()

            @always_inline
            fn outer_body[idx0: Int, idx1: Int]():
                let b_val = self.b_packed.simd_load[simd_size](
                    n_outer_idx,
                    tile_n_k_idx[1] + idx0,
                    idx1 * simd_size,
                ).cast[accum_type]()

                let c_idx = Index(idx_outer, idx1 * simd_size)
                var c_val = c_local.simd_load[simd_size](c_idx)

                c_val = fma[simd_size, accum_type](a_val[idx0], b_val, c_val)
                c_local.simd_store[simd_size](c_idx, c_val)

            unroll2[a_col_size, pack_inner_size // simd_size, outer_body]()

        unroll[a_row_size, _do]()

    @adaptive
    fn _accumulate[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
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
        assert_param_bool[a_col_size == 1]()
        # Seek outer indices in packed layout.
        let n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        let global_k = self.global_offset.K + tile_n_k_idx[1]

        # Prefetch B matrix.
        # TODO(#10919): Use `@parameter` if here, there is a bug where invalid
        # code is generated and accuracy is not maintained.
        if prefetch_b_distance > 0:

            @always_inline
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
        fn outer_body[idx0: Int, idx1: Int]():
            alias alignment = alignof[SIMD[simd_size, accum_type]]()
            let global_m = self.global_offset.M + idx0
            let c_idx = Index(idx0, idx1 * simd_size)

            let b_val = self.b_packed.aligned_simd_load[simd_size, alignment](
                n_outer_idx, tile_n_k_idx[1], idx1 * simd_size
            ).cast[accum_type]()

            let a_val = self.a.simd_load[1](global_m, global_k).cast[
                accum_type
            ]()

            var c_val = c_local.aligned_simd_load[simd_size, alignment](c_idx)

            c_val = fma[simd_size, accum_type](a_val, b_val, c_val)
            c_local.aligned_simd_store[simd_size, alignment](c_idx, c_val)

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

    @adaptive
    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        assert_param_bool[not has_neon()]()
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
            accum_type,
        ].aligned_stack_allocation[alignof[SIMD[simd_size, accum_type]]()]()

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
        assert_param_bool[has_neon()]()
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            2,
            create_dim_list(a_row_size, pack_inner_size),
            accum_type,
        ].aligned_stack_allocation[alignof[SIMD[simd_size, accum_type]]()]()

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


# Helper heuristic function to decide on tile size
#  Returns (TileN, TileK)
fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: Int,
    # Inner size of data layout.
    pack_inner_size: Int,
](n: Int, k: Int) -> StaticIntTuple[2]:
    """Helper heuristic function to decide on tile size to partition the matmul
    given the cache size and desired data layout.
        Args:
            pack_cache_size: Allocated space for packing elements, configuring as a
                function of target cache size desired.
            pack_inner_size: The desired inner dimension of the packed data
                layout.
            gemm_shape: The shape of the matmul problem size based on runtime
                input.
        Returns:
            The calculated tile size to partition the matmul as (TileN, TileK)
    """

    # Make sure outer dimension is at least 2
    let least_tile_n: Int = pack_inner_size * 2

    # Max tile K size based on smallest Tile N.
    let largest_tile_k = pack_cache_size // least_tile_n

    # Prioritize shape on K dimension, so try to fit in the whole
    #  input on the tile.
    let tile_k = min(largest_tile_k, k)

    # Calculate number of InnerSize to fit in tile_n dimension,
    #  guranteed to be at least 2.
    let max_tile_n_in_inner_size = pack_cache_size // tile_k // pack_inner_size
    let full_data_tile_n_in_inner_size = div_ceil(n, pack_inner_size)
    let tile_n_in_inner_size = min(
        max_tile_n_in_inner_size, full_data_tile_n_in_inner_size
    )

    # Calculate tile_n size.
    let tile_n = tile_n_in_inner_size * pack_inner_size

    return Index(tile_n, tile_k)


fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: Int,
    # Inner size of data layout.
    pack_inner_size: Int,
](global_tile_shape: GemmShape) -> StaticIntTuple[2]:
    return calculate_tile_n_k[pack_cache_size, pack_inner_size](
        global_tile_shape.N, global_tile_shape.K
    )


# Tiled Matmul Implementation.
# TODO: not yet supporting transpose_a
struct TiledMatmul[
    config: MatmulConfig,
    accum_type: DType,
    value_type: DType,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
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

    fn __copy__(self) -> Self:
        return Self {
            c: self.c,
            a: self.a,
            b: self.b,
            tile_n_k: self.tile_n_k,
            global_tile_shape: self.global_tile_shape,
            global_tile_offset: self.global_tile_offset,
            b_tile_generator: self.b_tile_generator,
        }

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[2, config.shape_c, accum_type],
        a: NDBuffer[2, config.shape_a, value_type],
        b: NDBuffer[2, config.shape_b, value_type],
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape = GemmShape {M: 0, N: 0, K: 0},
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
            config, accum_type, value_type, transpose_a, transpose_b, b_packed
        ] {
            c: c,
            a: a,
            b: b,
            tile_n_k: tile_n_k,
            global_tile_offset: global_tile_offset,
            global_tile_shape: global_tile_shape,
            b_tile_generator: BTileGenerator[
                config, value_type, transpose_b, b_packed
            ].get(b, tile_n_k),
        }

        matmul._run()

    fn _outer_m_loop_helper[
        skip_col_bound: Bool, m_loop_pack_inner_size: Int
    ](self, global_offset: GemmShape, sub_tile_n: Int, sub_tile_k: Int,):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

            Args:
                skip_col_bound(i1): Column dimension boundary check will be
                    statically skipped if true.
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
            Index(
                self.global_tile_shape.K,
                self.global_tile_shape.N,
                self.global_tile_shape.M,
            )
            + Index(
                self.global_tile_offset.K,
                self.global_tile_offset.N,
                self.global_tile_offset.M,
            )
            - Index(global_offset.K, global_offset.N, global_offset.M)
        )

        let b_packed_tile = self.b_tile_generator.get_tile[
            m_loop_pack_inner_size
        ](
            global_offset,
            Index(sub_tile_n, sub_tile_k),
            Index(knm_bounds[1], knm_bounds[0]),
        )

        # Launch the MLoop
        let sub_tile_n_k = Index(
            min(sub_tile_n, knm_bounds[1]), min(sub_tile_k, knm_bounds[0])
        )

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
            ].run(
                self.c,
                self.a,
                b_packed_tile,
                global_offset + GemmShape(row_offset, 0, 0),
                self.global_tile_offset + self.global_tile_shape,
                sub_tile_n_k,
            )

        tile[row_iteration, VariadicList[Int](config.a_row_size, 4, 3, 2, 1)](
            0,  # starting row offset
            knm_bounds[2],  # row bound
        )

    #  Pack a subtile of B and iterate through all the rows of C.
    fn _outer_m_loop[
        m_loop_pack_inner_size: Int
    ](self, global_offset: GemmShape, sub_tile_n: Int, sub_tile_k: Int,):
        """Pack a subtile of B and iterate through all the rows
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
        let valid_col_count = (
            self.global_tile_shape.N
            + self.global_tile_offset.N
            - global_offset.N
        )

        # The whole subtile is within valid columns,
        #  no need to check boundary when loading C
        #  on this tile.
        # TODO: this could be in finer granularity.
        @always_inline
        fn unswitched_mloop[static_switch: Bool]():
            self._outer_m_loop_helper[
                static_switch,  # skip_col_bound
                m_loop_pack_inner_size,
            ](global_offset, sub_tile_n, sub_tile_k)

        unswitch[unswitched_mloop](valid_col_count >= sub_tile_n)

    # Iterate on the N dimension of the gemm space.
    fn _outer_n_loop(
        self,
        global_offset: GemmShape,
        sub_tile_k: Int,
    ):
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

        @always_inline
        fn m_loop[secondary_tile_size: Int](col_idx: Int, tile_size_n: Int):
            self._outer_m_loop[secondary_tile_size](
                global_offset + GemmShape(0, col_idx, 0),
                tile_size_n,
                sub_tile_k,
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
        fn k_iteration(k_offset: Int, k_tile_size: Int):
            self._outer_n_loop(
                GemmShape(0, 0, k_offset) + self.global_tile_offset,
                k_tile_size,
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
            create_dim_list(tile_n // n_inner_size, tile_k, n_inner_size),
            value_type,
        )

    fn _run(self):
        """Wrapper utility function: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        self._outer_k_loop()


fn pack_b[
    transpose_b: Bool,
    simd_size: Int,
    inner_size: Int,
    type: DType,
    src_shape: DimList[2],
    dst_shape: DimList[2],
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
                    3, DimList[3].create_unknown(), type
                ](
                    dst_flat.data.offset(dst_offset),
                    create_dim_list(tile_n // inner_size, tile_k, inner_size),
                    type,
                )
                let valid_k = min(tile_k, k_in - idx_k)
                let valid_n = min(tile_n, n_in - idx_n)
                PackMatrixCols[
                    src_shape,
                    DimList[3].create_unknown(),
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
                    3, DimList[3].create_unknown(), type
                ](
                    dst_flat.data.offset(dst_offset),
                    create_dim_list(tile_n // inner_size, tile_k, inner_size),
                    type,
                )
                let valid_k_t = min(tile_k, k_in_t - idx_k_t)
                let valid_n_t = min(tile_n, n_in_t - idx_n_t)
                PackMatrixRows[
                    src_shape,
                    DimList[3].create_unknown(),
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
                alignof[SIMD[dtype_simd_width[type](), type]](),
            ]()

        return BTileGenerator[config, type, transpose_b, b_packed] {
            b: b, b_tile_stack_ptr: b_tile_stack_ptr, tile_n_k: tile_n_k
        }

    fn __copy__(self) -> Self:
        return Self {
            b: self.b,
            b_tile_stack_ptr: self.b_tile_stack_ptr,
            tile_n_k: self.tile_n_k,
        }

    fn get_tile[
        inner_size: Int
    ](
        self,
        global_offset: GemmShape,
        tile_dim_nk: StaticIntTuple[2],
        valid_data_dim_nk: StaticIntTuple[2],
    ) -> NDBuffer[3, config.packed_shape, type]:
        let tile_shape_nopack = create_dim_list(
            tile_dim_nk[0] // inner_size, tile_dim_nk[1], inner_size
        )
        let packed_b = NDBuffer[3, config.packed_shape, type](
            self.b_tile_stack_ptr,
            tile_shape_nopack,
            type,
        )

        @parameter
        if transpose_b & (not b_packed):
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
        elif (not transpose_b) & (not b_packed):
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
        elif b_packed & (not transpose_b):
            # Need to use tile_k that generator was initialized with.
            # When packing is done online, tile_dim_nk can vary in each call to
            # get_tile (if handling a residual K tile), but packing assumes that
            # tile_k is constant.
            let tile_shape_pack = create_dim_list(
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
