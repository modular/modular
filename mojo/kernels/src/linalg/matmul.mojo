# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_bool, debug_assert

from Buffer import (
    NDBuffer,
    Buffer,
    partial_simd_load,
    partial_simd_store,
    _raw_stack_allocation,
)
from BuildInfo import is_relwithdebinfo_build, is_debug_build
from Index import Index, StaticIntTuple
from Int import Int
from List import create_kgen_list, VariadicList
from Matrix import Matrix
from Memory import stack_allocation
from Pointer import DTypePointer
from Range import range
from SIMD import SIMD
from TargetInfo import simd_byte_width
from Transpose import transpose_inplace
from Tuple import StaticTuple
from Functional import tile, unswitch


fn get_pack_data_size() -> Int:
    """Utility to compute the number of elements to pack in each tile.
    Returns:
        The number of elements to pack.
    """

    if is_relwithdebinfo_build() or is_debug_build():
        # Only use the large cache size for release build as debug build may
        #  contain additional data could cause stack overflow.
        return 1024
    return 131_072


@register_passable
struct MatmulConfig:
    """Static configuration of tiled matmul algorithms."""

    # Static shape info of Operand A.
    var shape_a: __mlir_type[`!kgen.list<index[2]>`]

    # Static shape info of Operand B.
    var shape_b: __mlir_type[`!kgen.list<index[2]>`]

    # Static shape info of Operand C.
    var shape_c: __mlir_type[`!kgen.list<index[2]>`]

    # Static packed shape info of the packed buffer.
    var packed_shape: __mlir_type[`!kgen.list<index[3]>`]

    # Static info on simd vector size.
    var simd_size: __mlir_type.index

    # Static loop unrolling size on M dimension.
    var a_row_size: __mlir_type.index

    # Static inner dimension of packed data layout.
    var pack_inner_size: __mlir_type.index

    # Static info on number of elements to pack in the packing routine.
    var pack_data_size: __mlir_type.index


@register_passable
struct GemmShape:
    """Helper class to unpack gemm dimension and layout."""

    var M: Int
    var N: Int
    var K: Int

    # Construct from dynamic shaped input.
    @staticmethod
    fn get[
        shape_c: __mlir_type[`!kgen.list<index[2]>`],
        shape_a: __mlir_type[`!kgen.list<index[2]>`],
        shape_b: __mlir_type[`!kgen.list<index[2]>`],
        accum_type: __mlir_type.`!kgen.dtype`,
        value_type: __mlir_type.`!kgen.dtype`,
        transpose_a: Bool,
        transpose_b: Bool,
    ](
        c: NDBuffer[2, shape_c, accum_type],
        a: NDBuffer[2, shape_a, value_type],
        b: NDBuffer[2, shape_b, value_type],
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
        if transpose_a:
            gemm_shape.K = a.dim[0]()
        else:
            gemm_shape.K = a.dim[1]()
        return gemm_shape

    fn __clone__(self&) -> Self:
        return Self {M: self.M, N: self.N, K: self.K}

    fn __new__(m: Int, n: Int, k: Int) -> GemmShape:
        """Constructor of a gemm shape record by directly supplying the values.

        Args:
            m: M dimension of the gemm shape.
            n: N dimension of the gemm shape.
            k: K dimension of the gemm shape.

        Returns:
            The constructed shape record.
        """
        return GemmShape {M: m, N: n, K: k}

    fn __new__(index: StaticIntTuple[3]) -> GemmShape:
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
    shape_a: __mlir_type[`!kgen.list<index[2]>`],
    shape_b: __mlir_type[`!kgen.list<index[2]>`],
    shape_c: __mlir_type[`!kgen.list<index[2]>`],
    accum_type: __mlir_type.`!kgen.dtype`,
    value_type: __mlir_type.`!kgen.dtype`,
    transpose_a: Bool,
    transpose_b: Bool,
    epilogue_elemwise_func: __mlir_type[
        `!kgen.signature<<accum_type: dtype>(`,
        Int,  # Row
        `,`,
        Int,  # Col
        `,`,
        SIMD[1, `accum_type`],
        `) -> `,
        SIMD[1, `accum_type`],
        `>`,
    ],
    epilogue_rowise_func: __mlir_type[
        `!kgen.signature<<accum_type: dtype>(`,
        Int,  # Row
        `,`,
        Buffer[
            __mlir_attr.`#kgen.unknown : index`,
            `accum_type`,
        ],
        `) -> !lit.none>`,
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
        shape_c,
        shape_a,
        shape_b,
        accum_type,
        value_type,
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
                let a_val = matrix_a.__getitem__(m, k).cast[accum_type]()
                let b_val = matrix_b.__getitem__(k, n).cast[accum_type]()
                c_val += a_val * b_val
            c_val = epilogue_elemwise_func[accum_type](m, n, c_val)
            matrix_c.__setitem__(m, n, c_val)
            n += 1
        let row = Buffer[__mlir_attr.`#kgen.unknown : index`, accum_type](
            c.data.offset(m * gemm_shape.N).address, n
        )
        epilogue_rowise_func[accum_type](m, row)


# ===----------------------------------------------------------------------=== #
# Utilities.
# ===----------------------------------------------------------------------=== #

# Utility to compute inner block size that's divisible
#  by the block size, e.g. simd_size or TileSize.
fn round_down_to_block[
    block_size: __mlir_type.index
](original_size: Int) -> Int:
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
    original_shape: __mlir_type[`!kgen.list<index[2]>`],
    # packed matrix shape list
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    row_inner_size: __mlir_type.index,
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

    fn __clone__(self&) -> Self:
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
        assert_param[row_inner_size % simd_size == 0]()

        let instance = Self {
            packed_matrix: packed_matrix,
            original_matrix: original_matrix,
            global_offset: global_offset,
            pack_tile_dim: pack_tile_dim,
            valid_data_dim: valid_data_dim,
            valid_simd_dim: Index(
                round_down_to_block[simd_size](
                    Int.min(
                        valid_data_dim[0],
                        pack_tile_dim[0],
                    )
                ),
                round_down_to_block[simd_size](
                    Int.min(
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
            __mlir_attr[
                create_kgen_list[__mlir_type.index](simd_size, simd_size),
                __mlir_type.index,
                `[2]>`,
            ],
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
               transepose_buffer(NDBuffer): pre-allocated work space to hold
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
        for inner_row_idx in range(simd_size):
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound[0]):
                let row_gloal_index = Index(
                    start_idx_global[0] + inner_row_idx,
                    start_idx_global[1],
                )
                var row_data: SIMD[simd_size, type]
                if skip_col_bound:
                    # This is fastest path where both row and col bounds
                    #  are skipped so the code path is simd-in and simd-out
                    #  without any predicate.
                    row_data = self.original_matrix.simd_load[simd_size](
                        row_gloal_index
                    )
                else:
                    # Not skipping col bound, need to to a partial fill of
                    #  the transpose buffer row.
                    row_data = partial_simd_load[simd_size, type](
                        self.original_matrix._offset(row_gloal_index),
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

        # Transpose the buffered data
        transpose_inplace[simd_size, simd_size, type](transpose_buffer)

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        for transposed_inner_row_idx in range(simd_size):
            let transposed_data = transpose_buffer.simd_load[simd_size](
                Index(transposed_inner_row_idx, 0)
            )
            # compute the packed index
            let _row_outer = local_off_set[0] // row_inner_size
            let _row_inner = Int.remu(local_off_set[0], row_inner_size)

            if skip_col_bound or (transposed_inner_row_idx < write_bound[1]):
                self.packed_matrix.simd_store[simd_size](
                    Index(
                        _row_outer,
                        local_off_set[1] + transposed_inner_row_idx,
                        _row_inner,
                    ),
                    transposed_data,
                )
            # Out of bound columns are discarded as there's no allocation for them
            #  in the packed buffer.

    fn _pack(self):
        """Helper function: Allocates transpose workspace and launch the
        transpose helper function until all required data has been packed.
        """

        var transpose_buffer = NDBuffer[
            2,
            __mlir_attr[
                create_kgen_list[__mlir_type.index](simd_size, simd_size),
                __mlir_type.index,
                `[2]>`,
            ],
            type,
        ].aligned_stack_allocation[simd_byte_width().__as_mlir_index()]()

        let valid_tile_simd_dim = Index(
            Int.min(
                self.valid_simd_dim[0],
                self.pack_tile_dim[0],
            ),
            Int.min(
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
    original_shape: __mlir_type[`!kgen.list<index[2]>`],
    # packed matrix shape list
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    column_inner_size: __mlir_type.index,
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

        let instance = Self {
            packed_matrix: packed_matrix,
            original_matrix: original_matrix,
            global_offset: global_offset,
            pack_tile_dim: pack_tile_dim,
            valid_data_dim: valid_data_dim,
        }

        instance._pack()

    fn __clone__(self&) -> Self:
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
        # Fill all zero for this row.
        fill_zero: Bool,
    ](self, tile_row_idx: Int):
        """Helper function:  Packs a tiled row of original matrix into the
        packed buffer, with boundary checking. Boundary checking can be
        statically skipped., based on the parameters.
        Args:
            skip_col_bound(Bool): boundary check on y dimension will be
                skpped if true.
            fill_zero(Bool): the given row will be filled all zero if true.
            tile_row_idx(Int): row index of the row to pack within the tile of
                data to pack.
        """
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

            if fill_zero:
                # Statical fill zero case.
                data = SIMD[simd_size, type](0)
            elif skip_col_bound or (
                col_idx + simd_size <= self.valid_data_dim[1]
            ):
                # Whole SIMD vector within bound.
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
            let col_idx_inner = Int.remu(col_idx, column_inner_size)
            self.packed_matrix.simd_store[simd_size](
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
        var row_idx: Int = 0
        let valid_row_count = Int.min(
            self.valid_data_dim[0],
            self.pack_tile_dim[0],
        )

        @always_inline
        fn pack_unit[static_switch: Bool]():
            @parameter
            if static_switch:
                self._pack_row_helper[skip_col_bound, False](row_idx)
            else:
                self._pack_row_helper[True, False](row_idx)

        # Fill zero on the remaining rows on the tile.
        while row_idx < self.pack_tile_dim[0]:
            unswitch[pack_unit](row_idx < valid_row_count)
            row_idx += 1

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
    shape_a: __mlir_type[`!kgen.list<index[2]>`],
    shape_c: __mlir_type[`!kgen.list<index[2]>`],
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    accum_type: __mlir_type.`!kgen.dtype`,
    value_type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    a_row_size: __mlir_type.index,
    pack_inner_size: __mlir_type.index,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
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
        var instance = Self {
            c: c,
            a: a,
            b_packed: b_packed,
            global_offset: global_offset,
            tile_n_k: tile_n_k,
            c_bound: Index(global_bound.M, global_bound.N)
            - Index(global_offset.M, global_offset.N),
        }
        instance._run_inner_loop()

    fn __clone__(self&) -> Self:
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
            create_kgen_list[__mlir_type.index](a_row_size, pack_inner_size),
            accum_type,
        ],
    ):
        """Utility funcion on the inner loop. Initializes a local c buffer with
        all zeros.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
        """
        for row_idx in range(a_row_size):
            for col_idx in range(0, pack_inner_size, simd_size):
                c_local.simd_store[simd_size](
                    Index(row_idx, col_idx),
                    SIMD[simd_size, accum_type](0),
                )

    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](a_row_size, pack_inner_size),
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
        for row_idx in range(a_row_size):
            for col_idx in range(0, pack_inner_size, simd_size):
                let global_idx_pair = (
                    Index(self.global_offset.M, self.global_offset.N)
                    + tile_idx
                    + Index(row_idx, col_idx)
                )
                let global_idx = Index(
                    global_idx_pair[0],
                    global_idx_pair[1],
                )
                let local_idx = Index(row_idx, col_idx)

                # Load data from original matrix C.
                var c_data: SIMD[simd_size, accum_type] = 0
                if skip_boundary_check or (
                    Index(row_idx, col_idx + simd_size)
                    <= (self.c_bound - tile_idx)
                ):
                    # Use simd load if all within bound
                    c_data = self.c.simd_load[simd_size](global_idx)
                elif (row_idx + tile_idx[0]) < self.c_bound[0]:
                    # Use partial load if row inbound but col not
                    #  in simd bound.
                    c_data = partial_simd_load[simd_size, accum_type](
                        self.c._offset(global_idx),
                        0,
                        self.c_bound[1] - tile_idx[1] - col_idx,
                        0,
                    )
                else:
                    # Fill zero if row out of bound
                    c_data = SIMD[simd_size, accum_type](0)

                # Store data to local buffer.
                c_local.simd_store[simd_size](local_idx, c_data)

    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](a_row_size, pack_inner_size),
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
        for row_idx in range(a_row_size):
            for col_idx in range(0, pack_inner_size, simd_size):
                let global_idx_pair = (
                    Index(self.global_offset.M, self.global_offset.N)
                    + tile_idx
                    + Index(row_idx, col_idx)
                )
                let global_idx = Index(
                    global_idx_pair[0],
                    global_idx_pair[1],
                )
                let local_idx = Index(row_idx, col_idx)

                # Load data from original matrix C.
                var c_data = c_local.simd_load[simd_size](local_idx)

                if skip_boundary_check or (
                    Index(row_idx, col_idx + simd_size)
                    <= (self.c_bound - tile_idx)
                ):
                    # Use simd store if all within bound
                    self.c.simd_store[simd_size](global_idx, c_data)
                elif row_idx < (self.c_bound[0] - tile_idx[0]):
                    # Use partial store if row in bound but col not
                    #  in simd bound.
                    partial_simd_store[simd_size, accum_type](
                        self.c._offset(global_idx),
                        0,
                        self.c_bound[1] - tile_idx[1] - col_idx,
                        c_data,
                    )

    fn _accumulate(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](a_row_size, pack_inner_size),
            accum_type,
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Launch one tile of fma on the
        local accumulation buffer.

        Args:
            c_local(NDBuffer): pre-allocated local buffer for c partial
                sums.
            tile_n_k_idx(StaticIntTuple): index tuple with (n, k)
                coordinates within the current processing tile to index the
                packed B matrix.
        """
        # Seek outer indices in packed layout.
        let n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx[1]

        # Loop over local accumulator tiles.
        for col_idx in range(0, pack_inner_size, simd_size):
            let b_val = self.b_packed.simd_load[simd_size](
                Index(n_outer_idx, tile_n_k_idx[1], col_idx)
            ).cast[accum_type]()
            for row_idx in range(a_row_size):
                var global_m = self.global_offset.M + row_idx
                let a_val_scalar = self.a.simd_load[1](
                    Index(global_m, global_k)
                )
                let a_val = SIMD[simd_size, value_type](a_val_scalar).cast[
                    accum_type
                ]()

                var c_idx = Index(row_idx, col_idx)
                var c_val = c_local.simd_load[simd_size](c_idx)

                c_val = a_val.fma(b_val, c_val)
                c_local.simd_store[simd_size](c_idx, c_val)

    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](a_row_size, pack_inner_size),
            accum_type,
        ].stack_allocation()

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
                self._accumulate(c_local, Index(idx_n, idx_k))

            self._store_c_tile(c_local, Index(0, idx_n))


# Helper heuristic function to decide on tile size
#  Returns (TileN, TileK)
fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: __mlir_type.index,
    # Inner size of data layout.
    pack_inner_size: __mlir_type.index,
](gemm_shape: GemmShape) -> StaticIntTuple[2]:
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
    let largest_tile_k = Int(pack_cache_size) // least_tile_n

    # Prioritize shape on K dimension, so try to fit in the whole
    #  input on the tile.
    let tile_k = Int.min(largest_tile_k, gemm_shape.K)

    # Calculate number of InnerSize to fit in tile_n dimension,
    #  guranteed to be at least 2.
    let max_tile_n_in_inner_size = (
        Int(pack_cache_size) // tile_k // pack_inner_size
    )
    let full_data_tile_n_in_inner_size = (
        gemm_shape.N + pack_inner_size - 1
    ) // pack_inner_size
    let tile_n_in_inner_size = Int.min(
        max_tile_n_in_inner_size, full_data_tile_n_in_inner_size
    )

    # Calculate tile_n size.
    let tile_n = tile_n_in_inner_size * pack_inner_size

    return Index(tile_n, tile_k)


# Tiled Matmul Implementation.
# TODO: not yet supporting transpose_a
struct TiledMatmul[
    config: MatmulConfig,
    accum_type: __mlir_type.`!kgen.dtype`,
    value_type: __mlir_type.`!kgen.dtype`,
    transpose_a: Bool,
    transpose_b: Bool,
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

    fn __clone__(self&) -> Self:
        return Self {
            c: self.c,
            a: self.a,
            b: self.b,
            tile_n_k: self.tile_n_k,
            global_tile_shape: self.global_tile_shape,
            global_tile_offset: self.global_tile_offset,
        }

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[2, config.shape_c, accum_type],
        a: NDBuffer[2, config.shape_a, value_type],
        b: NDBuffer[2, config.shape_b, value_type],
    ):
        """Interface function to run tiled matmul on a given set of operands,
        pre-allocated output space and data layout tag.

        Args:
            c(NDBuffer): Pre-allocated buffer space for result.
            a(NDBuffer): Operand A of the matmul.
            b(NDBuffer): Operand B of the mamtul.
            transpose_a: True if a is in transposed layout.
            transpose_b: True if b is in transposed layout.
        """

        let global_tile_shape = GemmShape.get[
            config.shape_c,
            config.shape_a,
            config.shape_b,
            accum_type,
            value_type,
            transpose_a,
            transpose_b,
        ](c, a, b)

        Self.run(c, a, b, GemmShape(0, 0, 0), global_tile_shape)

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[2, config.shape_c, accum_type],
        a: NDBuffer[2, config.shape_a, value_type],
        b: NDBuffer[2, config.shape_b, value_type],
        global_tile_offset: GemmShape,
        global_tile_shape: GemmShape,
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
        ] {
            c: c,
            a: a,
            b: b,
            tile_n_k: tile_n_k,
            global_tile_offset: global_tile_offset,
            global_tile_shape: global_tile_shape,
        }

        matmul._run()

    fn _outer_m_loop_helper[
        skip_col_bound: Bool, m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, config.packed_shape, value_type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
    ):
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
        # pack B:
        if transpose_b:
            PackMatrixRows[
                config.shape_b,
                config.packed_shape,
                value_type,
                config.simd_size,
                m_loop_pack_inner_size,
            ].run(
                b_packed,
                self.b,
                # Input is [N, K]:
                # Starting global offset for packing.
                Index(global_offset.N, global_offset.K),
                Index(sub_tile_n, sub_tile_k),
                # Valid amount of input from the starting offset.
                Index(self.global_tile_shape.N, self.global_tile_shape.K)
                + Index(self.global_tile_offset.N, self.global_tile_offset.K)
                - Index(global_offset.N, global_offset.K),
            )
        else:
            PackMatrixCols[
                config.shape_b,
                config.packed_shape,
                value_type,
                config.simd_size,
                m_loop_pack_inner_size,
            ].run(
                b_packed,
                self.b,
                # Input is [K, N]:
                # Starting global offset for packing.
                Index(global_offset.K, global_offset.N),
                Index(sub_tile_k, sub_tile_n),
                # Valid amount of input from the starting offset.
                Index(self.global_tile_shape.K, self.global_tile_shape.N)
                + Index(self.global_tile_offset.K, self.global_tile_offset.N)
                - Index(global_offset.K, global_offset.N),
            )

        # Launch the MLoop
        let sub_tile_n_k = Index(sub_tile_n, sub_tile_k)
        let valid_row_count = (
            self.global_tile_shape.M
            + self.global_tile_offset.M
            - global_offset.M
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
                tile_size.__as_mlir_index(),
                m_loop_pack_inner_size,
                skip_col_bound,
            ].run(
                self.c,
                self.a,
                b_packed,
                global_offset + GemmShape(row_offset, 0, 0),
                self.global_tile_offset + self.global_tile_shape,
                sub_tile_n_k,
            )

        tile[row_iteration, VariadicList[Int](config.a_row_size, 4, 3, 2, 1)](
            0,  # starting row offset
            valid_row_count,  # row bound
        )

    #  Pack a subtile of B and iterate through all the rows of C.
    fn _outer_m_loop[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, config.packed_shape, value_type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
    ):
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
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)

        unswitch[unswitched_mloop](valid_col_count >= sub_tile_n)

    # Helper function:
    #  Iterate on the N dimension by steps of
    # size sub_tile_n with no crossing valid boundary.
    fn _outer_n_loop_helper[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, config.packed_shape, value_type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
        start_idx: Int,
        valid_col_count: Int,
    ) -> Int:
        """Helper function: Iterate on the N dimension by steps of size
            sub_tile_n without crossing valid boundary.

        Args:
            m_loop_pack_inner_size(index): Inner dimension of the packed data
                layout.
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_n(Int): Dynamic tile size to use on N dimension.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
            start_idx(Int): Starting index on N dimension.
            valid_col_count(Int): Number of valid columns remaining on the
                current processing tile.
        """
        var col_idx = start_idx
        while col_idx <= (valid_col_count - sub_tile_n):
            self._outer_m_loop[m_loop_pack_inner_size](
                b_packed,
                global_offset + GemmShape(0, col_idx, 0),
                sub_tile_n,
                sub_tile_k,
            )
            col_idx += sub_tile_n
        return col_idx

    # Iterate on the N dimension of the gemm space.
    fn _outer_n_loop(
        self,
        b_packed: NDBuffer[3, config.packed_shape, value_type],
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

        # Remap buffer indices for current tile.
        var remapped_bpacked = self._view_buffer_as(
            b_packed.data, tile_n, sub_tile_k, Int(config.pack_inner_size)
        )

        var col_idx: Int = 0
        # Proceed with the large tile:
        col_idx = self._outer_n_loop_helper[config.pack_inner_size](
            remapped_bpacked,
            global_offset,
            tile_n,
            sub_tile_k,
            col_idx,
            valid_col_count,
        )

        # Cover residual tiles.
        if col_idx < valid_col_count:
            remapped_bpacked = self._view_buffer_as(
                b_packed.data,
                config.simd_size,
                sub_tile_k,
                config.simd_size,
            )
            col_idx = self._outer_n_loop_helper[config.simd_size](
                remapped_bpacked,
                global_offset,
                config.simd_size,
                sub_tile_k,
                col_idx,
                valid_col_count,
            )

        # Cover the last sub simdsize tile:
        # This call will handle the sub-simd size boundary.
        if col_idx < valid_col_count:
            self._outer_m_loop[config.simd_size](
                remapped_bpacked,
                global_offset + GemmShape(0, col_idx, 0),
                config.simd_size,
                sub_tile_k,
            )

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(
        self, b_packed: NDBuffer[3, config.packed_shape, value_type]
    ):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        # Each tiled iteration on the k dimension.
        @always_inline
        fn k_iteration(k_offset: Int, k_tile_size: Int):
            self._outer_n_loop(
                b_packed,
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
        b_packed: DTypePointer[value_type],
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
            b_packed.address,
            create_kgen_list[__mlir_type.index](
                (tile_n // n_inner_size).__as_mlir_index(),
                tile_k.__as_mlir_index(),
                n_inner_size.__as_mlir_index(),
            ),
            value_type,
        )

    fn _run(self):
        """Wrapper utility funciton: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        # Allocate pack_b buffer.
        let _bpacked_data = _raw_stack_allocation[
            config.pack_data_size,  # Count.
            value_type,  # Data type.
            simd_byte_width().__as_mlir_index(),  # Alignment.
        ]()

        # Manually set the shape of packed B buffer:
        let mapped_bpacked = self._view_buffer_as(
            _bpacked_data,
            self.tile_n_k[0],
            self.tile_n_k[1],
            config.pack_inner_size,
        )
        self._outer_k_loop(mapped_bpacked)
