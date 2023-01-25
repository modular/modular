# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#  This file contains matmul kernel implementation details and utilities.
# ===----------------------------------------------------------------------=== #

from Buffer import (
    NDBuffer,
    Buffer,
    partial_simd_load,
    partial_simd_store,
    _raw_stack_allocation,
)
from MemoryUtilities import stack_allocation
from Bool import Bool
from Int import Int
from SIMD import SIMD
from Tuple import StaticTuple
from Pointer import DTypePointer
from Assert import assert_param, assert_param_bool
from Transpose import transpose_inplace
from Index import Index, StaticIntTuple
from TargetInfo import simd_byte_width
from List import create_kgen_list
from BuildInfo import is_debug_build


@interface
fn get_pack_data_size() -> Int:
    """Utility to compute the number of elements to pack in each tile.
    Returns:
        The number of elements to pack.
    """
    ...


@implements(get_pack_data_size)
fn get_pack_data_size_debug() -> Int:
    """Pack element counts in debug build. Use a small number to avoid
    stack overflow in asan builds.
    """
    assert_param_bool[is_debug_build()]
    return 1024


@implements(get_pack_data_size)
fn get_pack_data_size_release() -> Int:
    assert_param_bool[not is_debug_build()]
    """Pack element counts. Use a number that's proportion to the cache size.
    """
    # PackCacheSize (hard code to 512kB of f32,
    #  TODO: integrate cache size in sysinfo)
    return 131_072


struct GemmShape:
    """Helper class to unpack gemm dimension and layout."""

    var M: Int
    var N: Int
    var K: Int

    # Construct from dynamic shaped input.
    fn __new__[
        shape_c: __mlir_type[`!kgen.list<index[2]>`],
        shape_a: __mlir_type[`!kgen.list<index[2]>`],
        shape_b: __mlir_type[`!kgen.list<index[2]>`],
        type: __mlir_type.`!kgen.dtype`,
    ](
        c: NDBuffer[2, shape_c, type],
        a: NDBuffer[2, shape_a, type],
        b: NDBuffer[2, shape_b, type],
        transpose_a: Bool,
        transpose_b: Bool,
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
            index.__getitem__[0](),
            index.__getitem__[1](),
            index.__getitem__[2](),
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


struct _Matrix[
    shape: __mlir_type[`!kgen.list<index[2]>`],
    type: __mlir_type.`!kgen.dtype`,
    transposed: Bool,
]:
    """Utility to access matrix across layouts with
    unified indexing interface.
    """

    var data: NDBuffer[2, shape, type]

    fn __new__(
        data: NDBuffer[2, shape, type]
    ) -> _Matrix[shape, type, transposed]:
        """Constructor of a matrix based on a buffer and a transpose flag.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """

        return _Matrix[shape, type, transposed] {data: data}

    fn __getitem__(self, x: Int, y: Int) -> SIMD[1, type]:
        """Returns the data stored at the given untransposed coordinate.

        Args:
            x: The untransposed x coordinate.
            y: The untransposed y coordinate.

        Returns:
            The value stored at the coordinate.
        """
        if transposed:
            return self.data.__getitem__(Index(y, x).as_tuple())
        return self.data.__getitem__(Index(x, y).as_tuple())

    fn __setitem__(self, x: Int, y: Int, val: SIMD[1, type]):
        """Stores the data stored at the given untransposed coordinate.

        Args:
            x: The untransposed x coordinate.
            y: The untransposed y coordinate.
            val: The value to store.

        Returns:
            The value stored at the coordinate.
        """
        if transposed:
            self.data.__setitem__(Index(y, x).as_tuple(), val)
        else:
            self.data.__setitem__(Index(x, y).as_tuple(), val)


fn naive_matmul[
    shape_a: __mlir_type[`!kgen.list<index[2]>`],
    shape_b: __mlir_type[`!kgen.list<index[2]>`],
    shape_c: __mlir_type[`!kgen.list<index[2]>`],
    type: __mlir_type.`!kgen.dtype`,
    transpose_a: Bool,
    transpose_b: Bool,
](
    c: NDBuffer[2, shape_c, type],
    a: NDBuffer[2, shape_a, type],
    b: NDBuffer[2, shape_b, type],
):
    """Computes matrix multiplication with a naive algorithm.

    Args:
        c: Buffer with allocated output space.
        a: Buffer containing matrix operand A.
        b: Buffer containing matrix operand B.
        transpose_a: indicates if a is transposed.
        transpose_b: indicates if b is transposed.
    """
    var gemm_shape = GemmShape.__new__[shape_c, shape_a, shape_b, type](
        c, a, b, transpose_a, transpose_b
    )
    var matrix_a = _Matrix[shape_a, type, transpose_a](a)
    var matrix_b = _Matrix[shape_b, type, transpose_b](b)
    var matrix_c = _Matrix[shape_c, type, False](c)

    var m: Int = 0
    while m < gemm_shape.M:
        var n: Int = 0
        while n < gemm_shape.N:
            var c_val: SIMD[1, type] = 0
            var k: Int = 0
            while k < gemm_shape.K:
                var a_val = matrix_a.__getitem__(m, k)
                var b_val = matrix_b.__getitem__(k, n)
                c_val += a_val * b_val
                k += 1
            matrix_c.__setitem__(m, n, c_val)
            n += 1
        m += 1


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
    let num_of_blocks: Int = original_size // block_size
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
    var orig_matrix: NDBuffer[2, original_shape, type]
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
        orig_matrix: NDBuffer[2, original_shape, type],
        global_offset: StaticIntTuple[2],
        pack_tile_dim: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            orig_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            pack_tile_dim(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile.
        """
        let instance = PackMatrixRows[
            original_shape, packed_shape, type, simd_size, row_inner_size
        ](packed_matrix, orig_matrix, global_offset, pack_tile_dim)
        instance._pack()

    fn __new__(
        packed_matrix: NDBuffer[3, packed_shape, type],
        orig_matrix: NDBuffer[2, original_shape, type],
        global_offset: StaticIntTuple[2],
        pack_tile_dim: StaticIntTuple[2],
    ) -> PackMatrixRows[
        original_shape, packed_shape, type, simd_size, row_inner_size
    ]:
        """Constructor of the algorithm instance for configuring parameters.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            orig_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            pack_tile_dim(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile.
        """
        assert_param[row_inner_size % simd_size == 0]()
        # Assumes NumberOfRowToPack is divisible by row_inner_size
        # TODO: add dynamic checks.
        var packed: PackMatrixRows[
            original_shape, packed_shape, type, simd_size, row_inner_size
        ]
        packed.packed_matrix = packed_matrix
        packed.orig_matrix = orig_matrix
        packed.global_offset = global_offset
        packed.pack_tile_dim = pack_tile_dim

        # Calculate bound of valid data in original matrix.
        let valid_data_dim = Index(
            orig_matrix.dim[0]() - global_offset.__getitem__[0](),
            orig_matrix.dim[1]() - global_offset.__getitem__[1](),
        )
        # Calculate multiple-of-simd bound of valid data.
        let valid_simd_dim = Index(
            round_down_to_block[simd_size](
                Int.min(
                    valid_data_dim.__getitem__[0](),
                    pack_tile_dim.__getitem__[0](),
                )
            ),
            round_down_to_block[simd_size](
                Int.min(
                    valid_data_dim.__getitem__[1](),
                    pack_tile_dim.__getitem__[1](),
                )
            ),
        )

        packed.valid_data_dim = valid_data_dim
        packed.valid_simd_dim = valid_simd_dim
        return packed

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
        var start_idx_global = local_off_set + self.global_offset

        # Fill the simd_size x simd_size transpose buffer
        #  with un-transposed data.
        var inner_row_idx: Int = 0
        while inner_row_idx < simd_size:
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound.__getitem__[0]()):
                var row_gloal_index = Index(
                    start_idx_global.__getitem__[0]() + inner_row_idx,
                    start_idx_global.__getitem__[1](),
                ).as_tuple()
                var row_data: SIMD[simd_size, type]
                if skip_col_bound:
                    # This is fastest path where both row and col bounds
                    #  are skipped so the code path is simd-in and simd-out
                    #  without any predicate.
                    row_data = self.orig_matrix.simd_load[simd_size](
                        row_gloal_index
                    )
                else:
                    # Not skipping col bound, need to to a partial fill of
                    #  the transpose buffer row.
                    row_data = partial_simd_load[simd_size, type](
                        self.orig_matrix._offset(row_gloal_index),
                        0,  # no left bound.
                        read_bound.__getitem__[1](),
                        SIMD[1, type](0),
                    )

                transpose_buffer.simd_store[simd_size](
                    Index(inner_row_idx, 0).as_tuple(), row_data
                )
            else:
                # Row out of defined bound, fill the transpose buffer with zero
                transpose_buffer.simd_store[simd_size](
                    Index(inner_row_idx, 0).as_tuple(), SIMD[simd_size, type](0)
                )
            inner_row_idx += 1

        # Transpose the buffered data
        transpose_inplace[2, simd_size, simd_size, type](transpose_buffer)

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        var transposed_inner_row_idx: Int = 0
        while transposed_inner_row_idx < simd_size:
            let transposed_data = transpose_buffer.simd_load[simd_size](
                Index(transposed_inner_row_idx, 0).as_tuple()
            )
            # compute the packed index
            let _row_outer = local_off_set.__getitem__[0]() // row_inner_size
            let _row_inner = Int.remu(
                local_off_set.__getitem__[0](), row_inner_size
            )

            if skip_col_bound or (
                transposed_inner_row_idx < write_bound.__getitem__[1]()
            ):
                self.packed_matrix.simd_store[simd_size](
                    Index(
                        _row_outer,
                        local_off_set.__getitem__[1]()
                        + transposed_inner_row_idx,
                        _row_inner,
                    ).as_tuple(),
                    transposed_data,
                )
            # Out of bound columns are discarded as there's no allocation for them
            #  in the packed buffer.
            transposed_inner_row_idx += 1

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
                self.valid_simd_dim.__getitem__[0](),
                self.pack_tile_dim.__getitem__[0](),
            ),
            Int.min(
                self.valid_simd_dim.__getitem__[1](),
                self.pack_tile_dim.__getitem__[1](),
            ),
        )

        # # fill rows with valid data
        var row_idx0: Int = 0
        while row_idx0 < valid_tile_simd_dim.__getitem__[0]():
            var col_idx0: Int = 0
            while col_idx0 < valid_tile_simd_dim.__getitem__[1]():
                self._transpose_pack_helper[
                    # skip_row_bound, skip_col_bound
                    True,
                    True,
                ](
                    transpose_buffer,
                    # local offset
                    Index(row_idx0, col_idx0),
                )
                col_idx0 += simd_size

            # Pack residue and zero-ed columns.
            while col_idx0 < self.pack_tile_dim.__getitem__[1]():
                # TODO: this can be peeled further
                #  but cound be un-necessary as the tile
                #  level is also optimized.
                self._transpose_pack_helper[
                    # skip_row_bound, skip_col_bound
                    True,
                    False,
                ](
                    transpose_buffer,
                    # local offset
                    Index(row_idx0, col_idx0),
                )
                col_idx0 += simd_size

            row_idx0 += simd_size

        # If there's a few residue rows with valid data.
        #  pack the residue rows.
        if row_idx0 < self.pack_tile_dim.__getitem__[0]():
            var col_idx1: Int = 0
            while col_idx1 < valid_tile_simd_dim.__getitem__[1]():
                self._transpose_pack_helper[
                    # skip_row_bound, skip_col_bound
                    False,
                    True,
                ](
                    transpose_buffer,
                    # local offset
                    Index(row_idx0, col_idx1),
                )
                col_idx1 += simd_size

            # do a residue column if any.
            while col_idx1 < self.pack_tile_dim.__getitem__[1]():
                # do residue column
                self._transpose_pack_helper[
                    # skip_row_bound, skip_col_bound
                    False,
                    False,
                ](
                    transpose_buffer,
                    Index(row_idx0, col_idx1),  # local offset
                )
                col_idx1 += simd_size

        # TODO:
        #  This packing routine is intended to be used in mlas style matmul
        # so out of bound rows in tile never need to be zero filled. But
        # in general should add additional params controling how to handle
        # out of bound rows between valid_tile_simd_dim.__getitem__[0]() and packed_tile_dim_rows.


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
    var orig_matrix: NDBuffer[2, original_shape, type]
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
        orig_matrix: NDBuffer[2, original_shape, type],
        global_offset: StaticIntTuple[2],
        pack_tile_dim: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            orig_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            pack_tile_dim(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile.
        """
        let instance = PackMatrixCols[
            original_shape, packed_shape, type, simd_size, column_inner_size
        ](packed_matrix, orig_matrix, global_offset, pack_tile_dim)
        instance._pack()

    fn __new__(
        packed_matrix: NDBuffer[3, packed_shape, type],
        orig_matrix: NDBuffer[2, original_shape, type],
        global_offset: StaticIntTuple[2],
        pack_tile_dim: StaticIntTuple[2],
    ) -> PackMatrixCols[
        original_shape, packed_shape, type, simd_size, column_inner_size
    ]:
        """Constructor of the algorithm instance for configuring parameters.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            orig_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            pack_tile_dim(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile.
        """
        var pack: PackMatrixCols[
            original_shape, packed_shape, type, simd_size, column_inner_size
        ]
        pack.packed_matrix = packed_matrix
        pack.orig_matrix = orig_matrix
        pack.global_offset = global_offset
        pack.pack_tile_dim = pack_tile_dim
        pack.valid_data_dim = Index(
            orig_matrix.dim[0]() - global_offset.__getitem__[0](),
            orig_matrix.dim[1]() - global_offset.__getitem__[1](),
        )

        assert_param[column_inner_size % simd_size == 0]()
        # Also assumes that pack_tile_dim.__getitem__[1]() is divisible by column_inner_size.
        # TODO: add dynamic checks.
        return pack

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
        var col_idx: Int = 0
        while col_idx < self.pack_tile_dim.__getitem__[1]():
            # Decl the data to fill in packed buffer.
            var data: SIMD[simd_size, type]

            # Calculate global coordinates.
            let global_idx_pair = self.global_offset + Index(
                tile_row_idx, col_idx
            )
            let global_idx = Index(
                global_idx_pair.__getitem__[0](),
                global_idx_pair.__getitem__[1](),
            ).as_tuple()

            if fill_zero:
                # Statical fill zero case.
                data = SIMD[simd_size, type](0)
            elif skip_col_bound or (
                col_idx + simd_size <= self.valid_data_dim.__getitem__[1]()
            ):
                # Whole SIMD vector within bound.
                data = self.orig_matrix.simd_load[simd_size](global_idx)
            elif col_idx >= self.valid_data_dim.__getitem__[1]():
                # Starting point out of bound. Fill a zero vector.
                data = SIMD[simd_size, type](0)
            else:
                # Starting point within bound but cannot load a whole
                #  vector. Do a partial load.
                data = partial_simd_load[simd_size, type](
                    self.orig_matrix._offset(global_idx),
                    0,
                    self.valid_data_dim.__getitem__[1]() - col_idx,
                    SIMD[1, type](0),
                )

            # map to packed index
            let col_idx_outer = col_idx // column_inner_size
            let col_idx_inner = Int.remu(col_idx, column_inner_size)
            self.packed_matrix.simd_store[simd_size](
                Index(col_idx_outer, tile_row_idx, col_idx_inner).as_tuple(),
                data,
            )
            col_idx += simd_size

    fn _pack_helper[skip_col_bound: Bool](self):
        """Helper function: packs all the rows within the tile of data to pack
        with statical option to skip boundary check.
            Args:
                skip_col_bound: Boundary check on column dimension will be skipped
                    if true.
        """
        var row_idx: Int = 0
        let valid_row_count = Int.min(
            self.valid_data_dim.__getitem__[0](),
            self.pack_tile_dim.__getitem__[0](),
        )
        while row_idx < valid_row_count:
            self._pack_row_helper[skip_col_bound, False](row_idx)  # fill zero
            row_idx += 1
        # Fill zero on the remaining rows on the tile.
        while row_idx < self.pack_tile_dim.__getitem__[0]():
            self._pack_row_helper[
                True,  # skip read col bound.
                False,  # fill zero
            ](row_idx)
            row_idx += 1

    fn _pack(self):
        """Helper function: packs all the rows within the tile of data to pack"""
        # TODO:
        #  This packing routine can be further peeled and vectorized
        #    but dynamical tiling could cover some of the sub-optimality
        #    here. In a follow up should extend the blocking scheme here.
        if (
            self.pack_tile_dim.__getitem__[1]()
            < self.valid_data_dim.__getitem__[1]()
        ):
            # If the whole tile is within bound.
            #  skip all the column checks.
            self._pack_helper[
                # skip col bound check.
                True
            ]()
        else:
            # Do not skip checks if not the whole column tile
            #  is within bound.
            self._pack_helper[
                # skip row bound check.
                False
            ]()


struct MatmulInnerLoopBPacked[
    shape_a: __mlir_type[`!kgen.list<index[2]>`],
    shape_c: __mlir_type[`!kgen.list<index[2]>`],
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
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
    var c: NDBuffer[2, shape_c, type]
    var a: NDBuffer[2, shape_a, type]
    var b_packed: NDBuffer[3, packed_shape, type]
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
        c: NDBuffer[2, shape_c, type],
        a: NDBuffer[2, shape_a, type],
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.
        Args:
            c(NDBuffer): pre-allocated buffer space for packed result.
            a(NDBuffer): data buffer operand A.
            b(NDBuffer): data buffer operand B in packed layout.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            tile_n_k(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile of B.
        """
        var instance = MatmulInnerLoopBPacked[
            shape_a,
            shape_c,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            skip_boundary_check,
        ](c, a, b_packed, global_offset, tile_n_k)
        instance._run_inner_loop()

    fn __new__(
        c: NDBuffer[2, shape_c, type],
        a: NDBuffer[2, shape_a, type],
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ) -> MatmulInnerLoopBPacked[
        shape_a,
        shape_c,
        packed_shape,
        type,
        simd_size,
        a_row_size,
        pack_inner_size,
        skip_boundary_check,
    ]:
        """Constructor of the inner loop instance with parameter derivations.
        Args:
            c(NDBuffer): pre-allocated buffer space for packed result.
            a(NDBuffer): data buffer operand A.
            b(NDBuffer): data buffer operand B in packed layout.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            tile_n_k(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile of B.
        """
        var inner_loop: MatmulInnerLoopBPacked[
            shape_a,
            shape_c,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            skip_boundary_check,
        ]
        inner_loop.c = c
        inner_loop.a = a
        inner_loop.b_packed = b_packed
        inner_loop.global_offset = global_offset
        inner_loop.tile_n_k = tile_n_k
        inner_loop.c_bound = Index(c.dim[0](), c.dim[1]()) - Index(
            global_offset.M, global_offset.N
        )
        return inner_loop

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            type,
        ],
    ):
        """Utility funcion on the inner loop. Initializes a local c buffer with
        all zeros.
            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
        """
        var row_idx: Int = 0
        while row_idx < a_row_size:
            var col_idx: Int = 0
            while col_idx < pack_inner_size:
                c_local.simd_store[simd_size](
                    Index(row_idx, col_idx).as_tuple(), SIMD[simd_size, type](0)
                )
                col_idx += simd_size
            row_idx += 1

    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            type,
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
        var row_idx: Int = 0
        while row_idx < a_row_size:
            var col_idx: Int = 0
            while col_idx < pack_inner_size:
                var global_idx_pair = (
                    Index(self.global_offset.M, self.global_offset.N)
                    + tile_idx
                    + Index(row_idx, col_idx)
                )
                var global_idx = Index(
                    global_idx_pair.__getitem__[0](),
                    global_idx_pair.__getitem__[1](),
                ).as_tuple()
                var local_idx = Index(row_idx, col_idx).as_tuple()

                # Load data from original matrix C.
                var c_data = SIMD[simd_size, type](0)
                if skip_boundary_check or (
                    Index(row_idx, col_idx + simd_size)
                    <= (self.c_bound - tile_idx)
                ):
                    # Use simd load if all within bound
                    c_data = self.c.simd_load[simd_size](global_idx)
                elif (
                    row_idx + tile_idx.__getitem__[0]()
                ) < self.c_bound.__getitem__[0]():
                    # Use partial load if row inbound but col not
                    #  in simd bound.
                    c_data = partial_simd_load[simd_size, type](
                        self.c._offset(global_idx),
                        0,
                        self.c_bound.__getitem__[1]()
                        - tile_idx.__getitem__[1]()
                        - col_idx,
                        SIMD[1, type](0),
                    )
                else:
                    # Fill zero if row out of bound
                    c_data = SIMD[simd_size, type](0)

                # Store data to local buffer.
                c_local.simd_store[simd_size](local_idx, c_data)
                col_idx += simd_size
            row_idx += 1

    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            type,
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
        var row_idx: Int = 0
        while row_idx < a_row_size:
            var col_idx: Int = 0
            while col_idx < pack_inner_size:
                var global_idx_pair = (
                    Index(self.global_offset.M, self.global_offset.N)
                    + tile_idx
                    + Index(row_idx, col_idx)
                )
                var global_idx = Index(
                    global_idx_pair.__getitem__[0](),
                    global_idx_pair.__getitem__[1](),
                ).as_tuple()
                var local_idx = Index(row_idx, col_idx).as_tuple()

                # Load data from original matrix C.
                var c_data = c_local.simd_load[simd_size](local_idx)

                if skip_boundary_check or (
                    Index(row_idx, col_idx + simd_size)
                    <= (self.c_bound - tile_idx)
                ):
                    # Use simd store if all within bound
                    self.c.simd_store[simd_size](global_idx, c_data)
                elif row_idx < (
                    self.c_bound.__getitem__[0]() - tile_idx.__getitem__[0]()
                ):
                    # Use partial store if row in bound but col not
                    #  in simd bound.
                    partial_simd_store[simd_size, type](
                        self.c._offset(global_idx),
                        0,
                        self.c_bound.__getitem__[1]()
                        - tile_idx.__getitem__[1]()
                        - col_idx,
                        c_data,
                    )
                col_idx += simd_size
            row_idx += 1

    fn _accumulate(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            type,
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
        let n_outer_idx = tile_n_k_idx.__getitem__[0]() // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx.__getitem__[1]()

        # Loop over local accumulator tiles.
        var col_idx: Int = 0
        while col_idx < pack_inner_size:
            let b_val = self.b_packed.simd_load[simd_size](
                Index(
                    n_outer_idx, tile_n_k_idx.__getitem__[1](), col_idx
                ).as_tuple()
            )
            var row_idx: Int = 0
            while row_idx < a_row_size:
                var global_m = self.global_offset.M + row_idx
                let a_val_scalar = self.a.simd_load[1](
                    Index(global_m, global_k).as_tuple()
                )
                let a_val = SIMD[simd_size, type](a_val_scalar)

                var c_idx = Index(row_idx, col_idx).as_tuple()
                var c_val = c_local.simd_load[simd_size](c_idx)

                c_val = a_val.fma(b_val, c_val)
                c_local.simd_store[simd_size](c_idx, c_val)
                row_idx += 1
            col_idx += simd_size

    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        # Allocate accumulation buffer.
        var _c_data = _raw_stack_allocation[
            a_row_size * pack_inner_size * simd_size, type, 1
        ]()

        var c_local = NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            type,
        ](_c_data.address)

        var idx_n: Int = 0
        while idx_n < self.tile_n_k.__getitem__[0]():
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, Index(0, idx_n))

            # Iterate on tile K dimension.
            var idx_k: Int = 0

            # Not unrolled on K path.
            while idx_k < self.tile_n_k.__getitem__[1]():
                # accumulate data for this (n, k) index
                self._accumulate(c_local, Index(idx_n, idx_k))
                idx_k += 1

            self._store_c_tile(c_local, Index(0, idx_n))
            idx_n += pack_inner_size


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
    shape_a: __mlir_type[`!kgen.list<index[2]>`],
    shape_b: __mlir_type[`!kgen.list<index[2]>`],
    shape_c: __mlir_type[`!kgen.list<index[2]>`],
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    a_row_size: __mlir_type.index,
    pack_inner_size: __mlir_type.index,
    # Maximum number of elements of B that can fit
    #  in the packed buffer.
    pack_cache_size: __mlir_type.index,
    transpose_a: Bool,
    transpose_b: Bool,
]:
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    TODO: not yet supporting transpose_a.
    TODO: add tag based implementation dispatch.
    TODO: add fusion hooks.
    """

    var c: NDBuffer[2, shape_c, type]
    var a: NDBuffer[2, shape_a, type]
    var b: NDBuffer[2, shape_b, type]
    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]
    var gemm_shape: GemmShape

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[2, shape_c, type],
        a: NDBuffer[2, shape_a, type],
        b: NDBuffer[2, shape_b, type],
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
        var matmul = TiledMatmul[
            shape_a,
            shape_b,
            shape_c,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            pack_cache_size,
            transpose_a,
            transpose_b,
        ](c, a, b)
        matmul._run()

    fn __new__(
        c: NDBuffer[2, shape_c, type],
        a: NDBuffer[2, shape_a, type],
        b: NDBuffer[2, shape_b, type],
    ) -> TiledMatmul[
        shape_a,
        shape_b,
        shape_c,
        packed_shape,
        type,
        simd_size,
        a_row_size,
        pack_inner_size,
        pack_cache_size,
        transpose_a,
        transpose_b,
    ]:
        """Constructor of a tiled matmul instance with parameter derivation.

        Args:
            c(NDBuffer): Pre-allocated buffer space for result.
            a(NDBuffer): Operand A of the matmul.
            b(NDBuffer): Operand B of the mamtul.
            transpose_a: True if a is in transposed layout.
            transpose_b: True if b is in transposed layout.
        """
        var matmul: TiledMatmul[
            shape_a,
            shape_b,
            shape_c,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            pack_cache_size,
            transpose_a,
            transpose_b,
        ]
        matmul.c = c
        matmul.a = a
        matmul.b = b
        let _gemm_shape = GemmShape.__new__[shape_c, shape_a, shape_b, type](
            c, a, b, transpose_a, transpose_b
        )
        matmul.gemm_shape = _gemm_shape
        matmul.tile_n_k = calculate_tile_n_k[pack_cache_size, pack_inner_size](
            _gemm_shape
        )
        return matmul

    fn _outer_m_loop_row_helper[
        skip_col_bound: Bool,
        m_loop_pack_inner_size: __mlir_type.index,
        RowSize: __mlir_type.index,
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n_k: StaticIntTuple[2],
        start_idx: Int,
        valid_row_count: Int,
    ) -> Int:
        """
        Helper function: Process blocks of rows of the gemm space with the given
          RowBlock size until the given row block does not completely fit in
          valid operand bound.

            Args:
                skip_col_bound(i1): Column dimension boundary check will be
                    statically skipped if true.
                m_loop_pack_inner_size(index): Inner dimension of the packed data
                    layout.
                RowSize(index): Size of row blocks to proceed with on the tile.
                b_packed(NDBuffer): B matrix in packed layout.
                global_offset(GemmShape): 3D global offset within the whole
                    matmul problem space.
                sub_tile_n_k(StaticTuple): Dynamic tile size to use, in
                    (TileN, TileK).
                start_idx(Int): row idx to start from.
                valid_row_count(Int): number of valid rows to process from the
                    start_idx.
        """
        var row_idx = start_idx
        while row_idx <= (valid_row_count - RowSize):
            MatmulInnerLoopBPacked[
                shape_a,
                shape_c,
                packed_shape,
                type,
                simd_size,
                RowSize,
                m_loop_pack_inner_size,
                skip_col_bound,
            ].run(
                self.c,
                self.a,
                b_packed,
                global_offset + GemmShape(row_idx, 0, 0),
                sub_tile_n_k,
            )
            row_idx += RowSize
        return row_idx

    fn _outer_m_loop_helper[
        skip_col_bound: Bool, m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
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
                shape_b, packed_shape, type, simd_size, m_loop_pack_inner_size
            ].run(
                b_packed,
                self.b,
                # Input is [N, K]:
                Index(global_offset.N, global_offset.K),
                Index(sub_tile_n, sub_tile_k),
            )
        else:
            PackMatrixCols[
                shape_b, packed_shape, type, simd_size, m_loop_pack_inner_size
            ].run(
                b_packed,
                self.b,
                # Input is [K, N]:
                Index(global_offset.K, global_offset.N),
                Index(sub_tile_k, sub_tile_n),
            )

        # Launch the MLoop
        let sub_tile_n_k = Index(sub_tile_n, sub_tile_k)
        let valid_row_count = self.c.dim[0]() - global_offset.M

        # Launch largest row blocks possible and
        #  then reduce row size to maximizing unrolled tiles.
        var row_idx: Int = 0
        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, a_row_size
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 4
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 3
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 2
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 1
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

    #  Pack a subtile of B and iterate through all the rows of C.
    fn _outer_m_loop[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
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
        let valid_col_count = self.c.dim[1]() - global_offset.N

        # The whole subtile is within valid columns,
        #  no need to check boundary when loading C
        #  on this tile.
        # TODO: this could be in finer granularity.
        if valid_col_count >= sub_tile_n:
            self._outer_m_loop_helper[
                # skip_col_bound
                True,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)
        else:
            self._outer_m_loop_helper[
                # skip_col_bound
                False,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)

    # Helper function:
    #  Iterate on the N dimension by steps of
    # size sub_tile_n with no crossing valid boundary.
    fn _outer_n_loop_helper[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
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
        b_packed: NDBuffer[3, packed_shape, type],
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
        let valid_col_count: Int = self.c.dim[1]() - global_offset.N
        let tile_n: Int = self.tile_n_k.__getitem__[0]()

        # Remap buffer indices for current tile.
        var remapped_bpacked = self._view_buffer_as(
            b_packed, tile_n, sub_tile_k, Int(pack_inner_size)
        )

        var col_idx: Int = 0
        # Proceed with the large tile:
        col_idx = self._outer_n_loop_helper[pack_inner_size](
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
                b_packed, simd_size, sub_tile_k, simd_size
            )
            col_idx = self._outer_n_loop_helper[simd_size](
                remapped_bpacked,
                global_offset,
                simd_size,
                sub_tile_k,
                col_idx,
                valid_col_count,
            )

        # Cover the last sub simdsize tile:
        # This call will handle the sub-simd size boundary.
        if col_idx < valid_col_count:
            self._outer_m_loop[simd_size](
                remapped_bpacked,
                global_offset + GemmShape(0, col_idx, 0),
                simd_size,
                sub_tile_k,
            )

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(self, b_packed: NDBuffer[3, packed_shape, type]):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        let tile_k = self.tile_n_k.__getitem__[1]()
        let valid_k_count = self.gemm_shape.K
        var k_idx: Int = 0

        # Proceed with the largest K tile until crossing
        #  valid boundary.
        while k_idx <= (valid_k_count - tile_k):
            self._outer_n_loop(b_packed, GemmShape(0, 0, k_idx), tile_k)
            k_idx += tile_k

        # Launch another k tile to clean up the residue:
        let remaining_k = valid_k_count - k_idx

        # Do a residue tile if original gemm shape K is not
        #  a multiple of tile K.
        if remaining_k > 0:
            # TODO: possibly need to re-adjust N tile here, if the
            #  residue K is small then could use L2 cache better by
            #  having a wider N.
            self._outer_n_loop(b_packed, GemmShape(0, 0, k_idx), remaining_k)

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[3, packed_shape, type]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

            Args:
                b_packed(NDBuffer): B matrix in packed layout.
                tile_n(Int): Dynamic tile size to use on N dimension.
                tile_k(Int): Dynamic tile size to use on K dimension.
                n_inner_size(Int): Inner dimension size to use for the packed
                    data layout.
        """
        var new_b_packed = b_packed
        let n_outer = tile_n // n_inner_size
        new_b_packed.dynamic_shape.__setitem__[0](n_outer.__as_mlir_index())
        new_b_packed.dynamic_shape.__setitem__[1](tile_k.__as_mlir_index())
        new_b_packed.dynamic_shape.__setitem__[2](
            n_inner_size.__as_mlir_index()
        )
        return new_b_packed

    fn _run(self):
        """Wrapper utility funciton: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        # Allocate pack_b buffer.
        var _bpacked_data = _raw_stack_allocation[
            pack_cache_size,  # Count.
            type,  # Data type.
            simd_byte_width().__as_mlir_index(),  # Alignment.
        ]()

        var b_packed = NDBuffer[3, packed_shape, type](_bpacked_data.address)

        # Manually set the shape of packed B buffer:
        let mapped_bpacked = self._view_buffer_as(
            b_packed,
            self.tile_n_k.__getitem__[0](),
            self.tile_n_k.__getitem__[1](),
            pack_inner_size,
        )
        self._outer_k_loop(mapped_bpacked)
