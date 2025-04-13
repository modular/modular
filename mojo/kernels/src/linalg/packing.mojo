# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import align_down, align_up, ceildiv
from sys import alignof, has_neon, simdwidthof
from sys.intrinsics import PrefetchOptions

from algorithm import unswitch
from buffer.buffer import NDBuffer, partial_simd_load
from buffer.dimlist import DimList
from memory import UnsafePointer, memcpy, stack_allocation
from register import register_internal

from utils.index import Index, IndexList

from .apple_accelerate import use_apple_accelerate_lib
from .gemv import gemv
from .transpose import transpose, transpose_inplace
from .utils import (
    GemmShape,
    KernelConfig,
    _get_tile_n_k,
    dispatch_get_kernel_type,
    get_kernel_config,
    get_matmul_arch_factor,
    get_pack_data_size,
    get_packB_unroll_factor,
    use_i8mm_fn,
    use_vnni_fn,
)


@value
struct PackMatrixRows[
    original_mut: Bool, //,
    # original matrix shape list
    original_shape: DimList,
    # packed matrix shape list
    packed_shape: DimList,
    type: DType,
    simd_size: Int,
    row_inner_size: Int,
    packed_origin: MutableOrigin,
    original_origin: Origin[original_mut],
]:
    """Pack rows from a matrix into the mlas packed layout and
    extract inner vectors of rows into the packed inner dimension,
    e.g. extract tile [X, Y] and pack into [Xo][Y][Xi].
    """

    # packed matrix
    var packed_matrix: NDBuffer[type, 3, packed_origin, packed_shape]
    # original matrix:
    var original_matrix: NDBuffer[type, 2, original_origin, original_shape]
    # offsets in original matrix
    var global_offset: IndexList[2]
    # number of Row and Col to pack.
    #  in [Row, Col]
    var pack_tile_dim: IndexList[2]
    # valid data bound within the tile.
    var valid_data_dim: IndexList[2]
    # valid multiple-of-simd data bound within the tile.
    var valid_simd_dim: IndexList[2]

    # Interface method:
    #  run the packing and store to the given buffer.
    @staticmethod
    fn run(
        packed_matrix: NDBuffer[type, 3, packed_origin, packed_shape],
        original_matrix: NDBuffer[type, 2, original_origin, original_shape],
        global_offset: IndexList[2],
        pack_tile_dim: IndexList[2],
        valid_data_dim: IndexList[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            original_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(IndexList): offset to use when indexing the
                original matrix.
            pack_tile_dim(IndexList): 2D dimension tuple describing the
                size of the packed tile.
            valid_data_dim(IndexList): 2D dimension tuple describing the
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
            mut=True,
            type,
            2,
            _,
            DimList(simd_size, simd_size),
        ],
        local_off_set: IndexList[2],
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
               local_offset(IndexList): offset of the subtile to work on
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
        @parameter
        for idx in range(simd_size):
            alias inner_row_idx = idx
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound[0]):
                var row_global_index = Index(
                    start_idx_global[0] + inner_row_idx,
                    start_idx_global[1],
                )
                var row_data: SIMD[type, simd_size]

                @parameter
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
                    Index(inner_row_idx, 0), row_data
                )
            else:
                # Row out of defined bound, fill the transpose buffer with zero
                transpose_buffer.store[width=simd_size](
                    Index(inner_row_idx, 0), SIMD[type, simd_size](0)
                )

        # Transpose the buffered data
        transpose_inplace[simd_size, simd_size, type](transpose_buffer)

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        @parameter
        for idx in range(simd_size):
            var transposed_data = transpose_buffer.load[width=simd_size](
                Index(idx, 0)
            )
            # compute the packed index
            var _row_outer = local_off_set[0] // row_inner_size
            var _row_inner = local_off_set[0] % row_inner_size

            if skip_col_bound or (idx < write_bound[1]):
                self.packed_matrix.store[width=simd_size](
                    Index(
                        _row_outer,
                        local_off_set[1] + idx,
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
            type,
            2,
            MutableAnyOrigin,
            DimList(simd_size, simd_size),
        ].stack_allocation[alignment = alignof[SIMD[type, simd_size]]()]()

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
        var col_idx: Int

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
    original_mut: Bool, //,
    # original matrix shape list
    original_shape: DimList,
    # packed matrix shape list
    packed_shape: DimList,
    type: DType,
    simd_size: Int,
    column_inner_size: Int,
    use_vnni: Bool,
    use_i8mm: Bool,
    packed_origin: MutableOrigin,
    original_origin: Origin[original_mut],
]:
    """Pack columns from a matrix into the mlas packed layout and
    extract inner vectors of columns into the packed inner dimension,
    e.g. extracts [X, Y] and packs as [Yo][X][Yi].
    """

    # packed matrix
    var packed_matrix: NDBuffer[type, 3, packed_origin, packed_shape]
    # original matrix:
    var original_matrix: NDBuffer[type, 2, original_origin, original_shape]
    # offsets in original matrix:
    var global_offset: IndexList[2]
    # number of Row and Col to pack.
    #  in [Row, Col]
    var pack_tile_dim: IndexList[2]
    # valid data bound within the tile.
    var valid_data_dim: IndexList[2]

    # Interface function:
    @staticmethod
    fn run(
        packed_matrix: NDBuffer[type, 3, MutableAnyOrigin, packed_shape],
        original_matrix: NDBuffer[type, 2, MutableAnyOrigin, original_shape],
        global_offset: IndexList[2],
        pack_tile_dim: IndexList[2],
        valid_data_dim: IndexList[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix(NDBuffer): pre-allocated buffer space for packed
                data.
            original_matrix(NDBuffer): data buffer containing the original matrix
                to pack.
            global_offset(IndexList): offset to use when indexing the
                original matrix.
            pack_tile_dim(IndexList): 2D dimension tuple describing the
                size of the packed tile.
            valid_data_dim(IndexList): 2D dimension tuple describing the
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

                @parameter
                for i in range(unroll_factor):
                    prefetch_body[i]()

            @parameter
            for i in range(unroll_factor):
                pack_body[i]()
        else:
            for row_idx in range(row_start, valid_row_count):
                pack_vector(row_idx, col_start)

    fn _pack_vnni(self):
        """Copy the B tile from the original matrix to the packed buffer for VNNI.
        """
        constrained[use_vnni]()

        alias vnni_cols = 4

        var kc = self.valid_data_dim[0]
        var nc = self.valid_data_dim[1]
        var nr = column_inner_size
        for i in range(0, self.pack_tile_dim[0], vnni_cols):
            for j in range(self.pack_tile_dim[1] // nr):
                for p in range(nr):

                    @parameter
                    for l in range(vnni_cols):
                        var local_idx = Index(i + l, p + nr * j)
                        var val = 0 if local_idx[0] >= kc or local_idx[
                            1
                        ] >= nc else self.original_matrix[
                            self.global_offset + local_idx
                        ]
                        self.packed_matrix.store(
                            Index(j, i // vnni_cols, vnni_cols * p + l),
                            val,
                        )

    fn _pack_i8mm(self):
        alias i8mm_rows = 2
        alias i8mm_cols = 8

        constrained[use_i8mm]()
        var kc = self.valid_data_dim[0]
        var nc = self.valid_data_dim[1]
        alias nr = column_inner_size // 2
        for i in range(0, self.pack_tile_dim[0], i8mm_cols):
            for j in range(self.pack_tile_dim[1] // nr):
                for p in range(0, nr, i8mm_rows):
                    for i2 in range(i8mm_cols):
                        for p2 in range(i8mm_rows):
                            var local_idx = Index(i + i2, nr * j + p + p2)
                            var val = 0 if local_idx[0] >= kc or local_idx[
                                1
                            ] >= nc else self.original_matrix[
                                self.global_offset + local_idx
                            ]
                            self.packed_matrix.store[width=1](
                                Index(
                                    j,
                                    i // i8mm_cols,
                                    i8mm_cols * p + i8mm_cols * p2 + i2,
                                ),
                                val,
                            )

    fn _pack_default(self):
        """Copy the B tile from the original matrix to the packed buffer.
        Each iteration copies a block of shape (unroll_factor, simd_size)."""
        constrained[not use_vnni and not use_i8mm]()
        var valid_row_count = min(self.valid_data_dim[0], self.pack_tile_dim[0])
        alias unroll_factor = get_packB_unroll_factor()

        var row_idx: Int = 0
        var col_idx: Int

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


@always_inline
fn _pack_matmul_b_shape_func_impl[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_in_0: Bool,
    single_thread_blocking_override: Bool,
](
    b_input: NDBuffer[b_type, 2, _, b_shape], kernel_type_m: Int = 0
) -> IndexList[2]:
    """Sets in shape_ref the shape required by `pack_b`'s `b_packed_ref`
    argument.

    If transpose_b is True, this returns the un-transposed shape, since pack_b
    will un-transpose `b_ref` as part of the packing layout transformation."""

    var output = IndexList[2]()

    var n = b_input.dim(0) if transpose_in_0 else b_input.dim(1)
    var k = b_input.dim(1) if transpose_in_0 else b_input.dim(0)
    var tile_n_k = IndexList[2]()

    @parameter
    @always_inline
    fn dispatch_on_kernel_type[kernel_type: Bool]():
        alias config = get_kernel_config[
            a_type,
            b_type,
            c_type,
            kernel_type=kernel_type,
        ]()
        tile_n_k = _get_tile_n_k[
            a_type, b_type, c_type, config.kernel_cols, transpose_in_0
        ](b_input)

    dispatch_get_kernel_type[dispatch_on_kernel_type](kernel_type_m, n, k)

    @parameter
    if transpose_in_0:
        output[0] = b_input.dim(1)
        output[1] = b_input.dim(0)
    else:
        output[0] = b_input.dim(0)
        output[1] = b_input.dim(1)

    # If we are on MacOS with Float32 data types, we use apple_matmul
    # (which is a binding for cblas_sgemm that doesn't support packing) and a
    # special gemv for M=1 (apple_gemv).
    # So override packing, BUT pack functions will do transpose (facilitates
    # apple_gemv), so assign the transposed B dimensions.
    @parameter
    if not use_apple_accelerate_lib[c_type, a_type, b_type]():
        alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
        alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
        alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
        output[0] = align_up(output[0], factor)
        output[1] = align_up(output[1], tile_n_k[0])
    else:
        var tmp = output[0]
        output[0] = output[1]
        output[1] = tmp

    return output


@register_internal("pack_matmul_b_shape_func")
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
](b_input: NDBuffer[b_type, 2, _, b_shape]) -> IndexList[2]:
    # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()

    return _pack_matmul_b_shape_func_impl[
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
    dst: NDBuffer[mut=True, b_type, 2, _, dst_shape],
    src: NDBuffer[b_type, 2, _, src_shape],
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
            n_out % tile_n == 0,
            "N dimension of output must be padded to tile_n",
        )

        for idx_k in range(0, k_out, tile_k):
            var tile_k2 = align_up(min(tile_k, k_out - idx_k), factor)

            for idx_n in range(0, n_out, tile_n):
                var packed_dst_view = NDBuffer[b_type, 3](
                    dst_flat.data.offset(dst_offset),
                    DimList(
                        tile_n // inner_size2,
                        tile_k2 // factor,
                        inner_size2 * factor,
                    ),
                )
                var valid_k = min(tile_k2, k_in - idx_k)
                var valid_n = min(tile_n, n_in - idx_n)
                PackMatrixCols[
                    src_shape,
                    DimList.create_unknown[3](),
                    b_type,
                    simd_size,
                    inner_size,
                    use_vnni,
                    use_i8mm,
                    packed_dst_view.origin,
                    src.origin,
                ].run(
                    packed_dst_view,
                    src,
                    # Input is [K, N]:
                    # Starting global offset for packing.
                    Index(idx_k, idx_n),
                    Index(tile_k2, tile_n),
                    # Valid amount of input from the starting offset.
                    Index(valid_k, valid_n),
                )
                dst_offset += tile_n * tile_k2
    else:
        # _t = transpose, annoying WAR since variables can't have same name in if/else
        var k_in_t = src.dim[1]()
        var n_in_t = src.dim[0]()
        var k_out_t = dst.dim[0]()
        var n_out_t = dst.dim[1]()

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
    b_mut: Bool, //,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transposed: Bool,
    b_origin: Origin[b_mut],
    output_origin: MutableOrigin,
](
    b_input: NDBuffer[b_type, 2, b_origin, b_shape],
    output_buffer: NDBuffer[b_type, 2, output_origin],
    kernel_type_m: Int,
) raises:
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

        # If we are on MacOS with Float32 data types, we use apple_matmul
        # (which is a binding for cblas_sgemm that doesn't support packing) and a
        # special gemv for M=1 (apple_gemv).
        # So override packing, BUT do transpose (facilitates apple_gemv).
        @parameter
        if use_apple_accelerate_lib[c_type, a_type, b_type]():
            # If already transposed, skip transpose step and do a memcpy.
            @parameter
            if not transposed:
                var perm = NDBuffer[
                    DType.index, 1, MutableAnyOrigin, 2
                ].stack_allocation()
                perm[0] = 1
                perm[1] = 0

                transpose(output_buffer, b_input, perm.data)

            else:
                memcpy(output_buffer.data, b_input.data, n * k)
            return

        # The config (in particular inner size and tile_k) needs to EXACTLY match the
        # values used in the matmul algorithm consuming this packed b matrix

        @parameter
        @always_inline
        fn dispatch_on_kernel_type[kernel_type: Bool]():
            alias config = get_kernel_config[
                a_type,
                b_type,
                c_type,
                kernel_type=kernel_type,
            ]()
            var tile_n_k = _get_tile_n_k[
                a_type, b_type, c_type, config.kernel_cols, transposed
            ](b_input)
            pack_b[
                transposed,
                config.simd_size,
                config.kernel_cols,
                a_type,
                b_type,
                c_type,
                src_shape=b_shape,
                dst_shape = DimList.create_unknown[2](),
            ](output_buffer, b_input, tile_n_k[0], tile_n_k[1])

        dispatch_get_kernel_type[dispatch_on_kernel_type](kernel_type_m, n, k)


@register_internal("layout_transform_KN_to_KNkni")
fn pack_b_ndbuffer[
    b_mut: Bool, //,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    b_origin: Origin[b_mut],
    output_origin: MutableOrigin,
](
    b_input: NDBuffer[b_type, 2, b_origin, b_shape],
    output_buffer: NDBuffer[b_type, 2, output_origin],
) raises:
    # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()
    _pack_b_ndbuffer_impl[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transposed=False,
    ](b_input, output_buffer, kernel_type_m)


@register_internal("layout_transform_NK_to_KNkni")
fn pack_transposed_b_ndbuffer[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
](
    b_input: NDBuffer[b_type, 2, _, b_shape],
    output_buffer: NDBuffer[mut=True, b_type, 2],
) raises:
    # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()
    _pack_b_ndbuffer_impl[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transposed=True,
    ](b_input, output_buffer, kernel_type_m)


@value
struct BTileGenerator[
    mut: Bool, //,
    config: KernelConfig,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    shape: DimList,
    transpose_b: Bool,
    b_packed: Bool,
    origin: Origin[mut],
]:
    """Struct to encapsulate a tile of B that supports prepacking.

    If b_packed is true, calls to get_tile will return a buffer view from B.
    Otherwise, calls to get_tile will copy a tile from B into a stack allocated
    scratch buffer and return a view of that."""

    var b: NDBuffer[
        b_type, 2, origin, shape
    ]  # packed layout if b_packed is True
    var b_tile_stack_ptr: UnsafePointer[Scalar[b_type]]
    var tile_n_k: IndexList[2]

    # needs to be always_inline so b_tile_stack_ptr gets allocated on caller's stack
    @always_inline
    @staticmethod
    fn get(
        b: NDBuffer[b_type, 2, origin, shape], tile_n_k: IndexList[2]
    ) -> BTileGenerator[
        config, a_type, b_type, c_type, shape, transpose_b, b_packed, origin
    ]:
        var b_tile_stack_ptr = UnsafePointer[Scalar[b_type]]()

        debug_assert(
            not (transpose_b and b_packed),
            "b cannot be both transposed and pre-packed.",
        )

        @parameter
        if not b_packed:
            b_tile_stack_ptr = stack_allocation[
                get_pack_data_size[b_type](),
                b_type,
                alignof[SIMD[b_type, simdwidthof[b_type]()]](),
            ]()

        return BTileGenerator[
            config, a_type, b_type, c_type, shape, transpose_b, b_packed
        ](b, b_tile_stack_ptr, tile_n_k)

    fn get_tile[
        inner_size: Int
    ](
        self,
        global_offset: GemmShape,
        tile_dim_nk: IndexList[2],
        valid_data_dim_nk: IndexList[2],
    ) -> NDBuffer[b_type, 3, MutableAnyOrigin, config.packed_shape]:
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
        alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
        alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

        alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
        alias inner_size2 = inner_size // 2 if use_i8mm else inner_size

        var k = align_up(tile_dim_nk[1], factor)
        var tile_shape_nopack = DimList(
            tile_dim_nk[0] // inner_size2,
            k // factor,
            factor * inner_size2,
        )

        var packed_b = NDBuffer[b_type, 3, _, config.packed_shape](
            self.b_tile_stack_ptr, tile_shape_nopack
        )

        @parameter
        if transpose_b and not b_packed:
            PackMatrixRows[
                shape,
                config.packed_shape,
                b_type,
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
                shape,
                config.packed_shape,
                b_type,
                config.simd_size,
                inner_size,
                use_vnni,
                use_i8mm,
                packed_b.origin,
                origin,
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

            var factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
            alias inner_size2 = inner_size // 2 if use_i8mm else inner_size

            var tile_k = self.tile_n_k[1]
            var tile_k2 = align_up(
                min(self.tile_n_k[1], valid_data_dim_nk[1]), factor
            )

            var tile_shape_pack = DimList(
                self.tile_n_k[0] // inner_size2,
                tile_k2 // factor,
                inner_size2 * factor,
            )
            var tile_k_idx = global_offset.K // tile_k
            var b_flat = self.b.flatten()
            var n_padded = self.b.dim[1]()
            var b_tile_view = NDBuffer[b_type, 3, _, config.packed_shape](
                # tiles are ordered in row-major order
                # a bit of trickieness going on here, this works because:
                #   1. tile_k is the same for every thread (tile_n is not) since threads
                #       don't currently partition on the K dimension
                #   2. the n dimension of each thread's tile is gauranteed to be an
                #       exact multiple of the inner size
                #   3. each tile has dims [tile_n/inner, tile_k, inner]
                b_flat.data.offset(
                    tile_k_idx * tile_k * n_padded + global_offset.N * tile_k2
                ),
                tile_shape_pack,
            )
            return b_tile_view

        else:
            debug_assert(
                False, "unreachable, b_packed not supported with transpose_b"
            )

        return packed_b
