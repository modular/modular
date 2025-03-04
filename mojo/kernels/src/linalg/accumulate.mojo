# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray
from collections.optional import OptionalReg
from math import fma
from sys import alignof, prefetch
from sys.info import has_neon
from sys.intrinsics import PrefetchOptions

from algorithm.functional import tile
from buffer.buffer import NDBuffer, partial_simd_load, partial_simd_store
from memory import UnsafePointer

from utils import IndexList


# ===-----------------------------------------------------------------------===#
# Helper Functions
# ===-----------------------------------------------------------------------===#
# TODO: rename to _MatmulAccumulators?
struct _Accumulator[
    type: DType,
    num_rows: Int,
    num_cols: Int,
    simd_width: Int,
    row_start: Int = 0,
    row_stop: Int = num_rows,
]:
    """
    Parameters:
        type: DType of accumulator.
        num_rows: Number of rows in register tiling.
        num_cols: Number of columns in register tiling.
        simd_width: Number of lanes of a SIMD vector.
    """

    alias tile_columns = num_cols * simd_width

    # The output buffer, should have num_rows x num_cols x simd_width.
    var _storage: NDBuffer[type, 1, num_rows * num_cols * simd_width]

    @always_inline
    fn __init__(out self):
        constrained[(num_cols > 0) and (num_rows > 0) and (simd_width > 0)]()
        alias alignment = alignof[SIMD[type, simd_width]]()
        self._storage = NDBuffer[
            type, 1, num_rows * num_cols * simd_width
        ].stack_allocation[alignment=alignment]()

    @always_inline
    fn __init__(
        mut self,
        other_storage: NDBuffer[type, 1, num_rows * num_cols * simd_width],
    ):
        constrained[(num_cols > 0) and (num_rows > 0) and (simd_width > 0)]()
        self._storage = other_storage

    # NOTE: This is NOT a deepcopy; self uses the same _storage as other.
    @always_inline
    fn __copyinit__(out self, other: Self):
        constrained[(num_cols > 0) and (num_rows > 0) and (simd_width > 0)]()
        self._storage = other._storage

    @staticmethod
    @always_inline
    fn _storage_index(m: Int, n: Int) -> Int:
        return (m * num_cols + n) * simd_width

    @always_inline
    fn __getitem__(self, m: Int, n: Int) -> SIMD[type, simd_width]:
        return self._storage.load[width=simd_width](self._storage_index(m, n))

    @always_inline
    fn __setitem__(mut self, m: Int, n: Int, value: SIMD[type, simd_width]):
        self._storage.store(self._storage_index(m, n), value)

    @always_inline
    fn _partial_set[
        partial_width: Int
    ](mut self, offset: Int, value: SIMD[type, partial_width]):
        self._storage.store[width=partial_width](offset, value)

    @always_inline
    fn _partial_get[
        partial_width: Int
    ](mut self, idx: Int) -> SIMD[type, partial_width]:
        return self._storage.load[width=partial_width](idx)

    # In c+=(a*b), each of a, b, and c can have different types.
    @always_inline
    fn fma[
        a_type: DType, b_type: DType
    ](
        mut self,
        m: Int,
        n: Int,
        a: SIMD[a_type, simd_width],
        b: SIMD[b_type, simd_width],
    ):
        # TODO: the order of 'a' and 'b' in the following FMA and its impact on accuracy.
        self[m, n] = (b.cast[type]()).fma((a.cast[type]()), self[m, n])

    @always_inline
    fn _transfer[
        func: fn (
            m: Int, n: Int, ptr: UnsafePointer[Scalar[type]]
        ) capturing -> None
    ](mut self, base_ptr: UnsafePointer[Scalar[type]], stride: Int):
        var row_ptr = base_ptr

        @parameter
        for m in range(num_rows):

            @parameter
            for n in range(num_cols):
                func(m, n, row_ptr.offset(n * simd_width))
            row_ptr += stride

    # TODO: merge with load
    @always_inline
    fn load(mut self, base_ptr: UnsafePointer[Scalar[type]], stride: Int):
        @parameter
        @always_inline
        fn do_transfer(m: Int, n: Int, ptr: UnsafePointer[Scalar[type]]):
            self[m, n] = ptr.load[width=simd_width]()

        self._transfer[do_transfer](base_ptr, stride)

    @always_inline
    fn load(
        mut self,
        c_ptr: UnsafePointer[Scalar[type]],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
        skip_boundary_check: Bool = False,
    ):
        self._transfer[True](
            c_ptr, c_stride, tile_n_idx, c_bound, skip_boundary_check
        )

    @always_inline
    fn store(
        mut self,
        c_ptr: UnsafePointer[Scalar[type]],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
        skip_boundary_check: Bool = False,
    ):
        self._transfer[False](
            c_ptr, c_stride, tile_n_idx, c_bound, skip_boundary_check
        )

    @always_inline
    fn _transfer[
        is_load: Bool
    ](
        mut self,
        c_ptr: UnsafePointer[Scalar[type]],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
        skip_boundary_check: Bool,
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        if skip_boundary_check:

            @parameter
            if is_load:
                self.load(c_ptr_loc, c_stride)
            else:
                self.store(c_ptr_loc, c_stride)
        else:
            var transfer_count = min(
                c_bound[1] - tile_n_idx, num_cols * simd_width
            )
            var row_ptrs = InlineArray[UnsafePointer[Scalar[type]], num_rows](
                unsafe_uninitialized=True
            )

            @parameter
            for row in range(num_rows):
                row_ptrs[row] = c_ptr_loc + row * c_stride

            self._transfer_loop[0, is_load](
                transfer_count, row_ptrs.unsafe_ptr(), c_stride
            )

    @always_inline
    fn _transfer_columns[
        base_column: Int,
        column_count: Int,
        is_load: Bool,
    ](
        mut self,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[type]]],
        stride: Int,
    ):
        """Loads or stores one or more columns from the base column for each
        row of the tile."""
        alias column_step = min(column_count, simd_width)

        @parameter
        @always_inline
        fn body(row: Int, col: Int):
            @parameter
            if is_load:

                @parameter
                if has_neon():
                    var data = row_ptrs[row].load[width=column_step](col)
                    self._partial_set(row * Self.tile_columns + col, data)
                else:
                    var data = row_ptrs[0].load[width=column_step](
                        stride * row + col
                    )
                    self._partial_set(row * Self.tile_columns + col, data)
            else:
                var data = self._partial_get[column_step](
                    row * Self.tile_columns + col
                )

                @parameter
                if has_neon():
                    row_ptrs[row].store(col, data)
                else:
                    row_ptrs[0].store(stride * row + col, data)

        @parameter
        for row in range(num_rows):
            # Iterate twice for a pairwise load/store or once for any other access.

            @parameter
            for col in range(
                base_column, base_column + column_count, column_step
            ):
                body(row, col)

    @always_inline
    fn _transfer_loop[
        base_column: Int, is_load: Bool
    ](
        mut self,
        transfer_count: Int,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[type]]],
        stride: Int,
    ):
        """Loads/stores all pairwise vectors of the tile and dispatches the
        remaining non-pairwise elements."""
        alias tile_columns_remaining = Self.tile_columns - base_column
        # Support fusion of LDP/STP instructions by emitting pairs of load/store with neon
        alias column_groups = 2 if has_neon() else 1

        # vector instructions.
        @parameter
        if tile_columns_remaining >= column_groups * simd_width:
            if transfer_count >= base_column + column_groups * simd_width:
                self._transfer_columns[
                    base_column, column_groups * simd_width, is_load
                ](row_ptrs, stride)
                self._transfer_loop[
                    base_column + column_groups * simd_width, is_load
                ](transfer_count, row_ptrs, stride)
                return

        @parameter
        if tile_columns_remaining >= simd_width:

            @parameter
            if has_neon():
                self._transfer_tail[base_column, simd_width, is_load](
                    transfer_count, row_ptrs, stride
                )
            else:
                self._transfer_tail_mask[base_column, is_load](
                    transfer_count, row_ptrs, stride
                )

    @always_inline
    fn _transfer_tail[
        base_column: Int, tail_size: Int, is_load: Bool
    ](
        mut self,
        transfer_count: Int,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[type]]],
        stride: Int,
    ):
        """Loads/stores the last elements of the tile that cannot be accessed
        pairwise."""

        if transfer_count & tail_size:
            self._transfer_columns[base_column, tail_size, is_load](
                row_ptrs, stride
            )
            alias tile_columns_remaining = Self.tile_columns - base_column - tail_size

            @parameter
            if tile_columns_remaining >= tail_size // 2 and tail_size > 1:
                self._transfer_tail[
                    base_column + tail_size, tail_size // 2, is_load
                ](transfer_count, row_ptrs, stride)
            return

        @parameter
        if tail_size > 1:
            self._transfer_tail[base_column, tail_size // 2, is_load](
                transfer_count, row_ptrs, stride
            )

    @always_inline
    fn _transfer_tail_mask[
        base_column: Int, is_load: Bool
    ](
        mut self,
        transfer_count: Int,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[type]]],
        stride: Int,
    ):
        var tail_size = transfer_count - base_column

        @parameter
        for row in range(num_rows):
            alias col = base_column // simd_width

            @parameter
            if is_load:
                self[row, col] = partial_simd_load[simd_width](
                    row_ptrs[0].offset(stride * row + base_column),
                    0,
                    tail_size,
                    0,
                )
            else:
                partial_simd_store(
                    row_ptrs[0].offset(stride * row + base_column),
                    0,
                    tail_size,
                    self[row, col],
                )

    # TODO: merge with store
    @always_inline
    fn store(mut self, base_ptr: UnsafePointer[Scalar[type]], stride: Int):
        @parameter
        @always_inline
        fn do_transfer(m: Int, n: Int, ptr: UnsafePointer[Scalar[type]]):
            ptr.store(self[m, n])

        self._transfer[do_transfer](base_ptr, stride)

    # ===-------------------------------------------------------------------===#
    # Init/Load/Store register tiles
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn init(mut self):
        @parameter
        if type.is_floating_point():
            self.init(0.0)
        else:
            self.init(0)

    @always_inline
    fn init(mut self, val: Scalar[type]):
        # TODO: refactor with _transfer
        @parameter
        for m in range(num_rows):

            @parameter
            for n in range(num_cols):
                self[m, n] = val

    @always_inline
    fn load[
        dt: DType, //,
        partial_load: Bool = False,
    ](
        mut self,
        input: UnsafePointer[Scalar[dt], **_],
        input_stride: Int,
        partial_load_size: OptionalReg[Int] = None,
    ):
        """Load a register tile from the input buffer.

        Parameters:
            dt: DType of the input.
            partial_load: Whether load input partially.

        Args:
            input: UnsafePointer to input buffer.
            input_stride: Stride between input segments of size `num_cols * simd_width`.
            partial_load_size: Size of partial load for input.
        """

        # TODO: could we lift partial_load_size out of the loop?
        @parameter
        for i in range(num_rows):

            @parameter
            for j in range(num_cols):
                var input_ptr = input + i * input_stride + j * simd_width
                alias partial_load_last_vec = partial_load and (
                    j == num_cols - 1
                )

                # TODO: check if partial_load_size has value.
                self[i, j] = _simd_load_maybe_partial[
                    simd_width, partial_load_last_vec
                ](input_ptr, 0, partial_load_size).cast[type]()

    @always_inline
    fn store[
        dt: DType, //,
        partial_store: Bool = False,
    ](
        mut self,
        output: UnsafePointer[Scalar[dt], **_],
        output_stride: Int,
        partial_store_size: OptionalReg[Int] = None,
    ):
        """Load a register tile from the input buffer.

        Parameters:
            dt: DType of the output.
            partial_store: Whether store output partially.

        Args:
            output: UnsafePointer to output buffer.
            output_stride: Stride between output segments of size `num_cols * simd_width`.
            partial_store_size: Size of partial store to the output.
        """

        # TODO: could we lift partial_store_size out of the loop?
        @parameter
        for i in range(num_rows):

            @parameter
            for j in range(num_cols):
                alias partial_store_last_vec = partial_store and (
                    j == num_cols - 1
                )
                _simd_store_maybe_partial[simd_width, partial_store_last_vec](
                    output,
                    i * output_stride + j * simd_width,
                    self[i, j].cast[dt](),
                    partial_store_size,
                )

    # ===-------------------------------------------------------------------===#
    # Accumulation entry point.
    # ===-------------------------------------------------------------------===#
    @always_inline
    fn accumulate[
        a_type: DType,
        b_type: DType, //,
        prefetch_offset: OptionalReg[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[Scalar[a_type], **_],
        a_stride: Int,
        b: UnsafePointer[Scalar[b_type], **_],
        b_stride: Int,
        partial_load_b_size: OptionalReg[Int] = None,
    ):
        """Compute c += a * b with register tiling on SIMD ISAs.

        Parameters:
            a_type: DType of the a.
            b_type: DType of the b.
            prefetch_offset: The distance to  prefetch ahead.
            partial_load_b: Whether use partial load for B.
        Args:
            length: Number of elements in accumulation.
            a: The input buffer A.
            a_stride: A's stride between each `length` segment.
            b: The input buffer B.
            b_stride: B's stride between each `num_cols x simd_width` segment.
            partial_load_b_size: The partial load B size.
        """

        @parameter
        if has_neon():
            self._accumulate_neon[
                prefetch_offset=None,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_stride,
                b,
                b_stride,
                partial_load_b_size,
            )
        else:
            self._accumulate_x86_simd[
                prefetch_offset=prefetch_offset,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_stride,
                b,
                b_stride,
                partial_load_b_size,
            )

    @always_inline
    fn accumulate[
        a_type: DType,
        b_type: DType, //,
        # TODO: move the following params to accumulate function.
        prefetch_offset: OptionalReg[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[Scalar[a_type], **_],
        a_base_offsets: NDBuffer[DType.int32, 1, num_rows],
        a_offset: Int,
        b: UnsafePointer[Scalar[b_type], **_],
        b_stride: Int,
        partial_load_b_size: OptionalReg[Int] = None,
    ):
        """Compute c += a * b with register tiling on SIMD ISAs.

        This version applies to the cases where the rows in A are not separated
        evenly by a single stride. E.x. pointwise conv with stride > 1.

        Parameters:
            a_type: DType of the a.
            b_type: DType of the b.
            prefetch_offset: The distance to  prefetch ahead.
            partial_load_b: Whether use partial load for B.

        Args:
            length: Number of elements in accumulation.
            a: The input buffer A.
            a_base_offsets: Base offsets of rows in A.
            a_offset: Offset into A rows.
            b: The input buffer B.
            b_stride: B's stride between each `num_cols x simd_width` segment.
            partial_load_b_size: The partial load B size.


        The A offsets work as follow:

            a_base_offsets[0]: ------------------------------
            a_base_offsets[1]: ------------------------------
            ...
            a_base_offsets[2]: ------------------------------
            ...
            ...
            a_base_offsets[3]: ------------------------------
                                    ^                    ^
                                a_offset        a_offset + length
        """

        @parameter
        if has_neon():
            self._accumulate_neon[
                prefetch_offset=None,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_base_offsets,
                a_offset,
                b,
                b_stride,
                partial_load_b_size,
            )
        else:
            self._accumulate_x86_simd[
                prefetch_offset=prefetch_offset,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_base_offsets,
                a_offset,
                b,
                b_stride,
                partial_load_b_size,
            )

    # ===-------------------------------------------------------------------===#
    # Accumulation optimized for AVX2 and AVX512
    # ===-------------------------------------------------------------------===#

    # An example of accumulation with register tiling.
    #
    # B vector 0-3 -->         reg1     reg2     reg3     reg4
    #                       |========|========|========|========|
    # A point  0   ==> reg0 |  reg5  |  reg6  |  reg7  |  reg8  |
    #                       |--------|--------|--------|--------|
    # A point  1   ==> reg0 |  reg9  |  reg10 |  reg11 |  reg12 |
    #                       |--------|--------|--------|--------|
    # A point  2   ==> reg0 |  reg13 |  reg14 |  reg15 |  reg16 |
    #                       |--------|--------|--------|--------|
    # A point  3   ==> reg0 |  reg17 |  reg18 |  reg19 |  reg20 |
    #                       |--------|--------|--------|--------|
    # A point  4   ==> reg0 |  reg21 |  reg22 |  reg23 |  reg24 |
    #                       |--------|--------|--------|--------|
    # A point  5   ==> reg0 |  reg25 |  reg26 |  reg27 |  reg28 |
    #                       |--------|--------|--------|--------|
    #
    #    ==>      :         Broadcast a scalar into a SIMD register.
    #    -->      :         Load a SIMD vector from memory.
    # simd_width  :         |--------|
    # kernel_width:         |-----------------------------------|
    #
    # The accumulation proceeds as:
    #   for l in range(length):
    #       reg5 += reg0 * reg1
    #       reg6 += reg0 * reg2
    #       ...
    #
    # Note that we can reuse reg0 for different A points because of hardware's
    # register renaming.

    @always_inline
    fn _accumulate_x86_simd[
        a_type: DType,
        b_type: DType, //,
        prefetch_offset: OptionalReg[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[Scalar[a_type], **_],
        a_stride: Int,
        b: UnsafePointer[Scalar[b_type], **_],
        b_stride: Int,
        partial_load_b_size: OptionalReg[Int] = None,
    ):
        """Accumulation optimized for AVX512 and AVX2."""

        constrained[not has_neon()]()

        alias kernel_width = num_cols * simd_width
        var b_ptr = b

        for l in range(length):
            # prefetch
            @parameter
            if prefetch_offset:

                @parameter
                for j in range(num_cols):
                    prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](
                        b_ptr
                        + prefetch_offset.value() * kernel_width
                        + j * simd_width
                    )

            @parameter
            for i in range(row_start, row_stop):
                # Broadcast an scalar from A to a simd vector.
                var a_splat_vec = SIMD[a_type, simd_width](a[l + i * a_stride])

                @parameter
                for j in range(num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        simd_width, partial_load_b
                    ](b_ptr, j * simd_width, partial_load_b_size)

                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    self[i, j] = fma(
                        a_splat_vec.cast[type](),
                        b_vec.cast[type](),
                        self[i, j],
                    )

            b_ptr = b_ptr + b_stride

    @always_inline
    fn _accumulate_x86_simd[
        a_type: DType,
        b_type: DType, //,
        prefetch_offset: OptionalReg[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[Scalar[a_type], **_],
        a_base_offsets: NDBuffer[DType.int32, 1, num_rows],
        a_offset: Int,
        b: UnsafePointer[Scalar[b_type], **_],
        b_stride: Int,
        partial_load_b_size: OptionalReg[Int] = None,
    ):
        """Accumulation optimized for AVX512 and AVX2."""

        constrained[not has_neon()]()

        alias kernel_width = num_cols * simd_width
        var b_ptr = b

        for l in range(length):
            # prefetch
            @parameter
            if prefetch_offset:

                @parameter
                for j in range(num_cols):
                    prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](
                        b_ptr
                        + prefetch_offset.value() * kernel_width
                        + j * simd_width
                    )

            @parameter
            for i in range(row_start, row_stop):
                # Broadcast an scalar from A to a simd vector.
                var a_idx = Int(a_base_offsets[i]) + a_offset + l
                var a_splat_vec = SIMD[a_type, simd_width](a[a_idx])

                @parameter
                for j in range(num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        simd_width, partial_load_b
                    ](b_ptr, j * simd_width, partial_load_b_size)

                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    self[i, j] = fma(
                        a_splat_vec.cast[type](),
                        b_vec.cast[type](),
                        self[i, j],
                    )

            b_ptr = b_ptr + b_stride

    # ===-------------------------------------------------------------------===#
    # Accumulation optimized for NEON
    # ===-------------------------------------------------------------------===#

    # An example of accumulation with register tiling.
    #
    # B vector 0-3 -->                  reg6     reg7     reg6     reg7
    #                                |========|========|========|========|
    # A point  0   --> reg0, reg0[i] |  reg8  |  reg9  |  reg10 |  reg11 |
    #                                |--------|--------|--------|--------|
    # A point  1   --> reg1, reg1[i] |  reg12 |  reg13 |  reg14 |  reg15 |
    #                                |--------|--------|--------|--------|
    # A point  2   --> reg2, reg2[i] |  reg16 |  reg17 |  reg18 |  reg19 |
    #                                |--------|--------|--------|--------|
    # A point  3   --> reg3, reg3[i] |  reg20 |  reg21 |  reg22 |  reg23 |
    #                                |--------|--------|--------|--------|
    # A point  4   --> reg4, reg4[i] |  reg24 |  reg25 |  reg26 |  reg27 |
    #                                |--------|--------|--------|--------|
    # A point  5   --> reg5, reg5[i] |  reg28 |  reg29 |  reg30 |  reg31 |
    #                                |--------|--------|--------|--------|
    #
    #    -->      :         Load a SIMD vector from memory.
    # simd_width  :         |--------|
    # kernel_width:         |-----------------------------------|
    #
    #
    # The accumulation proceeds as:
    #   for i in range(lanes):
    #     for l in range(length):
    #         reg5 += reg0[i] * reg1
    #         reg6 += reg0[i] * reg2
    #         ...
    #
    # Neon FMA can take a lane of a register (reg0[i]). It's more efficient to load
    # A vectors first, then perform `num_lanes x num_rows x num_cols` FMA ops.
    #
    # We can reuse reg6, reg7 for different B vectors on Graviton3. This may spill
    # registers on Graviton2.

    @always_inline
    fn _accumulate_neon[
        a_type: DType,
        b_type: DType, //,
        prefetch_offset: OptionalReg[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[Scalar[a_type], **_],
        a_stride: Int,
        b: UnsafePointer[Scalar[b_type], **_],
        b_stride: Int,
        partial_load_b_size: OptionalReg[Int] = None,
    ):
        """Accumulation optimized for NEON."""
        constrained[has_neon()]()

        var b_ptr = b

        @parameter
        @always_inline
        fn micro_kernel[num_lanes: Int](offset: Int):
            var a_vecs = InlineArray[SIMD[a_type, num_lanes], num_rows](
                unsafe_uninitialized=True
            )

            # Load vectors of size num_lanes from input.
            @parameter
            for i in range(row_start, row_stop):
                a_vecs[i] = a.load[width=num_lanes](offset + i * a_stride)

            var b_ptr = b + offset * b_stride

            @parameter
            for lane in range(num_lanes):

                @parameter
                for j in range(num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        simd_width, partial_load_b
                    ](b_ptr, j * simd_width, partial_load_b_size)

                    @parameter
                    for i in range(row_start, row_stop):
                        # The following should be lifted to registers and show up as
                        # FMA instructions.
                        self[i, j] = fma[type, simd_width](
                            a_vecs[i][lane].cast[type](),
                            b_vec.cast[type](),
                            self[i, j],
                        )

                b_ptr = b_ptr + b_stride

        # Load vectors from A first. The remainder is handled one element at a time.
        tile[micro_kernel, VariadicList[Int](simd_width, 1)](0, length)

    @always_inline
    fn _accumulate_neon[
        a_type: DType,
        b_type: DType, //,
        prefetch_offset: OptionalReg[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[Scalar[a_type], **_],
        a_base_offsets: NDBuffer[DType.int32, 1, num_rows],
        a_offset: Int,
        b: UnsafePointer[Scalar[b_type], **_],
        b_stride: Int,
        partial_load_b_size: OptionalReg[Int] = None,
    ):
        """Accumulation optimized for NEON."""
        constrained[has_neon()]()

        var b_ptr = b

        @parameter
        @always_inline
        fn micro_kernel[num_lanes: Int](offset: Int):
            var a_vecs = InlineArray[SIMD[a_type, num_lanes], num_rows](
                unsafe_uninitialized=True
            )

            # Load vectors of size num_lanes from input.
            @parameter
            for i in range(row_start, row_stop):
                var a_idx = Int(a_base_offsets[i]) + a_offset + offset
                a_vecs[i] = a.load[width=num_lanes](a_idx)

            var b_ptr = b + offset * b_stride

            @parameter
            for lane in range(num_lanes):

                @parameter
                for j in range(num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        simd_width, partial_load_b
                    ](b_ptr, j * simd_width, partial_load_b_size)

                    @parameter
                    for i in range(row_start, row_stop):
                        # The following should be lifted to registers and show up as
                        # FMA instructions.
                        self[i, j] = fma[type, simd_width](
                            a_vecs[i][lane].cast[type](),
                            b_vec.cast[type](),
                            self[i, j],
                        )

                b_ptr += b_stride

        # Load vectors from A first. The remainder is handled one element at a time.
        tile[micro_kernel, VariadicList[Int](simd_width, 1)](0, length)


@always_inline
fn _simd_load_maybe_partial[
    dt: DType, //, simd_width: Int, partial_load: Bool
](
    ptr: UnsafePointer[Scalar[dt], **_],
    offset: Int,
    partial_load_size: OptionalReg[Int] = None,
) -> SIMD[dt, simd_width]:
    """Load a simd vector. The the vector may exceed the data's end, i.e.,
    offset + simd_width > end. In this case, if user specifies partial load, we
    will load partial values of size (end - offset), and fill the rest lanes
    with 0.

    One use case is in convolution when the output channel is NOT multiple of
    simd_width and is NOT padded with zeros at the end. We need to partially load
    the filter near the end.
    """

    @parameter
    if partial_load:
        return partial_simd_load[simd_width](
            ptr + offset, 0, partial_load_size.value(), 0
        )
    else:
        return ptr.load[width=simd_width](offset)


@always_inline
fn _simd_store_maybe_partial[
    dt: DType, //, simd_width: Int, partial_store: Bool
](
    ptr: UnsafePointer[Scalar[dt], **_],
    offset: Int,
    vec: SIMD[dt, simd_width],
    partial_store_size: OptionalReg[Int] = None,
):
    """Store a simd vector. The the vector may exceed the data's end, i.e.,
    offset + simd_width > end. In this case, if user specifies partial_store, we
    will store `partial_store_size` lanes of input vector.
    """

    @parameter
    if partial_store:
        # TODO: check if partial_store_size is present.
        return partial_simd_store[simd_width](
            ptr + offset, 0, partial_store_size.value(), vec
        )
    else:
        return ptr.store(offset, vec)
