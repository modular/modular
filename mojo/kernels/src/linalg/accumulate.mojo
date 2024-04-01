# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from math import fma
from sys.info import has_neon
from sys.intrinsics import PrefetchOptions

from algorithm.functional import tile
from buffer.buffer import Buffer, partial_simd_load, partial_simd_store
from memory import stack_allocation
from memory.unsafe import DTypePointer

# ===----------------------------------------------------------------------===#
# Helper Functions
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_load_maybe_partial[
    simd_size: Int, partial_load: Bool
](
    ptr: DTypePointer, offset: Int, partial_load_size: Optional[Int] = None
) -> SIMD[ptr.type, simd_size]:
    """Load a simd vector. The the vector may exceed the data's end, i.e.,
    offset + simd_size > end. In this case, if user specifies partial load, we
    will load partial values of size (end - offset), and fill the rest lanes
    with 0.

    One use case is in convolution when the output channel is NOT multiple of
    simd_size and is NOT padded with zeros at the end. We need to partially load
    the filter near the end.
    """

    @parameter
    if partial_load:
        return partial_simd_load[simd_size](
            ptr + offset, 0, partial_load_size.value(), 0.0
        )
    else:
        return ptr.load[width=simd_size](offset)


@always_inline
fn _simd_store_maybe_partial[
    simd_size: Int, partial_store: Bool
](
    ptr: DTypePointer,
    offset: Int,
    vec: SIMD[ptr.type, simd_size],
    partial_store_size: Optional[Int] = None,
):
    """Store a simd vector. The the vector may exceed the data's end, i.e.,
    offset + simd_size > end. In this case, if user specifies partial_store, we
    will store `partial_store_size` lanes of input vector.
    """

    @parameter
    if partial_store:
        # TODO: check if partial_store_size is present.
        return partial_simd_store[simd_size](
            ptr + offset, 0, partial_store_size.value(), vec
        )
    else:
        return ptr.store[width=simd_size](offset, vec)


# ===----------------------------------------------------------------------===#
# Accumulation entry point.
# ===----------------------------------------------------------------------===#


@always_inline
fn accumulate[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Optional[Int] = None,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_stride: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Optional[Int] = None,
):
    """Compute c += a * b with register tiling on SIMD ISAs.

    Parameters:
        num_rows: Number of rows in resigter tiling.
        num_cols: Number of columns in resigter tiling.
        simd_size: Number of lanes of a SIMD vector.
        prefetch_offset: The distance to  prefetch ahead.
        partial_load_b: Whether use partial load for B.

    Args:
        length: Number of elements in accumulation.
        c: The output buffer, should have num_rows x num_cols x simd_size.
        a: The input buffer A.
        a_stride: A's stride between each `length` segment.
        b: The input buffer B.
        b_stride: B's stride between each `num_cols x simd_size` segment.
        partial_load_b_size: The partial load B size.

    """

    @parameter
    if has_neon():
        _accumulate_neon[
            num_rows,
            num_cols,
            simd_size,
            prefetch_offset=None,
            partial_load_b=partial_load_b,
        ](
            length,
            c,
            a,
            a_stride,
            b,
            b_stride,
            partial_load_b_size,
        )
    else:
        _accumulate_x86_simd[
            num_rows,
            num_cols,
            simd_size,
            prefetch_offset=prefetch_offset,
            partial_load_b=partial_load_b,
        ](
            length,
            c,
            a,
            a_stride,
            b,
            b_stride,
            partial_load_b_size,
        )


@always_inline
fn accumulate[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Optional[Int] = None,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_base_offsets: Buffer[DType.int32, num_rows],
    a_offset: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Optional[Int] = None,
):
    """Compute c += a * b with register tiling on SIMD ISAs.

    This version applies to the cases where the rows in A are not separated
    evenly by a single stride. E.x. pointwise conv with stride > 1.

    Parameters:
        num_rows: Number of rows in resigter tiling.
        num_cols: Number of columns in resigter tiling.
        simd_size: Number of lanes of a SIMD vector.
        prefetch_offset: The distance to  prefetch ahead.
        partial_load_b: Whether use partial load for B.

    Args:
        length: Number of elements in accumulation.
        c: The output buffer, should have num_rows x num_cols x simd_size.
        a: The input buffer A.
        a_base_offsets: Base offsets of rows in A.
        a_offset: Offset into A rows.
        b: The input buffer B.
        b_stride: B's stride between each `num_cols x simd_size` segment.
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
        _accumulate_neon[
            num_rows,
            num_cols,
            simd_size,
            prefetch_offset=None,
            partial_load_b=partial_load_b,
        ](
            length,
            c,
            a,
            a_base_offsets,
            a_offset,
            b,
            b_stride,
            partial_load_b_size,
        )
    else:
        _accumulate_x86_simd[
            num_rows,
            num_cols,
            simd_size,
            prefetch_offset=prefetch_offset,
            partial_load_b=partial_load_b,
        ](
            length,
            c,
            a,
            a_base_offsets,
            a_offset,
            b,
            b_stride,
            partial_load_b_size,
        )


# ===----------------------------------------------------------------------===#
# Accumulation optimized for AVX2 and AVX512
# ===----------------------------------------------------------------------===#


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
# simd_size   :         |--------|
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
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Optional[Int] = None,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_stride: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Optional[Int] = None,
):
    """Accumulation optimized for AVX512 and AVX2."""

    constrained[not has_neon()]()

    @parameter
    if num_rows == 0 or num_cols == 0:
        return

    alias kernel_width = num_cols * simd_size

    var b_ptr = b

    for l in range(length):
        # prefetch
        @parameter
        if prefetch_offset:

            @unroll
            for j in range(num_cols):
                (
                    b_ptr
                    + prefetch_offset.value() * kernel_width
                    + j * simd_size
                ).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        @unroll
        for i in range(num_rows):
            # Broadcast an scalar from A to a simd vector.
            var a_splat_vec = SIMD[a.type, simd_size](a[l + i * a_stride])

            @unroll
            for j in range(num_cols):
                # Load a simd vector from B.
                var b_vec = _simd_load_maybe_partial[simd_size, partial_load_b](
                    b_ptr, j * simd_size, partial_load_b_size
                )

                # The following should be lifted to registers and show up as
                # FMA instructions.
                var c_ptr = c + i * kernel_width + j * simd_size
                c_ptr.store(
                    fma(
                        a_splat_vec.cast[c.type](),
                        b_vec.cast[c.type](),
                        c_ptr.load[width=simd_size](),
                    )
                )

        b_ptr = b_ptr + b_stride


@always_inline
fn _accumulate_x86_simd[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Optional[Int] = None,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_base_offsets: Buffer[DType.int32, num_rows],
    a_offset: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Optional[Int] = None,
):
    """Accumulation optimized for AVX512 and AVX2."""

    constrained[not has_neon()]()

    @parameter
    if num_rows == 0 or num_cols == 0:
        return

    alias kernel_width = num_cols * simd_size

    var b_ptr = b

    for l in range(length):
        # prefetch
        @parameter
        if prefetch_offset:

            @unroll
            for j in range(num_cols):
                (
                    b_ptr
                    + prefetch_offset.value() * kernel_width
                    + j * simd_size
                ).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        @unroll
        for i in range(num_rows):
            # Broadcast an scalar from A to a simd vector.
            var a_idx = a_base_offsets[i].value + a_offset + l
            var a_splat_vec = SIMD[a.type, simd_size](a[a_idx])

            @unroll
            for j in range(num_cols):
                # Load a simd vector from B.
                var b_vec = _simd_load_maybe_partial[simd_size, partial_load_b](
                    b_ptr, j * simd_size, partial_load_b_size
                )

                # The following should be lifted to registers and show up as
                # FMA instructions.
                var c_ptr = c + i * kernel_width + j * simd_size
                c_ptr.store(
                    fma(
                        a_splat_vec.cast[c.type](),
                        b_vec.cast[c.type](),
                        c_ptr.load[width=simd_size](),
                    )
                )

        b_ptr = b_ptr + b_stride


# ===----------------------------------------------------------------------===#
# Accumulation optimized for NEON
# ===----------------------------------------------------------------------===#


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
# simd_size   :         |--------|
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
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Optional[Int] = None,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_stride: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Optional[Int] = None,
):
    """Accumulation optimized for NEON."""

    constrained[has_neon()]()

    @parameter
    if num_rows == 0 or num_cols == 0:
        return

    alias kernel_width = num_cols * simd_size

    var b_ptr = b

    @parameter
    @always_inline
    fn micro_kernel[num_lanes: Int](offset: Int):
        var a_vecs = stack_allocation[num_rows, SIMD[a.type, num_lanes]]()

        # Load vectors of size num_lanes from input.
        @unroll
        for i in range(num_rows):
            a_vecs[i] = a.load[width=num_lanes](offset + i * a_stride)

        var b_ptr = b + offset * b_stride

        @unroll
        for lane in range(num_lanes):

            @unroll
            for j in range(num_cols):
                # Load a simd vector from B.
                var b_vec = _simd_load_maybe_partial[simd_size, partial_load_b](
                    b_ptr, j * simd_size, partial_load_b_size
                )

                @unroll
                for i in range(num_rows):
                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    var c_ptr = c + i * kernel_width + j * simd_size
                    c_ptr.store(
                        fma[c.type, simd_size](
                            a_vecs[i][lane].cast[c.type](),
                            b_vec.cast[c.type](),
                            c_ptr.load[width=simd_size](),
                        )
                    )

            b_ptr = b_ptr + b_stride

    # Load vectors from A first. The remainder is handled one element at a time.
    tile[micro_kernel, VariadicList[Int](simd_size, 1)](0, length)


@always_inline
fn _accumulate_neon[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Optional[Int] = None,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_base_offsets: Buffer[DType.int32, num_rows],
    a_offset: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Optional[Int] = None,
):
    """Accumulation optimized for NEON."""

    constrained[has_neon()]()

    @parameter
    if num_rows == 0 or num_cols == 0:
        return

    alias kernel_width = num_cols * simd_size

    var b_ptr = b

    @parameter
    @always_inline
    fn micro_kernel[num_lanes: Int](offset: Int):
        var a_vecs = stack_allocation[num_rows, SIMD[a.type, num_lanes]]()

        # Load vectors of size num_lanes from input.
        @unroll
        for i in range(num_rows):
            var a_idx = a_base_offsets[i].value + a_offset + offset
            a_vecs[i] = a.load[width=num_lanes](a_idx)

        var b_ptr = b + offset * b_stride

        @unroll
        for lane in range(num_lanes):

            @unroll
            for j in range(num_cols):
                # Load a simd vector from B.
                var b_vec = _simd_load_maybe_partial[simd_size, partial_load_b](
                    b_ptr, j * simd_size, partial_load_b_size
                )

                @unroll
                for i in range(num_rows):
                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    var c_ptr = c + i * kernel_width + j * simd_size
                    c_ptr.store(
                        fma[c.type, simd_size](
                            a_vecs[i][lane].cast[c.type](),
                            b_vec.cast[c.type](),
                            c_ptr.load[width=simd_size](),
                        )
                    )

            b_ptr = b_ptr + b_stride

    # Load vectors from A first. The remainder is handled one element at a time.
    tile[micro_kernel, VariadicList[Int](simd_size, 1)](0, length)


# ===----------------------------------------------------------------------===#
# Load/Store register tiles
# ===----------------------------------------------------------------------===#


@always_inline
fn init_register_tile[
    num_rows: Int, num_cols: Int, simd_size: Int
](tile: DTypePointer):
    """Initialize a register tile to zero.

    Parameters:
        num_rows: Number of rows in resigter tiling.
        num_cols: Number of columns in resigter tiling.
        simd_size: Number of lanes of a SIMD vector.

    Args:
        tile: Pointer to the register tile.
    """

    alias tile_width = num_cols * simd_size
    alias zero_vec = SIMD[tile.type, simd_size](0)

    @unroll
    for i in range(num_rows):

        @unroll
        for j in range(num_cols):
            tile.store(i * tile_width + j * simd_size, zero_vec)


@always_inline
fn load_register_tile[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    partial_load: Bool = False,
](
    tile: DTypePointer,
    input: DTypePointer,
    input_stride: Int,
    partial_load_size: Optional[Int] = None,
):
    """Load a register tile from the input buffer.

    Parameters:
        num_rows: Number of rows in resigter tiling.
        num_cols: Number of columns in resigter tiling.
        simd_size: Number of lanes of a SIMD vector.
        partial_load: Whether load input partially.

    Args:
        tile: Pointer to the register tile.
        input: Pointer to input buffer.
        input_stride: Stride between input segments of size `num_cols * simd_size`.
        partial_load_size: Size of partial load for input.
    """

    alias tile_width = num_cols * simd_size

    @always_inline
    @parameter
    fn body[i: Int, j: Int]():
        var input_ptr = input + i * input_stride + j * simd_size
        var tile_ptr = tile + i * tile_width + j * simd_size

        alias partial_load_last_vec = partial_load and (j == num_cols - 1)

        # TODO: check if partial_load_size has value.
        tile_ptr.store(
            _simd_load_maybe_partial[simd_size, partial_load_last_vec](
                input_ptr, 0, partial_load_size
            ).cast[tile_ptr.type]()
        )

    unroll[body, num_rows, num_cols]()


@always_inline
fn store_register_tile[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    partial_store: Bool = False,
](
    output: DTypePointer,
    output_stride: Int,
    tile: DTypePointer,
    partial_store_size: Optional[Int] = None,
):
    """Load a register tile from the input buffer.

    Parameters:
        num_rows: Number of rows in resigter tiling.
        num_cols: Number of columns in resigter tiling.
        simd_size: Number of lanes of a SIMD vector.
        partial_store: Whether store output partially.

    Args:
        output: Pointer to output buffer.
        output_stride: Stride between output segments of size `num_cols * simd_size`.
        tile: Pointer to the register tile.
        partial_store_size: Size of partial store to the output.
    """

    alias tile_width = num_cols * simd_size

    @always_inline
    @parameter
    fn body[i: Int, j: Int]():
        alias partial_store_last_vec = partial_store and (j == num_cols - 1)

        _simd_store_maybe_partial[simd_size, partial_store_last_vec](
            output,
            i * output_stride + j * simd_size,
            tile.load[width=simd_size](i * tile_width + j * simd_size).cast[
                output.type
            ](),
            partial_store_size,
        )

    unroll[body, num_rows, num_cols]()
