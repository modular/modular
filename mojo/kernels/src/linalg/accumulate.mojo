# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm.functional import tile
from memory.unsafe import DTypePointer
from memory.buffer import partial_simd_load
from memory import stack_allocation
from sys.info import has_neon
from sys.intrinsics import PrefetchOptions
from math import fma

# ===----------------------------------------------------------------------===#
# Helper Functions
# ===----------------------------------------------------------------------===#

# The default integer value for unused optional arguments/parameters.
alias UNUSED_INT = -1


@always_inline
fn _simd_load_maybe_partial[
    simd_size: Int, partial_load: Bool
](ptr: DTypePointer, offset: Int, partial_load_size: Int = UNUSED_INT) -> SIMD[
    ptr.type, simd_size
]:
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
            ptr + offset, 0, partial_load_size, 0.0
        )
    else:
        return ptr.simd_load[simd_size](offset)


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
fn accumulate_x86_simd[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Int = -1,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_stride: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Int = UNUSED_INT,
):
    """Compute c += a * b with register tiling on SIMD ISAs other than NEON.
    It has been optimized for AVX512 and AVX2.

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
    constrained[not has_neon()]()

    @parameter
    if num_rows == 0 or num_cols == 0:
        return

    alias kernel_width = num_cols * simd_size

    var b_ptr = b

    for l in range(length):
        # prefetch
        @parameter
        if prefetch_offset > 0:

            @unroll
            for i in range(num_cols):
                (b + prefetch_offset * kernel_width + i * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        @unroll
        for i in range(num_rows):

            @unroll
            for j in range(num_cols):
                # Broadcast an scalar from A to a simd vector.
                let a_splat_vec = SIMD[a.type, simd_size](a[l + i * a_stride])

                # Load a simd vector from B.
                let b_vec = _simd_load_maybe_partial[simd_size, partial_load_b](
                    b_ptr, j * simd_size, partial_load_b_size
                )

                # The following should be lifted to registers and show up as
                # FMA instructions.
                let c_ptr = c + i * kernel_width + j * simd_size
                c_ptr.simd_store(
                    fma(
                        a_splat_vec.cast[c.type](),
                        b_vec.cast[c.type](),
                        c_ptr.simd_load[simd_size](),
                    )
                )

        b_ptr = b_ptr + b_stride


# ===----------------------------------------------------------------------===#
# Accumulation optimized for AVX2 and AVX512
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
fn accumulate_neon[
    num_rows: Int,
    num_cols: Int,
    simd_size: Int,
    prefetch_offset: Int = -1,
    partial_load_b: Bool = False,
](
    length: Int,
    c: DTypePointer,
    a: DTypePointer,
    a_stride: Int,
    b: DTypePointer,
    b_stride: Int,
    partial_load_b_size: Int = UNUSED_INT,
):
    """Compute c += a * b with register tiling on SIMD ISAs other than NEON.
    It has been optimized for AVX512 and AVX2.

    Parameters:
        num_rows: Number of rows in resigter tiling.
        num_cols: Number of columns in resigter tiling.
        simd_size: Number of lanes of a SIMD vector.
        prefetch_offset: The distance to  prefetch ahead.
        partial_load_b: Whether use partial load for B.

    Args:
        length: Number of elements in accumulation.
        a: The input buffer A.
        b: The input buffer B.
        c: The output buffer, should have num_rows x num_cols x simd_size.
        a_stride: A's stride between each `length` segment.
        b_stride: B's stride between each `num_cols x simd_size` segment.
        b_end: B's end in it's contiguous dimension, i.e. last dim, row-majored.

    Don't use prefetch on Arm hardware for now.
    """
    constrained[has_neon()]()

    @parameter
    if num_rows == 0 or num_cols == 0:
        return

    alias kernel_width = num_cols * simd_size

    var b_ptr = b

    @parameter
    @always_inline
    fn micro_kernel[num_lanes: Int](offset: Int):
        let a_vecs = stack_allocation[num_rows, SIMD[a.type, num_lanes]]()

        # Load vectors of size num_lanes from input.
        @unroll
        for i in range(num_rows):
            a_vecs[i] = a.simd_load[num_lanes](offset + i * a_stride)

        var b_ptr = b + offset * b_stride

        @unroll
        for lane in range(num_lanes):

            @unroll
            for j in range(num_cols):
                # Load a simd vector from B.
                let b_vec = _simd_load_maybe_partial[simd_size, partial_load_b](
                    b_ptr, j * simd_size, partial_load_b_size
                )

                @unroll
                for i in range(num_rows):
                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    let c_ptr = c + i * kernel_width + j * simd_size
                    c_ptr.simd_store(
                        fma[c.type, simd_size](
                            a_vecs[i][lane].cast[c.type](),
                            b_vec.cast[c.type](),
                            c_ptr.simd_load[simd_size](),
                        )
                    )

            b_ptr = b_ptr + b_stride

    # Load vectors from A first. The remainder is handled one element at a time.
    tile[micro_kernel, VariadicList[Int](simd_size, 1)](0, length)
