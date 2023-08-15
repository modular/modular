# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module implements Transpose functions."""

from Assert import assert_param
from Buffer import Buffer, NDBuffer
from DType import DType
from Functional import (
    unroll,
    async_parallelize,
    sync_parallelize,
    vectorize,
    unroll,
    tile,
    unswitch,
)
from Index import StaticIntTuple, StaticTuple
from Intrinsics import strided_load, strided_store
from List import DimList, VariadicList
from LLCL import OutputChainPtr
from math import div_ceil, min
from Memory import memcpy
from Pointer import DTypePointer
from Range import range
from TargetInfo import sizeof, simdwidthof
from TypeUtilities import rebind
from SIMD import SIMD


fn _transpose_inplace_4x4[
    rows: Int,
    cols: Int,
    type: DType,
](bufloat0: NDBuffer[2, DimList(rows, cols), type]):
    assert_param[rows == 4]()
    assert_param[cols == 4]()
    let buf = rebind[
        NDBuffer[
            2,
            DimList(4, 4),
            type,
        ],
    ](bufloat0)

    let row0 = buf.simd_load[4](StaticIntTuple[2](0, 0))
    let row1 = buf.simd_load[4](StaticIntTuple[2](1, 0))
    let row2 = buf.simd_load[4](StaticIntTuple[2](2, 0))
    let row3 = buf.simd_load[4](StaticIntTuple[2](3, 0))

    let tmp0 = row0.shuffle[0, 1, 4, 5](row1)
    let tmp1 = row2.shuffle[0, 1, 4, 5](row3)
    let tmp2 = row0.shuffle[2, 3, 6, 7](row1)
    let tmp3 = row2.shuffle[2, 3, 6, 7](row3)

    let r0 = tmp0.shuffle[0, 2, 4, 6](tmp1)
    let r1 = tmp0.shuffle[1, 3, 5, 7](tmp1)
    let r2 = tmp2.shuffle[0, 2, 4, 6](tmp3)
    let r3 = tmp2.shuffle[1, 3, 5, 7](tmp3)

    buf.simd_store[4](StaticIntTuple[2](0, 0), r0)
    buf.simd_store[4](StaticIntTuple[2](1, 0), r1)
    buf.simd_store[4](StaticIntTuple[2](2, 0), r2)
    buf.simd_store[4](StaticIntTuple[2](3, 0), r3)


fn _transpose_inplace_8x8[
    rows: Int,
    cols: Int,
    type: DType,
](bufloat0: NDBuffer[2, DimList(rows, cols), type]):
    assert_param[rows == 8]()
    assert_param[cols == 8]()
    let buf = rebind[
        NDBuffer[
            2,
            DimList(8, 8),
            type,
        ],
    ](bufloat0)

    let row0 = buf.simd_load[8](StaticIntTuple[2](0, 0))
    let row1 = buf.simd_load[8](StaticIntTuple[2](1, 0))
    let row2 = buf.simd_load[8](StaticIntTuple[2](2, 0))
    let row3 = buf.simd_load[8](StaticIntTuple[2](3, 0))
    let row4 = buf.simd_load[8](StaticIntTuple[2](4, 0))
    let row5 = buf.simd_load[8](StaticIntTuple[2](5, 0))
    let row6 = buf.simd_load[8](StaticIntTuple[2](6, 0))
    let row7 = buf.simd_load[8](StaticIntTuple[2](7, 0))

    alias permute_0 = VariadicList[Int](0, 8, 1, 9, 4, 12, 5, 13)
    alias permute_1 = VariadicList[Int](2, 10, 3, 11, 6, 14, 7, 15)

    let k0 = row0._shuffle_list[permute_0](row1)
    let k1 = row0._shuffle_list[permute_1](row1)
    let k2 = row2._shuffle_list[permute_0](row3)
    let k3 = row2._shuffle_list[permute_1](row3)
    let k4 = row4._shuffle_list[permute_0](row5)
    let k5 = row4._shuffle_list[permute_1](row5)
    let k6 = row6._shuffle_list[permute_0](row7)
    let k7 = row6._shuffle_list[permute_1](row7)

    alias permute_2 = VariadicList[Int](0, 1, 8, 9, 4, 5, 12, 13)
    alias permute_3 = VariadicList[Int](2, 3, 10, 11, 6, 7, 14, 15)

    let k020 = k0._shuffle_list[permute_2](k2)
    let k021 = k0._shuffle_list[permute_3](k2)
    let k130 = k1._shuffle_list[permute_2](k3)
    let k131 = k1._shuffle_list[permute_3](k3)
    let k460 = k4._shuffle_list[permute_2](k6)
    let k461 = k4._shuffle_list[permute_3](k6)
    let k570 = k5._shuffle_list[permute_2](k7)
    let k571 = k5._shuffle_list[permute_3](k7)

    alias permute_4 = VariadicList[Int](0, 1, 2, 3, 8, 9, 10, 11)
    alias permute_5 = VariadicList[Int](4, 5, 6, 7, 12, 13, 14, 15)

    let r0 = k020._shuffle_list[permute_4](k460)
    let r1 = k021._shuffle_list[permute_4](k461)
    let r2 = k130._shuffle_list[permute_4](k570)
    let r3 = k131._shuffle_list[permute_4](k571)
    let r4 = k020._shuffle_list[permute_5](k460)
    let r5 = k021._shuffle_list[permute_5](k461)
    let r6 = k130._shuffle_list[permute_5](k570)
    let r7 = k131._shuffle_list[permute_5](k571)

    buf.simd_store[8](StaticIntTuple[2](0, 0), r0)
    buf.simd_store[8](StaticIntTuple[2](1, 0), r1)
    buf.simd_store[8](StaticIntTuple[2](2, 0), r2)
    buf.simd_store[8](StaticIntTuple[2](3, 0), r3)
    buf.simd_store[8](StaticIntTuple[2](4, 0), r4)
    buf.simd_store[8](StaticIntTuple[2](5, 0), r5)
    buf.simd_store[8](StaticIntTuple[2](6, 0), r6)
    buf.simd_store[8](StaticIntTuple[2](7, 0), r7)


fn _transpose_inplace_16x16[
    rows: Int,
    cols: Int,
    type: DType,
](bufloat0: NDBuffer[2, DimList(rows, cols), type]):
    assert_param[rows == 16]()
    assert_param[cols == 16]()
    let buf = rebind[
        NDBuffer[
            2,
            DimList(16, 16),
            type,
        ],
    ](bufloat0)

    alias permute_0 = VariadicList[Int](
        0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29
    )
    alias permute_1 = VariadicList[Int](
        2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31
    )
    alias permute_2 = VariadicList[Int](
        0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
    )
    alias permute_3 = VariadicList[Int](
        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
    )
    alias permute_4 = VariadicList[Int](
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
    )
    alias permute_5 = VariadicList[Int](
        4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    )
    alias permute_6 = VariadicList[Int](
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
    )
    alias permute_7 = VariadicList[Int](
        4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    )

    let row00 = buf.simd_load[16](StaticIntTuple[2](0, 0))
    let row01 = buf.simd_load[16](StaticIntTuple[2](1, 0))
    let row02 = buf.simd_load[16](StaticIntTuple[2](2, 0))
    let row03 = buf.simd_load[16](StaticIntTuple[2](3, 0))
    let row04 = buf.simd_load[16](StaticIntTuple[2](4, 0))
    let row05 = buf.simd_load[16](StaticIntTuple[2](5, 0))
    let row06 = buf.simd_load[16](StaticIntTuple[2](6, 0))
    let row07 = buf.simd_load[16](StaticIntTuple[2](7, 0))
    let row08 = buf.simd_load[16](StaticIntTuple[2](8, 0))
    let row09 = buf.simd_load[16](StaticIntTuple[2](9, 0))
    let row10 = buf.simd_load[16](StaticIntTuple[2](10, 0))
    let row11 = buf.simd_load[16](StaticIntTuple[2](11, 0))
    let row12 = buf.simd_load[16](StaticIntTuple[2](12, 0))
    let row13 = buf.simd_load[16](StaticIntTuple[2](13, 0))
    let row14 = buf.simd_load[16](StaticIntTuple[2](14, 0))
    let row15 = buf.simd_load[16](StaticIntTuple[2](15, 0))

    let k00 = row00._shuffle_list[permute_0](row01)
    let k01 = row00._shuffle_list[permute_1](row01)
    let k02 = row02._shuffle_list[permute_0](row03)
    let k03 = row02._shuffle_list[permute_1](row03)
    let k04 = row04._shuffle_list[permute_0](row05)
    let k05 = row04._shuffle_list[permute_1](row05)
    let k06 = row06._shuffle_list[permute_0](row07)
    let k07 = row06._shuffle_list[permute_1](row07)
    let k08 = row08._shuffle_list[permute_0](row09)
    let k09 = row08._shuffle_list[permute_1](row09)
    let k10 = row10._shuffle_list[permute_0](row11)
    let k11 = row10._shuffle_list[permute_1](row11)
    let k12 = row12._shuffle_list[permute_0](row13)
    let k13 = row12._shuffle_list[permute_1](row13)
    let k14 = row14._shuffle_list[permute_0](row15)
    let k15 = row14._shuffle_list[permute_1](row15)

    let j00 = k00._shuffle_list[permute_2](k02)
    let j01 = k00._shuffle_list[permute_3](k02)
    let j02 = k01._shuffle_list[permute_2](k03)
    let j03 = k01._shuffle_list[permute_3](k03)
    let j04 = k04._shuffle_list[permute_2](k06)
    let j05 = k04._shuffle_list[permute_3](k06)
    let j06 = k05._shuffle_list[permute_2](k07)
    let j07 = k05._shuffle_list[permute_3](k07)
    let j08 = k08._shuffle_list[permute_2](k10)
    let j09 = k08._shuffle_list[permute_3](k10)
    let j10 = k09._shuffle_list[permute_2](k11)
    let j11 = k09._shuffle_list[permute_3](k11)
    let j12 = k12._shuffle_list[permute_2](k14)
    let j13 = k12._shuffle_list[permute_3](k14)
    let j14 = k13._shuffle_list[permute_2](k15)
    let j15 = k13._shuffle_list[permute_3](k15)

    let t00 = j00._shuffle_list[permute_4](j04)
    let t01 = j01._shuffle_list[permute_4](j05)
    let t02 = j02._shuffle_list[permute_4](j06)
    let t03 = j03._shuffle_list[permute_4](j07)
    let t04 = j00._shuffle_list[permute_5](j04)
    let t05 = j01._shuffle_list[permute_5](j05)
    let t06 = j02._shuffle_list[permute_5](j06)
    let t07 = j03._shuffle_list[permute_5](j07)
    let t08 = j08._shuffle_list[permute_4](j12)
    let t09 = j09._shuffle_list[permute_4](j13)
    let t10 = j10._shuffle_list[permute_4](j14)
    let t11 = j11._shuffle_list[permute_4](j15)
    let t12 = j08._shuffle_list[permute_5](j12)
    let t13 = j09._shuffle_list[permute_5](j13)
    let t14 = j10._shuffle_list[permute_5](j14)
    let t15 = j11._shuffle_list[permute_5](j15)

    let r00 = t00._shuffle_list[permute_6](t08)
    let r01 = t01._shuffle_list[permute_6](t09)
    let r02 = t02._shuffle_list[permute_6](t10)
    let r03 = t03._shuffle_list[permute_6](t11)
    let r04 = t04._shuffle_list[permute_6](t12)
    let r05 = t05._shuffle_list[permute_6](t13)
    let r06 = t06._shuffle_list[permute_6](t14)
    let r07 = t07._shuffle_list[permute_6](t15)
    let r08 = t00._shuffle_list[permute_7](t08)
    let r09 = t01._shuffle_list[permute_7](t09)
    let r10 = t02._shuffle_list[permute_7](t10)
    let r11 = t03._shuffle_list[permute_7](t11)
    let r12 = t04._shuffle_list[permute_7](t12)
    let r13 = t05._shuffle_list[permute_7](t13)
    let r14 = t06._shuffle_list[permute_7](t14)
    let r15 = t07._shuffle_list[permute_7](t15)

    buf.simd_store[16](StaticIntTuple[2](0, 0), r00)
    buf.simd_store[16](StaticIntTuple[2](1, 0), r01)
    buf.simd_store[16](StaticIntTuple[2](2, 0), r02)
    buf.simd_store[16](StaticIntTuple[2](3, 0), r03)
    buf.simd_store[16](StaticIntTuple[2](4, 0), r04)
    buf.simd_store[16](StaticIntTuple[2](5, 0), r05)
    buf.simd_store[16](StaticIntTuple[2](6, 0), r06)
    buf.simd_store[16](StaticIntTuple[2](7, 0), r07)
    buf.simd_store[16](StaticIntTuple[2](8, 0), r08)
    buf.simd_store[16](StaticIntTuple[2](9, 0), r09)
    buf.simd_store[16](StaticIntTuple[2](10, 0), r10)
    buf.simd_store[16](StaticIntTuple[2](11, 0), r11)
    buf.simd_store[16](StaticIntTuple[2](12, 0), r12)
    buf.simd_store[16](StaticIntTuple[2](13, 0), r13)
    buf.simd_store[16](StaticIntTuple[2](14, 0), r14)
    buf.simd_store[16](StaticIntTuple[2](15, 0), r15)


fn _transpose_inplace_naive[
    rows: Int,
    cols: Int,
    type: DType,
](buf: NDBuffer[2, DimList(rows, cols), type]):
    for i in range(rows):
        for j in range(i + 1, cols):
            let tmp = buf[i, j]
            buf[StaticIntTuple[2](i, j)] = buf[j, i]
            buf[StaticIntTuple[2](j, i)] = tmp


fn transpose_inplace[
    rows: Int,
    cols: Int,
    type: DType,
](buf: NDBuffer[2, DimList(rows, cols), type]):
    # Reject sizes covered by specialized implementations
    assert_param[rows == cols]()

    @parameter
    if rows == 4:
        _transpose_inplace_4x4[rows, cols, type](buf)
    elif rows == 8:
        _transpose_inplace_8x8[rows, cols, type](buf)
    elif rows == 16:
        _transpose_inplace_16x16[rows, cols, type](buf)
    else:
        _transpose_inplace_naive[rows, cols, type](buf)


fn _permute_data[
    size: Int,
    type: DType,
](
    input: DTypePointer[type],
    output: DTypePointer[type],
    perms: DTypePointer[DType.index],
):
    """
    Ensures that output[i] = input[perms[i]] for i ∈ [0, size)
    """

    @always_inline
    @parameter
    fn body[idx: Int]():
        let perm_axis = perms.load(idx)[0].value
        let perm_data = input.load(perm_axis)
        output.store(idx, perm_data)

    unroll[size, body]()


fn _fill_strides[
    rank: Int,
    input_shape: DimList,
    type: DType,
](buf: NDBuffer[rank, input_shape, type], strides: DTypePointer[DType.index]):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    _fill_strides(buf, Buffer[rank, DType.index](strides))


fn _fill_strides[
    rank: Int,
    input_shape: DimList,
    type: DType,
](buf: NDBuffer[rank, input_shape, type], strides: Buffer[rank, DType.index]):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    assert_param[rank > 0]()
    strides[rank - 1] = 1

    @always_inline
    @parameter
    fn _fill_stride_at_idx[idx: Int]():
        alias axis = rank - idx - 2
        let next_axis_stride = strides[axis + 1]
        let next_axis_dim = buf.dim[axis + 1]()
        let curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride

    unroll[rank - 1, _fill_stride_at_idx]()


# ===------------------------------------------------------------------=== #
# Transpose Permutation simplification
# ===------------------------------------------------------------------=== #
@always_inline
fn _collapse_unpermuted_dims[
    rank: Int, tuple_size: Int
](
    inout simplified_shape: StaticIntTuple[tuple_size],
    inout simplified_perms: StaticIntTuple[tuple_size],
    dim: Int,
):
    let merged_dim = simplified_perms[dim]
    simplified_shape[merged_dim] = (
        simplified_shape[merged_dim] * simplified_shape[merged_dim + 1]
    )

    for j in range(merged_dim + 1, rank - 1):
        simplified_shape[j] = simplified_shape[j + 1]

    for i in range(rank):
        if simplified_perms[i] > merged_dim:
            simplified_perms[i] -= 1
    for k in range(dim + 1, rank - 1):
        simplified_perms[k] = simplified_perms[k + 1]
    simplified_shape[rank - 1] = 0
    simplified_perms[rank - 1] = 0


@always_inline
fn _delete_size_1_dim[
    rank: Int, tuple_size: Int
](
    inout simplified_shape: StaticIntTuple[tuple_size],
    inout simplified_perms: StaticIntTuple[tuple_size],
    dim: Int,
):
    for i in range(dim, rank - 1):
        simplified_shape[i] = simplified_shape[i + 1]

    var found_deleted: Bool = False
    for i in range(rank - 1):
        if simplified_perms[i] == dim:
            found_deleted = True
        if found_deleted:
            simplified_perms[i] = simplified_perms[i + 1]
        if simplified_perms[i] > dim:
            simplified_perms[i] -= 1

    simplified_shape[rank - 1] = 0
    simplified_perms[rank - 1] = 0


@always_inline
fn _simplify_transpose_perms_impl[
    rank: Int, tuple_size: Int
](
    inout simplified_rank: Int,
    inout simplified_shape: StaticIntTuple[tuple_size],
    inout simplified_perms: StaticIntTuple[tuple_size],
):
    @parameter
    if rank < 2:
        return

    else:
        for i in range(rank - 1):
            if simplified_perms[i] + 1 == simplified_perms[i + 1]:
                _collapse_unpermuted_dims[rank](
                    simplified_shape, simplified_perms, i
                )
                simplified_rank -= 1
                _simplify_transpose_perms_impl[rank - 1, tuple_size](
                    simplified_rank, simplified_shape, simplified_perms
                )
                return
            if simplified_shape[i] == 1:
                _delete_size_1_dim[rank](simplified_shape, simplified_perms, i)
                simplified_rank -= 1
                _simplify_transpose_perms_impl[rank - 1, tuple_size](
                    simplified_rank, simplified_shape, simplified_perms
                )
                return


@always_inline
fn _simplify_transpose_perms[
    rank: Int
](
    inout simplified_rank: Int,
    inout simplified_shape: StaticIntTuple[rank],
    inout simplified_perms: StaticIntTuple[rank],
):
    """Simplify the given permutation pattern.

    In some cases a permutation can be modeled by another permutation of a smaller rank.
    For instance, if we have
        shape=[1,3,200,200], perm = [0, 2, 3, 1]
    Then it is equivalent to:
        shape=[1,3,40000], perm = [0, 2, 1]
    Which in its turn is equivalent to:
        shape=[3,40000], perm = [1, 0]

    This function takes the original shape, permutation, and rank by reference,
    and updates their values to simplified ones.
    """
    _simplify_transpose_perms_impl[rank, rank](
        simplified_rank, simplified_shape, simplified_perms
    )


@always_inline
fn _convert_transpose_perms_to_static_int_tuple[
    rank: Int
](perms: DTypePointer[DType.index]) -> StaticIntTuple[rank]:
    var simplified_perms = StaticIntTuple[rank]()
    # TODO: unroll
    for j in range(rank):
        simplified_perms[j] = perms.load(j)[0].value
    return simplified_perms


# ===------------------------------------------------------------------=== #
# Functional additions
# ===------------------------------------------------------------------=== #
# TODO: Move to Memory.mojo
fn parallel_memcpy[
    type: DType
](
    dst_ptr: DTypePointer[type],
    src_ptr: DTypePointer[type],
    size: Int,
    task_size: Int,
    num_tasks: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    @always_inline
    fn _parallel_copy(thread_id: Int):
        let begin = task_size * thread_id
        let end = min(
            task_size * (thread_id + 1),
            size,
        )
        if begin >= size:
            return
        let to_copy = end - begin
        if to_copy <= 0:
            return

        memcpy(dst_ptr.offset(begin), src_ptr.offset(begin), to_copy)

    async_parallelize[_parallel_copy](out_chain, num_tasks)


# ===------------------------------------------------------------------=== #
#  Transpose special cases
# ===------------------------------------------------------------------=== #
@always_inline
fn _process_tile[
    tile_size_m: Int, tile_size_n: Int, type: DType
](
    m: Int,
    n: Int,
    M: Int,
    N: Int,
    out_ptr: DTypePointer[type],
    in_ptr: DTypePointer[type],
):

    let input_tile_offset = M * n + m
    let output_tile_offset = N * m + n

    let input_vals: StaticTuple[tile_size_n, SIMD[type, tile_size_m]]
    let output_vals: StaticTuple[tile_size_m, SIMD[type, tile_size_n]]

    @parameter
    @always_inline
    fn load_input_vals[count: Int]():
        input_vals[count] = in_ptr.simd_load[tile_size_m](
            input_tile_offset + M * count
        )

    unroll[tile_size_n, load_input_vals]()

    @parameter
    @always_inline
    fn compute_output_vals[m: Int, n: Int]():
        output_vals[m][n] = input_vals[n][m]

    unroll[tile_size_m, tile_size_n, compute_output_vals]()

    @parameter
    @always_inline
    fn store_output_vals[count: Int]():
        out_ptr.simd_store[tile_size_n](
            output_tile_offset + N * count, output_vals[count]
        )

    unroll[tile_size_m, store_output_vals]()


fn _transpose_2d_serial_tiled[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    simplified_input_shape: StaticIntTuple[rank],
    simplified_rank: Int,
    offset: Int,
    out_chain: OutputChainPtr,
):
    alias simd_width = simdwidthof[type]()

    @parameter
    if rank < 2:
        return
    # The input tile is MxN, the output tile is NxM.
    # We want to do:
    #   output[m, n] = input[n, m]
    # This is equivalent to:
    #   output[n*M + m] = input[m*N + n]
    # And we also have a global offset which needs to be added to both output
    # and input pointers.
    let N = simplified_input_shape[simplified_rank - 2]
    let M = simplified_input_shape[simplified_rank - 1]

    @parameter
    @always_inline
    fn process_tile[tile_size_m: Int, tile_size_n: Int](m: Int, n: Int):
        _process_tile[tile_size_m, tile_size_n, type](
            m, n, M, N, output.data.offset(offset), input.data.offset(offset)
        )

    alias tile_size = simd_width if simd_width <= 16 else 1
    tile[
        process_tile,
        VariadicList[Int](tile_size, 1),
        VariadicList[Int](tile_size, 1),
    ](0, 0, M, N)


@always_inline
fn _should_run_parallel(
    M: Int, N: Int, simd_width: Int, min_work_per_task: Int
) -> Bool:
    if N == 1:
        return False

    # Check if we can tile the space evenly
    if (N % simd_width) != 0 or (M % simd_width) != 0:
        return False

    let work_per_row = M * simd_width
    if min_work_per_task > work_per_row:
        # We will have to process several rows in each thread
        if (min_work_per_task % work_per_row) != 0:
            return False
        let rows_per_worker = div_ceil(min_work_per_task, work_per_row)
        if N // rows_per_worker < 4:
            return False

    return True


fn _transpose_2d_parallel_tiled[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    simplified_input_shape: StaticIntTuple[rank],
    simplified_rank: Int,
    offset: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    if rank < 2:
        return

    alias simd_width = simdwidthof[type]()
    let N = simplified_input_shape[simplified_rank - 2]
    let M = simplified_input_shape[simplified_rank - 1]
    alias min_work_per_task = 1024
    alias tile_size_m = simd_width if simd_width <= 16 else 1
    alias tile_size_n = simd_width if simd_width <= 16 else 1

    let n_unit_size = simd_width
    let m_unit_size = simd_width

    let n_tiles = N // n_unit_size
    let m_tiles = M // m_unit_size

    var rows_per_worker = 1  # Row in terms of tiles, i.e. we still take simd_width elements
    if min_work_per_task > M * simd_width:
        rows_per_worker = min_work_per_task // (M * simd_width)

    var work = div_ceil(n_tiles, rows_per_worker)

    let num_threads = out_chain.get_runtime().parallelism_level()

    let num_tasks = min(work, num_threads)

    let work_block_size = div_ceil(work, num_tasks)

    @parameter
    @always_inline
    fn _parallel_tile(thread_id: Int):
        let n_tile_begin = work_block_size * thread_id
        let n_tile_end = min(work_block_size * (thread_id + 1), work)

        for n_tile in range(n_tile_begin, n_tile_end):
            for m_tile in range(m_tiles):
                let m = tile_size_m * m_tile
                let n = tile_size_n * n_tile
                _process_tile[tile_size_m, tile_size_n, type](
                    m,
                    n,
                    M,
                    N,
                    output.data.offset(offset),
                    input.data.offset(offset),
                )

    async_parallelize[_parallel_tile](out_chain, num_tasks)


fn transpose_2d[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    simplified_input_shape: StaticIntTuple[rank],
    simplified_rank: Int,
    offset: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    if rank < 2:
        return

    alias simd_width = simdwidthof[type]()
    let N = simplified_input_shape[simplified_rank - 2]
    let M = simplified_input_shape[simplified_rank - 1]
    alias min_work_per_task = 1024

    if out_chain and _should_run_parallel(M, N, simd_width, min_work_per_task):
        _transpose_2d_parallel_tiled[rank, output_shape, input_shape](
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
            out_chain,
        )
    else:
        _transpose_2d_serial_tiled[rank, output_shape, input_shape](
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
            out_chain,
        )
        if out_chain:
            out_chain.mark_ready()
        return


fn _transpose_4d_swap_middle_helper[
    type: DType,
](
    dst_ptr: DTypePointer[type],
    src_ptr: DTypePointer[type],
    L: Int,
    M: Int,
    N: Int,
    K: Int,
    out_chain: OutputChainPtr,
):
    let work = L * M * N
    let memcpy_block_size = K
    let total_size = L * M * N * K

    alias KB = 1024

    # TODO: These parameters might be tuned
    alias min_work_per_task = 1 * KB
    alias min_work_for_parallel = 4 * min_work_per_task

    # TODO: take into account dimension K for parallelization.
    #
    # E.g. if we're transposing 2x3x8192 -> 3x2x8192, then parallelizing just
    # on dimensions M and N is not enough.
    if not out_chain or total_size <= min_work_for_parallel:
        for l in range(L):
            for m in range(M):
                for n in range(N):
                    # We want to do:
                    #   output[l, n, m, k] = input[l, m, n, k]
                    let in_off = l * M * N * K + m * N * K + n * K
                    let out_off = l * M * N * K + n * M * K + m * K
                    memcpy(dst_ptr.offset(out_off), src_ptr.offset(in_off), K)
        if out_chain:
            out_chain.mark_ready()
        return
    else:
        let num_threads = out_chain.get_runtime().parallelism_level()

        let num_tasks = min(work, num_threads)

        let work_block_size = div_ceil(work, num_tasks)

        @parameter
        @always_inline
        fn _parallel_copy(thread_id: Int):
            let begin = work_block_size * thread_id
            let end = min(work_block_size * (thread_id + 1), work)
            for block_idx in range(begin, end):
                let l = block_idx // (M * N)
                let block_idx_mn = block_idx % (M * N)
                let m = block_idx_mn // N
                let n = block_idx_mn % N

                let in_off = l * M * N * K + m * N * K + n * K
                let out_off = l * M * N * K + n * M * K + m * K
                memcpy(dst_ptr.offset(out_off), src_ptr.offset(in_off), K)

        async_parallelize[_parallel_copy](out_chain, num_tasks)


fn transpose_4d_swap_middle[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    simplified_input_shape: StaticIntTuple[rank],
    simplified_rank: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    if rank < 4:
        return
    # The input tile is LxMxNxK, the output tile is LxNxMxK.
    # We want to do:
    #   output[l, n, m, k] = input[l, m, n, k]
    let L = simplified_input_shape[simplified_rank - 4]
    let M = simplified_input_shape[simplified_rank - 3]
    let N = simplified_input_shape[simplified_rank - 2]
    let K = simplified_input_shape[simplified_rank - 1]
    let src_ptr = input.data.offset(0)
    let dst_ptr = output.data.offset(0)
    _transpose_4d_swap_middle_helper[type](
        dst_ptr, src_ptr, L, M, N, K, out_chain
    )


fn transpose_3d_swap_outer[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    simplified_input_shape: StaticIntTuple[rank],
    simplified_rank: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    if rank < 3:
        return
    # The input tile is MxNxK, the output tile is NxMxK.
    # We want to do:
    #   output[n, m, k] = input[m, n, k]
    # We use a 4d helper function for this, pretending that we have an outer
    # dimensions L=1.
    let M = simplified_input_shape[simplified_rank - 3]
    let N = simplified_input_shape[simplified_rank - 2]
    let K = simplified_input_shape[simplified_rank - 1]
    let src_ptr = input.data.offset(0)
    let dst_ptr = output.data.offset(0)
    _transpose_4d_swap_middle_helper[type](
        dst_ptr, src_ptr, 1, M, N, K, out_chain
    )


fn transpose_3d_swap_inner[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    simplified_input_shape: StaticIntTuple[rank],
    simplified_rank: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    if rank < 3:
        return
    # simplified perms must be 0, 2, 1
    var offset = 0
    let step = simplified_input_shape[
        simplified_rank - 2
    ] * simplified_input_shape[simplified_rank - 1]
    # TODO: parallelize this loop
    for i in range(simplified_input_shape[0]):
        _transpose_2d_serial_tiled[rank, output_shape, input_shape, type](
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
            out_chain,
        )
        offset += step
    if out_chain:
        out_chain.mark_ready()


fn transpose_trivial_memcpy[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    out_chain: OutputChainPtr,
):
    let src_ptr = input.data.offset(0)
    let dst_ptr = output.data.offset(0)

    alias KB = 1024
    alias min_work_per_task = 1 * KB
    alias min_work_for_parallel = 4 * min_work_per_task

    let total_size = output.size()

    if not out_chain or total_size <= min_work_for_parallel:
        memcpy(dst_ptr, src_ptr, total_size)
        if out_chain:
            out_chain.mark_ready()
    else:
        let work_units = div_ceil(total_size, min_work_per_task)
        let num_tasks = min(
            work_units, out_chain.get_runtime().parallelism_level()
        )
        let work_block_size = div_ceil(work_units, num_tasks)

        parallel_memcpy[type](
            dst_ptr,
            src_ptr,
            total_size,
            work_block_size * min_work_per_task,
            num_tasks,
            out_chain,
        )


# ===------------------------------------------------------------------=== #
#  Transpose generic strided implementation
# ===------------------------------------------------------------------=== #
fn _copy_with_strides[
    rank: Int,
    output_shape: DimList,
    type: DType,
](
    axis: Int,
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    input_strides: DTypePointer[DType.index],
    output_strides: DTypePointer[DType.index],
    input_offset: Int,
    output_offset: Int,
    out_chain: OutputChainPtr,
):
    """
    Copy data from `input` to `output`, starting at corresponding offsets,
    based on given strides.

    Args:
        output: the output buffer
        input: the input buffer
        input_strides: the stride at each input axis
        output_strides: the stride at each output axis
        input_offset: The offset at which input data starts
        output_offset: The offset at which output data starts
    """
    if axis + 1 > rank and out_chain:
        return out_chain.mark_error("out of range")

    let axis_dim = output.dim(axis)
    let input_axis_stride: Int = input_strides.load(axis)[0].value
    let output_axis_stride: Int = output_strides.load(axis)[0].value

    if axis + 1 == rank:
        var src_ptr = input.offset(input_offset)
        var dst_ptr = output.data.offset(output_offset)
        if input_axis_stride == 1 and output_axis_stride == 1:
            memcpy(dst_ptr, src_ptr, axis_dim)
        else:

            @always_inline
            @parameter
            fn _copy[simd_width: Int](offset: Int):
                strided_store(
                    strided_load[type, simd_width](src_ptr, input_axis_stride),
                    dst_ptr,
                    output_axis_stride,
                )
                src_ptr = src_ptr.offset(simd_width * input_axis_stride)
                dst_ptr = dst_ptr.offset(simd_width * output_axis_stride)

            vectorize[simdwidthof[type](), _copy](axis_dim)

        if out_chain:
            out_chain.mark_ready()
        return

    let next_axis = axis + 1

    alias KB = 1024

    # TODO: These parameters might be tuned
    alias min_work_per_task = 1 * KB
    alias min_work_for_parallel = 4 * min_work_per_task

    if (
        not out_chain
        or output.bytecount() <= min_work_for_parallel
        or axis_dim == 1
    ):
        var next_input_offset = input_offset
        var next_output_offset = output_offset
        for _ in range(axis_dim):
            _copy_with_strides[rank, output_shape, type](
                next_axis,
                output,
                input,
                input_strides,
                output_strides,
                next_input_offset,
                next_output_offset,
                OutputChainPtr(),
            )
            next_input_offset += input_axis_stride
            next_output_offset += output_axis_stride
        if out_chain:
            out_chain.mark_ready()
    else:
        let num_threads = out_chain.get_runtime().parallelism_level()
        let num_tasks = min(
            div_ceil(output.bytecount(), min_work_per_task), num_threads
        )

        let work = axis_dim
        let work_block_size = div_ceil(work, num_tasks)

        @always_inline
        @parameter
        fn _parallel_copy(thread_id: Int):

            var next_input_offset = (
                thread_id * work_block_size * input_axis_stride + input_offset
            )
            var next_output_offset = (
                thread_id * work_block_size * output_axis_stride + output_offset
            )

            for _ in range(
                work_block_size * thread_id,
                min(work_block_size * (thread_id + 1), work),
            ):
                _copy_with_strides[rank, output_shape, type](
                    next_axis,
                    output,
                    input,
                    input_strides,
                    output_strides,
                    next_input_offset,
                    next_output_offset,
                    OutputChainPtr(),
                )
                next_input_offset += input_axis_stride
                next_output_offset += output_axis_stride

        # TODO: transpose_strided is using stack allocated structueres and
        # so depends on us being synchronous. We need a better way to do this.
        sync_parallelize[_parallel_copy](out_chain, num_tasks)


fn transpose_strided[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    out_chain: OutputChainPtr,
):
    # Compute `permuted_input_strides`
    let input_strides = DTypePointer[DType.index].alloc(rank)
    let permuted_input_strides = DTypePointer[DType.index].alloc(rank)
    _fill_strides(input, input_strides)
    _permute_data[rank, DType.index](
        input_strides, permuted_input_strides, perms
    )
    # Compute `output_strides`
    let output_strides = DTypePointer[DType.index].alloc(rank)
    _fill_strides(output, output_strides)
    # Kickoff; for intuition on permuted input strides, note that
    #   transpose(output, input, [2, 0, 1])
    # guarantees
    #   (let isx denote input_stride_x, etc.)
    #   output[x, y, z] = input[z, x, y]
    # ~ output.at(offset(x*isx + y*isy + z*isz)) = input.at(offset(z*osx + x*osy + y*osz))
    # ~ output.at(offset(x*isx + y*isy + z*isz)) = input.at(offset(x*osy + y*osz + z*osx))
    # ~ output.at(offset([x, y, z], output_strides)) = input.at(offset([x, y, z], permuted_input_strides))
    # ~ output.at(offset(index, output_strides)) = input.at(offset(index, permuted_input_strides))
    alias init_axis = 0
    # NOTE: Synchronous, so the stack allocated input_strides, permuted_input_strings
    # and output_strides are safe to use.
    _copy_with_strides[rank, output_shape, type](
        init_axis,
        output,
        input.data,
        permuted_input_strides,
        output_strides,
        0,  # input_offset
        0,  # output_offset
        out_chain,
    )
    input_strides.free()
    permuted_input_strides.free()
    output_strides.free()


# ===------------------------------------------------------------------=== #
#  Transpose entry points
# ===------------------------------------------------------------------=== #
fn transpose[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
):
    """
    Permute the axis of `input` based on `perms`, and place the result in
    `output`.

    Example:
        transpose(output, input, [2, 0, 1])
        # guarantees output[x, y, z] = input[z, x, y]

    Parameters:
        rank: The rank of input and output buffers.
        output_shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
        type: The dtype of buffer elements.

    Args:
        output: The output buffer.
        input: The input buffer.
        perms: Permutation of the input axes.
    """
    transpose[rank, output_shape, input_shape, type](
        output, input, perms, OutputChainPtr()
    )


fn transpose[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
    out_chain: OutputChainPtr,
):
    """
    Permute the axis of `input` based on `perms`, and place the result in
    `output`.

    Example:
        transpose(output, input, [2, 0, 1])
        # guarantees output[x, y, z] = input[z, x, y]

    Parameters:
        rank: The rank of input and output buffers.
        output_shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
        type: The dtype of buffer elements.

    Args:
        output: The output buffer.
        input: The input buffer.
        perms: Permutation of the input axes.
        out_chain: The chain to attach to.
    """

    # If either input or output is not-contiguous, we need to use a general
    # strided implementation of transpose
    if not output.is_contiguous or not input.is_contiguous:
        return transpose_strided[rank, output_shape, input_shape, type](
            output, input, perms, out_chain
        )

    # If they are contiguous, we can try to recognize common special cases in
    # the desired permutation.
    # E.g.
    #   shape=[1,3,200,200], perm = [0, 2, 3, 1]
    # is equivalent to
    #   shape=[1,3,40000], perm = [0, 2, 1]
    #
    # And that just swaps two inner dimensions.
    var simplified_perms = _convert_transpose_perms_to_static_int_tuple[rank](
        perms
    )
    var simplified_shape = input.get_shape()
    var simplified_rank = rank
    _simplify_transpose_perms[rank](
        simplified_rank, simplified_shape, simplified_perms
    )

    if simplified_rank == 1:
        # memcpy
        return transpose_trivial_memcpy[rank, output_shape, input_shape, type](
            output, input, out_chain
        )
    # TODO: Reenable once #15947 is fixed.
    # elif simplified_rank == 2:
    #     # tiled transpose
    #     return transpose_2d[rank, output_shape, input_shape, type](
    #         output,
    #         input,
    #         perms,
    #         simplified_shape,
    #         simplified_rank,
    #         0,
    #         out_chain,
    #     )
    elif rank >= 3 and simplified_rank == 3:
        if (
            simplified_perms[0] == 0
            and simplified_perms[1] == 2
            and simplified_perms[2] == 1
        ):
            # batched tiled transpose
            return transpose_3d_swap_inner[
                rank, output_shape, input_shape, type
            ](
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
                out_chain,
            )
        elif (
            simplified_perms[0] == 1
            and simplified_perms[1] == 0
            and simplified_perms[2] == 2
        ):
            return transpose_3d_swap_outer[
                rank, output_shape, input_shape, type
            ](
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
                out_chain,
            )
    elif rank >= 4 and simplified_rank == 4:
        if (
            simplified_perms[0] == 0
            and simplified_perms[1] == 2
            and simplified_perms[2] == 1
            and simplified_perms[3] == 3
        ):
            return transpose_4d_swap_middle[
                rank, output_shape, input_shape, type
            ](
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
                out_chain,
            )
    transpose_strided[rank, output_shape, input_shape, type](
        output, input, perms, out_chain
    )
