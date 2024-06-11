# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import ceildiv
from sys.info import (
    has_avx512f,
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    is_apple_silicon,
    is_x86,
)
from sys.intrinsics import llvm_intrinsic

from algorithm import sync_parallelize, tile, vectorize
from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.accumulate import _Accumulator
from LinAlg.neon_intrinsics import _neon_dotprod_lane, _neon_matmul
from LinAlg.vnni_intrinsics import (
    dot_i8_to_i32_saturated_x86,
    dot_i16_to_i32_x86,
)
from memory import UnsafePointer
from memory.unsafe import DTypePointer

from utils import InlineArray
from utils.index import Index

from ._utils import roundeven_to_int32


@always_inline
fn _to_dtype_pointer[
    type: DType
](array: InlineArray[Scalar[type]]) -> DTypePointer[type]:
    return DTypePointer[type](array.unsafe_ptr())


struct _block_QK_K:
    alias quantized_k = 256

    @staticmethod
    fn calc_group_count[group_size: Int]() -> Int:
        return _block_QK_K.quantized_k // group_size


struct _block_Q4_K:
    alias group_size = 32
    alias group_count = _block_QK_K.calc_group_count[Self.group_size]()

    var base_scale: Float16
    var base_min: Float16
    var q_scales_and_mins: InlineArray[
        UInt8, (2 * _block_Q4_K.group_count * 6) // 8
    ]
    var q_bits: InlineArray[UInt8, _block_QK_K.quantized_k // 2]


struct _block_Q6_K:
    alias group_size = 16
    alias group_count = _block_QK_K.calc_group_count[Self.group_size]()

    var q_bits_lo: InlineArray[UInt8, _block_QK_K.quantized_k // 2]
    var q_bits_hi: InlineArray[UInt8, _block_QK_K.quantized_k // 4]
    var q_scales: InlineArray[Int8, _block_Q6_K.group_size]
    var base_scale: Float16


struct _packed_int4_array[size: Int]:
    """Packed storage for an array of uint4 data."""

    var lo: InlineArray[UInt8, size // 2]

    @always_inline
    fn pack(self, owned src_ptr: DTypePointer[DType.uint8]):
        """Packs the supplied external buffer to local storage."""
        constrained[(self.size % 4) == 0, "size should be multiple of 2"]()

        var lo_ptr = _to_dtype_pointer(self.lo)

        @parameter
        @always_inline
        fn do_pack[simd_width: Int](idx: Int):
            var packed_lo_bits = SIMD[DType.uint8, simd_width](0)

            @parameter
            for i in range(2):
                var bytes = SIMD[size=simd_width].load(src_ptr)
                src_ptr += simd_width

                packed_lo_bits |= (bytes & 15) << (i * 4)

            SIMD.store(lo_ptr, packed_lo_bits)
            lo_ptr += simd_width

        vectorize[do_pack, simdwidthof[DType.uint8]()](size // 2)

    @always_inline
    fn unpack(self, owned dst_ptr: DTypePointer[DType.uint8]):
        """Unpacks the local storage to the supplied external buffer."""
        constrained[(self.size % 2) == 0, "size should be multiple of 2"]()

        var lo_ptr = _to_dtype_pointer(self.lo)

        @parameter
        @always_inline
        fn do_unpack[simd_width: Int](idx: Int):
            var packed_lo_bits = SIMD[size=simd_width].load(lo_ptr)
            lo_ptr += simd_width

            @parameter
            for i in range(2):
                var lo_bits = (packed_lo_bits >> (i * 4)) & 15

                SIMD.store(dst_ptr, lo_bits)
                dst_ptr += simd_width

        vectorize[do_unpack, simdwidthof[DType.uint8]()](size // 2)


struct _packed_int6_array[size: Int]:
    """Packed storage for an array of uint6 data."""

    var lo: InlineArray[UInt8, size // 2]
    var hi: InlineArray[UInt8, size // 4]

    @always_inline
    fn pack(self, owned src_ptr: DTypePointer[DType.uint8]):
        """Packs the supplied external buffer to local storage."""
        constrained[(self.size % 4) == 0, "size should be multiple of 4"]()

        var hi_ptr = _to_dtype_pointer(self.hi)
        var lo_ptr = _to_dtype_pointer(self.lo)

        @parameter
        @always_inline
        fn do_pack[simd_width: Int](idx: Int):
            var packed_hi_bits = SIMD[DType.uint8, simd_width](0)

            @parameter
            for i in range(0, 4, 2):
                var packed_lo_bits = SIMD[DType.uint8, simd_width](0)

                @parameter
                for j in range(2):
                    var bytes = SIMD[size=simd_width].load(src_ptr)
                    src_ptr += simd_width

                    packed_lo_bits |= (bytes & 15) << (j * 4)
                    packed_hi_bits |= (bytes >> 4) << ((i + j) * 2)

                SIMD.store(lo_ptr, packed_lo_bits)
                lo_ptr += simd_width

            SIMD.store(hi_ptr, packed_hi_bits)
            hi_ptr += simd_width

        vectorize[do_pack, simdwidthof[DType.uint8]()](size // 4)

    @always_inline
    fn unpack[
        *, zero_point: UInt8 = 0
    ](self, owned dst_ptr: DTypePointer[DType.uint8]):
        """Unpacks the local storage to the supplied external buffer."""
        constrained[(self.size % 4) == 0, "size should be multiple of 4"]()

        var hi_ptr = _to_dtype_pointer(self.hi)
        var lo_ptr = _to_dtype_pointer(self.lo)

        @parameter
        @always_inline
        fn do_unpack[simd_width: Int](idx: Int):
            var packed_hi_bits = SIMD[size=simd_width].load(hi_ptr)
            hi_ptr += simd_width

            @parameter
            for i in range(0, 4, 2):
                var packed_lo_bits = SIMD[size=simd_width].load(lo_ptr)
                lo_ptr += simd_width

                @parameter
                for j in range(2):
                    var hi_bits = ((packed_hi_bits >> ((i + j) * 2)) & 3) << 4
                    var lo_bits = (packed_lo_bits >> (j * 4)) & 15

                    SIMD.store(dst_ptr, (hi_bits | lo_bits) - zero_point)
                    dst_ptr += simd_width

        vectorize[do_unpack, simdwidthof[DType.uint8]()](size // 4)


struct _block_Q4_K_packed[block_n: Int = 1]:
    var base_scales: InlineArray[Float16, block_n]
    var base_mins: InlineArray[Float16, block_n]
    var q_scales_and_mins: _packed_int6_array[
        2 * _block_Q4_K.group_count * block_n
    ]
    var q_bits: _packed_int4_array[_block_QK_K.quantized_k * block_n]


struct _block_Q6_K_packed[block_n: Int = 1]:
    var base_scales: InlineArray[Float16, block_n]
    var q_scales: InlineArray[Int8, _block_Q6_K.group_count * block_n]
    var q_bits: _packed_int6_array[_block_QK_K.quantized_k * block_n]


struct _block_Q8_K_packed[group_size: Int, tile_m: Int = 1]:
    alias group_count = _block_QK_K.calc_group_count[group_size]()

    var q_bits: InlineArray[Int8, _block_QK_K.quantized_k * tile_m]
    var scales: InlineArray[Float32, tile_m]
    var group_sums: InlineArray[Int16, Self.group_count * tile_m]


fn _quantize_a_Q8_K[
    group_size: Int, type: DType, *, interleave_group_sums: Bool = False
](a: NDBuffer[type, 2]) -> UnsafePointer[_block_Q8_K_packed[group_size]]:
    alias quantized_k = _block_QK_K.quantized_k
    alias group_count = quantized_k // group_size

    var M = a.dim[0]()
    var K = a.dim[1]()

    var packed_base_ptr = UnsafePointer[_block_Q8_K_packed[group_size]].alloc(
        M * (K // quantized_k)
    )
    var packed_ptr = packed_base_ptr

    for ko in range(0, K, quantized_k):
        var am_ptr = a.data + ko

        @parameter
        @always_inline
        fn process_rows[tile_m: Int](m: Int):
            constrained[
                sizeof[_block_Q8_K_packed[group_size]]() * tile_m
                == sizeof[_block_Q8_K_packed[group_size, tile_m]](),
                "tiled block size should be multiple of the single block size",
            ]()

            var block_ptr = packed_ptr.bitcast[
                _block_Q8_K_packed[group_size, tile_m]
            ]()
            var q_bits_ptr = _to_dtype_pointer(block_ptr[].q_bits)

            for row in range(tile_m):
                var max_value_simd = SIMD[type, group_size](Scalar[type].MIN)

                for g in range(group_count):
                    var fp_data = SIMD[size=group_size].load(
                        am_ptr, g * group_size
                    )
                    max_value_simd = abs(fp_data).max(max_value_simd)

                var max_value = max_value_simd.reduce_max()
                var scale = (max_value / 127.0).cast[DType.float32]()
                var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

                for g in range(group_count):
                    var fp_data = SIMD[size=group_size].load(
                        am_ptr, g * group_size
                    )
                    var q_data_i32 = roundeven_to_int32(fp_data * multiplier)
                    var q_data_i8 = q_data_i32.cast[DType.int8]()
                    var group_sum = q_data_i32.reduce_add()

                    SIMD.store(
                        q_bits_ptr,
                        g * tile_m * group_size + row * group_size,
                        q_data_i8,
                    )

                    @parameter
                    if interleave_group_sums:
                        block_ptr[].group_sums[
                            ((g >> 1) * tile_m + row) * 2 + (g & 1)
                        ] = group_sum.cast[DType.int16]()
                    else:
                        block_ptr[].group_sums[
                            g * tile_m + row
                        ] = group_sum.cast[DType.int16]()

                block_ptr[].scales[row] = scale

                am_ptr += K

            packed_ptr += tile_m

        tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)

    return packed_base_ptr


fn _expand_q_bits_lo[
    *, width: Int
](
    owned src_ptr: DTypePointer[DType.uint8],
    owned dst_ptr: DTypePointer[DType.uint8],
):
    for k in range(0, _block_QK_K.quantized_k // 2, width):
        var src_q_bits = SIMD[size=width].load(src_ptr)
        src_ptr += width

        @parameter
        for i in range(2):
            SIMD.store(dst_ptr, (src_q_bits >> (i * 4)) & 15)
            dst_ptr += width


fn _expand_and_merge_q_bits_hi[
    *, width: Int, bit_count: Int
](
    owned src_ptr: DTypePointer[DType.uint8],
    owned dst_ptr: DTypePointer[DType.uint8],
):
    alias values_per_byte = 8 // bit_count
    alias bit_mask = (1 << bit_count) - 1

    for k in range(0, _block_QK_K.quantized_k // values_per_byte, width):
        var src_q_bits = SIMD[size=width].load(src_ptr)
        src_ptr += width

        for i in range(values_per_byte):
            var dst_q_bits_lo = SIMD[size=width].load(dst_ptr)
            var dst_q_bits_hi = (src_q_bits & bit_mask) << 4
            src_q_bits >>= bit_count

            SIMD.store(dst_ptr, dst_q_bits_hi | dst_q_bits_lo)
            dst_ptr += width


fn _copy_column_q_bits_to_block[
    block_n: Int
](
    owned src_ptr: DTypePointer[DType.uint8],
    owned dst_ptr: DTypePointer[DType.uint8],
):
    """Interleaves the linear source buffer to the blocked destination
    buffer.
    """
    for k in range(0, _block_QK_K.quantized_k, 4):
        SIMD.store(dst_ptr, SIMD[size=4].load(src_ptr))
        src_ptr += 4
        dst_ptr += block_n * 4


fn _pack_block_Q4_K[
    block_n: Int
](
    owned src_ptr: UnsafePointer[_block_Q4_K],
    stride: Int,
    dst_ptr: UnsafePointer[_block_Q4_K_packed[block_n]],
):
    alias group_size = _block_Q4_K.group_size
    alias group_count = _block_Q4_K.group_count

    constrained[
        sizeof[_block_Q4_K]() * block_n
        == sizeof[_block_Q4_K_packed[block_n]](),
        "packed block size should be multiple of the unpacked block size",
    ]()

    var q_scales_buf = stack_allocation[group_count * block_n, DType.uint8]()
    var q_mins_buf = stack_allocation[group_count * block_n, DType.uint8]()
    var q_bits_block_buf = stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8
    ]()

    for n in range(block_n):
        dst_ptr[].base_scales[n] = src_ptr[].base_scale
        dst_ptr[].base_mins[n] = src_ptr[].base_min

        # Decode the packed 6-bit scales and minimums to a local working buffer.
        for g in range(group_count):
            var q_scale: UInt8
            var q_min: UInt8
            if g < 4:
                q_scale = src_ptr[].q_scales_and_mins[g] & 63
                q_min = src_ptr[].q_scales_and_mins[g + 4] & 63
            else:
                var q_scale_lo = src_ptr[].q_scales_and_mins[g + 4] & 15
                var q_min_lo = src_ptr[].q_scales_and_mins[g + 4] >> 4
                var q_scale_hi = src_ptr[].q_scales_and_mins[g - 4] >> 6
                var q_min_hi = src_ptr[].q_scales_and_mins[g - 0] >> 6
                q_scale = (q_scale_hi << 4) | q_scale_lo
                q_min = (q_min_hi << 4) | q_min_lo
            q_scales_buf[g * block_n + n] = q_scale
            q_mins_buf[g * block_n + n] = q_min

        var q_bits_column_buf = stack_allocation[
            _block_QK_K.quantized_k, DType.uint8
        ]()

        _expand_q_bits_lo[width=32](
            _to_dtype_pointer(src_ptr[].q_bits), q_bits_column_buf
        )
        _copy_column_q_bits_to_block[block_n](
            q_bits_column_buf, q_bits_block_buf + n * 4
        )

        src_ptr += stride

    # Allocate a staging buffer to pack the scales and minimums as a single
    # blob and to do processor specific reordering of the values for the
    # compute kernel.
    var q_scales_and_mins_buf = stack_allocation[
        2 * group_count * block_n, DType.uint8
    ]()
    var q_scales_reorder_buf = q_scales_and_mins_buf
    var q_mins_reorder_buf = q_scales_and_mins_buf + group_count * block_n

    # Scales are not currently transformed.
    memcpy(q_scales_reorder_buf, q_scales_buf, group_count * block_n)

    # Minimums are row interleaved with a stride to enable use of int16->int32
    # multiply/add instructions.
    #
    # For x86: The compute kernel uses `pmaddwd` + `paddd' (optimized to
    # `vpdpwssd` on processors that support VNNI). The two rows are interleaved
    # to form pairs of int16 values:
    #       [n0_g0 n0_g1 : n1_g0 n1_g1 : n2_g0 n2_g1 : n3_g0 n3_g1]
    #
    # For NEON: The compute kernel uses `smull(2)` and `smlal(2)` instructions
    # to do an `int16*int16` widening multiply/add to an int32 accumulator. The
    # two rows are split across the lower and upper halves of the register:
    #       [n0_g0 n1_g0 n2_g0 n3_g0 : n0_g1 n1_g1 n2_g1 n3_g1]
    for g in range(0, group_count, 2):
        var q_mins_row_0_ptr = q_mins_buf + g * block_n
        var q_mins_row_1_ptr = q_mins_row_0_ptr + block_n
        for n in range(block_n):
            var q_mins_row_0_val = q_mins_row_0_ptr[n]
            var q_mins_row_1_val = q_mins_row_1_ptr[n]

            @parameter
            if is_x86():
                var reorder_idx = g * block_n + n * 2
                q_mins_reorder_buf[reorder_idx + 0] = q_mins_row_0_val
                q_mins_reorder_buf[reorder_idx + 1] = q_mins_row_1_val
            elif has_neon():
                alias split_width = simdwidthof[DType.int32]()
                var n_idx_hi = n // split_width
                var n_idx_lo = n % split_width
                var reorder_idx = g * block_n + n_idx_hi * split_width * 2 + n_idx_lo
                q_mins_reorder_buf[reorder_idx + 0] = q_mins_row_0_val
                q_mins_reorder_buf[reorder_idx + split_width] = q_mins_row_1_val
            else:
                constrained[False, "unsupported architecture"]()

    dst_ptr[].q_scales_and_mins.pack(q_scales_and_mins_buf)
    dst_ptr[].q_bits.pack(q_bits_block_buf)


fn _pack_block_Q6_K[
    block_n: Int
](
    owned src_ptr: UnsafePointer[_block_Q6_K],
    stride: Int,
    dst_ptr: UnsafePointer[_block_Q6_K_packed[block_n]],
):
    alias group_count = _block_Q6_K.group_count

    constrained[
        sizeof[_block_Q6_K]() * block_n
        == sizeof[_block_Q6_K_packed[block_n]](),
        "packed block size should be multiple of the unpacked block size",
    ]()

    var q_bits_block_buf = stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8
    ]()

    for n in range(block_n):
        dst_ptr[].base_scales[n] = src_ptr[].base_scale

        for g in range(group_count):
            dst_ptr[].q_scales[g * block_n + n] = src_ptr[].q_scales[g]

        var q_bits_column_buf = stack_allocation[
            _block_QK_K.quantized_k, DType.uint8
        ]()

        _expand_q_bits_lo[width=64](
            _to_dtype_pointer(src_ptr[].q_bits_lo), q_bits_column_buf
        )
        _expand_and_merge_q_bits_hi[width=32, bit_count=2](
            _to_dtype_pointer(src_ptr[].q_bits_hi), q_bits_column_buf
        )
        _copy_column_q_bits_to_block[block_n](
            q_bits_column_buf, q_bits_block_buf + n * 4
        )

        src_ptr += stride

    dst_ptr[].q_bits.pack(q_bits_block_buf)


def matmul_Q4_K_pack_b(
    b: NDBuffer[DType.uint8, 2], b_packed: NDBuffer[DType.uint8, 2]
):
    var N = b.dim[0]()
    var K = b.dim[1]()
    var k_blocks = K // sizeof[_block_Q4_K]()

    alias simd_width = simdwidthof[DType.float32]()
    alias block_n = simd_width * 2

    var src_ptr = UnsafePointer[_block_Q4_K](address=int(b.data.address))
    var dst_ptr = UnsafePointer[_block_Q4_K_packed[block_n]](
        address=int(b_packed.data.address)
    )

    for kb in range(k_blocks):
        var src_n_ptr = src_ptr

        for n in range(0, N, block_n):
            _pack_block_Q4_K[block_n](src_n_ptr, k_blocks, dst_ptr)

            src_n_ptr += k_blocks * block_n
            dst_ptr += 1

        src_ptr += 1


def matmul_Q6_K_pack_b(
    b: NDBuffer[DType.uint8, 2], b_packed: NDBuffer[DType.uint8, 2]
):
    var N = b.dim[0]()
    var K = b.dim[1]()
    var k_blocks = K // sizeof[_block_Q6_K]()

    alias simd_width = simdwidthof[DType.float32]()
    alias block_n = simd_width * 2

    var src_ptr = UnsafePointer[_block_Q6_K](address=int(b.data.address))
    var dst_ptr = UnsafePointer[_block_Q6_K_packed[block_n]](
        address=int(b_packed.data.address)
    )

    for kb in range(k_blocks):
        var src_n_ptr = src_ptr

        for n in range(0, N, block_n):
            _pack_block_Q6_K[block_n](src_n_ptr, k_blocks, dst_ptr)

            src_n_ptr += k_blocks * block_n
            dst_ptr += 1

        src_ptr += 1


@always_inline
fn matmul_x86[
    group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
](
    a_ptr: DTypePointer[DType.int8],
    b_ptr: DTypePointer[DType.uint8],
    inout c_int32: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    var b_offset = 0

    @parameter
    for k in range(0, group_size, 4):

        @parameter
        for col in range(tile_n):
            var b_val = bitcast[DType.int32, simd_width](
                SIMD[size = simd_width * 4].load(b_ptr, b_offset)
            )
            b_offset += simd_width * 4

            @parameter
            for row in range(tile_m):
                var a_val = SIMD[DType.int32, simd_width](
                    bitcast[DType.int32, 1](
                        SIMD[size=4].load(a_ptr, row * group_size + k)
                    )
                )
                c_int32[row, col] = dot_i8_to_i32_saturated_x86(
                    c_int32[row, col], b_val, a_val
                )


@always_inline
fn matmul_neon_dotprod[
    group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
](
    a_ptr: DTypePointer[DType.int8],
    b_ptr: DTypePointer[DType.uint8],
    inout c_int32: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    var b_offset = 0

    @parameter
    for k in range(0, group_size, 16):
        var a_tile = InlineArray[SIMD[DType.int8, 16], tile_m](0)

        @parameter
        for row in range(tile_m):
            a_tile[row] = SIMD[size=16].load(a_ptr, row * group_size + k)

        @parameter
        for lane in range(4):

            @parameter
            for col in range(tile_n):
                var b_val = SIMD[size = simd_width * 4].load(
                    b_ptr, b_offset
                ).cast[DType.int8]()
                b_offset += simd_width * 4

                @parameter
                for row in range(tile_m):
                    c_int32[row, col] = _neon_dotprod_lane[lane](
                        c_int32[row, col], b_val, a_tile[row]
                    )


@always_inline
fn _apply_zero_point_correction[
    group_count: Int, tile_m: Int, tile_n: Int, simd_width: Int
](
    a_group_sums_ptr: DTypePointer[DType.int16],
    b_q_mins_ptr: DTypePointer[DType.uint8],
    b_base_mins_ptr: DTypePointer[DType.float16],
    inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    """Applies the zero point correction to the running float accumulator."""
    alias block_n = tile_n * simd_width

    var corrections = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()
    corrections.init()

    for g in range(0, group_count, 2):

        @parameter
        if is_x86():
            # Use `pmaddwd` + `paddd' (optimized to `vpdpwssd` on processors
            # that support VNNI) to multiply/add a pair of minimum values with
            # a pair of group sums from matrix A.
            @parameter
            for col in range(tile_n):
                # The minimum values vector is encoded as pairs of int16 values
                # from group_0 and group_1:
                #       [n0_g0 n0_g1 : n1_g0 n1_g1 : n2_g0 n2_g1 : n3_g0 n3_g1]
                var q_mins = SIMD[size = simd_width * 2].load(
                    b_q_mins_ptr, g * block_n + col * simd_width * 2
                ).cast[DType.int16]()

                @parameter
                for row in range(tile_m):
                    var a_group_sums = SIMD[size=2].load(
                        a_group_sums_ptr, g * tile_m + row * 2
                    )
                    corrections[row, col] = dot_i16_to_i32_x86(
                        corrections[row, col],
                        bitcast[DType.int32, simd_width](q_mins),
                        bitcast[DType.int32, 1](a_group_sums),
                    )

        elif has_neon():
            # Use `smull(2)` and `smlal(2)` instructions to do an `int16*int16`
            # widening multiply/add to an int32 accumulator.
            var group_sums = SIMD[size = tile_m * 2].load(
                a_group_sums_ptr + g * tile_m
            )

            @parameter
            for col in range(tile_n):
                # The minimum values vector is encoded as pairs of int16 values
                # from group_0 and group_1:
                #       [n0_g0 n1_g0 n2_g0 n3_g0 : n0_g1 n1_g1 n2_g1 n3_g1]
                var q_mins = SIMD[size = simd_width * 2].load(
                    b_q_mins_ptr, g * block_n + col * simd_width * 2
                ).cast[DType.int16]()

                # Logically slice the minimum values vector. This selects
                # between `smull` (lower half) or `smull2` (upper half).
                var q_mins_lo = q_mins.slice[simd_width, offset=0]()
                var q_mins_hi = q_mins.slice[simd_width, offset=simd_width]()

                @parameter
                for row in range(tile_m):
                    # Note: The ARM64 backend fuses `smull` with an int32 add to
                    # form `smlal` instructions. Also, the element broadcast is
                    # fused to with the instruction to generate the form
                    # `smlal r, a, b[lane]`. The instrinsic `vmlal_lane_s16` uses
                    # the same IR pattern to emit this instruction.
                    corrections[row, col] += llvm_intrinsic[
                        "llvm.aarch64.neon.smull.v4i32",
                        SIMD[DType.int32, simd_width],
                    ](q_mins_lo, SIMD[size=simd_width](group_sums[row * 2 + 0]))
                    corrections[row, col] += llvm_intrinsic[
                        "llvm.aarch64.neon.smull.v4i32",
                        SIMD[DType.int32, simd_width],
                    ](q_mins_hi, SIMD[size=simd_width](group_sums[row * 2 + 1]))

        else:
            constrained[False, "unsupported architecture"]()

    # Scale the correction value by the shared base minimum and update the
    # float accumulator.
    @parameter
    for col in range(tile_n):
        var base_mins = SIMD[size=simd_width].load(
            b_base_mins_ptr + col * simd_width
        ).cast[DType.float32]()

        @parameter
        for row in range(tile_m):
            c_float[row, col] -= (
                corrections[row, col].cast[DType.float32]() * base_mins
            )


fn _matmul_Q4_K[
    tile_n: Int
](
    owned a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q4_K.group_size]],
    b_ptr: UnsafePointer[_block_Q4_K_packed[]],
    owned c_ptr: DTypePointer[DType.float32],
    M: Int,
    N: Int,
    accumulate: Bool,
):
    alias group_size = _block_Q4_K.group_size
    alias group_count = _block_Q4_K.group_count

    alias simd_width = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_width]]()
    alias block_n = tile_n * simd_width

    var b_q_bits = stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8, alignment=alignment
    ]()
    var b_q_scales_and_mins_buf = stack_allocation[
        2 * group_count * block_n, DType.uint8, alignment=alignment
    ]()

    var b_tile_ptr = b_ptr.bitcast[_block_Q4_K_packed[block_n]]()

    # Convert the packed bit arrays into local arrays that can be efficiently
    # used by the inner kernel.
    b_tile_ptr[].q_bits.unpack(b_q_bits)
    b_tile_ptr[].q_scales_and_mins.unpack(b_q_scales_and_mins_buf)

    var b_q_scales_ptr = b_q_scales_and_mins_buf
    var b_q_mins_ptr = b_q_scales_and_mins_buf + group_count * block_n

    @parameter
    @always_inline
    fn process_rows[tile_m: Int](m: Int):
        var a_tile_ptr = a_ptr.bitcast[_block_Q8_K_packed[group_size, tile_m]]()

        var c_int32_block = _Accumulator[
            DType.int32, tile_m, tile_n, simd_width
        ]()

        c_int32_block.init()

        var a_q_bits_ptr = _to_dtype_pointer(a_tile_ptr[].q_bits)
        var b_q_bits_ptr = b_q_bits

        for g in range(group_count):
            var c_int32_group = _Accumulator[
                DType.int32, tile_m, tile_n, simd_width
            ]()

            c_int32_group.init()

            # Matrix multiply a single group of the block.
            @parameter
            if is_x86():
                matmul_x86[group_size](
                    a_q_bits_ptr, b_q_bits_ptr, c_int32_group
                )
            else:
                matmul_neon_dotprod[group_size](
                    a_q_bits_ptr, b_q_bits_ptr, c_int32_group
                )

            a_q_bits_ptr += tile_m * group_size
            b_q_bits_ptr += block_n * group_size

            # Scale the accumulator for this group and add to the block level
            # accumulators.
            @parameter
            for col in range(tile_n):
                var b_q_scale_val = SIMD[size=simd_width].load(
                    b_q_scales_ptr, col * simd_width + g * block_n
                ).cast[DType.int32]()

                @parameter
                for row in range(tile_m):
                    c_int32_block[row, col] += (
                        c_int32_group[row, col] * b_q_scale_val
                    )

        var c_float = _Accumulator[DType.float32, tile_m, tile_n, simd_width]()

        # Convert to floating point and apply the block scale of matrix B.
        @parameter
        for col in range(tile_n):
            var b_scale = SIMD[size=simd_width].load(
                _to_dtype_pointer(b_tile_ptr[].base_scales).offset(
                    col * simd_width
                )
            ).cast[DType.float32]()

            @parameter
            for row in range(tile_m):
                c_float[row, col] = (
                    c_int32_block[row, col].cast[DType.float32]() * b_scale
                )

        _apply_zero_point_correction[group_count](
            _to_dtype_pointer(a_tile_ptr[].group_sums),
            b_q_mins_ptr,
            _to_dtype_pointer(b_tile_ptr[].base_mins),
            c_float,
        )

        var a_scales_ptr = _to_dtype_pointer(a_tile_ptr[].scales)

        # Apply the block scale of matrix A.
        @parameter
        if has_neon():
            # NEON supports a multiply instruction that can broadcast from a
            # vector element, so help the compiler produce that by doing a
            # vector load.
            var a_scale = SIMD[size=tile_m].load(a_scales_ptr)

            @parameter
            for row in range(tile_m):

                @parameter
                for col in range(tile_n):
                    c_float[row, col] *= a_scale[row]

        else:

            @parameter
            for row in range(tile_m):
                var a_scale = a_scales_ptr[row]

                @parameter
                for col in range(tile_n):
                    c_float[row, col] *= a_scale

        var c_float2 = _Accumulator[DType.float32, tile_m, tile_n, simd_width]()

        if accumulate:
            c_float2.load(c_ptr, N)

            @parameter
            for col in range(tile_n):

                @parameter
                for row in range(tile_m):
                    c_float[row, col] += c_float2[row, col]

        c_float.store(c_ptr, N)

        a_ptr += tile_m
        c_ptr += tile_m * N

    tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)


fn matmul_Q4_K(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
):
    alias simd_width = simdwidthof[DType.float32]()

    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()
    var k_blocks = K // _block_QK_K.quantized_k

    var a_packed_base_ptr = _quantize_a_Q8_K[
        _block_Q4_K.group_size, interleave_group_sums=True
    ](a)

    alias grain_size = 64

    var num_workers = ceildiv(N, grain_size)

    @parameter
    fn task_func(task_id: Int):
        var task_n_start = task_id * grain_size
        var task_n_count = min(N - task_n_start, grain_size)

        var a_packed_ptr = a_packed_base_ptr
        var b_packed_ptr = UnsafePointer[_block_Q4_K_packed[]](
            address=int(b.data.address)
        )

        for k_block in range(k_blocks):
            var bn_packed_ptr = b_packed_ptr + task_n_start
            var cn_ptr = c.data + task_n_start
            var accumulate = k_block > 0

            @parameter
            @always_inline
            fn process_cols[tile_n: Int](n_idx: Int):
                var n = task_n_start + n_idx * simd_width

                _matmul_Q4_K[tile_n](
                    a_packed_ptr, bn_packed_ptr, cn_ptr, M, N, accumulate
                )

                bn_packed_ptr += tile_n * simd_width
                cn_ptr += tile_n * simd_width

            tile[process_cols, VariadicList[Int](2, 1)](
                0, ceildiv(task_n_count, simd_width)
            )

            a_packed_ptr += M
            b_packed_ptr += N

    sync_parallelize[task_func](num_workers)

    a_packed_base_ptr.free()


fn _matmul_Q6_K[
    tile_n: Int
](
    owned a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q6_K.group_size]],
    b_ptr: UnsafePointer[_block_Q6_K_packed[]],
    owned c_ptr: DTypePointer[DType.float32],
    M: Int,
    N: Int,
    accumulate: Bool,
):
    alias group_size = _block_Q6_K.group_size
    alias group_count = _block_Q6_K.group_count

    alias simd_width = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_width]]()
    alias block_n = tile_n * simd_width

    var b_q_bits = stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8, alignment=alignment
    ]()

    var b_tile_ptr = b_ptr.bitcast[_block_Q6_K_packed[block_n]]()

    # NEON has support for s8s8 dot products, so shift the quantized bits down
    # to avoid performing any zero point corrections.
    alias b_zero_point = 32 if has_neon() else 0

    # Convert the packed bit arrays into local arrays that can be efficiently
    # used by the inner kernel.
    b_tile_ptr[].q_bits.unpack[zero_point=b_zero_point](b_q_bits)

    @parameter
    @always_inline
    fn process_rows[tile_m: Int](m: Int):
        var a_tile_ptr = a_ptr.bitcast[_block_Q8_K_packed[group_size, tile_m]]()

        var c_int32_block = _Accumulator[
            DType.int32, tile_m, tile_n, simd_width
        ]()

        c_int32_block.init()

        var a_q_bits_ptr = _to_dtype_pointer(a_tile_ptr[].q_bits)
        var b_q_bits_ptr = b_q_bits

        for g in range(group_count):
            var c_int32_group = _Accumulator[
                DType.int32, tile_m, tile_n, simd_width
            ]()

            c_int32_group.init()

            # Matrix multiply a single group of the block.
            @parameter
            if is_x86():
                # Initialize the accumulators with the zero point correction
                # values. This is necessary for x86 as there are no VNNI
                # instructions for s8s8.
                @parameter
                for row in range(tile_m):
                    var group_sum = a_tile_ptr[].group_sums[
                        g * tile_m + row
                    ].cast[DType.int32]()
                    var correction_val = SIMD[DType.int32, simd_width](
                        -32 * group_sum
                    )

                    @parameter
                    for col in range(tile_n):
                        c_int32_group[row, col] = correction_val

                matmul_x86[group_size](
                    a_q_bits_ptr, b_q_bits_ptr, c_int32_group
                )
            else:
                matmul_neon_dotprod[group_size](
                    a_q_bits_ptr, b_q_bits_ptr, c_int32_group
                )

            a_q_bits_ptr += tile_m * group_size
            b_q_bits_ptr += block_n * group_size

            var b_q_scales_ptr = _to_dtype_pointer(b_tile_ptr[].q_scales)

            # Scale the accumulator for this group and add to the block level
            # accumulators.
            @parameter
            for col in range(tile_n):
                var b_q_scale_val = SIMD[size=simd_width].load(
                    b_q_scales_ptr, col * simd_width + g * block_n
                ).cast[DType.int32]()

                @parameter
                for row in range(tile_m):
                    c_int32_block[row, col] += (
                        c_int32_group[row, col] * b_q_scale_val
                    )

        var c_float = _Accumulator[DType.float32, tile_m, tile_n, simd_width]()

        # Convert to floating point and apply the block scale of matrix B.
        @parameter
        for col in range(tile_n):
            var b_scale = SIMD[size=simd_width].load(
                _to_dtype_pointer(b_tile_ptr[].base_scales).offset(
                    col * simd_width
                )
            ).cast[DType.float32]()

            @parameter
            for row in range(tile_m):
                c_float[row, col] = (
                    c_int32_block[row, col].cast[DType.float32]() * b_scale
                )

        c_int32_block.init()

        var a_scales_ptr = _to_dtype_pointer(a_tile_ptr[].scales)

        # Apply the block scale of matrix A.
        @parameter
        if has_neon():
            # NEON supports a multiply instruction that can broadcast from a
            # vector element, so help the compiler produce that by doing a
            # vector load.
            var a_scale = SIMD[size=tile_m].load(a_scales_ptr)

            @parameter
            for row in range(tile_m):

                @parameter
                for col in range(tile_n):
                    c_float[row, col] *= a_scale[row]

        else:

            @parameter
            for row in range(tile_m):
                var a_scale = a_scales_ptr[row]

                @parameter
                for col in range(tile_n):
                    c_float[row, col] *= a_scale

        var c_float2 = _Accumulator[DType.float32, tile_m, tile_n, simd_width]()

        if accumulate:
            c_float2.load(c_ptr, N)

            @parameter
            for col in range(tile_n):

                @parameter
                for row in range(tile_m):
                    c_float[row, col] += c_float2[row, col]

        c_float.store(c_ptr, N)

        a_ptr += tile_m
        c_ptr += tile_m * N

    tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)


fn matmul_Q6_K(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
):
    alias simd_width = simdwidthof[DType.float32]()

    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()
    var k_blocks = K // _block_QK_K.quantized_k

    var a_packed_base_ptr = _quantize_a_Q8_K[_block_Q6_K.group_size](a)

    alias grain_size = 64

    var num_workers = ceildiv(N, grain_size)

    @parameter
    fn task_func(task_id: Int):
        var task_n_start = task_id * grain_size
        var task_n_count = min(N - task_n_start, grain_size)

        var a_packed_ptr = a_packed_base_ptr
        var b_packed_ptr = UnsafePointer[_block_Q6_K_packed[]](
            address=int(b.data.address)
        )

        for k_block in range(k_blocks):
            var bn_packed_ptr = b_packed_ptr + task_n_start
            var cn_ptr = c.data + task_n_start
            var accumulate = k_block > 0

            @parameter
            @always_inline
            fn process_cols[tile_n: Int](n_idx: Int):
                var n = task_n_start + n_idx * simd_width

                _matmul_Q6_K[tile_n](
                    a_packed_ptr, bn_packed_ptr, cn_ptr, M, N, accumulate
                )

                bn_packed_ptr += tile_n * simd_width
                cn_ptr += tile_n * simd_width

            tile[process_cols, VariadicList[Int](2, 1)](
                0, ceildiv(task_n_count, simd_width)
            )

            a_packed_ptr += M
            b_packed_ptr += N

    sync_parallelize[task_func](num_workers)

    a_packed_base_ptr.free()
