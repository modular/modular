# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import InlineArray
from math import ceildiv
from sys import (
    CompilationTarget,
    alignof,
    has_avx512f,
    has_neon,
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    is_apple_silicon,
    simdwidthof,
    sizeof,
)
from sys.intrinsics import llvm_intrinsic

from algorithm import sync_parallelize, tile, vectorize
from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.accumulate import _Accumulator
from linalg.neon_intrinsics import _neon_dotprod_lane, _neon_matmul
from linalg.utils import partition_work
from linalg.vnni_intrinsics import (
    dot_i8_to_i32_saturated_x86,
    dot_i16_to_i32_x86,
)
from memory import UnsafePointer, bitcast, stack_allocation
from runtime.asyncrt import parallelism_level

from utils.index import Index

from ._utils import roundeven_to_int32


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


struct _packed_bit_array[bit_width: Int, block_m: Int, block_n: Int]:
    """Packed storage for an array of bit data.

    Logically, the array has a size of block_m by block_n. Physically, block_m
    is a multiple of the tuple width (corresponding to VNNI or equivalent
    instructions) and the tuple width is projected into the N dimension.
    This allows the SIMD width extracted from this array to be consistent with
    the native SIMD width for int32/float32 types without needing to play games
    with this array's dimensions.
    """

    alias _size = block_m * block_n
    alias _simd_width = simdwidthof[DType.uint8]()
    alias _tuple_width = 4
    alias _packed_stride = block_n * Self._tuple_width
    alias _tile_n = Self._packed_stride // Self._simd_width

    var bits: InlineArray[UInt8, Self._size * bit_width // 8]

    """
    For the 4-bit encoding, the following encoding is used (one lane of the
    SIMD register is depicted) where two rows of M are bundled together:
        [ b3 b2 b1 b0 a3 a2 a1 a0 ]
    """

    @always_inline
    fn _pack_int4(mut self, owned src_ptr: UnsafePointer[UInt8, **_]):
        constrained[bit_width == 4]()
        constrained[(block_m % (2 * Self._tuple_width)) == 0]()

        var bits_ptr = self.bits.unsafe_ptr()

        for _m in range(0, block_m, 2 * Self._tuple_width):

            @parameter
            for col in range(Self._tile_n):
                var packed_bits = SIMD[DType.uint8, Self._simd_width](0)

                @parameter
                for i in range(2):
                    var bytes = (src_ptr + i * Self._packed_stride).load[
                        width = Self._simd_width
                    ]()
                    packed_bits |= bytes << (i * 4)

                src_ptr += Self._simd_width

                bits_ptr.store(packed_bits)
                bits_ptr += Self._simd_width

            src_ptr += Self._packed_stride

    @always_inline
    fn _unpack_int4(mut self, owned dst_ptr: UnsafePointer[UInt8, **_]):
        constrained[bit_width == 4]()
        constrained[(block_m % (2 * Self._tuple_width)) == 0]()

        var bits_ptr = self.bits.unsafe_ptr()

        for _ in range(0, block_m, 2 * Self._tuple_width):

            @parameter
            for col in range(Self._tile_n):
                var packed_bits = bits_ptr.load[width = Self._simd_width]()
                bits_ptr += Self._simd_width

                @parameter
                for i in range(2):
                    var bytes = (packed_bits >> (i * 4)) & 15
                    (dst_ptr + i * Self._packed_stride).store(bytes)

                dst_ptr += Self._simd_width

            dst_ptr += Self._packed_stride

    # For the 6-bit encoding, the following encoding is used (one lane of the
    # SIMD register is depicted) where four rows of M are bundled together:
    #     [ d1 d0 a5 a4 a3 a2 a1 a0 ]
    #     [ d3 d2 b5 b4 b3 b2 b1 b0 ]
    #     [ d5 d6 c5 c4 c3 c2 c1 c0 ]

    @always_inline
    fn _pack_int6(mut self, owned src_ptr: UnsafePointer[UInt8, **_]):
        constrained[bit_width == 6]()
        constrained[(block_m % (4 * Self._tuple_width)) == 0]()

        var bits_ptr = self.bits.unsafe_ptr()

        for _m in range(0, block_m, 4 * Self._tuple_width):
            var src_col_ptr = src_ptr

            @parameter
            for col in range(Self._tile_n):
                var hi_bytes = (src_col_ptr + 3 * Self._packed_stride).load[
                    width = Self._simd_width
                ]()

                @parameter
                for i in range(3):
                    var bytes = (src_col_ptr + i * Self._packed_stride).load[
                        width = Self._simd_width
                    ]()
                    var packed_bits = bytes | (((hi_bytes >> (i * 2)) & 3) << 6)

                    bits_ptr.store(packed_bits)
                    bits_ptr += Self._simd_width

                src_col_ptr += Self._simd_width

            src_ptr += Self._packed_stride * 4

    @always_inline
    fn _unpack_int6[
        zero_point: UInt8
    ](mut self, owned dst_ptr: UnsafePointer[UInt8, **_]):
        constrained[bit_width == 6]()
        constrained[(block_m % (4 * Self._tuple_width)) == 0]()

        var bits_ptr = self.bits.unsafe_ptr()

        for _m in range(0, block_m, 4 * Self._tuple_width):
            var dst_col_ptr = dst_ptr

            @parameter
            for col in range(Self._tile_n):
                var hi_bytes = SIMD[DType.uint8, size = Self._simd_width](0)

                @parameter
                for i in range(3):
                    var packed_bits = bits_ptr.load[width = Self._simd_width]()
                    bits_ptr += Self._simd_width

                    (dst_col_ptr + i * Self._packed_stride).store(
                        (packed_bits & 63) - zero_point,
                    )

                    hi_bytes |= (packed_bits >> 6) << (i * 2)

                (dst_col_ptr + 3 * Self._packed_stride).store(
                    hi_bytes - zero_point,
                )

                dst_col_ptr += Self._simd_width

            dst_ptr += Self._packed_stride * 4

    @always_inline
    fn pack(mut self, owned src_ptr: UnsafePointer[UInt8, **_]):
        """Packs the supplied external buffer to local storage."""
        constrained[(Self._packed_stride % Self._simd_width) == 0]()

        @parameter
        if bit_width == 4:
            return self._pack_int4(src_ptr)
        elif bit_width == 6:
            return self._pack_int6(src_ptr)
        else:
            constrained[False, "unsupported bit width"]()

    @always_inline
    fn unpack[
        *, zero_point: UInt8 = 0
    ](mut self, owned dst_ptr: UnsafePointer[UInt8, **_]):
        """Unpacks the local storage to the supplied external buffer."""
        constrained[(Self._packed_stride % Self._simd_width) == 0]()

        @parameter
        if bit_width == 4:
            constrained[zero_point == 0, "zero point not implemented"]()
            return self._unpack_int4(dst_ptr)
        elif bit_width == 6:
            return self._unpack_int6[zero_point](dst_ptr)
        else:
            constrained[False, "unsupported bit width"]()


struct _block_Q4_K_packed[block_n: Int = 1]:
    var base_scales: InlineArray[Float16, block_n]
    var base_mins: InlineArray[Float16, block_n]
    var q_scales_and_mins: _packed_bit_array[
        6, 2 * _block_Q4_K.group_count, block_n
    ]
    var q_bits: _packed_bit_array[4, _block_QK_K.quantized_k, block_n]


struct _block_Q6_K_packed[block_n: Int = 1]:
    var base_scales: InlineArray[Float16, block_n]
    var q_scales: InlineArray[Int8, _block_Q6_K.group_count * block_n]
    var q_bits: _packed_bit_array[6, _block_QK_K.quantized_k, block_n]


struct _block_Q8_K_packed[group_size: Int, tile_m: Int = 1]:
    alias group_count = _block_QK_K.calc_group_count[group_size]()

    var q_bits: InlineArray[Int8, _block_QK_K.quantized_k * tile_m]
    var scales: InlineArray[Float32, tile_m]
    var group_sums: InlineArray[Int16, Self.group_count * tile_m]


fn _quantize_a_Q8_K[
    group_size: Int, type: DType, *, interleave_group_sums: Bool = False
](a: NDBuffer[type, 2, **_]) -> UnsafePointer[
    _block_Q8_K_packed[group_size], mut = a.mut, origin = a.origin
]:
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
            var q_bits_ptr = block_ptr[].q_bits.unsafe_ptr()

            for row in range(tile_m):
                var max_value_simd = SIMD[type, group_size](Scalar[type].MIN)

                for g in range(group_count):
                    var fp_data = am_ptr.load[width=group_size](g * group_size)
                    max_value_simd = max(abs(fp_data), max_value_simd)

                var max_value = max_value_simd.reduce_max()
                var scale = (max_value / 127.0).cast[DType.float32]()
                var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

                for g in range(group_count):
                    var fp_data = am_ptr.load[width=group_size](g * group_size)
                    var q_data_i32 = roundeven_to_int32(fp_data * multiplier)
                    var q_data_i8 = q_data_i32.cast[DType.int8]()
                    var group_sum = q_data_i32.reduce_add()

                    q_bits_ptr.store(
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
    owned src_ptr: UnsafePointer[UInt8, **_],
    owned dst_ptr: UnsafePointer[UInt8, **_],
):
    for _k in range(0, _block_QK_K.quantized_k // 2, width):
        var src_q_bits = src_ptr.load[width=width]()
        src_ptr += width

        @parameter
        for i in range(2):
            dst_ptr.store((src_q_bits >> (i * 4)) & 15)
            dst_ptr += width


fn _expand_and_merge_q_bits_hi[
    *, width: Int, bit_count: Int
](
    owned src_ptr: UnsafePointer[UInt8, **_],
    owned dst_ptr: UnsafePointer[UInt8, **_],
):
    alias values_per_byte = 8 // bit_count
    alias bit_mask = (1 << bit_count) - 1

    for _k in range(0, _block_QK_K.quantized_k // values_per_byte, width):
        var src_q_bits = src_ptr.load[width=width]()
        src_ptr += width

        for _ in range(values_per_byte):
            var dst_q_bits_lo = dst_ptr.load[width=width]()
            var dst_q_bits_hi = (src_q_bits & bit_mask) << 4
            src_q_bits >>= bit_count

            dst_ptr.store(dst_q_bits_hi | dst_q_bits_lo)
            dst_ptr += width


fn _copy_column_q_bits_to_block[
    block_n: Int
](
    owned src_ptr: UnsafePointer[UInt8, **_],
    owned dst_ptr: UnsafePointer[UInt8, **_],
):
    """Interleaves the linear source buffer to the blocked destination
    buffer.
    """
    for _k in range(0, _block_QK_K.quantized_k, 4):
        dst_ptr.store(src_ptr.load[width=4]())
        src_ptr += 4
        dst_ptr += block_n * 4


fn _pack_block_Q4_K[
    block_n: Int,
    src_origin: MutableOrigin,
    dst_origin: MutableOrigin,
    alignment: Int,
](
    owned src_ptr: UnsafePointer[
        _block_Q4_K, origin=src_origin, alignment=alignment
    ],
    stride: Int,
    mut dst_ptr: UnsafePointer[
        _block_Q4_K_packed[block_n], origin=dst_origin, alignment=alignment
    ],
):
    alias group_size = _block_Q4_K.group_size
    alias group_count = _block_Q4_K.group_count

    constrained[
        sizeof[_block_Q4_K]() * block_n
        == sizeof[_block_Q4_K_packed[block_n]](),
        "packed block size should be multiple of the unpacked block size",
    ]()

    var q_scales_buf = InlineArray[UInt8, group_count * block_n](
        uninitialized=True
    )
    var q_mins_buf = InlineArray[UInt8, group_count * block_n](
        uninitialized=True
    )
    var q_bits_block_buf = InlineArray[
        UInt8, _block_QK_K.quantized_k * block_n
    ](uninitialized=True)

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

        var q_bits_column_buf = InlineArray[UInt8, _block_QK_K.quantized_k](
            uninitialized=True
        )

        _expand_q_bits_lo[width=32](
            src_ptr[].q_bits.unsafe_ptr(), q_bits_column_buf.unsafe_ptr()
        )
        _copy_column_q_bits_to_block[block_n](
            q_bits_column_buf.unsafe_ptr(),
            q_bits_block_buf.unsafe_ptr() + n * 4,
        )

        src_ptr += stride

    # Allocate a staging buffer to pack the scales and minimums as a single
    # blob and to do processor specific reordering of the values for the
    # compute kernel.
    var q_scales_and_mins_buf = InlineArray[UInt8, 2 * group_count * block_n](
        uninitialized=True
    )
    var q_scales_reorder_buf = q_scales_and_mins_buf.unsafe_ptr()
    var q_mins_reorder_buf = q_scales_and_mins_buf.unsafe_ptr() + group_count * block_n

    # Scales are not currently transformed.
    memcpy(
        q_scales_reorder_buf,
        q_scales_buf.unsafe_ptr(),
        group_count * block_n,
    )

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
        var q_mins_row_0_ptr = q_mins_buf.unsafe_ptr() + g * block_n
        var q_mins_row_1_ptr = q_mins_row_0_ptr + block_n
        for n in range(block_n):
            var q_mins_row_0_val = q_mins_row_0_ptr[n]
            var q_mins_row_1_val = q_mins_row_1_ptr[n]

            @parameter
            if CompilationTarget.is_x86():
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

    dst_ptr[].q_scales_and_mins.pack(q_scales_and_mins_buf.unsafe_ptr())
    dst_ptr[].q_bits.pack(q_bits_block_buf.unsafe_ptr())


fn _pack_block_Q6_K[
    block_n: Int,
    src_origin: MutableOrigin,
    dst_origin: MutableOrigin,
    alignment: Int,
](
    owned src_ptr: UnsafePointer[
        _block_Q6_K, origin=src_origin, alignment=alignment
    ],
    stride: Int,
    mut dst_ptr: UnsafePointer[
        _block_Q6_K_packed[block_n], origin=dst_origin, alignment=alignment
    ],
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
            src_ptr[].q_bits_lo.unsafe_ptr(), q_bits_column_buf
        )
        _expand_and_merge_q_bits_hi[width=32, bit_count=2](
            src_ptr[].q_bits_hi.unsafe_ptr(), q_bits_column_buf
        )
        _copy_column_q_bits_to_block[block_n](
            q_bits_column_buf, q_bits_block_buf + n * 4
        )

        src_ptr += stride

    dst_ptr[].q_bits.pack(q_bits_block_buf)


def matmul_Q4_K_pack_b[
    b_origin: MutableOrigin, b_packed_origin: MutableOrigin
](
    b: NDBuffer[DType.uint8, 2, b_origin],
    b_packed: NDBuffer[DType.uint8, 2, b_packed_origin],
):
    var N = b.dim[0]()
    var K = b.dim[1]()
    var k_blocks = K // sizeof[_block_Q4_K]()

    alias simd_width = simdwidthof[DType.float32]()
    alias block_n = simd_width * 2

    var src_ptr = b.data.bitcast[_block_Q4_K]()
    var dst_ptr = b_packed.data.bitcast[_block_Q4_K_packed[block_n]]()

    for _kb in range(k_blocks):
        var src_n_ptr = src_ptr

        for _n in range(0, N, block_n):
            _pack_block_Q4_K[block_n, b_origin, b_packed_origin](
                src_n_ptr, k_blocks, dst_ptr
            )

            src_n_ptr += k_blocks * block_n
            dst_ptr += 1

        src_ptr += 1


def matmul_Q6_K_pack_b[
    b_origin: MutableOrigin, b_packed_origin: MutableOrigin
](
    b: NDBuffer[DType.uint8, 2, b_origin],
    b_packed: NDBuffer[DType.uint8, 2, b_packed_origin],
):
    var N = b.dim[0]()
    var K = b.dim[1]()
    var k_blocks = K // sizeof[_block_Q6_K]()

    alias simd_width = simdwidthof[DType.float32]()
    alias block_n = simd_width * 2

    var src_ptr = b.data.bitcast[_block_Q6_K]()
    var dst_ptr = b_packed.data.bitcast[_block_Q6_K_packed[block_n]]()

    for _kb in range(k_blocks):
        var src_n_ptr = src_ptr

        for _n in range(0, N, block_n):
            _pack_block_Q6_K[block_n](src_n_ptr, k_blocks, dst_ptr)

            src_n_ptr += k_blocks * block_n
            dst_ptr += 1

        src_ptr += 1


@always_inline
fn _matmul_group_stream_x86[
    tile_k: Int,
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
    group_size: Int,
    stream_b_vals_fn: fn (
        mut b_vals: InlineArray[
            SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
        ]
    ) capturing [_] -> None,
](
    a_q_bits_ptr: UnsafePointer[Int8],
    mut c_int32_group: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    var b_vals = InlineArray[
        SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
    ](0)

    @parameter
    for k in range(0, group_size, tile_k * 4):
        stream_b_vals_fn(b_vals)

        @parameter
        for tk in range(tile_k):

            @parameter
            for col in range(tile_n):

                @parameter
                for row in range(tile_m):
                    var a_val = SIMD[DType.int32, simd_width](
                        bitcast[DType.int32, 1](
                            (a_q_bits_ptr + row * group_size + k + tk * 4).load[
                                width=4
                            ]()
                        )
                    )
                    c_int32_group[row, col] = dot_i8_to_i32_saturated_x86(
                        c_int32_group[row, col],
                        bitcast[DType.int32, simd_width](
                            b_vals[col * tile_k + tk]
                        ),
                        a_val,
                    )


@always_inline
fn _matmul_group_stream_neon_dotprod[
    tile_k: Int,
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
    group_size: Int,
    stream_b_vals_fn: fn (
        mut b_vals: InlineArray[
            SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
        ]
    ) capturing [_] -> None,
](
    a_q_bits_ptr: UnsafePointer[Int8],
    mut c_int32_group: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    var b_vals = InlineArray[
        SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
    ](0)

    @parameter
    for k in range(0, group_size, 16):
        var a_tile = InlineArray[SIMD[DType.int8, 16], tile_m](0)

        @parameter
        for row in range(tile_m):
            a_tile[row] = (a_q_bits_ptr + row * group_size + k).load[width=16]()

        @parameter
        for lane in range(0, 4, tile_k):
            stream_b_vals_fn[](b_vals)

            @parameter
            for tk in range(tile_k):

                @parameter
                for col in range(tile_n):

                    @parameter
                    for row in range(tile_m):
                        c_int32_group[row, col] = _neon_dotprod_lane[lane + tk](
                            c_int32_group[row, col],
                            b_vals[col * tile_k + tk].cast[DType.int8](),
                            a_tile[row],
                        )


@always_inline
fn _matmul_group_stream[
    tile_k: Int,
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    origins: OriginSet, //,
    group_size: Int,
    stream_b_vals_fn: fn (
        mut b_vals: InlineArray[
            SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
        ]
    ) capturing [origins] -> None,
](
    a_q_bits_ptr: UnsafePointer[Int8],
    mut c_int32_group: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    constrained[tile_k.is_power_of_two() and tile_k <= 4]()

    @parameter
    if CompilationTarget.is_x86():
        return _matmul_group_stream_x86[group_size, stream_b_vals_fn](
            a_q_bits_ptr, c_int32_group
        )
    elif has_neon():
        return _matmul_group_stream_neon_dotprod[group_size, stream_b_vals_fn](
            a_q_bits_ptr, c_int32_group
        )
    else:
        constrained[False, "unsupported architecture"]()


@always_inline
fn _matmul_group_unpacked[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
    group_size: Int,
](
    a_q_bits_ptr: UnsafePointer[Int8],
    mut b_q_bits_ptr: UnsafePointer[UInt8],
    mut c_int32_group: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    """Streaming matrix multiplication where the B matrix has been unpacked to
    local storage.
    """

    @parameter
    fn stream_b_vals(
        mut b_vals: InlineArray[SIMD[DType.uint8, simd_width * 4], tile_n * 1]
    ):
        @parameter
        for col in range(tile_n):
            b_vals[col] = b_q_bits_ptr.load[width = simd_width * 4]()
            b_q_bits_ptr += simd_width * 4

    _matmul_group_stream[
        tile_k=1,
        tile_m=tile_m,
        tile_n=tile_n,
        simd_width=simd_width,
        group_size,
        stream_b_vals,
    ](a_q_bits_ptr, c_int32_group)


@always_inline
fn _apply_base_scales[
    tile_m: Int, tile_n: Int, simd_width: Int
](
    b_base_scales_ptr: UnsafePointer[Float16],
    c_int32_block: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
    mut c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    # Convert to floating point and apply the block scale of matrix B.
    @parameter
    for col in range(tile_n):
        var b_scale = (b_base_scales_ptr + col * simd_width).load[
            width=simd_width
        ]().cast[DType.float32]()

        @parameter
        for row in range(tile_m):
            c_float[row, col] = (
                c_int32_block[row, col].cast[DType.float32]() * b_scale
            )


@always_inline
fn _apply_zero_point_correction[
    group_count: Int, tile_m: Int, tile_n: Int, simd_width: Int
](
    a_group_sums_ptr: UnsafePointer[Int16],
    b_q_mins_ptr: UnsafePointer[UInt8],
    b_base_mins_ptr: UnsafePointer[Float16],
    mut c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    """Applies the zero point correction to the running float accumulator."""
    alias block_n = tile_n * simd_width

    var corrections = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()
    corrections.init()

    for g in range(0, group_count, 2):

        @parameter
        if CompilationTarget.is_x86():
            # Use `pmaddwd` + `paddd' (optimized to `vpdpwssd` on processors
            # that support VNNI) to multiply/add a pair of minimum values with
            # a pair of group sums from matrix A.
            @parameter
            for col in range(tile_n):
                # The minimum values vector is encoded as pairs of int16 values
                # from group_0 and group_1:
                #       [n0_g0 n0_g1 : n1_g0 n1_g1 : n2_g0 n2_g1 : n3_g0 n3_g1]
                var q_mins = b_q_mins_ptr.load[width = simd_width * 2](
                    g * block_n + col * simd_width * 2
                ).cast[DType.int16]()

                @parameter
                for row in range(tile_m):
                    var a_group_sums = a_group_sums_ptr.load[width=2](
                        g * tile_m + row * 2
                    )
                    corrections[row, col] = dot_i16_to_i32_x86(
                        corrections[row, col],
                        bitcast[DType.int32, simd_width](q_mins),
                        bitcast[DType.int32, 1](a_group_sums),
                    )

        elif has_neon():
            # Use `smull(2)` and `smlal(2)` instructions to do an `int16*int16`
            # widening multiply/add to an int32 accumulator.
            var group_sums = (a_group_sums_ptr + g * tile_m).load[
                width = tile_m * 2
            ]()

            @parameter
            for col in range(tile_n):
                # The minimum values vector is encoded as pairs of int16 values
                # from group_0 and group_1:
                #       [n0_g0 n1_g0 n2_g0 n3_g0 : n0_g1 n1_g1 n2_g1 n3_g1]
                var q_mins = b_q_mins_ptr.load[width = simd_width * 2](
                    g * block_n + col * simd_width * 2
                ).cast[DType.int16]()

                # Logically slice the minimum values vector. This selects
                # between `smull` (lower half) or `smull2` (upper half).
                var q_mins_lo_hi = q_mins.split()

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
                    ](
                        q_mins_lo_hi[0],
                        SIMD[size=simd_width](group_sums[row * 2 + 0]),
                    )
                    corrections[row, col] += llvm_intrinsic[
                        "llvm.aarch64.neon.smull.v4i32",
                        SIMD[DType.int32, simd_width],
                    ](
                        q_mins_lo_hi[1],
                        SIMD[size=simd_width](group_sums[row * 2 + 1]),
                    )

        else:
            constrained[False, "unsupported architecture"]()

    # Scale the correction value by the shared base minimum and update the
    # float accumulator.
    @parameter
    for col in range(tile_n):
        var base_mins = (b_base_mins_ptr + col * simd_width).load[
            width=simd_width
        ]().cast[DType.float32]()

        @parameter
        for row in range(tile_m):
            c_float[row, col] -= (
                corrections[row, col].cast[DType.float32]() * base_mins
            )


@always_inline
fn _apply_a_scales[
    tile_m: Int, tile_n: Int, simd_width: Int
](
    a_scales_ptr: UnsafePointer[Float32],
    mut c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    @parameter
    if has_neon():
        # NEON supports a multiply instruction that can broadcast from a
        # vector element, so help the compiler produce that by doing a
        # vector load.
        var a_scale = a_scales_ptr.load[width=tile_m]()

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


@always_inline
fn _accumulate_and_store[
    tile_m: Int, tile_n: Int, simd_width: Int
](
    c_ptr: UnsafePointer[Float32],
    N: Int,
    accumulate: Bool,
    mut c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    if accumulate:
        var c_existing = _Accumulator[
            DType.float32, tile_m, tile_n, simd_width
        ]()

        c_existing.load(c_ptr, N)

        @parameter
        for col in range(tile_n):

            @parameter
            for row in range(tile_m):
                c_float[row, col] += c_existing[row, col]

    c_float.store(c_ptr, N)


@always_inline
fn _matmul_group_packed_Q4_K[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
](
    a_q_bits_ptr: UnsafePointer[Int8],
    mut b_q_bits_ptr: UnsafePointer[UInt8],
    mut c_int32_group: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    alias group_size = _block_Q4_K.group_size
    alias tile_k = 2

    @parameter
    fn stream_b_vals(
        mut b_vals: InlineArray[
            SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
        ]
    ):
        @parameter
        for col in range(tile_n):
            var packed_bits = b_q_bits_ptr.load[width = simd_width * 4]()
            b_q_bits_ptr += simd_width * 4

            @parameter
            for i in range(2):
                var bytes = (packed_bits >> (i * 4)) & 15
                b_vals[col * tile_k + i] = bytes

    _matmul_group_stream[
        tile_k=tile_k,
        tile_m=tile_m,
        tile_n=tile_n,
        simd_width=simd_width,
        group_size,
        stream_b_vals,
    ](a_q_bits_ptr, c_int32_group)


@always_inline
fn _matmul_Q4_K_tile[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
    matmul_group_fn: fn (
        a_ptr: UnsafePointer[Int8],
        mut c_int32: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
    ) capturing [_] -> None,
](
    a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q4_K.group_size]],
    b_ptr: UnsafePointer[_block_Q4_K_packed[]],
    b_q_scales_and_mins_buf: UnsafePointer[UInt8],
    c_ptr: UnsafePointer[Float32],
    N: Int,
    accumulate: Bool,
):
    alias group_size = _block_Q4_K.group_size
    alias group_count = _block_Q4_K.group_count

    alias block_n = tile_n * simd_width

    var a_tile_ptr = a_ptr.bitcast[_block_Q8_K_packed[group_size, tile_m]]()
    var b_tile_ptr = b_ptr.bitcast[_block_Q4_K_packed[block_n]]()

    var b_q_scales_ptr = b_q_scales_and_mins_buf
    var b_q_mins_ptr = b_q_scales_and_mins_buf + group_count * block_n

    var c_int32_block = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()

    c_int32_block.init()

    var a_q_bits_ptr = a_tile_ptr[].q_bits.unsafe_ptr()

    for g in range(group_count):
        var c_int32_group = _Accumulator[
            DType.int32, tile_m, tile_n, simd_width
        ]()

        c_int32_group.init()

        # Matrix multiply a single group of the block.
        matmul_group_fn(a_q_bits_ptr, c_int32_group)

        a_q_bits_ptr += tile_m * group_size

        # Scale the accumulator for this group and add to the block level
        # accumulators.
        @parameter
        for col in range(tile_n):
            var b_q_scale_val = b_q_scales_ptr.load[width=simd_width](
                col * simd_width + g * block_n
            ).cast[DType.int32]()

            @parameter
            for row in range(tile_m):
                c_int32_block[row, col] += (
                    c_int32_group[row, col] * b_q_scale_val
                )

    var c_float = _Accumulator[DType.float32, tile_m, tile_n, simd_width]()

    _apply_base_scales(
        b_tile_ptr[].base_scales.unsafe_ptr(), c_int32_block, c_float
    )

    _apply_zero_point_correction[group_count](
        a_tile_ptr[].group_sums.unsafe_ptr(),
        b_q_mins_ptr,
        b_tile_ptr[].base_mins.unsafe_ptr(),
        c_float,
    )

    _apply_a_scales(a_tile_ptr[].scales.unsafe_ptr(), c_float)

    _accumulate_and_store(c_ptr, N, accumulate, c_float)


fn _matmul_Q4_K_columns[
    tile_n: Int, simd_width: Int
](
    owned a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q4_K.group_size]],
    b_ptr: UnsafePointer[_block_Q4_K_packed[]],
    owned c_ptr: UnsafePointer[Float32],
    M: Int,
    N: Int,
    accumulate: Bool,
):
    alias group_size = _block_Q4_K.group_size
    alias group_count = _block_Q4_K.group_count

    alias alignment = alignof[SIMD[DType.float32, simd_width]]()
    alias block_n = tile_n * simd_width

    var b_tile_ptr = b_ptr.bitcast[_block_Q4_K_packed[block_n]]()

    # Unpack the scales and minimums to uint8 values.
    var b_q_scales_and_mins_buf = stack_allocation[
        2 * group_count * block_n, DType.uint8, alignment=alignment
    ]()
    b_tile_ptr[].q_scales_and_mins.unpack(b_q_scales_and_mins_buf)

    # Fast path for M=1 that avoids materializing the unpacked weights.
    if M == 1:
        var b_q_bits_ptr = b_tile_ptr[].q_bits.bits.unsafe_ptr()

        @parameter
        fn matmul_group_packed(
            a_q_bits_ptr: UnsafePointer[Int8],
            mut c_int32_group: _Accumulator[DType.int32, 1, tile_n, simd_width],
        ):
            _matmul_group_packed_Q4_K(a_q_bits_ptr, b_q_bits_ptr, c_int32_group)

        _matmul_Q4_K_tile[matmul_group_packed](
            a_ptr, b_ptr, b_q_scales_and_mins_buf, c_ptr, N, accumulate
        )
        _ = b_q_bits_ptr

        return

    # Unpack the quantized bits to uint8 values.
    var b_q_bits = stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8, alignment=alignment
    ]()
    b_tile_ptr[].q_bits.unpack(b_q_bits)

    @parameter
    @__copy_capture(b_tile_ptr, b_q_scales_and_mins_buf, b_q_bits)
    @always_inline
    fn process_rows[tile_m: Int](m: Int):
        var b_q_bits_ptr = b_q_bits

        @parameter
        fn matmul_group_unpacked(
            a_ptr: UnsafePointer[Int8],
            mut c_int32_group: _Accumulator[
                DType.int32, tile_m, tile_n, simd_width
            ],
        ):
            _matmul_group_unpacked[group_size](
                a_ptr, b_q_bits_ptr, c_int32_group
            )

        _matmul_Q4_K_tile[matmul_group_unpacked](
            a_ptr, b_ptr, b_q_scales_and_mins_buf, c_ptr, N, accumulate
        )
        _ = b_q_bits_ptr

        a_ptr += tile_m
        c_ptr += tile_m * N

    tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)


@always_inline
fn _matmul_group_packed_Q6_K[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
    *,
    zero_point: UInt8,
](
    a_q_bits_ptr: UnsafePointer[Int8],
    mut b_q_bits_ptr: UnsafePointer[UInt8],
    mut c_int32_group: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
):
    alias group_size = _block_Q6_K.group_size
    alias tile_k = 4

    @parameter
    fn stream_b_vals(
        mut b_vals: InlineArray[
            SIMD[DType.uint8, simd_width * 4], tile_n * tile_k
        ]
    ):
        @parameter
        for col in range(tile_n):
            var hi_bytes = SIMD[DType.uint8, size = simd_width * 4](0)

            @parameter
            for i in range(3):
                var packed_bits = b_q_bits_ptr.load[width = simd_width * 4]()
                b_q_bits_ptr += simd_width * 4

                var bytes = packed_bits & 63
                b_vals[col * tile_k + i] = bytes - zero_point

                hi_bytes |= (packed_bits >> 6) << (i * 2)

            b_vals[col * tile_k + 3] = hi_bytes - zero_point

    _matmul_group_stream[
        tile_k=4,
        tile_m=tile_m,
        tile_n=tile_n,
        simd_width=simd_width,
        group_size,
        stream_b_vals,
    ](a_q_bits_ptr, c_int32_group)


@always_inline
fn _matmul_Q6_K_tile[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int, //,
    matmul_group_fn: fn (
        a_ptr: UnsafePointer[Int8],
        mut c_int32_group: _Accumulator[
            DType.int32, tile_m, tile_n, simd_width
        ],
    ) capturing [_] -> None,
](
    a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q6_K.group_size]],
    b_ptr: UnsafePointer[_block_Q6_K_packed[]],
    c_ptr: UnsafePointer[Float32],
    N: Int,
    accumulate: Bool,
):
    alias group_size = _block_Q6_K.group_size
    alias group_count = _block_Q6_K.group_count

    alias block_n = tile_n * simd_width

    var a_tile_ptr = a_ptr.bitcast[_block_Q8_K_packed[group_size, tile_m]]()
    var b_tile_ptr = b_ptr.bitcast[_block_Q6_K_packed[block_n]]()

    var c_int32_block = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()

    c_int32_block.init()

    var a_q_bits_ptr = a_tile_ptr[].q_bits.unsafe_ptr()

    for g in range(group_count):
        var c_int32_group = _Accumulator[
            DType.int32, tile_m, tile_n, simd_width
        ]()

        c_int32_group.init()

        @parameter
        if CompilationTarget.is_x86():
            # Initialize the accumulators with the zero point correction
            # values. This is necessary for x86 as there are no VNNI
            # instructions for s8s8.
            @parameter
            for row in range(tile_m):
                var group_sum = a_tile_ptr[].group_sums[g * tile_m + row].cast[
                    DType.int32
                ]()
                var correction_val = SIMD[DType.int32, simd_width](
                    -32 * group_sum
                )

                @parameter
                for col in range(tile_n):
                    c_int32_group[row, col] = correction_val

        # Matrix multiply a single group of the block.
        matmul_group_fn(a_q_bits_ptr, c_int32_group)

        a_q_bits_ptr += tile_m * group_size

        var b_q_scales_ptr = b_tile_ptr[].q_scales.unsafe_ptr()

        # Scale the accumulator for this group and add to the block level
        # accumulators.
        @parameter
        for col in range(tile_n):
            var b_q_scale_val = b_q_scales_ptr.load[width=simd_width](
                col * simd_width + g * block_n
            ).cast[DType.int32]()

            @parameter
            for row in range(tile_m):
                c_int32_block[row, col] += (
                    c_int32_group[row, col] * b_q_scale_val
                )

    var c_float = _Accumulator[DType.float32, tile_m, tile_n, simd_width]()

    _apply_base_scales(
        b_tile_ptr[].base_scales.unsafe_ptr(), c_int32_block, c_float
    )

    _apply_a_scales(a_tile_ptr[].scales.unsafe_ptr(), c_float)

    _accumulate_and_store(c_ptr, N, accumulate, c_float)


fn _matmul_Q6_K_columns[
    tile_n: Int, simd_width: Int
](
    owned a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q6_K.group_size]],
    b_ptr: UnsafePointer[_block_Q6_K_packed[]],
    owned c_ptr: UnsafePointer[Float32],
    M: Int,
    N: Int,
    accumulate: Bool,
):
    alias group_size = _block_Q6_K.group_size
    alias group_count = _block_Q6_K.group_count

    alias alignment = alignof[SIMD[DType.float32, simd_width]]()
    alias block_n = tile_n * simd_width

    var b_tile_ptr = b_ptr.bitcast[_block_Q6_K_packed[block_n]]()

    # NEON has support for s8s8 dot products, so shift the quantized bits down
    # to avoid performing any zero point corrections.
    alias b_zero_point = 32 if has_neon() else 0

    # Fast path for M=1 that avoids materializing the unpacked weights.
    if M == 1:
        var b_q_bits_ptr = b_tile_ptr[].q_bits.bits.unsafe_ptr()

        @parameter
        fn matmul_group_packed(
            a_q_bits_ptr: UnsafePointer[Int8],
            mut c_int32_group: _Accumulator[DType.int32, 1, tile_n, simd_width],
        ):
            _matmul_group_packed_Q6_K[zero_point=b_zero_point](
                a_q_bits_ptr, b_q_bits_ptr, c_int32_group
            )

        _matmul_Q6_K_tile[matmul_group_packed](
            a_ptr, b_ptr, c_ptr, N, accumulate
        )
        _ = b_q_bits_ptr

        return

    # Unpack the quantized bits to uint8 values.
    var b_q_bits = stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8, alignment=alignment
    ]()
    b_tile_ptr[].q_bits.unpack[zero_point=b_zero_point](b_q_bits)

    @parameter
    @__copy_capture(b_tile_ptr, b_q_bits)
    @always_inline
    fn process_rows[tile_m: Int](m: Int):
        var b_q_bits_ptr = b_q_bits

        @parameter
        fn matmul_group_unpacked(
            a_ptr: UnsafePointer[Int8],
            mut c_int32_group: _Accumulator[
                DType.int32, tile_m, tile_n, simd_width
            ],
        ):
            _matmul_group_unpacked[group_size](
                a_ptr, b_q_bits_ptr, c_int32_group
            )

        _matmul_Q6_K_tile[matmul_group_unpacked](
            a_ptr, b_ptr, c_ptr, N, accumulate
        )
        _ = b_q_bits_ptr

        a_ptr += tile_m
        c_ptr += tile_m * N

    tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)


@always_inline
fn _matmul_Qb_K[
    group_size: Int,
    b_type: AnyType, //,
    columns_fn: fn[tile_n: Int, simd_width: Int] (
        owned a_ptr: UnsafePointer[_block_Q8_K_packed[group_size]],
        b_ptr: UnsafePointer[b_type],
        owned c_ptr: UnsafePointer[Float32],
        M: Int,
        N: Int,
        accumulate: Bool,
    ) -> None,
    *,
    interleave_group_sums: Bool = False,
](
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
        group_size, interleave_group_sums=interleave_group_sums
    ](a)

    alias grain_size = simd_width * 2

    var work_count = ceildiv(N, grain_size)
    var num_workers = min(work_count, parallelism_level())

    @parameter
    @__copy_capture(
        a_packed_base_ptr, k_blocks, M, N, K, work_count, num_workers
    )
    fn task_func(task_id: Int):
        var block_range = partition_work(task_id, num_workers, work_count, 1)
        var task_n_start = block_range[0] * grain_size
        var task_n_count = block_range[1] * grain_size

        var a_packed_ptr = a_packed_base_ptr
        var b_packed_ptr = b.data.bitcast[b_type]()

        for k_block in range(k_blocks):
            var bn_packed_ptr = b_packed_ptr + task_n_start
            var cn_ptr = c.data + task_n_start
            var accumulate = k_block > 0

            @parameter
            @always_inline
            fn process_cols[tile_n: Int](n_idx: Int):
                columns_fn[tile_n, simd_width](
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


fn matmul_Q4_K(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
):
    _matmul_Qb_K[_matmul_Q4_K_columns, interleave_group_sums=True](a, b, c)


fn matmul_Q6_K(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
):
    _matmul_Qb_K[_matmul_Q6_K_columns](a, b, c)
