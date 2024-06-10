# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import ceildiv
from sys.info import (
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    is_apple_silicon,
    is_x86,
)

from algorithm import sync_parallelize, tile
from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.accumulate import _Accumulator
from LinAlg.neon_intrinsics import _neon_dotprod_lane, _neon_matmul
from LinAlg.vnni_intrinsics import dot_i8_to_i32_saturated_x86
from memory.unsafe import DTypePointer

from utils import InlineArray
from utils.index import Index

from ._utils import roundeven_to_int32

alias K_BATCH_SIZE = 512
"""Defines the batch size of K used to pack A and unpack B weights."""


def matmul_qint4_pack_b[
    group_size: Int
](b: NDBuffer[DType.uint8, 2], b_rot: NDBuffer[DType.uint8, 2]):
    alias n_tiles = 2
    alias n_groups = n_tiles * simdwidthof[DType.float32]()
    alias bytes_per_group_int4 = DType.float16.sizeof() + (group_size // 2)

    var N = b.dim[0]()
    var K = b.dim[1]() // bytes_per_group_int4 * group_size

    if N % 32 != 0:
        raise ("N must be a multiple of 32")

    var k_groups = ceildiv(K, group_size)

    var src_ptr = b.data
    var dst_ptr = b_rot.data

    for n in range(0, N, n_groups):
        for nn in range(n_groups):
            var dst_k_ptr = dst_ptr
            for k in range(0, K, group_size):
                var scale = Scalar.load(src_ptr.bitcast[DType.float16]())
                Scalar.store(
                    dst_k_ptr.bitcast[DType.float16]().offset(nn), scale
                )
                src_ptr += DType.float16.sizeof()
                dst_k_ptr += DType.float16.sizeof() * n_groups

                var b_data_i4 = SIMD[size = group_size // 2].load(src_ptr)
                src_ptr += group_size // 2

                var b_data_i8_lo = (b_data_i4 & 15)
                var b_data_i8_hi = (b_data_i4 >> 4)
                var b_data_i8 = b_data_i8_lo.join(b_data_i8_hi)

                @parameter
                for i in range(0, group_size, 8):
                    var b_tuple_lo = b_data_i8.slice[4, offset=i]()
                    var b_tuple_hi = b_data_i8.slice[4, offset = i + 4]()
                    var b_tuple = (b_tuple_lo << 0) + (b_tuple_hi << 4)
                    SIMD.store(dst_k_ptr.offset(4 * nn), b_tuple)
                    dst_k_ptr += 4 * n_groups

        dst_ptr += n_groups * k_groups * bytes_per_group_int4


fn _quantize_a_block[
    group_size: Int, aq_type: DType, type: DType
](a_ptr: DTypePointer[type]) -> (
    SIMD[aq_type, group_size],
    Scalar[DType.float32],
):
    alias a_zero_point = 128 if aq_type.is_unsigned() else 0

    var fp_data = SIMD[size=group_size].load(a_ptr)
    var max_value = abs(fp_data).reduce_max()
    var scale = (max_value / 127.0).cast[DType.float32]()
    var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

    var quant_data_s8 = roundeven_to_int32(fp_data * multiplier).cast[
        DType.int8
    ]()
    var quant_data = quant_data_s8.cast[aq_type]() + a_zero_point

    return (quant_data, scale)


fn _quantize_a_buffer[
    group_size: Int,
    type: DType,
    aq_type: DType,
    *,
    aq_interleave: Int = group_size,
](
    a: NDBuffer[type, 2],
    a_quant: NDBuffer[aq_type, 2],
    a_scale: NDBuffer[DType.float32, 2],
):
    """Converts a floating point buffer to a symmetrically quantized
    representation. The data is in a packed layout that can be efficiently
    indexed by the matrix multiply kernels.
    """
    constrained[
        (group_size % aq_interleave) == 0,
        "interleave must be a factor of group size",
    ]()

    var M = a.dim[0]()
    var K = a.dim[1]()

    var a_quant_ptr = a_quant.data
    var a_scale_ptr = a_scale.data

    # Pack the quantized integers and scales in batches of K.
    for ko in range(0, K, K_BATCH_SIZE):
        var ko_count = min(K_BATCH_SIZE, K - ko)

        var am_ptr = a.data + ko

        @parameter
        @always_inline
        fn process_rows[tile_m: Int](m: Int):
            for row in range(tile_m):
                var ak_quant_ptr = a_quant_ptr + row * aq_interleave
                var ak_scale_ptr = a_scale_ptr + row

                for ki in range(0, ko_count, group_size):
                    var quant_data: SIMD[aq_type, group_size]
                    var scale: Scalar[DType.float32]
                    (quant_data, scale) = _quantize_a_block[
                        group_size, aq_type
                    ](am_ptr.offset(ki))

                    # Interleave this local block to the output buffer.
                    #
                    # This supports the i8mm use case where the instruction
                    # expects a 2x8 matrix of data loaded from two rows. This
                    # loop slices and outputs data at the `tile_m` stride.
                    #
                    # For the non-i8mm use case, no interleaving occurs and
                    # this is a simple store.
                    #
                    # For either case, when M=1, the data layout is effectively
                    # a flat array of data. The M=1 kernels assume this and
                    # ignore the K batching and interleave concepts.
                    @parameter
                    for i in range(0, group_size, aq_interleave):
                        SIMD.store(
                            ak_quant_ptr,
                            quant_data.slice[aq_interleave, offset=i](),
                        )
                        ak_quant_ptr += tile_m * aq_interleave

                    Scalar.store(ak_scale_ptr, scale)
                    ak_scale_ptr += tile_m

                am_ptr += K

            a_quant_ptr += tile_m * ko_count
            a_scale_ptr += tile_m * (ko_count // group_size)

        tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)


fn _unpack_weights[
    group_size: Int,
    tile_n: Int,
    simd_width: Int,
    needs_correction: Bool,
    is_i8mm: Bool,
](
    _b_s8_ptr: DTypePointer[DType.int8],
    _b_packed_ptr: DTypePointer[DType.uint8],
    _b_scale_ptr: DTypePointer[DType.float32],
    _b_correction_ptr: DTypePointer[DType.int32],
    batch_k: Int,
):
    var b_s8_ptr = _b_s8_ptr
    var b_packed_ptr = _b_packed_ptr
    var b_scale_ptr = _b_scale_ptr
    var b_correction_ptr = _b_correction_ptr

    for ko in range(0, batch_k, group_size):

        @parameter
        for col in range(tile_n):
            var b_scale = SIMD[size=simd_width].load(
                b_packed_ptr.bitcast[DType.float16](), col * simd_width
            ).cast[DType.float32]()
            SIMD.store(b_scale_ptr, col * simd_width, b_scale)

        b_scale_ptr += tile_n * simd_width
        b_packed_ptr += DType.float16.sizeof() * tile_n * simd_width

        var b_column_sums = InlineArray[SIMD[DType.int32, simd_width], tile_n](
            0
        )

        for k in range(0, group_size, 8):

            @parameter
            for col in range(tile_n):
                var b_data_packed = SIMD[size = simd_width * 4].load(
                    b_packed_ptr, col * simd_width * 4
                ).cast[DType.uint8]()
                var b_data_i4_lo = (b_data_packed & 15).cast[DType.int8]() - 8
                var b_data_i4_hi = (b_data_packed >> 4).cast[DType.int8]() - 8

                @parameter
                if needs_correction:
                    alias a_zero_point = SIMD[DType.uint8, simd_width * 4](128)

                    b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                        b_column_sums[col],
                        bitcast[DType.int32, simd_width](a_zero_point),
                        bitcast[DType.int32, simd_width](b_data_i4_lo),
                    )
                    b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                        b_column_sums[col],
                        bitcast[DType.int32, simd_width](a_zero_point),
                        bitcast[DType.int32, simd_width](b_data_i4_hi),
                    )

                @parameter
                if is_i8mm:
                    var intl = bitcast[DType.int32, simd_width](
                        b_data_i4_lo
                    ).interleave(bitcast[DType.int32, simd_width](b_data_i4_hi))
                    b_data_i4_lo = bitcast[DType.int8, simd_width * 4](
                        intl.slice[simd_width, offset=0]()
                    )
                    b_data_i4_hi = bitcast[DType.int8, simd_width * 4](
                        intl.slice[simd_width, offset=simd_width]()
                    )

                    SIMD.store(b_s8_ptr, col * simd_width * 8, b_data_i4_lo)
                    SIMD.store(
                        b_s8_ptr,
                        col * simd_width * 8 + (tile_n // 2) * simd_width * 4,
                        b_data_i4_hi,
                    )

                else:
                    SIMD.store(b_s8_ptr, col * simd_width * 4, b_data_i4_lo)
                    SIMD.store(
                        b_s8_ptr,
                        col * simd_width * 4 + tile_n * simd_width * 4,
                        b_data_i4_hi,
                    )

            b_s8_ptr += 2 * tile_n * simd_width * 4
            b_packed_ptr += tile_n * simd_width * 4

        @parameter
        if needs_correction:

            @parameter
            for col in range(tile_n):
                SIMD.store(
                    b_correction_ptr, simd_width * col, -b_column_sums[col]
                )

            b_correction_ptr += tile_n * simd_width


@always_inline
fn _scale_and_accumulate[
    group_size: Int,
    b_scale_type: DType,
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
](
    a_scale_ptr: DTypePointer[DType.float32],
    b_scale_ptr: DTypePointer[b_scale_type],
    inout c_int32: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
    inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    var b_scale = InlineArray[SIMD[DType.float32, simd_width], tile_n](0)

    # Load the per-column scale values for the B matrix.
    @parameter
    for col in range(tile_n):
        b_scale[col] = (
            SIMD[size=simd_width]
            .load(b_scale_ptr, col * simd_width)
            .cast[DType.float32]()
        )

    @parameter
    @always_inline
    fn apply_a_scale[row: Int](a_scale: Float32):
        @parameter
        for col in range(tile_n):
            c_float[row, col] += (
                c_int32[row, col].cast[DType.float32]() * a_scale * b_scale[col]
            )

    # Convert and rescale the integer accumulators and accumulate to the output
    # float accumulators.
    @parameter
    if has_neon():
        # NEON supports a multiply instruction that can broadcast from a
        # vector element, so help the compiler produce that by doing a vector
        # load.
        var a_scale = SIMD[size=tile_m].load(a_scale_ptr)

        @parameter
        for row in range(tile_m):
            apply_a_scale[row](a_scale[row])

    else:

        @parameter
        for row in range(tile_m):
            apply_a_scale[row](Scalar.load(a_scale_ptr, row))


trait _MatmulQInt4Kernel:
    @staticmethod
    fn aq_type() -> DType:
        """Returns the type to use for representing quantized A data."""
        ...

    @staticmethod
    fn aq_tuple_type() -> DType:
        """Returns the type to use for representing tuples of quantized A data.
        """
        ...

    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[DType.float32, 2],
    ):
        ...

    @staticmethod
    fn process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        inout c_float: _Accumulator[DType.float32, 1, tile_n, simd_width],
    ):
        ...

    @staticmethod
    fn process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_base_ptr: DTypePointer[DType.int8],
        b_ptr: DTypePointer[DType.float32],
        b_correction_ptr: DTypePointer[DType.int32],
        inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
    ):
        ...


struct _MatmulQInt4Kernel_x86(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    fn aq_type() -> DType:
        return DType.uint8

    @always_inline
    @staticmethod
    fn aq_tuple_type() -> DType:
        return DType.int32

    @always_inline
    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[DType.float32, 2],
    ):
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @staticmethod
    fn process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        inout c_float: _Accumulator[DType.float32, 1, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[DType.int32, 1, tile_n, simd_width]()

        c_int32.init()

        # Skip over the float16 scales.
        var b_offset = DType.float16.sizeof() * tile_n * simd_width

        var b_column_sums = InlineArray[SIMD[DType.int32, simd_width], tile_n](
            0
        )

        @parameter
        for k in range(0, group_size, 8):
            var a_val_lo = bitcast[DType.int32, 1](SIMD[size=4].load(a_ptr, k))
            var a_val_hi = bitcast[DType.int32, 1](
                SIMD[size=4].load(a_ptr, k + 4)
            )

            @parameter
            for col in range(tile_n):
                var b_data_packed = SIMD[size = simd_width * 4].load(
                    b_ptr, b_offset
                ).cast[DType.uint8]()
                b_offset += simd_width * 4

                var b_data_i4_lo = (b_data_packed & 15).cast[DType.int8]() - 8
                var b_data_i4_hi = (b_data_packed >> 4).cast[DType.int8]() - 8

                alias a_zero_point = SIMD[DType.uint8, simd_width * 4](128)

                b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                    b_column_sums[col],
                    bitcast[DType.int32, simd_width](a_zero_point),
                    bitcast[DType.int32, simd_width](b_data_i4_lo),
                )
                b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                    b_column_sums[col],
                    bitcast[DType.int32, simd_width](a_zero_point),
                    bitcast[DType.int32, simd_width](b_data_i4_hi),
                )

                c_int32[0, col] = dot_i8_to_i32_saturated_x86(
                    c_int32[0, col],
                    SIMD[DType.int32, simd_width](a_val_lo),
                    bitcast[DType.int32, simd_width](b_data_i4_lo),
                )
                c_int32[0, col] = dot_i8_to_i32_saturated_x86(
                    c_int32[0, col],
                    SIMD[DType.int32, simd_width](a_val_hi),
                    bitcast[DType.int32, simd_width](b_data_i4_hi),
                )

        @parameter
        for col in range(tile_n):
            c_int32[0, col] -= b_column_sums[col]

        var b_scale_ptr = b_ptr.bitcast[DType.float16]()

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )

    @always_inline
    @staticmethod
    fn process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        b_scale_ptr: DTypePointer[DType.float32],
        b_correction_ptr: DTypePointer[DType.int32],
        inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()

        # Initialize the integer accumulators with the zero point corrections.
        @parameter
        for col in range(tile_n):
            var correction_val = SIMD[size=simd_width].load(
                b_correction_ptr, col * simd_width
            )

            @parameter
            for row in range(tile_m):
                c_int32[row, col] = correction_val

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
                        c_int32[row, col], a_val, b_val
                    )

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


struct _MatmulQInt4Kernel_neon_dotprod(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    fn aq_type() -> DType:
        return DType.int8

    @always_inline
    @staticmethod
    fn aq_tuple_type() -> DType:
        return DType.int32

    @always_inline
    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[DType.float32, 2],
    ):
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @staticmethod
    fn process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        inout c_float: _Accumulator[DType.float32, 1, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[DType.int32, 1, tile_n, simd_width]()

        c_int32.init()

        # Skip over the float16 scales.
        var b_offset = DType.float16.sizeof() * tile_n * simd_width

        @parameter
        for k in range(0, group_size, 16):
            var a_val = SIMD[size=16].load(a_ptr, k)

            @parameter
            for lane in range(0, 4, 2):

                @parameter
                for col in range(tile_n):
                    var b_data_packed = SIMD[size = simd_width * 4].load(
                        b_ptr, b_offset
                    ).cast[DType.uint8]()
                    b_offset += simd_width * 4

                    var b_data_i4_lo = (b_data_packed & 15).cast[
                        DType.int8
                    ]() - 8
                    var b_data_i4_hi = (b_data_packed >> 4).cast[
                        DType.int8
                    ]() - 8

                    c_int32[0, col] = _neon_dotprod_lane[lane](
                        c_int32[0, col], b_data_i4_lo, a_val
                    )
                    c_int32[0, col] = _neon_dotprod_lane[lane + 1](
                        c_int32[0, col], b_data_i4_hi, a_val
                    )

        var b_scale_ptr = b_ptr.bitcast[DType.float16]()

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )

    @always_inline
    @staticmethod
    fn process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        b_scale_ptr: DTypePointer[DType.float32],
        b_correction_ptr: DTypePointer[DType.int32],
        inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()

        c_int32.init()

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
                    )
                    b_offset += simd_width * 4

                    @parameter
                    for row in range(tile_m):
                        c_int32[row, col] = _neon_dotprod_lane[lane](
                            c_int32[row, col], b_val, a_tile[row]
                        )

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


struct _MatmulQInt4Kernel_neon_i8mm(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    fn aq_type() -> DType:
        return DType.int8

    @always_inline
    @staticmethod
    fn aq_tuple_type() -> DType:
        return DType.int64

    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[DType.float32, 2],
    ):
        # Interleave the quantized data to produce the block format required
        # for the NEON `smmla` instruction.
        return _quantize_a_buffer[group_size, aq_interleave=8](
            a, a_quant, a_scale
        )

    @staticmethod
    fn process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        inout c_float: _Accumulator[DType.float32, 1, tile_n, simd_width],
    ):
        # The data layout for quantized A data is identical for the NEON dot
        # product kernel when M=1, so delegate to that implementation.
        _MatmulQInt4Kernel_neon_dotprod.process_group_packed[group_size](
            a_ptr, a_scale_ptr, b_ptr, c_float
        )

    @always_inline
    @staticmethod
    fn process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        b_scale_ptr: DTypePointer[DType.float32],
        b_correction_ptr: DTypePointer[DType.int32],
        inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
    ):
        alias block_m = max(tile_m // 2, 1)
        var c_int32_block = _Accumulator[
            DType.int32, block_m, tile_n * 2, simd_width
        ]()

        c_int32_block.init()

        var a_offset = 0
        var b_offset = 0

        @parameter
        for k in range(0, group_size, 8):
            var a_tile = InlineArray[SIMD[DType.int8, simd_width * 4], block_m](
                0
            )

            @parameter
            if tile_m > 1:

                @parameter
                for row in range(block_m):
                    a_tile[row] = SIMD[size = simd_width * 4].load(
                        a_ptr, a_offset
                    )
                    a_offset += simd_width * 4
            else:
                var a_val = SIMD[size = simd_width * 2].load(a_ptr, a_offset)
                a_tile[0] = rebind[SIMD[DType.int8, simd_width * 4]](
                    a_val.join(SIMD[DType.int8, simd_width * 2](0))
                )
                a_offset += simd_width * 2

            @parameter
            for col in range(tile_n * 2):
                var b_val = SIMD[size = simd_width * 4].load(b_ptr, b_offset)
                b_offset += simd_width * 4

                @parameter
                for row in range(block_m):
                    c_int32_block[row, col] = _neon_matmul(
                        c_int32_block[row, col], a_tile[row], b_val
                    )

        var c_int32 = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()

        # Swizzle 2x2 blocks to 1x4 vectors:
        # [a0 a1 b0 b1] [a2 a3 b2 b3] -> [a0 a1 a2 a3] [b0 b1 b2 b3]
        #
        # Note that these linear accumulators have a lifetime that overlaps the
        # blocked accumulators from above. Only an extra register is needed to
        # do the swizzling.
        @parameter
        for row in range(0, tile_m, 2):

            @parameter
            for col in range(tile_n):
                var c_val_0 = c_int32_block[row // 2, col * 2]
                var c_val_1 = c_int32_block[row // 2, col * 2 + 1]

                c_int32[row, col] = c_val_0.shuffle[0, 1, 4, 5](c_val_1)

                @parameter
                if tile_m > 1:
                    c_int32[row + 1, col] = c_val_0.shuffle[2, 3, 6, 7](c_val_1)

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


fn _matmul_qint4_m_1[
    kernel: _MatmulQInt4Kernel,
    group_size: Int,
    aq_type: DType,
    b_static_shape: DimList = DimList.create_unknown[2](),
](
    a_quant: NDBuffer[aq_type, 2],
    a_scale: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2, b_static_shape],
    c: NDBuffer[DType.float32, 2],
):
    alias simd_width = simdwidthof[DType.float32]()
    alias bytes_per_group_int4 = DType.float16.sizeof() + (group_size // 2)

    var N = b.dim[0]()
    var K = a_quant.dim[1]()
    var k_groups = K // group_size

    var b_ptr = b.data.bitcast[DType.int8]()

    alias grain_size = 64

    var num_workers = ceildiv(N, grain_size)

    @parameter
    @__copy_capture(N, K, k_groups)
    fn task_func(task_id: Int):
        var task_n_start = task_id * grain_size
        var task_n_count = min(N - task_n_start, grain_size)

        @parameter
        @always_inline
        fn process_cols[tile_n: Int](n_idx: Int):
            var n = task_n_start + n_idx * simd_width

            var c_float = _Accumulator[DType.float32, 1, tile_n, simd_width]()

            c_float.init()

            var ak_ptr = a_quant.data.bitcast[DType.int8]()
            var ak_scale_ptr = a_scale.data
            var bk_ptr = b_ptr + n * k_groups * bytes_per_group_int4

            for k in range(0, K, group_size):
                kernel.process_group_packed[group_size](
                    ak_ptr, ak_scale_ptr, bk_ptr, c_float
                )

                ak_ptr += group_size
                ak_scale_ptr += 1
                bk_ptr += tile_n * simd_width * bytes_per_group_int4

            c_float.store(c._offset(Index(0, n)), N)

        tile[process_cols, VariadicList[Int](2, 1)](
            0, ceildiv(task_n_count, simd_width)
        )

    sync_parallelize[task_func](num_workers)


fn _matmul_qint4_m_any[
    kernel: _MatmulQInt4Kernel,
    group_size: Int,
    aq_type: DType,
    b_static_shape: DimList = DimList.create_unknown[2](),
](
    a_quant: NDBuffer[aq_type, 2],
    a_scale: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2, b_static_shape],
    c: NDBuffer[DType.float32, 2],
):
    alias simd_width = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_width]]()
    alias bytes_per_group_int4 = DType.float16.sizeof() + (group_size // 2)

    var M = a_quant.dim[0]()
    var N = b.dim[0]()
    var K = a_quant.dim[1]()
    var k_groups = K // group_size

    alias grain_size = 64

    var num_workers = ceildiv(N, grain_size)

    @parameter
    @__copy_capture(M, N, K)
    fn task_func(task_id: Int):
        var task_n_start = task_id * grain_size
        var task_n_count = min(N - task_n_start, grain_size)

        var b_ptr = b.data

        for ko in range(0, K, K_BATCH_SIZE):
            var ko_count = min(K_BATCH_SIZE, K - ko)
            var ko_group = ko // group_size

            @parameter
            @always_inline
            fn process_cols[tile_n: Int](n_idx: Int):
                var n = task_n_start + n_idx * simd_width

                alias k_batch_groups = K_BATCH_SIZE // group_size

                var b_s8_buf = stack_allocation[
                    K_BATCH_SIZE * tile_n * simd_width,
                    DType.int8,
                    alignment=alignment,
                ]()
                var b_scale_buf = stack_allocation[
                    k_batch_groups * tile_n * simd_width,
                    DType.float32,
                    alignment=alignment,
                ]()

                # If the A matrix is quantized using an unsigned data type,
                # then a zero point correction is required to the block int32
                # accumulator.
                alias needs_correction = aq_type.is_unsigned()

                var b_correction_buf = stack_allocation[
                    k_batch_groups * tile_n * simd_width,
                    DType.int32,
                    alignment=alignment,
                ]() if needs_correction else DTypePointer[DType.int32]()

                _unpack_weights[
                    group_size,
                    tile_n,
                    simd_width,
                    needs_correction=needs_correction,
                    is_i8mm = kernel.aq_tuple_type() == DType.int64,
                ](
                    b_s8_buf,
                    b_ptr
                    + (n * k_groups + ko_group * tile_n * simd_width)
                    * bytes_per_group_int4,
                    b_scale_buf,
                    b_correction_buf,
                    ko_count,
                )

                var ak_ptr = a_quant.data + ko * M
                var ak_scale_ptr = a_scale.data + ko_group * M

                @parameter
                @always_inline
                fn process_rows[tile_m: Int](m: Int):
                    var c_ptr = c._offset(Index(m, n))
                    var c_float = _Accumulator[
                        DType.float32, tile_m, tile_n, simd_width
                    ]()

                    if ko == 0:
                        c_float.init()
                    else:
                        c_float.load(c_ptr, N)

                    var bk_s8_ptr = b_s8_buf
                    var bk_scale_ptr = b_scale_buf
                    var bk_correction_ptr = b_correction_buf

                    for ki in range(0, ko_count, group_size):
                        kernel.process_group_unpacked[group_size](
                            rebind[DTypePointer[DType.int8]](ak_ptr),
                            ak_scale_ptr,
                            bk_s8_ptr,
                            bk_scale_ptr,
                            bk_correction_ptr,
                            c_float,
                        )

                        ak_ptr += tile_m * group_size
                        ak_scale_ptr += tile_m
                        bk_s8_ptr += group_size * tile_n * simd_width
                        bk_scale_ptr += tile_n * simd_width
                        bk_correction_ptr += tile_n * simd_width

                    c_float.store(c_ptr, N)

                tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)

            tile[process_cols, VariadicList[Int](2, 1)](
                0, ceildiv(task_n_count, simd_width)
            )

    sync_parallelize[task_func](num_workers)


fn _matmul_qint4[
    kernel: _MatmulQInt4Kernel,
    group_size: Int,
    b_static_shape: DimList = DimList.create_unknown[2](),
](
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2, b_static_shape],
    c: NDBuffer[DType.float32, 2],
):
    alias simd_width = simdwidthof[DType.float32]()
    alias alignment = alignof[SIMD[DType.float32, simd_width]]()

    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()
    var k_groups = K // group_size

    alias aq_type = kernel.aq_type()

    var a_quant_base_ptr = DTypePointer[aq_type].alloc(
        M * K, alignment=alignment
    )
    var a_scale_base_ptr = DTypePointer[DType.float32].alloc(M * k_groups)

    var a_quant = NDBuffer[aq_type, 2](a_quant_base_ptr, Index(M, K))
    var a_scale = NDBuffer[DType.float32, 2](
        a_scale_base_ptr, Index(M, k_groups)
    )

    kernel.quantize_a_buffer[group_size](a, a_quant, a_scale)

    if M == 1:
        _matmul_qint4_m_1[kernel, group_size](a_quant, a_scale, b, c)
    else:
        _matmul_qint4_m_any[kernel, group_size](a_quant, a_scale, b, c)

    a_quant_base_ptr.free()
    a_scale_base_ptr.free()


fn matmul_qint4[
    group_size: Int,
    b_static_shape: DimList = DimList.create_unknown[2](),
](
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2, b_static_shape],
    c: NDBuffer[DType.float32, 2],
):
    @parameter
    fn kernel_dispatch[kernel: _MatmulQInt4Kernel]():
        return _matmul_qint4[kernel, group_size=group_size](a, b, c)

    @parameter
    if is_x86():
        kernel_dispatch[_MatmulQInt4Kernel_x86]()
    elif has_neon_int8_matmul() and not is_apple_silicon():
        kernel_dispatch[_MatmulQInt4Kernel_neon_i8mm]()
    elif has_neon_int8_dotprod():
        kernel_dispatch[_MatmulQInt4Kernel_neon_dotprod]()
    else:
        constrained[False, "unsupported architecture"]()
