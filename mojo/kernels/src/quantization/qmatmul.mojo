# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from algorithm import sync_parallelize, tile
from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.accumulate import _Accumulator
from LinAlg.neon_intrinsics import _neon_dotprod_lane, _neon_matmul
from LinAlg.vnni_intrinsics import dot_i8_to_i32_saturated_x86
from math import align_down, ceildiv
from memory.unsafe import DTypePointer
from sys.info import (
    has_avx512f,
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    is_x86,
    is_apple_silicon,
)
from sys.intrinsics import llvm_intrinsic
from utils import StaticTuple
from utils.index import Index


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
                var scale = src_ptr.bitcast[DType.float16]().load()
                dst_k_ptr.bitcast[DType.float16]().offset(nn).store(scale)
                src_ptr += DType.float16.sizeof()
                dst_k_ptr += DType.float16.sizeof() * n_groups

                var b_data_i4 = src_ptr.load[width = group_size // 2]()
                src_ptr += group_size // 2

                var b_data_i8_lo = (b_data_i4 & 15)
                var b_data_i8_hi = (b_data_i4 >> 4)
                var b_data_i8 = b_data_i8_lo.join(b_data_i8_hi)

                @parameter
                @always_inline
                fn repack_fn[i: Int]():
                    var b_tuple_lo = b_data_i8.slice[4, offset = i * 8]()
                    var b_tuple_hi = b_data_i8.slice[4, offset = i * 8 + 4]()
                    var b_tuple = (b_tuple_lo << 0) + (b_tuple_hi << 4)
                    dst_k_ptr.offset(4 * nn).store(b_tuple)
                    dst_k_ptr += 4 * n_groups

                unroll[repack_fn, group_size // 8]()

        dst_ptr += n_groups * k_groups * bytes_per_group_int4


fn _quantize_a_block[
    group_size: Int, aq_type: DType, scale_type: DType, type: DType
](a_ptr: DTypePointer[type]) -> (SIMD[aq_type, group_size], Scalar[scale_type]):
    alias a_zero_point = 128 if aq_type.is_unsigned() else 0

    @parameter
    @always_inline
    fn roundeven_to_int32(
        x: SIMD[type, group_size]
    ) -> SIMD[DType.int32, group_size]:
        alias simd_width = simdwidthof[type]()

        # Use the AVX512 instruction `vcvtps2dq` with embedded rounding control
        # set to do rounding to nearest with ties to even (roundeven). This
        # replaces a `vrndscaleps` and `vcvttps2dq` instruction pair.
        @parameter
        if has_avx512f() and type == DType.float32 and group_size >= simd_width:
            var x_i32 = SIMD[DType.int32, group_size]()

            @parameter
            @always_inline
            fn cvtps2dq_fn[idx: Int]():
                alias i = idx * simd_width
                var part = llvm_intrinsic[
                    "llvm.x86.avx512.mask.cvtps2dq.512",
                    SIMD[DType.int32, simd_width],
                    has_side_effect=False,
                ](
                    x.slice[simd_width, offset=i](),
                    SIMD[DType.int32, simd_width](0),
                    Int16(-1),  # no mask
                    Int32(8),  # round to nearest
                )
                x_i32 = x_i32.insert[offset=i](part)

            unroll[cvtps2dq_fn, group_size // simd_width]()
            return x_i32

        # Use the NEON instruction `fcvtns` to fuse the conversion to int32
        # with rounding to nearest with ties to even (roundeven). This
        # replaces a `frintn` and `fcvtzs` instruction pair.
        @parameter
        if has_neon() and type == DType.float32 and group_size >= simd_width:
            var x_i32 = SIMD[DType.int32, group_size]()

            @parameter
            @always_inline
            fn fcvtns_fn[idx: Int]():
                alias i = idx * simd_width
                var part = llvm_intrinsic[
                    "llvm.aarch64.neon.fcvtns.v4i32.v4f32",
                    SIMD[DType.int32, simd_width],
                    has_side_effect=False,
                ](x.slice[simd_width, offset=i]())
                x_i32 = x_i32.insert[offset=i](part)

            unroll[fcvtns_fn, group_size // simd_width]()
            return x_i32

        return x.roundeven().cast[DType.int32]()

    var fp_data = a_ptr.load[width=group_size]()
    var max_value = abs(fp_data).reduce_max()
    var scale = (max_value / 127.0).cast[scale_type]()
    var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

    var quant_data_s8 = roundeven_to_int32(fp_data * multiplier).cast[
        DType.int8
    ]()
    var quant_data = quant_data_s8.cast[aq_type]() + a_zero_point

    return (quant_data, scale)


fn _quantize_a_buffer[
    group_size: Int, type: DType, aq_type: DType, scale_type: DType
](
    a: NDBuffer[type, 2],
    a_quant: NDBuffer[aq_type, 2],
    a_scale: NDBuffer[scale_type, 2],
):
    var M = a.dim[0]()
    var K = a.dim[1]()

    var a_ptr = a.data
    var a_quant_ptr = a_quant.data
    var a_scale_ptr = a_scale.data

    # Dynamically quantize the input in blocks of 32 bytes. Uses symmetric
    # quantization.
    for m in range(M):
        for k in range(0, K, group_size):
            var quant_data: SIMD[aq_type, group_size]
            var scale: Scalar[scale_type]
            (quant_data, scale) = _quantize_a_block[
                group_size, aq_type, scale_type
            ](a_ptr)

            a_quant_ptr.store(quant_data)
            a_scale_ptr.store(scale)

            a_ptr += group_size
            a_quant_ptr += group_size
            a_scale_ptr += 1


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

        @unroll
        for col in range(tile_n):
            var b_scale = (
                b_packed_ptr.bitcast[DType.float16]()
                .load[width=simd_width](col * simd_width)
                .cast[DType.float32]()
            )
            b_scale_ptr.store(col * simd_width, b_scale)

        b_scale_ptr += tile_n * simd_width
        b_packed_ptr += DType.float16.sizeof() * tile_n * simd_width

        var b_column_sums = StaticTuple[SIMD[DType.int32, simd_width], tile_n](
            0
        )

        for k in range(0, group_size, 8):

            @unroll
            for col in range(tile_n):
                var b_data_packed = b_packed_ptr.load[width = simd_width * 4](
                    col * simd_width * 4
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

                    b_s8_ptr.store(col * simd_width * 8, b_data_i4_lo)
                    b_s8_ptr.store(
                        col * simd_width * 8 + (tile_n // 2) * simd_width * 4,
                        b_data_i4_hi,
                    )

                else:
                    b_s8_ptr.store(col * simd_width * 4, b_data_i4_lo)
                    b_s8_ptr.store(
                        col * simd_width * 4 + tile_n * simd_width * 4,
                        b_data_i4_hi,
                    )

            b_s8_ptr += 2 * tile_n * simd_width * 4
            b_packed_ptr += tile_n * simd_width * 4

        @parameter
        if needs_correction:

            @unroll
            for col in range(tile_n):
                b_correction_ptr.store(simd_width * col, -b_column_sums[col])

            b_correction_ptr += tile_n * simd_width


@always_inline
fn _scale_and_accumulate[
    group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
](
    K: Int,
    a_scale_ptr: DTypePointer[DType.float32],
    b_scale_ptr: DTypePointer[DType.float32],
    inout c_int32: _Accumulator[DType.int32, tile_m, tile_n, simd_width],
    inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
):
    var b_scale = StaticTuple[SIMD[DType.float32, simd_width], tile_n]()

    # Load the per-column scale values for the B matrix.
    @unroll
    for col in range(tile_n):
        b_scale[col] = b_scale_ptr.load[width=simd_width](col * simd_width)

    # Convert and rescale the integer accumulators and accumulate to the output
    # float accumulators.
    @unroll
    for row in range(tile_m):
        var a_scale = a_scale_ptr.load(row * (K // group_size))

        @unroll
        for col in range(tile_n):
            c_float[row, col] += (
                c_int32[row, col].cast[DType.float32]() * a_scale * b_scale[col]
            )


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
    fn aq_row_interleave[tile_m: Int]() -> Int:
        ...

    @staticmethod
    fn quantize_a_buffer[
        group_size: Int,
        type: DType,
        aq_type: DType,
        scale_type: DType,
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[scale_type, 2],
    ):
        ...

    @staticmethod
    fn process_group[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        K: Int,
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_base_ptr: DTypePointer[DType.int8],
        b_scale_ptr: DTypePointer[DType.float32],
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
    fn aq_row_interleave[tile_m: Int]() -> Int:
        return 1

    @always_inline
    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType, scale_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[scale_type, 2],
    ):
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @always_inline
    @staticmethod
    fn process_group[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        K: Int,
        a_ptr: DTypePointer[DType.int8],
        a_scale_ptr: DTypePointer[DType.float32],
        b_ptr: DTypePointer[DType.int8],
        b_scale_ptr: DTypePointer[DType.float32],
        b_correction_ptr: DTypePointer[DType.int32],
        inout c_float: _Accumulator[DType.float32, tile_m, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[DType.int32, tile_m, tile_n, simd_width]()

        # Initialize the integer accumulators with the zero point corrections.
        @unroll
        for col in range(tile_n):
            var correction_val = b_correction_ptr.load[width=simd_width](
                col * simd_width
            )

            @unroll
            for row in range(tile_m):
                c_int32[row, col] = correction_val

        var b_offset = 0

        @unroll
        for k in range(0, group_size, 4):

            @unroll
            for col in range(tile_n):
                var b_val = bitcast[DType.int32, simd_width](
                    b_ptr.load[width = simd_width * 4](b_offset)
                )
                b_offset += simd_width * 4

                @unroll
                for row in range(tile_m):
                    var a_val = SIMD[DType.int32, simd_width](
                        bitcast[DType.int32, 1](
                            a_ptr.load[width=4](row * K + k)
                        )
                    )
                    c_int32[row, col] = dot_i8_to_i32_saturated_x86(
                        c_int32[row, col], a_val, b_val
                    )

        _scale_and_accumulate[group_size](
            K, a_scale_ptr, b_scale_ptr, c_int32, c_float
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
    fn aq_row_interleave[tile_m: Int]() -> Int:
        return 1

    @always_inline
    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType, scale_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[scale_type, 2],
    ):
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @always_inline
    @staticmethod
    fn process_group[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        K: Int,
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

        @unroll
        for k in range(0, group_size, 16):
            var a_tile = StaticTuple[SIMD[DType.int8, 16], tile_m]()

            @unroll
            for row in range(tile_m):
                a_tile[row] = a_ptr.load[width=16](K * row + k)

            @parameter
            @always_inline
            fn dotprod_fn[idx: Int]():
                @unroll
                for col in range(tile_n):
                    var b_val = b_ptr.load[width = simd_width * 4](b_offset)
                    b_offset += simd_width * 4

                    @unroll
                    for row in range(tile_m):
                        c_int32[row, col] = _neon_dotprod_lane[idx](
                            c_int32[row, col], b_val, a_tile[row]
                        )

            unroll[dotprod_fn, 4]()

        _scale_and_accumulate[group_size](
            K, a_scale_ptr, b_scale_ptr, c_int32, c_float
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

    @always_inline
    @staticmethod
    fn aq_row_interleave[tile_m: Int]() -> Int:
        return tile_m

    @staticmethod
    fn quantize_a_buffer[
        group_size: Int, type: DType, aq_type: DType, scale_type: DType
    ](
        a: NDBuffer[type, 2],
        a_quant: NDBuffer[aq_type, 2],
        a_scale: NDBuffer[scale_type, 2],
    ):
        var M = a.dim[0]()
        var K = a.dim[1]()

        var a_ptr = a.data
        var a_quant_ptr = a_quant.data
        var a_scale_ptr = a_scale.data

        @parameter
        @always_inline
        fn process_rows[tile_m: Int](m: Int):
            alias aq_tuple_type = DType.int64
            alias aq_tuple_size = aq_tuple_type.sizeof()
            alias aq_tuple_stride = tile_m * aq_tuple_size

            for row in range(tile_m):
                var ak_quant_ptr = a_quant_ptr + row * aq_tuple_size

                for k in range(0, K, group_size):
                    var quant_data: SIMD[aq_type, group_size]
                    var scale: Scalar[scale_type]
                    (quant_data, scale) = _quantize_a_block[
                        group_size, aq_type, scale_type
                    ](a_ptr)

                    var quant_data_tuple = bitcast[
                        aq_tuple_type, group_size // aq_tuple_size
                    ](quant_data)

                    @parameter
                    @always_inline
                    fn store_strided_tuple[idx: Int]():
                        ak_quant_ptr.bitcast[aq_tuple_type]().store(
                            quant_data_tuple[idx]
                        )
                        ak_quant_ptr += aq_tuple_stride

                    unroll[store_strided_tuple, group_size // aq_tuple_size]()

                    a_scale_ptr.store(scale)

                    a_ptr += group_size
                    a_scale_ptr += 1

            a_quant_ptr += tile_m * K

        tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)

    @always_inline
    @staticmethod
    fn process_group[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        K: Int,
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

        @unroll
        for k in range(0, group_size, 8):
            var a_tile = StaticTuple[
                SIMD[DType.int8, simd_width * 4], block_m
            ]()

            @parameter
            if tile_m > 1:

                @unroll
                for row in range(block_m):
                    a_tile[row] = a_ptr.load[width = simd_width * 4](a_offset)
                    a_offset += simd_width * 4
            else:
                var a_val = a_ptr.load[width = simd_width * 2](a_offset)
                a_tile[0] = rebind[SIMD[DType.int8, simd_width * 4]](
                    a_val.join(SIMD[DType.int8, simd_width * 2](0))
                )
                a_offset += simd_width * 2

            @unroll
            for col in range(tile_n * 2):
                var b_val = b_ptr.load[width = simd_width * 4](b_offset)
                b_offset += simd_width * 4

                @unroll
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
        @unroll
        for row in range(0, tile_m, 2):

            @unroll
            for col in range(tile_n):
                var c_val_0 = c_int32_block[row // 2, col * 2]
                var c_val_1 = c_int32_block[row // 2, col * 2 + 1]

                c_int32[row, col] = c_val_0.shuffle[0, 1, 4, 5](c_val_1)

                @parameter
                if tile_m > 1:
                    c_int32[row + 1, col] = c_val_0.shuffle[2, 3, 6, 7](c_val_1)

        _scale_and_accumulate[group_size](
            K, a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


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
    alias bytes_per_group_int4 = DType.float16.sizeof() + (group_size // 2)

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

    alias grain_size = 64

    var num_workers = ceildiv(N, grain_size)

    @parameter
    @__copy_capture(M, N, K)
    fn task_func(task_id: Int):
        var task_n_start = task_id * grain_size
        var task_n_count = min(N - task_n_start, grain_size)

        alias k_batch = 512
        alias k_batch_groups = k_batch // group_size

        var b_ptr = b.data

        for ko in range(0, K, k_batch):
            var ko_group = ko // group_size
            var k_unpack_count = min(k_batch, K - ko)

            @parameter
            @always_inline
            fn process_cols[tile_n: Int](n_idx: Int):
                var n = task_n_start + n_idx * simd_width

                var b_s8_buf = stack_allocation[
                    k_batch * tile_n * simd_width,
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
                    k_unpack_count,
                )

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

                    # Some kernels (i8mm) may interleave the quantized A data,
                    # so this scales the starting offset and strides to access
                    # the current position.
                    alias aq_row_interleave = kernel.aq_row_interleave[tile_m]()

                    var ak_ptr = a_quant.data + m * K + aq_row_interleave * ko
                    var ak_scale_ptr = a_scale.data + m * k_groups + ko_group
                    var bk_s8_ptr = b_s8_buf
                    var bk_scale_ptr = b_scale_buf
                    var bk_correction_ptr = b_correction_buf

                    for ki in range(0, k_unpack_count, group_size):
                        kernel.process_group[group_size](
                            K,
                            rebind[DTypePointer[DType.int8]](ak_ptr),
                            ak_scale_ptr,
                            bk_s8_ptr,
                            bk_scale_ptr,
                            bk_correction_ptr,
                            c_float,
                        )

                        ak_ptr += aq_row_interleave * group_size
                        ak_scale_ptr += 1
                        bk_s8_ptr += group_size * tile_n * simd_width
                        bk_scale_ptr += tile_n * simd_width
                        bk_correction_ptr += tile_n * simd_width

                    c_float.store(c_ptr, N)

                tile[process_rows, VariadicList[Int](4, 2, 1)](0, M)

            tile[process_cols, VariadicList[Int](2, 1)](
                0, ceildiv(task_n_count, simd_width)
            )

    sync_parallelize[task_func](num_workers)

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
