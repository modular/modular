# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import align_down, ceil, ceildiv
from sys.info import has_avx2, has_neon_int8_dotprod, sizeof

from algorithm import sync_parallelize
from bit import is_power_of_two
from buffer import NDBuffer
from buffer.buffer import prod_dims
from buffer.list import DimList
from linalg.neon_intrinsics import _neon_dotprod
from linalg.vnni_intrinsics import dot_i8_to_i32_saturated_x86
from memory.unsafe import bitcast

from utils import Index, StaticIntTuple, StaticTuple, unroll


@always_inline
fn _to_StaticTuple[
    type: DType, size: Int
](data: SIMD[type, size]) -> StaticTuple[Scalar[type], size]:
    """Convert SIMD to StaticTuple."""

    @parameter
    if size == 1:
        return StaticTuple[_, size](data[0])
    elif size == 2:
        return StaticTuple[_, size](data[0], data[1])
    elif size == 4:
        return StaticTuple[_, size](data[0], data[1], data[2], data[3])
    elif size == 8:
        return StaticTuple[_, size](
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
        )
    elif size == 16:
        return StaticTuple[_, size](
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
        )
    else:
        constrained[size == 32]()
        return StaticTuple[_, size](
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
            data[16],
            data[17],
            data[18],
            data[19],
            data[20],
            data[21],
            data[22],
            data[23],
            data[24],
            data[25],
            data[26],
            data[27],
            data[28],
            data[29],
            data[30],
            data[31],
        )


@always_inline
fn _to_SIMD[
    type: DType, size: Int
](data: StaticTuple[Scalar[type], size]) -> SIMD[type, size]:
    """Convert StaticTuple to SIMD."""

    @parameter
    if size == 1:
        return SIMD[type, size](data[0])
    elif size == 2:
        return SIMD[type, size](data[0], data[1])
    elif size == 4:
        return SIMD[type, size](data[0], data[1], data[2], data[3])
    elif size == 8:
        return SIMD[type, size](
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
        )
    elif size == 16:
        return SIMD[type, size](
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
        )
    else:
        constrained[size == 32]()
        return SIMD[type, size](
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
            data[16],
            data[17],
            data[18],
            data[19],
            data[20],
            data[21],
            data[22],
            data[23],
            data[24],
            data[25],
            data[26],
            data[27],
            data[28],
            data[29],
            data[30],
            data[31],
        )


@always_inline
fn calculate_symmetric_vector[
    input_dtype: DType, simd_width: Int, output_bits: Int
](data: SIMD[input_dtype, simd_width]) -> (
    SIMD[DType.uint8, simd_width],
    SIMD[input_dtype, 1],
):
    """
    Symmetrically quantizes the given SIMD vector `data` with input type
    `input_dtype` and `simd_width` elements, assuming we want
    the results to fit in an unsigned integer of size `output_bits`.

    Parameters:
        input_dtype: The dtype of the input tensor.
        simd_width: The width of the SIMD input.
        output_bits: The bits we want to fit the unsigned integral result in.

    Args:
        data: The input SIMD we want to quantize.

    Returns:
      A vector of the quantized values.
      The associated scale factor.
    """
    constrained[
        output_bits >= 1 and output_bits <= 8,
        "expected a scalar type",
    ]()
    constrained[
        input_dtype.is_floating_point(),
        "expect accumulating over floating point only.",
    ]()
    var max_value = data.reduce_max()
    var min_value = data.reduce_min()
    var result_range = max(max_value, -min_value)

    # quantize as if we were signed, and convert back to unsigned later
    var positive_steps = (1 << (output_bits - 1)) - 1
    var negative_steps = (1 << (output_bits - 1))

    # Handle the case we only have one type of value in `data`, e.g.
    # data = [12., 12., 12., 12.], then the scale is 1/12.0 and the quantized
    # data is ~[1, 1, 1, 1] + `negative_steps`
    var f32_scale = (max_value / 1.0) if (
        result_range == 0
    ) else result_range / positive_steps

    # TODO: consider clipping values
    var data_rounded = (data / f32_scale).roundeven().cast[DType.int8]()

    # each bit pattern in `data_quantized`
    var data_quantized = (data_rounded + negative_steps).cast[DType.uint8]()

    return data_quantized, f32_scale


struct Q4sym[
    group_size: Int,
    float_dtype: DType = DType.float32,
]:
    """
    Q4sym: compresses values of type `float_dtype` to 4bit unsigned integers
    which have been dynamically symmetrically quantized with the given scale
    factor.

    `group_size` determines the number of elements which share quantization
    parameters.

    We store things in a strided fashion:
    Example:

        Assume `group_size = 8` and we want to process uint4 numbers:
        A, B, C, D, E, F, G, H which have associated bits aaaa, bbbb, cccc, ....

           eeeeaaaa|ffffbbbb|ggggcccc|hhhhdddd

    To uncompress to floating point, take the decoded uint4 value, subtract
    the implicit zero-point of 2^4=8, and multiply by the scale factor.

    Parameters:
        group_size: The number of encoded numbers stored in this struct.
        float_dtype: The floating point dtype this struct works with.
    """

    var scale: StaticTuple[UInt8, 2]
    """The FP16 scale of the group, stored as individual bytes."""

    var bits: StaticTuple[UInt8, group_size // 2]
    """The bits of the encoded uint4 numbers."""

    @staticmethod
    @always_inline
    fn _check_constraints():
        # TODO
        constrained[
            is_power_of_two(group_size), "`group_size` must be a power of 2."
        ]()
        constrained[
            group_size == 8 or group_size == 16 or group_size == 32,
            "Only support some `group_sizes`",
        ]()
        constrained[
            float_dtype.is_floating_point(), "Must be floating point type"
        ]()

    @always_inline
    fn __init__(inout self):
        """Construct a default initialized Q4sym."""
        self.scale = StaticTuple[UInt8, 2]()
        self.bits = StaticTuple[UInt8, group_size // 2]()
        self._check_constraints()

    @always_inline
    fn __init__(inout self, data: SIMD[float_dtype, group_size]):
        """
        Construct an encoded Q4sym from data.

        Args:
            data: The floating point data to encode and store.
        """
        var quantization_tuple = calculate_symmetric_vector[
            float_dtype, group_size, 4
        ](data)
        var qdata = quantization_tuple[0]
        var f_scale = quantization_tuple[1]

        # TODO: add warning if we overflow/underflow
        var f16_scale = f_scale.cast[DType.float16]()
        self.scale = _to_StaticTuple(bitcast[DType.uint8, 2](f16_scale))
        self.bits = _to_StaticTuple(self._encode_bits(qdata))
        self._check_constraints()

    @staticmethod
    @always_inline
    fn _encode_bits(
        qdata: SIMD[DType.uint8, group_size]
    ) -> SIMD[DType.uint8, group_size // 2]:
        var lo_hi = qdata.split()
        return lo_hi[0] | (lo_hi[1] << 4)

    @always_inline
    fn _decode_bits(inout self) -> SIMD[DType.uint8, group_size]:
        # Extract the lower 4 bits of all bits in the `l_bits` format
        var bits_simd = _to_SIMD[DType.uint8, group_size // 2](self.bits)
        var bits_upper = (bits_simd & 0xF0) >> 4
        var bits_lower = (bits_simd & 0x0F)
        return rebind[SIMD[DType.uint8, group_size]](
            bits_lower.join(bits_upper)
        )

    @always_inline
    fn decode_scale(inout self) -> Float16:
        """
        Obtain the scale factor.

        Returns:
          The decoded scale factor.
        """
        # We avoid bit-casting directly, as the code-generation might hit a
        # path which attempts to bitcast 2xui8 --> f16 which is not typically supported
        var scale_bytes: SIMD[DType.uint8, 2] = _to_SIMD[DType.uint8, 2](
            self.scale
        )

        # NOTE: this may break on different endian systems...
        var upcast_bytes: SIMD[DType.uint16, 2] = scale_bytes.cast[
            DType.uint16
        ]()
        upcast_bytes[1] = upcast_bytes[1] << 8
        var final_result = upcast_bytes.reduce_add()
        var scale_decoded = bitcast[DType.float16, 1](final_result)
        return scale_decoded

    @always_inline
    fn decode_unsigned(inout self) -> SIMD[DType.uint8, group_size]:
        """
        Decode the stored uint4 numbers to uint8.

        Returns:
          The decoded stored numbers as uint8 numbers. These have an implicit
          zero-point of 8.
        """
        # Obtain the unsigned quantized values, these have a zp of 8
        return self._decode_bits()

    @always_inline
    fn decode_signed(inout self) -> SIMD[DType.int8, group_size]:
        """
        Decode the stored uint4 numbers to requantized int4 numbers.

        This is done by simply subtracting an implicit zp of 8 from the
        unsigned decoding.

        Returns:
          The decoded stored numbers as int8 numbers. These have a zero-point of
          0.
        """
        var decoded_result = self.decode_unsigned()
        return decoded_result.cast[DType.int8]() - 8

    @always_inline
    fn decode_fully(inout self) -> SIMD[float_dtype, group_size]:
        """
        Decode the stored numbers into floating point representation.

        Returns:
          The decoded numbers.
        """
        # Obtain the fully dequantized values
        var signed_result = self.decode_signed()
        var scale_decoded = self.decode_scale().cast[float_dtype]()
        var answer = scale_decoded.cast[float_dtype]() * signed_result.cast[
            float_dtype
        ]()
        return answer

    # TODO: support other axis of quantization, right now assume inner-most dim.
    # TODO: support axis which not divisible by group_size
    @staticmethod
    fn quantize_and_write_to_tensor[
        rank: Int,
    ](
        input_tensor: NDBuffer[float_dtype, rank],
        output_tensor: NDBuffer[DType.uint8, rank],
        input_shape: StaticIntTuple[rank],
    ):
        """
        Encodes the floating point numbers in `input_tensor` along the
        inner-most dimension and writes the result to output_tensor.

        Parameters:
            rank: The rank of the input and output tensors.

        Args:
            input_tensor: The input tensor we are encoding.
            output_tensor: The output tensor containing the encoded input.
                The shape of the output should be the same as the input
                except along the inner dimension where if the original inner
                dimension was `d`, the corresponding output dimension should be:
                    ceil(`d` / group_size) * sizeof(self).
            input_shape: The shape of the input tensor.
        """
        # TODO: check contiguous inputs and outputs

        # Read and quantize `input_tensor`` to blocked format, dump the raw
        # struct/block into `output_tensor`
        var size_of_block = sizeof[Q4sym[group_size, float_dtype]]()
        debug_assert(
            input_shape[rank - 1] % group_size == 0,
            "Only support fully divisible dimensions right now.",
        )

        var blob_output_ptr = output_tensor.data.address
        var base_block_ptr = UnsafePointer(blob_output_ptr.address).bitcast[
            Q4sym[group_size, float_dtype]
        ]()

        # as we support only inner-most dim, treat like rank-2 tensor
        var outer_stride = prod_dims[0, rank - 1](input_tensor)
        var input_inner_stride = input_shape[rank - 1]
        var output_inner_stride = ceildiv(input_shape[rank - 1], group_size)

        # TODO: vectorize parallelize, blah blah blah
        for i in range(outer_stride):
            for j in range(output_inner_stride):
                var flat_index_input = input_inner_stride * i + j * group_size
                var loaded_group = SIMD[size=group_size].load(
                    input_tensor.data, flat_index_input
                )

                var flat_index_output = output_inner_stride * i + j
                var output_ptr = base_block_ptr + flat_index_output

                # TODO: use the memory more directly instead of memcpy
                var encoded_data = Q4sym[group_size, float_dtype](loaded_group)
                var src_ptr = UnsafePointer.address_of(encoded_data).bitcast[
                    address_space = output_ptr.address_space
                ]()
                memcpy(output_ptr, src_ptr, 1)
                _ = encoded_data^

    @staticmethod
    fn dequantize_and_write_to_tensor[
        rank: Int,
    ](
        input_tensor: NDBuffer[DType.uint8, rank],
        output_tensor: NDBuffer[float_dtype, rank],
        output_shape: StaticIntTuple[rank],
    ):
        """
        Encodes the floating point numbers in `input_tensor` along the
        inner-most dimension and writes the result to output_tensor.

        Parameters:
            rank: The rank of the input and output tensors.

        Args:
            input_tensor: The input tensor we are decoding.
            output_tensor: The output tensor containing the decoded input.
            output_shape: The shape of the output tensor.
        """
        # Read and dequantize `input_tensor` which are the bytes of the raw
        # blocked format. Write the corresponding results to `output_tensor`
        debug_assert(
            output_shape[rank - 1] % group_size == 0,
            "Only support fully divisible dimensions right now.",
        )

        # TODO: check contiguous inputs and outputs

        var uint8_input_ptr = input_tensor.data.address
        var base_block_ptr = UnsafePointer(uint8_input_ptr.address).bitcast[
            Q4sym[group_size, float_dtype]
        ]()

        # as we support only inner-most dim, treat like rank-2 tensor
        var output_inner_dim = output_shape[rank - 1]
        var outer_dim = prod_dims[0, rank - 1](output_tensor)

        # Note: this is calculated assuming a pointer of Q4Sym's
        var input_inner_dim = ceildiv(output_inner_dim, group_size)

        # TODO: vectorize parallelize, blah blah blah
        for i in range(outer_dim):
            for j in range(input_inner_dim):
                var flat_index_input = input_inner_dim * i + j
                var encoded = Q4sym[group_size, float_dtype]()
                memcpy(
                    UnsafePointer.address_of(encoded),
                    base_block_ptr + flat_index_input,
                    1,
                )

                var flat_index_output = output_inner_dim * i + j * group_size
                SIMD[size=group_size].store(
                    output_tensor.data,
                    flat_index_output,
                    encoded.decode_fully(),
                )


fn _block_quantize_a[
    group_size: Int,
    type: DType,
    scale_type: DType,
](
    a: NDBuffer[type, 2],
    a_quant: NDBuffer[DType.int8, 2],
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
            var fp_data = SIMD[size=group_size].load(a_ptr)
            var max_value = abs(fp_data).reduce_max()
            var scale = max_value / 127.0
            var multiplier = Scalar[type](
                127.0
            ) / max_value if max_value != 0.0 else 0.0

            var quant_data = (fp_data * multiplier).roundeven().cast[
                DType.int8
            ]()
            SIMD.store(a_quant_ptr, quant_data)
            Scalar.store(a_scale_ptr, scale.cast[scale_type]())

            a_ptr = a_ptr.offset(group_size)
            a_quant_ptr = a_quant_ptr.offset(group_size)
            a_scale_ptr = a_scale_ptr.offset(1)


@always_inline
fn _matmul_int4_dotprod[
    simd_width: Int,
    group_size: Int,
](
    a: SIMD[DType.int8, group_size],
    b: SIMD[DType.int8, group_size],
) -> SIMD[
    DType.int32, simd_width
]:
    @always_inline
    @parameter
    fn dotprod[
        simd_width: Int
    ](
        a: SIMD[DType.int8, simd_width * 4],
        b: SIMD[DType.int8, simd_width * 4],
        c: SIMD[DType.int32, simd_width],
    ) -> SIMD[DType.int32, simd_width]:
        var c_local = c

        @parameter
        if has_avx2():
            c_local = dot_i8_to_i32_saturated_x86(
                c_local,
                bitcast[DType.int32, simd_width](b),
                bitcast[DType.int32, simd_width](a),
            )
            c_local = dot_i8_to_i32_saturated_x86(
                c_local,
                bitcast[DType.int32, simd_width](
                    SIMD[DType.uint8, simd_width * 4](8)
                ),
                bitcast[DType.int32, simd_width](-a),
            )
            return c_local

        # Adjust the `b` unpacked weights from values in the unsigned range
        # [0:15] to the signed range [-8:7].
        var b_s8 = b - 8

        @parameter
        if has_neon_int8_dotprod() and simd_width == 4:
            c_local = rebind[SIMD[DType.int32, simd_width]](
                _neon_dotprod(
                    rebind[SIMD[DType.int32, 4]](c_local),
                    rebind[SIMD[DType.int8, 16]](a),
                    rebind[SIMD[DType.int8, 16]](b_s8),
                )
            )
            return c_local

        @parameter
        for i in range(4):
            c_local += (
                a.slice[simd_width, offset = i * simd_width]().cast[
                    DType.int32
                ]()
                * b_s8.slice[simd_width, offset = i * simd_width]().cast[
                    DType.int32
                ]()
            )
        return c_local

    var c = SIMD[DType.int32, simd_width](0)

    @parameter
    for i in range(ceildiv(group_size, simd_width * 4)):
        alias idx = i * simd_width * 4
        var a_slice = a.slice[simd_width * 4, offset=idx]()
        var b_slice = b.slice[simd_width * 4, offset=idx]()
        c = dotprod(a_slice, b_slice, c)

    return c


@always_inline
fn _process_rows[
    group_size: Int,
    row_count: Int,
    type: DType,
](
    a_quant: NDBuffer[DType.int8, 2],
    a_scale: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[type, 2],
    n: Int,
):
    alias block_size = sizeof[Q4sym[group_size, type]]()
    alias simd_width = min(simdwidthof[DType.int32](), group_size // 4)

    var M = a_quant.dim[0]()
    var K = a_quant.dim[1]()
    var N_packed_bytes = b.dim[1]()
    var k_groups = K // group_size

    var accum_fp_tile = NDBuffer[
        type, 2, DimList(row_count, simd_width)
    ].stack_allocation()

    @parameter
    @always_inline
    fn dequantize_simd_int4(
        b_ptr: DTypePointer[DType.uint8], b_scale: SIMD[type, 1]
    ) -> SIMD[DType.int8, group_size]:
        var b_data_i4 = SIMD[size = group_size // 2].load(b_ptr.offset(2))
        var b_data_i8_lo = ((b_data_i4 >> 4)).cast[DType.int8]()
        var b_data_i8_hi = ((b_data_i4 & 15)).cast[DType.int8]()

        return rebind[SIMD[DType.int8, group_size]](
            b_data_i8_hi.join(b_data_i8_lo)
        )

    # 3. Outer matmul loop over `M`.
    for m in range(M):
        var a_quant_ptr = a_quant._offset(Index(m, 0))
        var a_scale_ptr = a_scale._offset(Index(m, 0))
        var b_ptr = b._offset(Index(n, 0))

        @parameter
        for row in range(row_count):
            accum_fp_tile.store(Index(row, 0), SIMD[type, simd_width](0))

        # 4. Inner loop over `K // block_size_bytes`.
        for k in range(k_groups):
            var a_data_i8 = SIMD[size=group_size].load(a_quant_ptr)
            var a_scale = a_scale_ptr[0].cast[type]()

            # 5. Process `row_batch_count` rows of `b` at a time.
            @parameter
            for row in range(row_count):
                # Dequantize a group of Q4_0 nibbles to int8.
                var b_row_ptr = b_ptr.offset(row * N_packed_bytes)
                var b_scale = bitcast[DType.float16, 1](
                    SIMD[size=2].load(b_row_ptr)
                ).cast[type]()
                var b_data_i8 = dequantize_simd_int4(b_row_ptr, b_scale)

                var c_data_i32 = _matmul_int4_dotprod[simd_width, group_size](
                    a_data_i8, b_data_i8
                )
                var accum_fp = accum_fp_tile.load[width=simd_width](
                    Index(row, 0)
                )
                accum_fp += c_data_i32.cast[type]() * a_scale * b_scale
                accum_fp_tile.store(Index(row, 0), accum_fp)

            a_quant_ptr = a_quant_ptr.offset(group_size)
            a_scale_ptr = a_scale_ptr.offset(1)
            b_ptr = b_ptr.offset(block_size)

        @parameter
        for row in range(row_count):
            var accum_fp = accum_fp_tile.load[width=simd_width](Index(row, 0))
            c.store(Index(m, row + n), accum_fp.reduce_add())


fn ggml_q4_0_matmul_impl[
    group_size: Int,
    type: DType,
](
    a: NDBuffer[type, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[type, 2],
) raises:
    # Matmul parallelized for the token generation case where matmuls have the
    # following shape:
    # 1xK . KxN

    # Overview:
    # 1. Quantize activations from float to int8.
    # 2. Parallelize on `N`.
    # 3. The outer matmul loop is over `M`, which is usually 1.
    # 4. The inner loop is over `K // block_size_bytes`.
    # 5. Additionally, the inner loop processes `row_batch_count=4` rows of `b`
    #    at a time, unrolled in the inner loop.

    alias block_size = sizeof[Q4sym[group_size, type]]()

    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()
    var k_groups = K // group_size

    if b.dim[1]() % block_size != 0:
        raise Error("unexpected b shape")

    # 1. Quantize incoming activations `a` from float to int8.
    var a_quant_base_ptr = DTypePointer[DType.int8].alloc(M * K)
    var a_scale_base_ptr = DTypePointer[DType.float32].alloc(M * k_groups)

    var a_quant = NDBuffer[DType.int8, 2](a_quant_base_ptr, Index(M, K))
    var a_scale = NDBuffer[DType.float32, 2](
        a_scale_base_ptr, Index(M, k_groups)
    )

    _block_quantize_a[group_size](a, a_quant, a_scale)

    # Set the minimum number of rows processed by each additional thread.
    alias grain_size = 32

    # 2. Parallelize on `N`.
    var num_workers = ceildiv(N, grain_size)

    @__copy_capture(N, a_quant, a_scale)
    @parameter
    @always_inline
    fn task_func(task_id: Int):
        alias row_batch_count = 4

        var start_item = task_id * grain_size
        var end_item = min(N, start_item + grain_size)
        var end_batch_item = align_down(end_item, row_batch_count)

        for n in range(start_item, end_batch_item, row_batch_count):
            # Process rows of `b`.
            _process_rows[group_size, row_batch_count](
                a_quant, a_scale, b, c, n
            )

        for n in range(end_batch_item, end_item):
            _process_rows[group_size, 1](a_quant, a_scale, b, c, n)

    sync_parallelize[task_func](num_workers)

    a_quant_base_ptr.free()
    a_scale_base_ptr.free()


######
# Q4_K
######


struct block_QK_K:
    alias quantized_k = 256


struct block_Q4_K:
    alias group_size = 32
    alias group_count = block_QK_K.quantized_k // Self.group_size

    var base_scale: Float16
    var base_min: Float16
    var q_scales_and_mins: InlineArray[
        UInt8, (2 * block_Q4_K.group_count * 6) // 8
    ]
    # 256 total elements / 8 groups => 32 elements per group.
    var q_bits: InlineArray[UInt8, block_QK_K.quantized_k // 2]


fn scale_min_k4(
    src_ptr: UnsafePointer[block_Q4_K], g: Int
) -> (Float32, Float32):
    if g < 4:
        var q_scale = src_ptr[].q_scales_and_mins[g] & 63
        var q_min = src_ptr[].q_scales_and_mins[g + 4] & 63

        return q_scale.cast[DType.float32](), q_min.cast[DType.float32]()
    else:
        var q_scale_lo = src_ptr[].q_scales_and_mins[g + 4] & 15
        var q_min_lo = src_ptr[].q_scales_and_mins[g + 4] >> 4
        var q_scale_hi = src_ptr[].q_scales_and_mins[g - 4] >> 6
        var q_min_hi = src_ptr[].q_scales_and_mins[g - 0] >> 6
        var q_scale = (q_scale_hi << 4) | q_scale_lo
        var q_min = (q_min_hi << 4) | q_min_lo

        return q_scale.cast[DType.float32](), q_min.cast[DType.float32]()


fn q4_k_dequantize_impl(
    input_tensor: NDBuffer[DType.uint8, 2],
    output_tensor: NDBuffer[DType.float32, 2],
):
    alias group_nelems = block_Q4_K.group_size
    # 2 elements per byte.
    alias group_nbytes = group_nelems // 2
    alias block_nelems = block_QK_K.quantized_k
    alias block_nbytes = sizeof[block_Q4_K]()

    var num_blocks = input_tensor.num_elements() // block_nbytes
    var input_q4_k_ptr = UnsafePointer[block_Q4_K](
        address=int(input_tensor.data.address)
    )
    var output_ptr = UnsafePointer[Float32](
        address=int(output_tensor.data.address)
    )
    for block_idx in range(num_blocks):
        var src_ptr = input_q4_k_ptr + block_idx
        var dst_ptr = output_ptr + (block_idx * block_nelems)

        # d
        var b_scale = src_ptr[].base_scale.cast[DType.float32]()
        # min
        var b_min = src_ptr[].base_min.cast[DType.float32]()

        # Process 2 groups at a time to load 6-bit scales/mins.
        @parameter
        for group_idx in range(0, block_Q4_K.group_count, 2):
            var q_scale: Float32
            var q_min: Float32

            # group_idx: low bits
            q_scale, q_min = scale_min_k4(src_ptr, group_idx)
            var d1 = b_scale * q_scale
            var m1 = b_min * q_min

            # group_idx + 1: high bits
            q_scale, q_min = scale_min_k4(src_ptr, group_idx + 1)
            var d2 = b_scale * q_scale
            var m2 = b_min * q_min

            var q_bits_idx = group_idx * group_nbytes
            var dst_idx = group_idx * group_nelems

            # Dequantize 1st group bits.
            @parameter
            for elem_idx in range(group_nelems):
                dst_ptr[dst_idx + elem_idx] = (
                    d1
                    * (src_ptr[].q_bits[q_bits_idx + elem_idx] & 0xF).cast[
                        DType.float32
                    ]()
                    - m1
                )

            # Dequantize 2nd group bits.
            @parameter
            for elem_idx in range(group_nelems):
                dst_ptr[dst_idx + group_nelems + elem_idx] = (
                    d2
                    * (src_ptr[].q_bits[q_bits_idx + elem_idx] >> 4).cast[
                        DType.float32
                    ]()
                    - m2
                )


######
# Q6_K
######


@always_inline
fn _to_dtype_pointer[
    type: DType
](array: InlineArray[Scalar[type]]) -> DTypePointer[type]:
    return DTypePointer[type](array.unsafe_ptr())


struct block_Q6_K:
    alias group_size = 16
    alias group_count = block_QK_K.quantized_k // Self.group_size

    # Low 4 bits.
    var q_bits_lo: InlineArray[UInt8, block_QK_K.quantized_k // 2]
    # High 2 bits.
    var q_bits_hi: InlineArray[UInt8, block_QK_K.quantized_k // 4]
    # int8 scales.
    var q_scales: InlineArray[Int8, block_Q6_K.group_size]
    # Superblock scale.
    var base_scale: Float16


fn q6_k_dequantize_impl(
    input_tensor: NDBuffer[DType.uint8, 2],
    output_tensor: NDBuffer[DType.float32, 2],
    output_shape: StaticIntTuple[2],
):
    alias group_nelems = block_Q6_K.group_size
    alias block_nelems = block_QK_K.quantized_k
    alias block_nbytes = sizeof[block_Q6_K]()

    var num_blocks = (output_shape[0] * output_shape[1]) // block_nelems
    var input_q6_k_ptr = UnsafePointer[block_Q6_K](
        address=int(input_tensor.data.address)
    )
    var dst_ptr = UnsafePointer[Float32](
        address=int(output_tensor.data.address)
    )
    var dst_idx = 0
    for block_idx in range(num_blocks):
        var src_ptr = input_q6_k_ptr + block_idx

        var d = src_ptr[].base_scale.cast[DType.float32]()

        var ql = src_ptr[].q_bits_lo.unsafe_ptr()
        var qh = src_ptr[].q_bits_hi.unsafe_ptr()
        var sc = src_ptr[].q_scales.unsafe_ptr()

        # Process 8 groups at a time.
        @parameter
        for _ in range(0, block_Q6_K.group_count, 8):

            @parameter
            for l in range(32):
                var sc_idx = l // 16
                var q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)).cast[
                    DType.int8
                ]() - 32
                var q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)).cast[
                    DType.int8
                ]() - 32
                var q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)).cast[
                    DType.int8
                ]() - 32
                var q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)).cast[
                    DType.int8
                ]() - 32

                dst_ptr[l + 0] = (
                    d
                    * sc[sc_idx + 0].cast[DType.float32]()
                    * q1.cast[DType.float32]()
                )
                dst_ptr[l + 32] = (
                    d
                    * sc[sc_idx + 2].cast[DType.float32]()
                    * q2.cast[DType.float32]()
                )
                dst_ptr[l + 64] = (
                    d
                    * sc[sc_idx + 4].cast[DType.float32]()
                    * q3.cast[DType.float32]()
                )
                dst_ptr[l + 96] = (
                    d
                    * sc[sc_idx + 6].cast[DType.float32]()
                    * q4.cast[DType.float32]()
                )

            dst_ptr += 128
            dst_idx += 128
            ql += 64
            qh += 32
            sc += 8
