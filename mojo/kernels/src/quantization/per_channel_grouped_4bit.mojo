# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import roundeven, max, abs, ceil, floor, div_ceil
from sys.info import sizeof, alignof
from memory.unsafe import bitcast
from memory.buffer import NDBuffer, prod_dims
from math import is_power_of_2


@always_inline
fn _to_StaticTuple[
    type: DType, size: Int
](data: SIMD[type, size]) -> StaticTuple[size, SIMD[type, 1]]:
    """Convert SIMD to StaticTuple."""

    @parameter
    if size == 1:
        return StaticTuple[size](data[0])
    elif size == 2:
        return StaticTuple[size](data[0], data[1])
    elif size == 4:
        return StaticTuple[size](data[0], data[1], data[2], data[3])
    elif size == 8:
        return StaticTuple[size](
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
        return StaticTuple[size](
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
        return StaticTuple[size](
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
](data: StaticTuple[size, SIMD[type, 1]]) -> SIMD[type, size]:
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
    let max_value = data.reduce_max()
    let min_value = data.reduce_min()
    let result_range = max(max_value, -min_value)

    # quantize as if we were signed, and convert back to unsigned later
    let positive_steps = (1 << (output_bits - 1)) - 1
    let negative_steps = (1 << (output_bits - 1))

    # Handle the case we only have one type of value in `data`, e.g.
    # data = [12., 12., 12., 12.], then the scale is 1/12.0 and the quantized
    # data is ~[1, 1, 1, 1] + `negative_steps`
    let f32_scale = (max_value / 1.0) if (
        result_range == 0
    ) else result_range / positive_steps

    # TODO: consider clipping values
    let data_rounded = roundeven(data / f32_scale).cast[DType.int8]()

    # each bit pattern in `data_quantized`
    let data_quantized = (data_rounded + negative_steps).cast[DType.uint8]()

    return data_quantized, f32_scale


struct Q4sym[group_size: Int, float_dtype: DType = DType.float32]:
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

        Then `bits` is 4 bytes and has a layout like:
           aaaaeeee|bbbbffff|ccccgggg|ddddhhhh


    To uncompress to floating point, take the decoded uint4 value, subtract
    the implicit zero-point of 2^4=8, and muliply by the scale factor.

    Parameters:
        group_size: The number of encoded numbers stored in this struct.
        float_dtype: The floating point dtype this struct works with.
    """

    var scale: StaticTuple[2, UInt8]
    """The FP16 scale of the group, stored as individual bytes."""

    var bits: StaticTuple[group_size // 2, UInt8]
    """The bits of the encoded uint4 numbers."""

    @staticmethod
    @always_inline
    fn _check_constraints():
        # TODO
        constrained[
            is_power_of_2(group_size), "`group_size` must be a power of 2."
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
        self.scale = StaticTuple[2, UInt8]()
        self.bits = StaticTuple[group_size // 2, UInt8]()
        self._check_constraints()

    @always_inline
    fn __init__(inout self, data: SIMD[float_dtype, group_size]):
        """
        Construct an encoded Q4sym from data.

        Args:
            data: The floating point data to encode and store.
        """
        let quantization_tuple = calculate_symmetric_vector[
            float_dtype, group_size, 4
        ](data)
        let qdata = quantization_tuple.get[0, SIMD[DType.uint8, group_size]]()
        let f_scale = quantization_tuple.get[1, SIMD[float_dtype, 1]]()

        # TODO: add warning if we overflow/underflow
        let f16_scale = f_scale.cast[DType.float16]()
        self.scale = _to_StaticTuple(bitcast[DType.uint8, 2](f16_scale))
        self.bits = _to_StaticTuple(self._encode_bits(qdata))
        self._check_constraints()

    @staticmethod
    @always_inline
    fn _encode_bits(
        qdata: SIMD[DType.uint8, group_size]
    ) -> SIMD[DType.uint8, group_size // 2]:
        let lower_elements = qdata.slice[group_size // 2]() << 4
        let upper_elements = qdata.slice[group_size // 2](group_size // 2)
        return lower_elements | upper_elements

    @always_inline
    fn _decode_bits(inout self) -> SIMD[DType.uint8, group_size]:
        # Extract the lower 4 bits of all bits in the `l_bits` format
        let bits_simd = _to_SIMD[DType.uint8, group_size // 2](self.bits)
        let bits_upper = (bits_simd & 0xF0) >> 4
        let bits_lower = (bits_simd & 0x0F)
        return rebind[SIMD[DType.uint8, group_size]](
            bits_upper.join(bits_lower)
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
        let scale_bytes: SIMD[DType.uint8, 2] = _to_SIMD[DType.uint8, 2](
            self.scale
        )

        # NOTE: this may break on different endian systems...
        var upcast_bytes: SIMD[DType.uint16, 2] = scale_bytes.cast[
            DType.uint16
        ]()
        upcast_bytes[1] = upcast_bytes[1] << 8
        let final_result = upcast_bytes.reduce_add()
        let scale_decoded = bitcast[DType.float16, 1](final_result)
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
        let decoded_result = self.decode_unsigned()
        return decoded_result.cast[DType.int8]() - 8

    @always_inline
    fn decode_fully(inout self) -> SIMD[float_dtype, group_size]:
        """
        Decode the stored numbers into floating point representation.

        Returns:
          The decoded numbers.
        """
        # Obtain the fully dequantized values
        let signed_result = self.decode_signed()
        let scale_decoded = self.decode_scale().cast[float_dtype]()
        let answer = scale_decoded.cast[float_dtype]() * signed_result.cast[
            float_dtype
        ]()
        return answer

    # TODO: support other axis of quantization, right now assume inner-most dim.
    # TODO: support axis which not divisible by group_size
    @staticmethod
    fn quantize_and_write_to_tensor[
        rank: Int,
    ](
        input_tensor: NDBuffer[
            rank,
            DimList.create_unknown[rank](),
            float_dtype,
        ],
        output_tensor: NDBuffer[
            rank, DimList.create_unknown[rank](), DType.uint8
        ],
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
        let size_of_block = sizeof[Q4sym[group_size, float_dtype]]()
        debug_assert(
            input_shape[rank - 1] % group_size == 0,
            "Only support fully divisible dimensions right now.",
        )

        let blob_output_ptr = output_tensor.data.address
        let base_block_ptr = bitcast[Q4sym[group_size, float_dtype]](
            blob_output_ptr
        )

        # as we support only inner-most dim, treat like rank-2 tensor
        let outer_stride = prod_dims[0, rank - 1](input_tensor)
        let input_inner_stride = input_shape[rank - 1]
        let output_inner_stride = div_ceil(input_shape[rank - 1], group_size)

        # TODO: vectorize parallelize, blah blah blah
        for i in range(outer_stride):
            for j in range(output_inner_stride):
                let flat_index_input = input_inner_stride * i + j * group_size
                let loaded_group = input_tensor.data.simd_load[group_size](
                    flat_index_input
                )

                let flat_index_output = output_inner_stride * i + j
                let output_ptr = base_block_ptr + flat_index_output

                # TODO: use the memory more directly instead of memcpy
                var encoded_data = Q4sym[group_size, float_dtype](loaded_group)
                let src_ptr = Pointer.address_of(
                    encoded_data
                ).address_space_cast[output_ptr.address_space]()
                memcpy(output_ptr, src_ptr, 1)

    @staticmethod
    fn dequantize_and_write_to_tensor[
        rank: Int,
    ](
        input_tensor: NDBuffer[
            rank, DimList.create_unknown[rank](), DType.uint8
        ],
        output_tensor: NDBuffer[
            rank,
            DimList.create_unknown[rank](),
            float_dtype,
        ],
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

        let uint8_input_ptr = input_tensor.data.address
        let base_block_ptr = bitcast[Q4sym[group_size, float_dtype]](
            uint8_input_ptr
        )

        # as we support only inner-most dim, treat like rank-2 tensor
        let output_inner_dim = output_shape[rank - 1]
        let outer_dim = prod_dims[0, rank - 1](output_tensor)

        # Note: this is calculated assuming a pointer of Q5Sym's
        let input_inner_dim = div_ceil(output_inner_dim, group_size)

        # TODO: vectorize parallelize, blah blah blah
        for i in range(outer_dim):
            for j in range(input_inner_dim):
                let flat_index_input = input_inner_dim * i + j
                var encoded = Q4sym[group_size, float_dtype]()
                memcpy(
                    Pointer.address_of(encoded),
                    base_block_ptr + flat_index_input,
                    1,
                )

                let flat_index_output = output_inner_dim * i + j * group_size
                output_tensor.data.simd_store[group_size](
                    flat_index_output, encoded.decode_fully()
                )


from math import min, align_down
from Neon import _neon_dotprod
from VNNI import dot_i8_to_i32_saturated_x86
from algorithm import sync_parallelize
from utils.index import Index
from sys.info import has_avx2, has_neon_int8_dotprod


fn _block_quantize_a[
    group_size: Int,
    type: DType,
    scale_type: DType,
](
    a: NDBuffer[2, DimList.create_unknown[2](), type],
    a_quant: NDBuffer[2, DimList.create_unknown[2](), DType.int8],
    a_scale: NDBuffer[2, DimList.create_unknown[2](), scale_type],
):
    let M = a.dim[0]()
    let K = a.dim[1]()

    var a_ptr = a.data
    var a_quant_ptr = a_quant.data
    var a_scale_ptr = a_scale.data

    # Dynamically quantize the input in blocks of 32 bytes. Uses symmetric
    # quantization.
    for m in range(M):
        for k in range(0, K, group_size):
            let fp_data = a_ptr.simd_load[group_size]()
            let max_value = abs(fp_data).reduce_max()
            let scale = max_value / 127.0
            let multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

            let quant_data = roundeven(fp_data * multiplier).cast[DType.int8]()
            a_quant_ptr.simd_store(quant_data)
            a_scale_ptr.simd_store(scale.cast[scale_type]())

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
        let b_s8 = b - 8

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

        @unroll
        for idx in range(4):
            c_local += (
                a.slice[simd_width](idx * simd_width).cast[DType.int32]()
                * b_s8.slice[simd_width](idx * simd_width).cast[DType.int32]()
            )
        return c_local

    var c = SIMD[DType.int32, simd_width](0)

    @unroll
    for idx in range(0, group_size, simd_width * 4):
        let a_slice = a.slice[simd_width * 4](idx)
        let b_slice = b.slice[simd_width * 4](idx)
        c = dotprod(a_slice, b_slice, c)

    return c


@always_inline
fn _process_rows[
    group_size: Int,
    row_count: Int,
    type: DType,
](
    a_quant: NDBuffer[2, DimList.create_unknown[2](), DType.int8],
    a_scale: NDBuffer[2, DimList.create_unknown[2](), DType.float32],
    b: NDBuffer[2, DimList.create_unknown[2](), DType.uint8],
    c: NDBuffer[2, DimList.create_unknown[2](), type],
    m: Int,
):
    alias block_size = sizeof[Q4sym[group_size, type]]()
    alias simd_width = min(simdwidthof[DType.int32](), group_size // 4)

    let N = b.dim[0]()
    let K = a_quant.dim[1]()
    let k_groups = K // group_size

    let accum_fp_tile = NDBuffer[
        2, DimList(row_count, simd_width), type
    ].stack_allocation()

    for n in range(N):
        var a_quant_ptr = a_quant._offset(Index(m, 0))
        var a_scale_ptr = a_scale._offset(Index(m, 0))
        var b_ptr = b._offset(Index(n, 0))

        @unroll
        for row in range(row_count):
            accum_fp_tile.simd_store(Index(row, 0), SIMD[type, simd_width](0))

        for k in range(0, k_groups):
            let b_data_i4 = b_ptr.offset(2).simd_load[group_size // 2]()
            let b_scale = bitcast[DType.float16, 1](
                b_ptr.offset(0).simd_load[2]()
            ).cast[type]()
            let b_data_i8_lo = ((b_data_i4 >> 4)).cast[DType.int8]()
            let b_data_i8_hi = ((b_data_i4 & 15)).cast[DType.int8]()
            let b_data_i8 = b_data_i8_lo.join(b_data_i8_hi)

            @unroll
            for row in range(row_count):
                let a_data_i8 = a_quant_ptr.offset(row * K).simd_load[
                    group_size
                ]()
                let a_scale = a_scale_ptr.offset(row * k_groups)[0].cast[type]()

                let c_data_i32 = _matmul_int4_dotprod[simd_width, group_size](
                    a_data_i8, rebind[SIMD[DType.int8, group_size]](b_data_i8)
                )
                var accum_fp = accum_fp_tile.simd_load[simd_width](
                    Index(row, 0)
                )
                accum_fp += c_data_i32.cast[type]() * a_scale * b_scale
                accum_fp_tile.simd_store(Index(row, 0), accum_fp)

            a_quant_ptr = a_quant_ptr.offset(group_size)
            a_scale_ptr = a_scale_ptr.offset(1)
            b_ptr = b_ptr.offset(block_size)

        @unroll
        for row in range(row_count):
            let accum_fp = accum_fp_tile.simd_load[simd_width](Index(row, 0))
            c.simd_store(Index(row + m, n), accum_fp.reduce_add())


fn matmul_int4[
    group_size: Int,
    type: DType,
](
    a: NDBuffer[2, DimList.create_unknown[2](), type],
    b: NDBuffer[2, DimList.create_unknown[2](), DType.uint8],
    c: NDBuffer[2, DimList.create_unknown[2](), type],
    out_chain: OutputChainPtr,
):
    alias block_size = sizeof[Q4sym[group_size, type]]()

    let M = a.dim[0]()
    let N = b.dim[0]()
    let K = a.dim[1]()
    let k_groups = K // group_size

    if b.dim[1]() % block_size != 0:
        return out_chain._mark_error_old("unexpected b shape")

    let a_quant_base_ptr = DTypePointer[DType.int8].alloc(M * K)
    let a_scale_base_ptr = DTypePointer[DType.float32].alloc(M * k_groups)

    let a_quant = NDBuffer[2, DimList.create_unknown[2](), DType.int8](
        a_quant_base_ptr, Index(M, K)
    )
    let a_scale = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        a_scale_base_ptr, Index(M, k_groups)
    )

    _block_quantize_a[group_size](a, a_quant, a_scale)

    alias grain_size = 8
    let num_workers = div_ceil(M, grain_size)

    @parameter
    @always_inline
    fn task_func(task_id: Int):
        alias row_batch_count = 4

        let start_item = task_id * grain_size
        let end_item = min(M, start_item + grain_size)
        let end_batch_item = align_down(end_item, row_batch_count)

        for m in range(start_item, end_batch_item, row_batch_count):
            _process_rows[group_size, row_batch_count](
                a_quant, a_scale, b, c, m
            )

        for m in range(end_batch_item, end_item):
            _process_rows[group_size, 1](a_quant, a_scale, b, c, m)

    sync_parallelize[task_func](out_chain, num_workers)

    a_quant_base_ptr.free()
    a_scale_base_ptr.free()
