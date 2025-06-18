# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from math import ceil, ceildiv
from sys.info import sizeof

from buffer import NDBuffer
from buffer.buffer import prod_dims
from memory import bitcast, memcpy

from utils import IndexList, StaticTuple


@always_inline
fn _to_StaticTuple[
    type: DType, size: Int
](data: SIMD[type, size]) -> StaticTuple[Scalar[type], size]:
    """Convert SIMD to StaticTuple."""

    var res = StaticTuple[Scalar[type], size]()

    @parameter
    for i in range(size):
        res[i] = data[i]
    return res


@always_inline
fn _to_SIMD[
    type: DType, size: Int
](data: StaticTuple[Scalar[type], size]) -> SIMD[type, size]:
    """Convert StaticTuple to SIMD."""
    var res = SIMD[type, size]()

    @parameter
    for i in range(size):
        res[i] = data[i]
    return res


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
    var negative_steps = 1 << (output_bits - 1)

    # Handle the case we only have one type of value in `data`, e.g.
    # data = [12., 12., 12., 12.], then the scale is 1/12.0 and the quantized
    # data is ~[1, 1, 1, 1] + `negative_steps`
    var f32_scale = (max_value / 1.0) if (
        result_range == 0
    ) else result_range / positive_steps

    # TODO: consider clipping values
    var data_rounded = round(data / f32_scale).cast[DType.int8]()

    # each bit pattern in `data_quantized`
    var data_quantized = (data_rounded + negative_steps).cast[DType.uint8]()

    return data_quantized, f32_scale


struct Q4sym[
    group_size: Int,
    float_dtype: DType = DType.float32,
](Defaultable):
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
            group_size.is_power_of_two(), "`group_size` must be a power of 2."
        ]()
        constrained[
            group_size == 8 or group_size == 16 or group_size == 32,
            "Only support some `group_sizes`",
        ]()
        constrained[
            float_dtype.is_floating_point(), "Must be floating point type"
        ]()

    @always_inline
    fn __init__(out self):
        """Construct a default initialized Q4sym."""
        self.scale = StaticTuple[UInt8, 2]()
        self.bits = StaticTuple[UInt8, group_size // 2]()
        self._check_constraints()

    @always_inline
    @implicit
    fn __init__(out self, data: SIMD[float_dtype, group_size]):
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
    fn _decode_bits(mut self) -> SIMD[DType.uint8, group_size]:
        # Extract the lower 4 bits of all bits in the `l_bits` format
        var bits_simd = _to_SIMD[DType.uint8, group_size // 2](self.bits)
        var bits_upper = (bits_simd & 0xF0) >> 4
        var bits_lower = bits_simd & 0x0F
        return rebind[SIMD[DType.uint8, group_size]](
            bits_lower.join(bits_upper)
        )

    @always_inline
    fn decode_scale(mut self) -> Float16:
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
    fn decode_unsigned(mut self) -> SIMD[DType.uint8, group_size]:
        """
        Decode the stored uint4 numbers to uint8.

        Returns:
          The decoded stored numbers as uint8 numbers. These have an implicit
          zero-point of 8.
        """
        # Obtain the unsigned quantized values, these have a zp of 8
        return self._decode_bits()

    @always_inline
    fn decode_signed(mut self) -> SIMD[DType.int8, group_size]:
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
    fn decode_fully(mut self) -> SIMD[float_dtype, group_size]:
        """
        Decode the stored numbers into floating point representation.

        Returns:
          The decoded numbers.
        """
        # Obtain the fully dequantized values
        var signed_result = self.decode_signed()
        var scale_decoded = self.decode_scale().cast[float_dtype]()
        var answer = (
            scale_decoded.cast[float_dtype]()
            * signed_result.cast[float_dtype]()
        )
        return answer

    # TODO: support other axis of quantization, right now assume inner-most dim.
    # TODO: support axis which not divisible by group_size
    @staticmethod
    fn quantize_and_write_to_tensor[
        rank: Int,
    ](
        input_tensor: NDBuffer[float_dtype, rank],
        output_tensor: NDBuffer[DType.uint8, rank],
        input_shape: IndexList[rank],
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

        var blob_output_ptr = output_tensor.data
        var base_block_ptr = UnsafePointer(blob_output_ptr).bitcast[
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
                var loaded_group = input_tensor.data.load[width=group_size](
                    flat_index_input
                )

                var flat_index_output = output_inner_stride * i + j
                var output_ptr = base_block_ptr + flat_index_output

                # TODO: use the memory more directly instead of memcpy
                var encoded_data = Q4sym[group_size, float_dtype](loaded_group)
                var src_ptr = UnsafePointer(to=encoded_data).address_space_cast[
                    output_ptr.address_space
                ]()
                memcpy(output_ptr, src_ptr, 1)
                _ = encoded_data^

    @staticmethod
    fn dequantize_and_write_to_tensor[
        rank: Int, //
    ](
        input_tensor: NDBuffer[DType.uint8, rank],
        output_tensor: NDBuffer[float_dtype, rank],
        output_shape: IndexList[rank],
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

        var uint8_input_ptr = input_tensor.data
        var base_block_ptr = UnsafePointer(uint8_input_ptr).bitcast[
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
                    UnsafePointer(to=encoded),
                    base_block_ptr + flat_index_input,
                    1,
                )

                var flat_index_output = output_inner_dim * i + j * group_size
                output_tensor.data.store(
                    flat_index_output,
                    encoded.decode_fully(),
                )


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
    output_tensor: NDBuffer[mut=True, DType.float32, 2],
):
    alias group_nelems = block_Q4_K.group_size
    # 2 elements per byte.
    alias group_nbytes = group_nelems // 2
    alias block_nelems = block_QK_K.quantized_k
    alias block_nbytes = sizeof[block_Q4_K]()

    var num_blocks = input_tensor.num_elements() // block_nbytes
    var input_q4_k_ptr = input_tensor.data.bitcast[block_Q4_K]()
    var output_ptr = output_tensor.data.bitcast[Float32]()
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
    output_tensor: NDBuffer[mut=True, DType.float32, 2],
    output_shape: IndexList[2],
):
    alias group_nelems = block_Q6_K.group_size
    alias block_nelems = block_QK_K.quantized_k
    alias block_nbytes = sizeof[block_Q6_K]()

    var num_blocks = (output_shape[0] * output_shape[1]) // block_nelems
    var input_q6_k_ptr = input_tensor.data.bitcast[block_Q6_K]()
    var dst_ptr = output_tensor.data.bitcast[Float32]()
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
