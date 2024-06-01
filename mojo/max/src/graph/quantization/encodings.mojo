# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Implementations of quantization encodings."""

from tensor import Tensor, TensorShape
from utils import InlineArray

from .quantization_encoding import QuantizationEncoding


struct BFloat16Encoding(QuantizationEncoding):
    """The bfloat16 quantization encoding.

    Like float32, the bfloat16 encoding uses 8 bits to store the exponent
    value, so it has the same numeric range as float32. However, it has just 7
    bits for the mantissa (compared to 23 bits available in float32), so it has
    less precision for the fractional part. This is often a better trade-off
    for ML applications, compared to traditional float16, which has less
    numeric range because it uses only 5 bits to store the exponent (though it
    has better precision with 10 bits for the mantissa)."""

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision input tensor to bfloat16.

        Only supports quantizing from float16 and float32, using a direct
        elementwise cast.

        Args:
            tensor: Full-precision tensor to quantize to bfloat16.

        Returns:
            Quantized bfloat16 tensor.
        """
        if not tensor.num_elements():
            return Tensor[DType.uint8]()

        # Quantize to bfloat16 via elementwise cast.
        quantized = tensor.astype[DType.bfloat16]()

        # Compute bytes buffer shape as the tensor shape with 2 bytes per
        # bfloat16 element in the innermost dimension.
        # Note that this implies the storage is row major.
        tensor_shape = tensor.shape()
        buff_dims = List[Int]()
        for i in range(tensor_shape.rank() - 1):
            buff_dims.append(tensor_shape[i])

        buff_dims.append(2 * tensor_shape[-1])

        return Tensor(
            TensorShape(buff_dims^),
            quantized._steal_ptr().bitcast[DType.uint8](),
        )

    @staticmethod
    fn id() -> String:
        """Identifier for the bfloat16 quantized encoding."""
        return "bfloat16"


struct Float32Encoding(QuantizationEncoding):
    """The float32 quantization encoding.

    This encoding is essentially an identity operation.
    It exists in order to be a default case for code that is generic over
    quantization encoding.
    """

    @staticmethod
    def quantize(_tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Unimplemented quantize method for float32.

        Since float32 is an identity encoding, it shouldn't define a quantize method.
        In particular, float32 values should be used with non-quantized ops,
        which expect dtype float32.
        This is in contrast to quantized ops, which expect dtype uint8 operands.
        So raise an exception here to avoid accidental bugs.
        """
        raise "float32 quantize intentionally not implemented"

    @staticmethod
    fn id() -> String:
        """Identifier for the float32 quantized encoding."""
        return "float32"


@value
struct _BlockQ40:
    """4-bit quantization.

    Constraints:
        The data layout must exactly match `block_q4_0` from ggml-quants.h.
    """

    alias QK4_0 = 32
    """Number of elements per Q4_0 block."""

    var d: Float16
    """Delta."""
    var qs: InlineArray[UInt8, Self.QK4_0 // 2]
    """Nibbles / quants."""

    def __init__(
        inout self,
        d: Float16,
        qs: InlineArray[UInt8, Self.QK4_0 // 2],
    ):
        constrained[sizeof[Self]() == sizeof[Float16]() + (Self.QK4_0 // 2)]()

        self.d = d
        self.qs = qs

    @staticmethod
    fn elements_per_block() -> Int:
        """Returns the number of elements per Q4_0 block."""
        return Self.QK4_0


struct Q4_0Encoding(QuantizationEncoding):
    """The Q4_0 quantization encoding.

    Q4_0 is a block quantization scheme originally designed for
    [GGML](https://ggml.ai) in which each element (number) is reduced to an
    unsigned, fixed-point, 4-bit value. Multiple quantized elements are packed
    together in a block, all using the same float16 scale.

    For example, suppose that we have a block of N = 8 numbers A, B, C, D, E,
    F, G, and H, with associated bits aaaa, bbbb, and so on. Then, within that
    32-bit block of 8 elements, the elements are packed as follows:

    ```
    eeeeaaaa|ffffbbbb|ggggcccc|hhhhdddd
    ```
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to Q4_0.

        Args:
            tensor: Full-precision tensor to quantize to Q4_0.

        Returns:
            Quantized Q4_0 tensor.
        """
        if not tensor.num_elements():
            return Tensor[DType.uint8]()

        alias elems_per_block = _BlockQ40.elements_per_block()
        tensor_shape = tensor.shape()
        cols = tensor_shape[-1]
        if cols % elems_per_block != 0:
            raise "num elements in row must be a multiple of Q4_0 block size"

        # Q4_0 quantizes row-wise, so compute the output shape as the same as
        # the input shape, except with the last dimension packed as _BlockQ40.
        buff_dims = List[Int]()
        for i in range(tensor_shape.rank() - 1):
            buff_dims.append(tensor_shape[i])
        # Compute number of bytes in last block, which is packed.
        buff_dims.append((cols // elems_per_block) * sizeof[_BlockQ40]())

        # Allocate the output buffer and interpret it as an array of _BlockQ40.
        quantized = Tensor[DType.uint8](TensorShape(buff_dims^))
        quantized_ptr = rebind[UnsafePointer[_BlockQ40]](quantized.unsafe_ptr())

        tensor_ptr = tensor.unsafe_ptr()

        # Iterate over all blocks in the tensor.
        for block_idx in range(tensor.num_elements() // elems_per_block):
            # Track max and abs(max) over the block.
            block_abs_max = Float32(0.0)
            block_max = Float32(0.0)

            # Find and set the max over the block.
            for i in range(elems_per_block):
                val = tensor_ptr[block_idx * elems_per_block + i]
                if block_abs_max < abs(val):
                    block_abs_max = abs(val)
                    block_max = val

            # Compute float16 scale and its inverse to scale elems to [-8, 8].
            d = block_max / -8
            inv_d = 1.0 / d if d else Float32(0)

            # Write scale to output buffer.
            quantized_ptr[block_idx].d = d.cast[DType.float16]()

            @parameter
            for elem_idx in range(elems_per_block // 2):
                # x0: first half of block.
                x0 = inv_d * tensor[block_idx * elems_per_block + elem_idx]
                # x1: second half of block.
                x1 = (
                    inv_d
                    * tensor[
                        block_idx * elems_per_block
                        + elems_per_block // 2
                        + elem_idx
                    ]
                )

                # Offset by 8.5 and clamp to [0, 15].
                x0_int4 = min(15, (x0 + 8.5).cast[DType.int8]())
                x1_int4 = min(15, (x1 + 8.5).cast[DType.int8]())

                # Write first half elem to low bits, second half elem to high.
                low_bits = x0_int4.cast[DType.uint8]()
                high_bits = x1_int4.cast[DType.uint8]() << 4
                quantized_ptr[block_idx].qs[elem_idx] = low_bits | high_bits

        return quantized

    @staticmethod
    fn id() -> String:
        """Identifier for the Q4_0 quantized encoding."""
        return "q4_0"


# Note that there is a compile definition in ggml-quants.h that allows setting
# `QK_K=64`, which is useful for models with rows unaligned to 256 bits.
alias QK_K = 256
"""Size of superblock quantized elements, in bytes."""

alias K_SCALE_SIZE = 12
"""Size of superblock scales and mins, in bytes."""


@value
struct _BlockQ4K:
    """4-bit quantization.

    8 blocks of 32 elements each.
    Weights are represented as `x = a * q + b`.
    Effectively 4.5 bits per weight.

    Constraints:
        The data layout must exactly match `block_q4_K` from ggml-quants.h.
    """

    var d: Float16
    """Super-block scale for quantized scales."""

    var dmin: Float16
    """Super-block scale for quantized mins."""

    var scales: InlineArray[UInt8, K_SCALE_SIZE]
    """Scales and mins, quantized with 6 bits."""

    var qs: InlineArray[UInt8, QK_K // 2]
    """4-bit quants."""

    def __init__(
        inout self,
        d: Float16,
        dmin: Float16,
        scales: InlineArray[UInt8, K_SCALE_SIZE],
        qs: InlineArray[UInt8, QK_K // 2],
    ):
        constrained[
            sizeof[Self]() == 2 * sizeof[Float16]() + K_SCALE_SIZE + QK_K // 2
        ]()

        self.d = d
        self.dmin = dmin
        self.scales = scales
        self.qs = qs


struct Q4_KEncoding(QuantizationEncoding):
    """The Q4_K quantization encoding."""

    @staticmethod
    def quantize(_tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to Q4_K.

        The quantize method is not yet implemented.
        However, since Q4_K quantized ops are supported, Q4_KEncoding is still
        provided to allow code to be generic over quantization encoding type.
        """
        raise "quantize not implemented for Q4_KEncoding. Please file an issue!"

    @staticmethod
    fn id() -> String:
        """Identifier for the Q4_K quantized encoding."""
        return "q4_k"


@value
struct _BlockQ6K:
    """6-bit quantization.

    16 blocks of 16 elements each.
    Weights are represented as `x = a * q`.
    Effectively 6.5625 bits per weight.

    Constraints:
        The data layout must exactly match `block_q6_K` from ggml-quants.h.
    """

    var ql: InlineArray[UInt8, QK_K // 2]
    """Quants: lower 4 bits."""

    var qh: InlineArray[UInt8, QK_K // 2]
    """Quants: upper 2 bits."""
    var scales: InlineArray[Int8, QK_K // 16]
    """Scales: quantized with 8 bits."""

    var d: Float16
    """Super-block scale."""

    def __init__(
        inout self,
        ql: InlineArray[UInt8, QK_K // 2],
        qh: InlineArray[UInt8, QK_K // 2],
        scales: InlineArray[Int8, QK_K // 16],
        d: Float16,
    ):
        constrained[
            sizeof[Self]()
            == (2 * (QK_K // 2)) + (QK_K // 16) + sizeof[Float16]()
        ]()

        self.ql = ql
        self.qh = qh
        self.scales = scales
        self.d = d


struct Q6_KEncoding(QuantizationEncoding):
    """The Q6_K quantization encoding."""

    @staticmethod
    def quantize(_tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to Q6_K.

        The quantize method is not yet implemented.
        However, since Q6_K quantized ops are supported, Q6_KEncoding is still
        provided to allow code to be generic over quantization encoding type.
        """
        raise "quantize not implemented for Q6_KEncoding. Please file an issue!"

    @staticmethod
    fn id() -> String:
        """Identifier for the Q6_K quantized encoding."""
        return "q6_k"
