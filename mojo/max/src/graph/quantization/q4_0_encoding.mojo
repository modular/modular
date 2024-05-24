# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Describes the quantization encoding for Q4_0 matching that in GGML.

Q4_0 is a block quantization scheme where elements are 4-bit nibbles.
Then each block consists of N elements and a float16 scale.

Suppose that we have a block of N = 8 numbers A, B, C, D, E, F, G, and H, with
associated bits aaaa, bbbb, and so on.
Then, within that block of 8 elements, the elements are packed as follows:

    eeeeaaaa|ffffbbbb|ggggcccc|hhhhdddd
"""
from buffer import Buffer
from tensor import Tensor, TensorShape
from utils import InlineArray

from .quantization_encoding import QuantizationEncoding


@value
struct BlockQ40:
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
    """The Q4_0 quantization encoding."""

    @staticmethod
    def quantize(dequantized: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `dequantized` to Q4_0.

        Args:
            dequantized: Full-precision tensor to quantize to Q4_0.

        Returns:
            Quantized Q4_0 tensor.
        """
        if not dequantized.num_elements():
            return Tensor[DType.uint8]()

        alias elems_per_block = BlockQ40.elements_per_block()
        tensor_shape = dequantized.shape()
        cols = tensor_shape[-1]
        if cols % elems_per_block != 0:
            raise "num elements in row must be a multiple of Q4_0 block size"

        # Q4_0 quantizes row-wise, so compute the output shape as the same as
        # the input shape, except with the last dimension packed as BlockQ40.
        buff_dims = List[Int]()
        for i in range(tensor_shape.rank() - 1):
            buff_dims.append(tensor_shape[i])
        # Compute number of bytes in last block, which is packed.
        buff_dims.append((cols // elems_per_block) * sizeof[BlockQ40]())

        # Allocate the output buffer and interpret it as an array of BlockQ40.
        quantized = Tensor[DType.uint8](TensorShape(buff_dims^))
        quantized_ptr = rebind[UnsafePointer[BlockQ40]](quantized.unsafe_ptr())

        dequantized_ptr = dequantized.unsafe_ptr()

        # Iterate over all blocks in the tensor.
        for block_idx in range(dequantized.num_elements() // elems_per_block):
            # Track max and abs(max) over the block.
            block_abs_max = Float32(0.0)
            block_max = Float32(0.0)

            # Find and set the max over the block.
            for i in range(elems_per_block):
                val = dequantized_ptr[block_idx * elems_per_block + i]
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
                x0 = inv_d * dequantized[block_idx * elems_per_block + elem_idx]
                # x1: second half of block.
                x1 = (
                    inv_d
                    * dequantized[
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
    def id() -> String:
        """Identifier for the Q4_0 quantized encoding."""
        return "Q4_0"
