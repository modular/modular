# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Implementation of the bfloat16 quantization encoding.

This defines an API for quantization to bfloat16, which is a direct elementwise
cast.
The API is useful because it conforms to the QuantizationEncoding trait.
"""
from max.tensor import Tensor, TensorShape

from .quantization_encoding import QuantizationEncoding


struct BFloat16Encoding(QuantizationEncoding):
    """The bfloat16 quantization encoding."""

    @staticmethod
    def quantize(dequantized: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `dequantized` to bfloat16.

        Only supports quantizing from float16 and float32.

        Args:
            dequantized: Full-precision tensor to quantize to bfloat16.

        Returns:
            Quantized bfloat16 tensor.
        """
        if not dequantized.num_elements():
            return Tensor[DType.uint8]()

        # Quantize to bfloat16 via elementwise cast.
        quantized = dequantized.astype[DType.bfloat16]()

        # Compute bytes buffer shape as the tensor shape with 2 bytes per
        # bfloat16 element in the innermost dimension.
        # Note that this implies the storage is row major.
        tensor_shape = dequantized.shape()
        buff_dims = List[Int]()
        for i in range(tensor_shape.rank() - 1):
            buff_dims.append(tensor_shape[i])

        buff_dims.append(2 * tensor_shape[-1])

        return Tensor(
            TensorShape(buff_dims^),
            quantized._steal_ptr().bitcast[DType.uint8](),
        )

    @staticmethod
    def id() -> String:
        """Identifier for the bfloat16 quantized encoding."""
        return "bfloat16"
