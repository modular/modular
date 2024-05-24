# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Implements the bfloat16 quantization encoding."""
from tensor import Tensor, TensorShape

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
    def id() -> String:
        """Identifier for the bfloat16 quantized encoding."""
        return "bfloat16"
