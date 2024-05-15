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
from max.tensor import Tensor

from .quantization_encoding import QuantizationEncoding


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
        # TODO(FFE-317): Implement Q4_0 quantization.
        return abort[Tensor[DType.uint8]](
            "Q4_0 quantization isn't implemented yet"
        )

    @staticmethod
    def id() -> String:
        """Identifier for the Q4_0 quantized encoding."""
        return "Q4_0"
