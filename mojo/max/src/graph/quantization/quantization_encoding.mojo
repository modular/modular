# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Quantization module defining the interface for quantized types.

The main interface for defining a new quantized type is
`QuantizationEncoding.quantize`.
This takes a full-precision tensor represented as float32 and quantizes it
according to the encoding.
The resulting quantized tensor is represented as a bytes tensor.
For that reason, the `QuantizationEncoding` must know how to translate between
the tensor shape and its corresponding quantized buffer shape.

For example consider the following:

```mojo
from tensor import Tensor
from max.graph.quantization import Q4_0Encoding

var tensor: Tensor[DType.float32]
# Initialize `tensor`.

# Quantize using the `Q4_0` quantization encoding.
var quantized: Tensor[DType.uint8] = Q4_0Encoding.quantize(tensor)

# Now `quantized`'s storage is packed according to the `Q4_0` encoding.
# `quantized` can be used to create graph constants and serialized to disk.
```
"""

from tensor import Tensor


trait QuantizationEncoding:
    """Describes the encoding for a data type that can be quantized.

    The `QuantizationEncoding` trait implicitly knows the relationship between
    the buffer and tensor layouts of the quantized encoding for the type.
    So in particular, the quantize API takes in a tensor with a logical shape
    in terms of its elements.
    Then it returns a uint8 tensor with a different shape that instead
    describes the shape of the bytes storage buffer after applying the
    quantization encoding.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor `tensor` to the quantized
        type associated with this `QuantizationEncoding` instance.

        Args:
            tensor: Full-precision tensor to quantize.

        Returns:
            A `Tensor` quantized to the quantized storage format of this
            `QuantizationEncoding` instance.
        """
        ...

    @staticmethod
    fn id() -> String:
        """Returns a unique string identifier for this quantization encoding."""
        ...
