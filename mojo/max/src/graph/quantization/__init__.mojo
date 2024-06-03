# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""APIs to quantize graph tensors.

This package includes a generic quantization encoding interface and some
quantization encodings that conform to it, such as bfloat16 and Q4_0 encodings.

The main interface for defining a new quantized type is
`QuantizationEncoding.quantize()`. This takes a full-precision tensor
represented as float32 and quantizes it according to the encoding. The
resulting quantized tensor is represented as a bytes tensor. For that reason,
the `QuantizationEncoding` must know how to translate between the tensor shape
and its corresponding quantized buffer shape.

For example, this code quantizes a tensor with the Q4_0 encoding:

```mojo
from tensor import Tensor
from max.graph.quantization import Q4_0Encoding

var tensor: Tensor[DType.float32]
# Initialize `tensor`.

# Quantize using the `Q4_0` quantization encoding.
var quantized: Tensor[DType.uint8] = Q4_0Encoding.quantize(tensor)

# Now `quantized` is packed according to the `Q4_0` encoding and can be
# used to create graph constants and serialized to disk.
```

Specific ops in the MAX Graph API that use quantization can be found in the
[`ops.quantized_ops`](/max/reference/mojo/graph/ops/quantized_ops) module. You
can also add a quantized node in your graph with
[`Graph.quantize()`](/max/reference/mojo/graph/graph/Graph#quantize).

To save the quantized tensors to disk, use
[`graph.checkpoint.save()`](/max/reference/mojo/graph/checkpoint/save_load/save).
"""

from .quantization_encoding import QuantizationEncoding
from .encodings import (
    _BlockQ40 as BlockQ40,
    BFloat16Encoding,
    Float32Encoding,
    Q4_0Encoding,
    Q4_KEncoding,
    Q6_KEncoding,
)
