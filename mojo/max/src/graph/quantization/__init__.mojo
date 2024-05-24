# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Quantization in MAX graphs.

This includes a generic quantization encoding interface.
Also, this module includes specific encodings conforming to the quantization
encoding interface, such as bfloat16 and Q4_0 encodings.

Ops in the MAX Graph API that use quantization can be found in the
[`ops.quantized_ops`](/max/reference/mojo/graph/ops/quantized_ops) module.
These ops require that you specify the quantization encoding you want to use
with the op. You can use the encodings defined here, such as `BFloat16Encoding`,
or you can define your own encoding by implementing it with the
`QuantizationEncoding` trait.

You also may add a quantized node in your graph with
[`Graph.quantize()`](/max/reference/mojo/graph/graph/Graph#quantize).

"""

from .quantization_encoding import QuantizationEncoding
from .bfloat16_encoding import BFloat16Encoding
from .q4_0_encoding import BlockQ40, Q4_0Encoding
