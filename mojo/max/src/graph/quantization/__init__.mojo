# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Quantization in MAX graphs.

This includes a generic quantization encoding interface.
Also, this module includes specific encodings conforming to the quantization
encoding interface, such as bfloat16 and Q4_0 encodings.
"""

from .quantization_encoding import QuantizationEncoding
from .bfloat16_encoding import BFloat16Encoding
from .q4_0_encoding import Q4_0Encoding
