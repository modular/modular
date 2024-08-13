# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Quantization support for MAX Graph."""

import enum


class QuantizationEncoding(enum.Enum):
    """Quantization encodings supported by MAX Graph."""

    Q4_0 = "Q4_0"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
