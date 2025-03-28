# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Quantization support for MAX Graph."""

import enum
from dataclasses import dataclass


# Can't put this directly on the enum because it then becomes unhashable.
@dataclass(frozen=True)
class BlockParameters:
    elements_per_block: int
    block_size: int


# TODO: BlockParameters should be integrated into this class
@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for specifying quantization parameters that affect inference."""

    quant_method: str
    bits: int
    group_size: int
    desc_act: bool = False
    sym: bool = False


class QuantizationEncoding(enum.Enum):
    """Quantization encodings supported by MAX Graph."""

    Q4_0 = "Q4_0"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    GPTQ = "GPTQ"

    @property
    def block_parameters(self) -> BlockParameters:
        return _BLOCK_PARAMETERS[self]

    @property
    def elements_per_block(self) -> int:
        """Number of elements per block.

        All quantization types currently supported by MAX Graph are
        block-based: groups of a fixed number of elements are formed, and each
        group is quantized together into a fixed-size output block.  This value
        is the number of elements gathered into a block.
        """
        return self.block_parameters.elements_per_block

    @property
    def block_size(self) -> int:
        """Number of bytes in encoded representation of block.

        All quantization types currently supported by MAX Graph are
        block-based: groups of a fixed number of elements are formed, and each
        group is quantized together into a fixed-size output block.  This value
        is the number of bytes resulting after encoding a single block.
        """
        return self.block_parameters.block_size

    @property
    def name(self) -> str:
        return self.value.lower()

    @property
    def is_gguf(self) -> bool:
        return self in [
            QuantizationEncoding.Q4_0,
            QuantizationEncoding.Q4_K,
            QuantizationEncoding.Q5_K,
            QuantizationEncoding.Q6_K,
        ]


_BLOCK_PARAMETERS: dict[QuantizationEncoding, BlockParameters] = {
    # Block: d, q (4 bits)
    QuantizationEncoding.Q4_0: BlockParameters(32, 2 + 32 // 2),
    # Block: d, dmin, scales, q (4 bits)
    QuantizationEncoding.Q4_K: BlockParameters(256, 2 + 2 + 12 + 256 // 2),
    # Block: d, dmin, scales, qh (4 bits), qs (1 bit)
    QuantizationEncoding.Q5_K: BlockParameters(
        256, 2 + 2 + 12 + 256 // 2 + 256 // 8
    ),
    # Block: ql, qh, scales, d
    QuantizationEncoding.Q6_K: BlockParameters(
        256, 256 // 2 + 256 // 4 + 256 // 16 + 2
    ),
}
