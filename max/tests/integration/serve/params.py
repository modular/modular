# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The parameters used when parameterizing a test."""

from max.driver import Device


class ModelParams:
    """The model parameters passed via a pytest fixture."""

    def __init__(
        self,
        weight_path: str,
        max_length: int,
        max_new_tokens: int,
        device: Device,
        encoding: str,
    ):
        self.weight_path = weight_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.encoding = encoding
