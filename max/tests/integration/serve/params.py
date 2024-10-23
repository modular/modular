# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The parameters used when parameterizing a test."""

from llama3 import DeviceSpec


class ModelParams:
    """The model parameters passed via a pytest fixture."""

    def __init__(
        self,
        weight_path: str,
        max_length: int,
        max_new_tokens: int,
        device_spec: DeviceSpec,
        encoding: str,
    ):
        self.weight_path = weight_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device_spec = device_spec
        self.encoding = encoding
