# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Modular engine provides methods to load and execute AI models."""

from .api import (
    DType,
    InferenceSession,
    Model,
    TensorSpec,
    TorchInputSpec,
)
from max._engine.core import __version__
