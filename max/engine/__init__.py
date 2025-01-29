# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Modular engine provides methods to load and execute AI models."""

from max._engine.core import __version__

from .api import (
    InferenceSession,
    LogLevel,
    Model,
    MojoValue,
    TensorSpec,
    TorchInputSpec,
)
