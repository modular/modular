# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Modular engine provides methods to load and execute AI models."""

from max._core.engine import __version__  # type: ignore

from .api import (
    InferenceSession,
    LogLevel,
    Model,
    MojoValue,
    TensorSpec,
    TorchInputSpec,
)
