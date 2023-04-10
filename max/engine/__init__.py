# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from enum import Enum
from pathlib import Path
from sys import version_info

from .api import (
    DType,
    InferenceSession,
    Model,
    ModelKind,
    TensorFlowLoadOptions,
    TensorSpec,
    TFLiteLoadOptions,
    TFSavedModelVersion,
    TorchLoadOptions,
    version_string,
)

__doc__ = (
    "Modular engine provides methods to load and execute saved models from"
    " TensorFlow, PyTorch and ONNX."
)
__version__ = version_string

del api
