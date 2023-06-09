# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .api import (
    DType,
    InferenceSession,
    Model,
    TensorSpec,
    TorchLoadOptions,
    version_string,
)

__doc__ = (
    "Modular engine provides methods to load and execute saved models from"
    " TensorFlow, PyTorch and ONNX."
)
__version__ = version_string

del api
