# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from enum import Enum
from pathlib import Path
from sys import version_info

from .api import DType
from .api import InferenceSession
from .api import Model
from .api import TensorSpec
from .api import version_string

__doc__ = (
    "Modular engine provides methods to compile and execute saved models from"
    " TensorFlow, PyTorch and ONNX."
)
__version__ = version_string

del api
