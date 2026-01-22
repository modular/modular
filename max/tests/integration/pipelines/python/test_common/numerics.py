# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Utilities for controlling numerical precision and numerical operations"""

import functools
import threading
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import ParamSpec

# Add a lock for thread safety
_tf32_lock = threading.RLock()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def pytorch_disable_tf32_dtype(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Thread-safe decorator which disables TF32 for PyTorch code.

    PyTorch uses the TensorFloat32 precision by default on modern NVIDIA GPUs.
    MAX uses Float32, and this difference can mask real numerical issues on
    our comparison tests.

    See: https://docs.pytorch.org/docs/stable/notes/cuda.html
    This decorator disables TF32 precision for the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _tf32_lock:
            # Store original flag values
            original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            original_cudnn_tf32 = torch.backends.cudnn.allow_tf32

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original flag values
                torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
                torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

    return wrapper


def log_softmax(
    x: npt.NDArray[np.floating], axis: int = -1
) -> npt.NDArray[np.floating]:
    """Compute the logarithm of the softmax function.

    This implementation uses the identity log(softmax(x)) = x - log(sum(exp(x)))
    with numerical stability improvements to prevent overflow/underflow.

    Args:
        x: Input array
        axis: Axis to compute values along

    Returns:
        Array with same shape as x, representing log(softmax(x))
    """
    # Subtract max value for numerical stability (prevents exp overflow)
    x_max = np.amax(x, axis=axis, keepdims=True)

    # Compute exp(x - x_max) which is now safe from overflow
    shifted_x = x - x_max
    exp_shifted = np.exp(shifted_x)

    # Suppress -inf warnings from log(0)
    # This can happen when input contains extreme negative values (-inf),
    # which become 0 after exp() operation
    with np.errstate(divide="ignore"):
        sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
        log_sum_exp = np.log(sum_exp)

    # Final result: x - x_max - log(sum(exp(x - x_max)))
    # This is mathematically equivalent to log(softmax(x))
    return shifted_x - log_sum_exp
