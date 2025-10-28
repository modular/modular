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

"""Custom exceptions for MAX serving infrastructure."""

from __future__ import annotations


class CUDAOOMError(RuntimeError):
    """Custom exception for CUDA out-of-memory errors with helpful guidance."""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format a comprehensive error message with actionable solutions."""
        return """
GPU ran out of memory during model execution.

This typically happens when:
1. The model's runtime memory usage is too large for your GPU's memory
2. The batch size is too large for the available memory
3. The sequence length (max_length) is too large

Suggested solutions:
1. Reduce --device-memory-utilization to a smaller value
2. Reduce batch size with --max-batch-size parameter
3. Reduce sequence length with --max-length parameter
4. Reduce prefill chunk size with --prefill-chunk-size parameter
"""


def detect_and_wrap_cuda_oom(exception: Exception) -> Exception:
    """
    Detect CUDA OOM errors and wrap them in a more helpful exception.

    This function checks if the given exception is a CUDA out-of-memory error
    and wraps it in a CUDAOOMError with helpful guidance if so.

    Args:
        exception: The exception to check

    Returns:
        CUDAOOMError if it's a CUDA OOM, otherwise the original exception
    """
    # Check for the specific CUDA OOM error pattern in ValueError exceptions
    error_message = str(exception)
    if (
        isinstance(exception, ValueError)
        and "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY" in error_message
    ):
        return CUDAOOMError(exception)

    # Return the original exception if it's not a CUDA OOM
    return exception
