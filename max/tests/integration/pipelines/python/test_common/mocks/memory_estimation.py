# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TypeVar
from unittest.mock import patch

from max.pipelines.lib import MemoryEstimator
from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_R = TypeVar("_R")


def mock_estimate_memory_footprint(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mock the MemoryEstimator.estimate_memory_footprint method.

    This decorator works with both sync and async functions.
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with (
                patch.object(
                    MemoryEstimator, "estimate_memory_footprint", return_value=0
                ),
                patch.object(
                    MemoryEstimator,
                    "max_supported_sequence_length",
                    return_value=None,
                ),
            ):
                return await func(*args, **kwargs)

        return async_wrapper  # type: ignore
    else:

        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with (
                patch.object(
                    MemoryEstimator, "estimate_memory_footprint", return_value=0
                ),
                patch.object(
                    MemoryEstimator,
                    "max_supported_sequence_length",
                    return_value=None,
                ),
            ):
                return func(*args, **kwargs)

        return wrapper
