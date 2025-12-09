# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
            with patch.object(
                MemoryEstimator, "estimate_memory_footprint", return_value=0
            ):
                return await func(*args, **kwargs)

        return async_wrapper  # type: ignore
    else:

        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with patch.object(
                MemoryEstimator, "estimate_memory_footprint", return_value=0
            ):
                return func(*args, **kwargs)

        return wrapper
