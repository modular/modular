# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar
from unittest.mock import patch

from max.pipelines.lib import MEMORY_ESTIMATOR
from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_R = TypeVar("_R")


def mock_estimate_memory_footprint(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch.object(
            MEMORY_ESTIMATOR, "estimate_memory_footprint", return_value=0
        ):
            return func(*args, **kwargs)

    return wrapper
