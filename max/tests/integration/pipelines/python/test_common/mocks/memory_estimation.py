# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from functools import wraps
from unittest.mock import patch

from max.pipelines.lib import MEMORY_ESTIMATOR


def mock_estimate_memory_footprint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch.object(
            MEMORY_ESTIMATOR, "estimate_memory_footprint", return_value=0
        ):
            return func(*args, **kwargs)

    return wrapper
