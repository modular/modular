# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for loading and ensuring registry works in test suite."""

from collections.abc import Generator
from functools import wraps
from typing import Callable, TypeVar

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib.registry import PipelineRegistry
from typing_extensions import ParamSpec


@pytest.fixture(scope="session")
def pipeline_registry() -> Generator[PipelineRegistry]:
    """
    A pytest fixture that manages the registry of production models for testing purposes.

    This fixture performs the following action:
    - If the registry is empty, it registers all production models before yielding control to the test.

    Usage:
    def test_function(manage_production_models):
        # Test code here

    Note:
    - This fixture is particularly useful for tests that depend on or interact with the model registry.
    - It ensures that all production models are registered before the test runs, providing a consistent testing environment.
    - The registry state is maintained across tests, allowing for potential cumulative effects or shared state between tests.
    """
    yield PIPELINE_REGISTRY


_P = ParamSpec("_P")
_R = TypeVar("_R")


def prepare_registry(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        PIPELINE_REGISTRY.reset()
        result = func(*args, **kwargs)

        return result

    return wrapper
