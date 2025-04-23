# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for loading and ensuring registry works in test suite."""

from functools import wraps

import pytest
from max.pipelines import PIPELINE_REGISTRY


@pytest.fixture(scope="session")
def pipeline_registry():
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


def prepare_registry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        PIPELINE_REGISTRY.reset()
        result = func(*args, **kwargs)

        return result

    return wrapper
