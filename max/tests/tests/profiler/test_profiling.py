# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import unittest.mock

import pytest
from max.profiler import Trace, traced


def test_profiling() -> None:
    """Tests that profiling functions do not error."""
    with Trace("foo"):
        pass

    @traced(message="baz", color="red")
    def foo() -> None:
        # The span is named "baz".
        pass

    @traced
    def bar() -> None:
        # The span is named "bar".
        pass

    foo()
    bar()

    Trace("I'm here").mark()


def test_profiling_disabled() -> None:
    with unittest.mock.patch(
        "max.profiler.tracing.is_profiling_enabled", return_value=False
    ) as m:
        test_profiling()


@pytest.mark.asyncio
async def test_async_profiling() -> None:
    """Tests that profiling async functions doesn't error."""

    async def bar() -> None:
        pass

    @traced(message="baz", color="red")
    async def foo() -> None:
        await bar()

    with Trace("potato"):
        await foo()
