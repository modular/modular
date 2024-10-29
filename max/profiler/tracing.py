# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tracing Python functions."""

from __future__ import annotations

import functools
import inspect
import os
from typing import Callable

from max._profiler import Trace


def traced(
    func: Callable | None = None,
    *,
    message: str | None = None,
    color: str = "blue",
) -> Callable:
    """Decorator for creating a profiling span for `func`.

    Args:
        func: function to profile.
        message: name of the profiling span; defaults to function name if None.
        color: color of the profilng span.

    Returns:
        Callable that is `func` wrapped in a trace object.

    Examples:
        @traced(message="baz", color="red")
        def foo() -> None:
            # The span is named "baz".
            pass

        @traced
        def bar() -> None:
            # The span is named "bar".
            pass
    """
    if not is_profiling_enabled():
        return func if func is not None else lambda f: func  # type: ignore

    if func is None:
        return lambda f: traced(f, message=message, color=color)

    # Default to the function name if message wasn't passed.
    message = message if message is not None else func.__name__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with Trace(message, color):
                return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Trace(message, color):
                return func(*args, **kwargs)

    return wrapper


def is_profiling_enabled() -> bool:
    """Returns true if profiling is enabled via `MODULAR_ENABLE_PROFILING = 1`
    """
    enable_profiling: str | None = os.getenv("MODULAR_ENABLE_PROFILING")
    truthy_values: list[str] = ["1", "t", "true", "yes", "y"]
    return (
        enable_profiling is not None
        and enable_profiling.lower() in truthy_values
    )
