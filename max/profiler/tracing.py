# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tracing Python functions."""

from __future__ import annotations

import functools
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
    if func is None:
        return lambda f: traced(f, message=message, color=color)

    # Default to the function name if message wasn't passed.
    message = message if message is not None else func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Trace(message, color):
            return func(*args, **kwargs)

    return wrapper
