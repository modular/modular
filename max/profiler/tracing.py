# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tracing Python functions."""

from __future__ import annotations

import functools
import inspect
from types import TracebackType
from typing import Callable, TypeVar, overload

from max._core.profiler import Trace, is_profiling_enabled

_FuncType = TypeVar("_FuncType", bound=Callable)

# For the list of valid colors, take a look at the struct `Color` in:
# `open-source/max/mojo/stdlib/stdlib/gpu/host/_tracing.mojo`


@overload
def traced(
    func: _FuncType,
    *,
    message: str | None = None,
    color: str = "modular_purple",
) -> _FuncType: ...


@overload
def traced(
    func: None = None,
    *,
    message: str | None = None,
    color: str = "modular_purple",
) -> Callable[[_FuncType], _FuncType]: ...


def traced(
    func: Callable | None = None,
    *,
    message: str | None = None,
    color: str = "modular_purple",
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

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if is_profiling_enabled():
                with Trace(message, color):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_profiling_enabled():
                with Trace(message, color):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

    return wrapper


class Tracer:
    """
    A stack of Trace objects that allows for nested tracing spans
    without having to indent code via nested `with Trace(name):` statements.

    Note: The `Tracer` object can also be used as a context manager the same way
    the `Trace` object can.

    Example:
    ```python
    tracer = Tracer("foo", color="modular_purple")
    tracer.push("bar")
    # ...
    tracer.pop()

    with Tracer("foo", color="modular_purple") as tracer:
        # The parent span is named "foo".
        tracer.push("bar")
        # The sub-span is named "bar".
        tracer.pop()
    ```
    """

    def __init__(
        self, message: str | None = None, color: str = "modular_purple"
    ) -> None:
        """
        Initialize the stack.
        Optionally push a new trace onto the stack if message is not None.
        """
        self.trace_stack: list[Trace | None] = []
        self.push(message, color)

    def push(
        self, message: str | None = None, color: str = "modular_purple"
    ) -> None:
        """
        Push a new trace onto the stack.
        None is pushed if profiling is disabled or if message is None.
        """
        if not is_profiling_enabled() or message is None:
            self.trace_stack.append(None)
        else:
            trace = Trace(message, color)
            self.trace_stack.append(trace)
            trace.__enter__()

    def pop(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """
        Pop a trace off the stack and call its __exit__ method, optionally
        passing through exception information for in context manager.
        """
        trace = self.trace_stack.pop()
        if trace is not None:
            trace.__exit__(exc_type, exc_value, traceback)

    def next(self, message: str, color: str = "modular_purple") -> None:
        """
        Pop current then push a new trace with the next message.
        """
        self.pop()
        self.push(message, color)

    def cleanup(self) -> None:
        """
        Pop all traces that were pushed.
        """
        while self.trace_stack:
            self.pop()

    def __del__(self) -> None:
        self.cleanup()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    def mark(self) -> None:
        """
        Mark the current trace.
        """
        assert self.trace_stack, "stack underflow in Tracer.mark()"
        if self.trace_stack[-1] is not None:
            self.trace_stack[-1].mark()
