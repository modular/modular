# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Provides MOTR tracing utilities for profiling Python and async code.

Includes decorators, context managers, and async-safe tracing functionality
for integration with the MAX profiling system.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar, overload

from max._core.profiler import (
    MotrTrace,
    is_profiling_enabled,
    set_motr_thread_local_stack,
)

_FuncType = TypeVar("_FuncType", bound=Callable[..., Any])

# Context variable to store only the user portion of async trace stack
_async_user_stack: ContextVar[list[int]] = ContextVar(
    "_async_user_stack",
    default=[],  # noqa: B039
)


class _TraceContext:
    """Manages profiling traces safely across async and synchronous contexts.

    Handles proper trace stack management for both async and sync code paths,
    ensuring trace hierarchy is maintained correctly in async environments.
    """

    def __init__(self, trace_message: str, color: str = "modular_purple"):
        """Initializes a new trace context.

        Args:
            trace_message: The name for the profiling span.
            color: The color to display in profiling visualization.
                Defaults to "modular_purple".
        """
        self.trace_message = trace_message
        self.color = color
        self.trace: Optional[MotrTrace] = None
        self.user_stack_token: Optional[Token[list[int]]] = None

    async def __aenter__(self) -> _TraceContext:
        return self.__enter__()

    async def __aexit__(self, *exc) -> None:
        self.__exit__(*exc)

    def __enter__(self) -> _TraceContext:
        if not is_profiling_enabled():
            return self

        # Create a copy of the current async stack and set it as the new context.
        # The token allows us to restore the original stack when exiting.
        # This ensures nested traces don't interfere with each other's stack state.
        self.user_stack_token = _async_user_stack.set(
            list(_async_user_stack.get())
        )

        user_stack = _async_user_stack.get()
        set_motr_thread_local_stack(user_stack)
        # Create a new trace with the given message and color.
        # The 0 parent id tells MOTR to use the thread_id from the stack we just set
        self.trace = MotrTrace(self.trace_message, self.color, 0).__enter__()
        user_stack.append(self.trace.id)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self.trace:
            return

        try:
            set_motr_thread_local_stack(_async_user_stack.get())
            self.trace.__exit__(exc_type, exc_value, traceback)
        finally:
            if self.user_stack_token:
                _async_user_stack.reset(self.user_stack_token)
                # Sync C++ with the restored Python state
                set_motr_thread_local_stack(_async_user_stack.get())


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
    func: _FuncType | None = None,
    *,
    message: str | None = None,
    color: str = "modular_purple",
) -> _FuncType | Callable[[_FuncType], _FuncType]:
    """Decorator that creates a profiling span around function execution.

    Supports both synchronous and asynchronous functions with automatic
    detection. Creates trace spans that integrate with the MOTR profiling
    system.

    Args:
        func: The function to wrap with profiling. If :obj:`None`, returns a
            decorator factory.
        message: The name for the profiling span. Defaults to the function
            name if not provided.
        color: The color to display in profiling visualization. Defaults
            to "modular_purple".

    Returns:
        The decorated function or a decorator factory if func is :obj:`None`.

    .. code-block:: python

        @traced(message="custom_name", color="modular_purple")
        def my_function():
            # The span is named "custom_name".
            pass

        @traced
        async def async_function():
            # The span is named "async_function".
            pass
    """
    if func is None:
        return lambda f: traced(f, message=message, color=color)

    # Default to the function name if message wasn't passed.
    message = message if message is not None else func.__name__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):  # noqa: ANN202
            async with _TraceContext(message, color):
                return await func(*args, **kwargs)

        return async_wrapper

    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):  # noqa: ANN202
            with _TraceContext(message, color):
                return func(*args, **kwargs)

        return sync_wrapper


class Tracer:
    """A stack-based tracer for creating nested profiling spans.

    Allows nested tracing without deep indentation through push/pop operations.
    Can be used as a context manager or manually managed.

    .. code-block:: python

        tracer = Tracer("parent_span", color="modular_purple")
        tracer.push("child_span")
        # ... work ...
        tracer.pop()

        # Or as context manager:
        with Tracer("parent_span") as tracer:
            # The parent span is named "parent_span".
            tracer.push("child_span")
            # The sub-span is named "child_span".
            tracer.pop()
    """

    def __init__(
        self, message: str | None = None, color: str = "modular_purple"
    ) -> None:
        """Initializes a new tracer stack.

        Args:
            message: The initial trace message to push onto the stack. If
                :obj:`None`, creates an empty tracer.
            color: The color for the initial trace span. Defaults to "modular_purple".
        """
        self.trace_stack: list[Optional[MotrTrace]] = []
        self.push(message, color)

    def push(
        self, message: str | None = None, color: str = "modular_purple"
    ) -> None:
        """Pushes a new trace span onto the stack.

        Args:
            message: The trace span name. If :obj:`None` or profiling is
                disabled, pushes a placeholder entry.
            color: The color for the trace span. Defaults to "modular_purple".
        """
        if not is_profiling_enabled() or message is None:
            # Push None for empty traces but still maintain the stack
            self.trace_stack.append(None)
            return

        user_stack = _async_user_stack.get()
        set_motr_thread_local_stack(user_stack)
        trace = MotrTrace(message, color, 0).__enter__()
        self.trace_stack.append(trace)
        user_stack.append(trace.id)

    def pop(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Pops the most recent trace span from the stack and closes it.

        Args:
            exc_type: Exception type if called during exception handling.
            exc_value: Exception instance if called during exception handling.
            traceback: Traceback object if called during exception handling.

        Raises:
            AssertionError: If called when the stack is empty.
        """
        assert len(self.trace_stack), (
            "stack underflow: pop called without a push"
        )
        trace = self.trace_stack.pop()
        if trace is None:
            return
        # Set the C++ stack to the user stack so that the
        # trace is closed with the correct parent id
        set_motr_thread_local_stack(_async_user_stack.get())
        trace.__exit__(exc_type, exc_value, traceback)
        # Pop the trace id from the user stack
        _async_user_stack.get().pop()

    def next(self, message: str, color: str = "modular_purple") -> None:
        """Pops the current trace and pushes a new one with the given message.

        Args:
            message: The name for the new trace span.
            color: The color for the new trace span. Defaults to "modular_purple".
        """
        self.pop()
        self.push(message, color)

    def mark(self) -> None:
        """Marks the current trace span with a timestamp.

        This is useful for marking atomic events within a span.
        Currently, the mark has no metadata associated with it.

        Raises:
            AssertionError: If called when the stack is empty.
        """
        assert len(self.trace_stack), (
            "stack underflow: mark called without a push"
        )
        if self.trace_stack[-1] is not None:
            self.trace_stack[-1].mark()

    def __del__(self) -> None:
        """Cleans up any remaining traces when the tracer is garbage collected.

        Ensures that traces are properly closed even if the user forgets to
        call :obj:`pop` or use the tracer as a context manager.
        """
        self.__exit__(None, None, None)

    def __enter__(self) -> Tracer:
        """Returns self for context manager protocol."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Pop all traces that were pushed
        while self.trace_stack:
            self.pop(exc_type, exc_value, traceback)


# ---------------------------------------------------------------------------
# AsyncIO integration: ensure every new Task starts with the correct MOTR
# parent-id stack.  Installing a custom event-loop policy at import time means
# applications do not need to call anything explicitly.
# ---------------------------------------------------------------------------


def _motr_task_factory(loop: asyncio.AbstractEventLoop, coro, *args, **kwargs):  # noqa: ANN001, ANN202
    """Creates asyncio tasks with proper MOTR trace context synchronization.

    Every freshly-created Task inherits the caller's ContextVars, including
    :obj:`_async_user_stack`. When spawning a Task inside an async trace, this
    factory pushes the Python stack down to the C++ side so that the first
    synchronous :obj:`MotrTrace` in the Task derives the correct parent id.

    Args:
        loop: The event loop creating the task.
        coro: The coroutine to wrap in a task.
        args: Additional positional arguments for task creation.
        kwargs: Additional keyword arguments for task creation.

    Returns:
        A new asyncio Task with synchronized trace context.
    """

    # Always sync C++ stack with inherited context
    set_motr_thread_local_stack(_async_user_stack.get())

    return asyncio.Task(coro, loop=loop, *args, **kwargs)  # noqa: B026


class _MotrEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Event loop policy that installs MOTR task factory on each loop.

    Extends the default asyncio event loop policy to automatically install
    :obj:`_motr_task_factory` on newly created event loops, ensuring proper
    trace context management for all asyncio tasks.
    """

    def new_event_loop(self):  # noqa: ANN202
        """Creates a new event loop with MOTR task factory installed."""
        loop = super().new_event_loop()
        loop.set_task_factory(_motr_task_factory)
        return loop


def _install_motr_policy_once() -> None:
    """Installs the MOTR event loop policy if not already installed.

    Replaces the global asyncio event loop policy only if it is not already
    a :obj:`_MotrEventLoopPolicy` instance.
    """

    if not isinstance(asyncio.get_event_loop_policy(), _MotrEventLoopPolicy):
        asyncio.set_event_loop_policy(_MotrEventLoopPolicy())


# Install immediately at import time.
_install_motr_policy_once()


# ---------------------------------------------------------------------------
# uvloop Integration: Ensure MOTR trace context propagation with uvloop
# ---------------------------------------------------------------------------
#
# uvloop ships with its own EventLoopPolicy that doesn't use our custom
# task factory (_motr_task_factory).
#
# Without this patch:
# 1. Users who install uvloop and call uvloop.run() or uvloop.install()
#    would get uvloop's policy instead of our _MotrEventLoopPolicy
# 2. New asyncio tasks would lose MOTR trace context inheritance
# 3. Profiling spans would be disconnected across task boundaries
#
# The solution is to monkey-patch uvloop.EventLoopPolicy at import time
# to create a hybrid that combines uvloop's performance with our
# MOTR task factory integration.
def _patch_uvloop_policy() -> None:
    """Patches uvloop to integrate with MOTR trace context management.

    uvloop provides its own EventLoopPolicy that doesn't include our MOTR
    task factory. This function dynamically creates a hybrid EventLoopPolicy that:
    1. Inherits from uvloop's policy for performance benefits
    2. Adds our _motr_task_factory for proper trace context inheritance
    3. Replaces uvloop.EventLoopPolicy so future uvloop.run() calls work

    The patching is safe because:
    - It only runs if uvloop is actually installed
    - It preserves all uvloop functionality
    - It's idempotent (won't patch twice)
    - It only affects future event loop creation
    """
    try:
        import uvloop  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return  # uvloop not installed, nothing to patch

    import typing as _t

    _uv: _t.Any = uvloop  # cast to Any for dynamic attribute access

    # Skip if we already patched it (idempotent operation)
    if getattr(_uv, "_MOTR_PATCHED", False):
        return

    # Dynamically build a subclass of uvloop's policy.
    # This preserves all uvloop optimizations while adding MOTR integration.
    class _MotrUvloopPolicy(_uv.EventLoopPolicy):
        """uvloop event loop policy with MOTR task factory integration.

        This hybrid policy combines uvloop's event loop implementation
        with MOTR's trace context management. Every event loop created by
        this policy will automatically install the MOTR task factory,
        ensuring proper trace hierarchy across asyncio tasks.

        Inherits all uvloop optimizations while adding seamless profiling
        integration.
        """

        def new_event_loop(self):  # noqa: ANN202
            """Creates a new event loop with MOTR task factory installed."""
            loop = super().new_event_loop()
            loop.set_task_factory(_motr_task_factory)
            return loop

    # Replace uvloop.EventLoopPolicy with our MOTR-aware hybrid.
    # This affects uvloop.run() and uvloop.install() calls.
    setattr(_uv, "EventLoopPolicy", _MotrUvloopPolicy)  # noqa: B010
    _uv._MOTR_PATCHED = True  # Mark as patched for idempotency

    # If the current global policy is already a uvloop policy, replace it
    # with our MOTR-aware version. This handles the case where uvloop was
    # imported and installed before this module was imported.
    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, _uv.EventLoopPolicy):
        # Replace the existing uvloop policy with our enhanced version
        asyncio.set_event_loop_policy(_MotrUvloopPolicy())


_patch_uvloop_policy()
