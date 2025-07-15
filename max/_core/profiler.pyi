# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence

class Trace:
    """
    Context manager for creating profiling spans.

    Examples:
        >>> with Trace("foo", color="modular_purple"):
        >>>   # Run `bar()` inside the profiling span.
        >>>   bar()
        >>> # The profiling span ends when the context manager exits.
    """

    def __init__(self, message: str, color: str = "modular_purple") -> None:
        """
        Constructs and initializes the underlying Mojo Trace object.

        Args:
            message: name of the span.
            color: color of the span.
        """

    def __enter__(self) -> Trace:
        """Begins a profiling event."""

    def __exit__(
        self,
        exc_type: object | None = None,
        exc_value: object | None = None,
        traceback: object | None = None,
    ) -> None:
        """Ends a profiling event."""

    def mark(self) -> None:
        """Marks an event in the trace timeline."""

def is_profiling_enabled() -> bool:
    """Returns whether profiling is enabled."""

def set_gpu_profiling_state(arg: str, /) -> None:
    """Sets the GPU profiling state."""

class MotrTrace:
    """
    Context manager for creating profiling spans with explicit parent ID.

    Examples:
        >>> with MotrTrace("foo", color="blue", parentId=123):
        >>>   # Run `bar()` inside the profiling span.
        >>>   bar()
        >>> # The profiling span ends when the context manager exits.
    """

    def __init__(self, message: str, color: str, parentId: int) -> None:
        """
        Constructs and initializes the underlying Motr Trace object.

        Args:
            message: name of the span.
            color: color of the span.
            parentId: parent id of the span.
        """

    def __enter__(self) -> MotrTrace:
        """Begins a profiling event."""

    def __exit__(
        self,
        exc_type: object | None = None,
        exc_value: object | None = None,
        traceback: object | None = None,
    ) -> None:
        """Ends a profiling event."""

    def mark(self) -> None:
        """Marks an event in the trace timeline."""

    @property
    def id(self) -> int:
        """Get the trace ID."""

def get_motr_thread_local_stack() -> list[int]:
    """
    Gets the current thread-local parent ID stack.

    Returns:
        List[int]: The current MOTR thread-local stack of parent IDs.
    """

def set_motr_thread_local_stack(stack: Sequence[int]) -> None:
    """
    Sets the current MOTR thread-local parent ID stack.

    Args:
        stack (List[int]): The stack of parent IDs to set.

    Returns:
        None
    """

def is_motr_enabled() -> bool:
    """
    Returns whether MOTR tracing is compiled in.

    Returns:
        bool: True if MOTR is enabled, False otherwise.
    """
