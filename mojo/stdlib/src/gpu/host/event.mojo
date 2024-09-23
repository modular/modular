# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This Event implements the event utilities."""

from os import abort
from time import perf_counter_ns

from ._utils import _check_error, _EventHandle, _StreamHandle
from .context import Context
from .stream import Stream

# ===----------------------------------------------------------------------===#
# Flag
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Flag:
    var _value: UInt32

    alias DEFAULT = Self(0)
    """Default event creation flag."""

    alias BLOCKING_SYNC = Self(1)
    """Specifies that the created event should use blocking synchronization."""

    alias DISABLE_TIMING = Self(2)
    """Event will not record timing data."""

    alias INTERPROCESS = Self(4)
    """Event is suitable for interprocess use. Flag.DISABLE_TIMING must be
    set."""

    @always_inline("nodebug")
    fn __init__(inout self, value: UInt32):
        self._value = value

    @always_inline("nodebug")
    fn __or__(self, rhs: Self) -> Self:
        """Returns `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return Self {_value: self._value | rhs._value}


# ===----------------------------------------------------------------------===#
# Event
# ===----------------------------------------------------------------------===#


struct Event:
    var _event: _EventHandle
    var cuda_dll: CudaDLL

    @always_inline
    fn __init__(inout self, ctx: Context, flags: Flag = Flag.DEFAULT) raises:
        """Creates an event for the current CUDA context."""
        self.cuda_dll = ctx.cuda_dll
        self._event = _EventHandle()
        _check_error(self.cuda_dll.cuCtxPushCurrent(ctx.ctx))
        _check_error(
            self.cuda_dll.cuEventCreate(
                UnsafePointer.address_of(self._event), flags
            )
        )

    @always_inline
    fn __del__(owned self):
        """Destroys a specified CUDA event."""

        try:
            if not self._event:
                return

            _check_error(self.cuda_dll.cuEventDestroy(self._event))
            self._event = _EventHandle()
        except e:
            abort(e.__str__())

    @always_inline
    fn sync(self) raises:
        """Waits until the completion of all work currently capturend in a particular event.
        """
        _check_error(self.cuda_dll.cuEventSynchronize(self._event))

    @always_inline
    fn record(self, stream: Stream) raises:
        """Captures the contents of a stream in the events object at the time of this call.
        """
        _check_error(self.cuda_dll.cuEventRecord(self._event, stream.stream))

    @always_inline
    fn elapsed(self, other: Event) raises -> Float32:
        """Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
        """

        var ms = Float32(0)
        _check_error(
            self.cuda_dll.cuEventElapsedTime(
                UnsafePointer.address_of(ms), self._event, other._event
            )
        )
        return ms


@always_inline
@parameter
fn time_function[func: fn () capturing [_] -> None]() -> Int:
    """Time a function using CPU timer."""

    var start = perf_counter_ns()
    func()
    return perf_counter_ns() - start
