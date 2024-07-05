# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This Event implements the event utilities."""

from os import abort
from time import perf_counter_ns

from memory.unsafe import DTypePointer

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
    var cuda_dll: Optional[CudaDLL]

    @always_inline
    fn __init__(inout self, ctx: Context, flags: Flag = Flag.DEFAULT) raises:
        """Creates an event for the current CUDA context."""

        self.__init__(flags, ctx.cuda_dll)

    @always_inline
    fn __init__(
        inout self,
        flags: Flag = Flag.DEFAULT,
        cuda_dll: Optional[CudaDLL] = None,
    ) raises:
        """Creates an event for the current CUDA context."""

        self.cuda_dll = cuda_dll
        self._event = _EventHandle()

        var cuEventCreate = self.cuda_dll.value().cuEventCreate if self.cuda_dll else cuEventCreate.load()
        _check_error(
            cuEventCreate(UnsafePointer.address_of(self._event), flags)
        )

    @always_inline
    fn __del__(owned self):
        """Destroys a specified CUDA event."""

        try:
            if not self._event:
                return

            var cuEventCreate = self.cuda_dll.value().cuEventDestroy if self.cuda_dll else cuEventDestroy.load()
            _check_error(cuEventCreate(self._event))
            self._event = _EventHandle()
        except e:
            abort(e.__str__())

    @always_inline
    fn sync(self) raises:
        """Waits until the completion of all work currently capturend in a particular event.
        """

        var cuEventSynchronize = self.cuda_dll.value().cuEventSynchronize if self.cuda_dll else cuEventSynchronize.load()
        _check_error(cuEventSynchronize(self._event))

    @always_inline
    fn record(self, stream: Stream) raises:
        """Captures the contents of a stream in the events object at the time of this call.
        """

        var cuEventRecord = self.cuda_dll.value().cuEventRecord if self.cuda_dll else cuEventRecord.load()
        _check_error(cuEventRecord(self._event, stream.stream))

    @always_inline
    fn elapsed(self, other: Event) raises -> Float32:
        """Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
        """

        var cuEventElapsedTime = self.cuda_dll.value().cuEventElapsedTime if self.cuda_dll else cuEventElapsedTime.load()
        var ms = Float32(0)
        _check_error(
            cuEventElapsedTime(
                UnsafePointer.address_of(ms), self._event, other._event
            )
        )
        return ms


# ===----------------------------------------------------------------------===#
# time_function
# ===----------------------------------------------------------------------===#


@always_inline
@parameter
fn time_function[func: fn (Stream) capturing -> None](stream: Stream) -> Int:
    """Time a function using CUDA Events timer."""

    try:
        var start = Event()
        var end = Event()

        start.record(stream)
        func(stream)
        end.record(stream)
        end.sync()

        var msec = start.elapsed(end)

        return int(msec * 1_000_000)
    except e:
        print("CUDA timing error:", e)
        return -1


@always_inline
@parameter
fn time_function[
    func: fn (Stream) raises capturing -> None
](stream: Stream) raises -> Int:
    """Time a function using CUDA Events timer."""

    var start = Event()
    var end = Event()

    start.record(stream)
    func(stream)
    end.record(stream)
    end.sync()

    var msec = start.elapsed(end)

    return int(msec * 1_000_000)


@always_inline
@parameter
fn time_function[func: fn () capturing -> None]() -> Int:
    """Time a function using CPU timer."""

    var start = perf_counter_ns()
    func()
    return perf_counter_ns() - start
