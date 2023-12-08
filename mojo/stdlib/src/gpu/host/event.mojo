# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This Event implements the event utilities."""

from memory.unsafe import DTypePointer, Pointer

from ._utils import _check_error, _get_dylib_function
from gpu.host.stream import Stream, _StreamImpl
from time import now
from debug._debug import trap


# ===----------------------------------------------------------------------===#
# Flag
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Flag:
    var _value: UInt32

    alias DEFAULT = Flag(0)
    """Default event creation flag."""

    alias BLOCKING_SYNC = Flag(1)
    """Specifies that the created event should use blocking synchronization."""

    alias DISABLE_TIMING = Flag(2)
    """Event will not record timing data."""

    alias INTERPROCESS = Flag(4)
    """Event is suitable for interprocess use. Flag.DISABLE_TIMING must be
    set."""

    @always_inline("nodebug")
    fn __init__(value: UInt32) -> Self:
        return Self {_value: value}

    @always_inline("nodebug")
    fn __or__(self, rhs: Flag) -> Flag:
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


@value
@register_passable("trivial")
struct _EventImpl:
    var handle: DTypePointer[DType.invalid]

    @always_inline
    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    @always_inline
    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    @always_inline
    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


@register_passable
struct Event:
    var _event: _EventImpl

    @always_inline
    fn __init__(flags: Flag = Flag.DEFAULT) raises -> Self:
        var event = _EventImpl()

        _check_error(
            _get_dylib_function[fn (Pointer[_EventImpl], Flag) -> Result](
                "cuEventCreate"
            )(Pointer.address_of(event), flags)
        )
        return Self {_event: event}

    @always_inline
    fn __del__(owned self):
        try:
            if not self._event:
                return
            _check_error(
                _get_dylib_function[fn (_EventImpl) -> Result](
                    "cuEventDestroy_v2"
                )(self._event)
            )
            self._event = _EventImpl()
        except e:
            trap(e.__str__())

    @always_inline
    fn sync(self) raises:
        _check_error(
            _get_dylib_function[fn (_EventImpl) -> Result](
                "cuEventSynchronize"
            )(self._event)
        )

    @always_inline
    fn record(self, stream: Stream) raises:
        _check_error(
            _get_dylib_function[fn (_EventImpl, _StreamImpl) -> Result](
                "cuEventRecord"
            )(self._event, stream.stream)
        )

    @always_inline
    fn elapsed(self, other: Event) raises -> Float32:
        var ms = Float32(0)
        _check_error(
            _get_dylib_function[
                fn (Pointer[Float32], _EventImpl, _EventImpl) -> Result
            ]("cuEventElapsedTime")(
                Pointer.address_of(ms), self._event, other._event
            )
        )
        return ms


# ===----------------------------------------------------------------------===#
# time_function
# ===----------------------------------------------------------------------===#


@always_inline
@parameter
fn time_function[func: fn (Stream) capturing -> None](stream: Stream) -> Int:
    try:
        let start = Event()
        let end = Event()

        start.record(stream)
        func(stream)
        end.record(stream)
        end.sync()

        let msec = start.elapsed(end)

        return int(msec * 1_000_000)
    except e:
        print("CUDA timing error:", e)
        return -1


@always_inline
@parameter
fn time_function[
    func: fn (Stream) raises capturing -> None
](stream: Stream) raises -> Int:
    let start = Event()
    let end = Event()

    start.record(stream)
    func(stream)
    end.record(stream)
    end.sync()

    let msec = start.elapsed(end)

    return int(msec * 1_000_000)


@always_inline
@parameter
fn time_function[func: fn () capturing -> None]() -> Int:
    let start = now()
    func()
    return now() - start
