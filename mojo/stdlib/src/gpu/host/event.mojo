# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This Event implements the event utilities."""

from memory.unsafe import DTypePointer, Pointer

from ._utils import _check_error, _get_dylib, _get_dylib_function


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


@value
@register_passable("trivial")
struct _EventImpl:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

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
    fn __del__(owned self) raises:
        if not self._event:
            return
        _check_error(
            _get_dylib_function[fn (_EventImpl) -> Result]("cuEventDestroy")(
                self._event
            )
        )
        self._event = _EventImpl()

    @always_inline
    fn sync(self) raises:
        _check_error(
            _get_dylib_function[fn (_EventImpl) -> Result](
                "cuEventSynchronize"
            )(self._event)
        )
