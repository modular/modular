# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA stream operations."""

from memory.unsafe import DTypePointer, Pointer

from ._utils import _check_error, _get_dylib_function

# ===----------------------------------------------------------------------===#
# StreamImpl
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _StreamImpl:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


# ===----------------------------------------------------------------------===#
# Stream
# ===----------------------------------------------------------------------===#


struct Stream[is_borrowed: Bool = False]:
    var stream: _StreamImpl

    fn __init__(inout self, stream: _StreamImpl):
        self.stream = stream

    fn __init__(inout self, flags: Int = 0) raises:
        var stream = _StreamImpl()

        _check_error(
            _get_dylib_function[fn (Pointer[_StreamImpl], Int32) -> Result](
                "cuStreamCreate"
            )(Pointer.address_of(stream), Int32(0))
        )

        self.stream = stream

    fn __del__(owned self) raises:
        @parameter
        if is_borrowed:
            return
        if self.stream:
            _check_error(
                _get_dylib_function[fn (_StreamImpl) -> Result](
                    "cuStreamDestroy"
                )(self.stream)
            )

    fn __moveinit__(inout self, owned existing: Self):
        self.stream = existing.stream
        existing.stream = _StreamImpl()

    fn __takeinit__(inout self, inout existing: Self):
        self.stream = existing.stream
        existing.stream = _StreamImpl()

    fn synchronize(inout self) raises:
        if self.stream:
            _check_error(
                _get_dylib_function[fn (_StreamImpl) -> Result](
                    "cuStreamSynchronize"
                )(self.stream)
            )
