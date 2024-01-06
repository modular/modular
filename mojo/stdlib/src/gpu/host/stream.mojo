# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA stream operations."""

from memory.unsafe import DTypePointer, Pointer
from debug._debug import trap

from ._utils import _check_error, _get_dylib_function


@always_inline
fn _get_current_stream() -> DTypePointer[DType.invalid]:
    return external_call[
        "KGEN_CompilerRT_LLCL_GetCurrentStream", DTypePointer[DType.invalid]
    ]()


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


struct Stream:
    var stream: _StreamImpl
    var owning: Bool

    @staticmethod
    fn get_current_stream() -> Stream:
        return Stream(_StreamImpl(_get_current_stream()))

    fn __init__(inout self, stream: _StreamImpl):
        self.stream = stream
        self.owning = False

    fn __init__(inout self, flags: Int = 0) raises:
        var stream = _StreamImpl()

        _check_error(
            _get_dylib_function[fn (Pointer[_StreamImpl], Int32) -> Result](
                "cuStreamCreate"
            )(Pointer.address_of(stream), Int32(0))
        )

        self.stream = stream
        self.owning = True

    fn __del__(owned self):
        try:
            if self.owning and self.stream:
                _check_error(
                    _get_dylib_function[fn (_StreamImpl) -> Result](
                        "cuStreamDestroy"
                    )(self.stream)
                )
        except e:
            trap(e.__str__())

    fn __moveinit__(inout self, owned existing: Self):
        self.stream = existing.stream
        self.owning = existing.owning
        existing.stream = _StreamImpl()
        existing.owning = False

    fn synchronize(inout self) raises:
        if self.stream:
            _check_error(
                _get_dylib_function[fn (_StreamImpl) -> Result](
                    "cuStreamSynchronize"
                )(self.stream)
            )
