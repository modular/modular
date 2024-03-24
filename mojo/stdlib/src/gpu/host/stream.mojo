# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA stream operations."""

from os import abort

from memory.unsafe import DTypePointer, Pointer

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
struct _StreamImpl(Boolable):
    var handle: DTypePointer[DType.invalid]

    fn __init__(
        inout self,
        handle: DTypePointer[DType.invalid] = DTypePointer[DType.invalid](),
    ):
        self.handle = handle

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


# ===----------------------------------------------------------------------===#
# Stream
# ===----------------------------------------------------------------------===#


struct Stream(CollectionElement):
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
            _get_dylib_function[
                "cuStreamCreate", fn (Pointer[_StreamImpl], Int32) -> Result
            ]()(Pointer.address_of(stream), Int32(0))
        )

        self.stream = stream
        self.owning = True

    fn __del__(owned self):
        try:
            if self.owning and self.stream:
                _check_error(
                    _get_dylib_function[
                        "cuStreamDestroy", fn (_StreamImpl) -> Result
                    ]()(self.stream)
                )
        except e:
            abort(e.__str__())

    fn __copyinit__(inout self, existing: Self):
        self.stream = existing.stream
        self.owning = False

    fn __moveinit__(inout self, owned existing: Self):
        self.stream = existing.stream
        self.owning = existing.owning
        existing.stream = _StreamImpl()
        existing.owning = False

    fn synchronize(inout self) raises:
        """Wait until a CUDA stream's tasks are completed."""
        if self.stream:
            _check_error(
                _get_dylib_function[
                    "cuStreamSynchronize", fn (_StreamImpl) -> Result
                ]()(self.stream)
            )
