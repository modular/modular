# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA context operations."""

from os import abort

from memory.unsafe import DTypePointer, Pointer

from ._utils import _check_error, _get_dylib_function
from .device import Device

# ===----------------------------------------------------------------------===#
# Context
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _ContextImpl(Boolable):
    var handle: DTypePointer[DType.invalid]

    fn __init__(inout self):
        self.handle = DTypePointer[DType.invalid]()

    fn __init__(inout self, handle: DTypePointer[DType.invalid]):
        self.handle = handle

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct Context:
    var ctx: _ContextImpl

    fn __init__(inout self) raises:
        self.__init__(Device())

    fn __init__(inout self, device: Device, flags: Int = 0) raises:
        var ctx = _ContextImpl()

        _check_error(
            _get_dylib_function[
                "cuCtxCreate_v2",
                fn (Pointer[_ContextImpl], Int32, Device) -> Result,
            ]()(Pointer.address_of(ctx), flags, device)
        )
        self.ctx = ctx

    fn __del__(owned self):
        try:
            if self.ctx:
                _check_error(
                    _get_dylib_function[
                        "cuCtxDestroy_v2", fn (_ContextImpl) -> Result
                    ]()(self.ctx)
                )
        except e:
            abort(e.__str__())

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx = existing.ctx
        existing.ctx = _ContextImpl()
