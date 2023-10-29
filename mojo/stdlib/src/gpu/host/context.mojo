# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA context operations."""

from memory.unsafe import DTypePointer, Pointer

from .device import Device
from ._utils import _check_error, _get_dylib_function

# ===----------------------------------------------------------------------===#
# Context
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _ContextImpl:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

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
                fn (Pointer[_ContextImpl], Int32, Device) -> Result
            ]("cuCtxCreate_v2")(Pointer.address_of(ctx), flags, device)
        )
        self.ctx = ctx

    fn __del__(owned self) raises:
        if self.ctx:
            _check_error(
                _get_dylib_function[fn (_ContextImpl) -> Result](
                    "cuCtxDestroy_v2"
                )(self.ctx)
            )

    fn __enter__(owned self) -> Self:
        return self ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx = existing.ctx
        existing.ctx = _ContextImpl()

    fn __takeinit__(inout self, inout existing: Self):
        self.ctx = existing.ctx
        existing.ctx = _ContextImpl()
