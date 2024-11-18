# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from max._utils import call_dylib_func
from sys.ffi import DLHandle
from .session import InferenceSession
from ._compilation import CCompiledModel
from ._status import Status


@value
@register_passable("trivial")
struct CModel:
    """Mojo representation of Engine's AsyncModel pointer.
    Useful for C inter-op.
    """

    var ptr: UnsafePointer[NoneType]

    alias FreeModelFnName = "M_freeModel"
    alias WaitForModelFnName = "M_waitForModel"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn await_model(self, lib: DLHandle) raises:
        var status = Status(lib)
        call_dylib_func(
            lib, Self.WaitForModelFnName, self.ptr, status.borrow_ptr()
        )
        if status:
            raise status.__str__()

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeModelFnName, self)
