# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import *
from ._context import *
from ._compilation import CCompiledModel
from .tensor_map import *
from ._tensor_impl import *
from .session import InferenceSession


@value
@register_passable("trivial")
struct CModel:
    """Mojo representation of Engine's AsyncModel pointer.
    Useful for C inter-op.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeModelFnName = "M_freeModel"
    alias WaitForModelFnName = "M_waitForModel"

    fn await_model(self, lib: DLHandle) raises:
        let status = Status(lib)
        call_dylib_func(
            lib, Self.WaitForModelFnName, self.ptr, status.borrow_ptr()
        )
        if status:
            raise status.__str__()

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeModelFnName, self)
