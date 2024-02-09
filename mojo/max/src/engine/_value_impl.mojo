# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._tensor_impl import CTensor
from ._utils import call_dylib_func


@value
@register_passable("trivial")
struct CValue:
    """Represents an AsyncValue pointer from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias _GetTensorFnName = "M_getTensorFromValue"
    alias _GetBoolFnName = "M_getBoolFromValue"
    alias _FreeValueFnName = "M_freeValue"

    fn get_c_tensor(self, borrowed lib: DLHandle) -> CTensor:
        """Get CTensor within value."""
        return call_dylib_func[CTensor](lib, Self._GetTensorFnName, self)

    fn get_bool(self, borrowed lib: DLHandle) -> Bool:
        """Get bool within value."""
        return call_dylib_func[Bool](lib, Self._GetBoolFnName, self)

    fn free(self, borrowed lib: DLHandle):
        """Free value."""
        call_dylib_func(lib, Self._FreeValueFnName, self)
