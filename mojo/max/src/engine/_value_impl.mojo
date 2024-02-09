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
    alias _GetListFnName = "M_getListFromValue"
    alias _FreeValueFnName = "M_freeValue"

    fn get_c_tensor(self, borrowed lib: DLHandle) -> CTensor:
        """Get CTensor within value."""
        return call_dylib_func[CTensor](lib, Self._GetTensorFnName, self)

    fn get_bool(self, borrowed lib: DLHandle) -> Bool:
        """Get bool within value."""
        return call_dylib_func[Bool](lib, Self._GetBoolFnName, self)

    fn get_list(self, borrowed lib: DLHandle) -> CList:
        """Get list within value."""
        return call_dylib_func[CList](lib, Self._GetListFnName, self)

    fn free(self, borrowed lib: DLHandle):
        """Free value."""
        call_dylib_func(lib, Self._FreeValueFnName, self)


@value
@register_passable("trivial")
struct CList:
    """Represents an AsyncList pointer from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias _GetSizeFnName = "M_getListSize"
    alias _GetValueFnName = "M_getListValue"
    alias _AppendFnName = "M_appendToList"
    alias _FreeFnName = "M_freeList"

    fn get_size(self, borrowed lib: DLHandle) -> Int:
        """Get number of elements in list."""
        return call_dylib_func[Int](lib, Self._GetSizeFnName, self)

    fn get_value(self, borrowed lib: DLHandle, index: Int) -> CValue:
        """Get value by index in list."""
        return call_dylib_func[CValue](lib, Self._GetValueFnName, self, index)

    fn append(self, borrowed lib: DLHandle, value: CValue):
        """Get value by index in list."""
        return call_dylib_func(lib, Self._AppendFnName, self, value)

    fn free(self, borrowed lib: DLHandle):
        """Free list."""
        call_dylib_func(lib, Self._FreeFnName, self)
