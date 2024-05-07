# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle

from ._status import Status
from ._utils import call_dylib_func, CString
from ._tensor_impl import CTensor
from ._value_impl import CValue


@value
@register_passable("trivial")
struct CTensorMap:
    """Represents AsyncTensorMap ptr from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias FreeAsyncTensorMapFnName = "M_freeAsyncTensorMap"
    alias BorrowTensorIntoFnName = "M_borrowTensorInto"
    alias BorrowValueIntoFnName = "M_borrowValueInto"
    alias GetTensorByNameFromFnName = "M_getTensorByNameFrom"
    alias GetValueByNameFromFnName = "M_getValueByNameFrom"
    alias GetTensorMapSizeFnName = "M_getTensorMapSize"

    fn get_tensor_by_name(self, name: CString, lib: DLHandle) raises -> CTensor:
        var status = Status(lib)
        var tensor = call_dylib_func[CTensor](
            lib, Self.GetTensorByNameFromFnName, self, name, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return tensor

    fn get_value_by_name(self, name: CString, lib: DLHandle) raises -> CValue:
        var status = Status(lib)
        var value = call_dylib_func[CValue](
            lib, Self.GetValueByNameFromFnName, self, name, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return value

    fn borrow_tensor_by_name(
        self,
        ptr: DTypePointer[DType.invalid],
        spec: EngineTensorSpec,
        lib: DLHandle,
    ) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            Self.BorrowTensorIntoFnName,
            self,
            ptr,
            spec._borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

    fn borrow_value_by_name(
        self,
        name: String,
        ptr: DTypePointer[DType.invalid],
        lib: DLHandle,
    ) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            Self.BorrowValueIntoFnName,
            self,
            name.unsafe_ptr(),
            ptr,
            status.borrow_ptr(),
        )
        _ = name
        if status:
            raise status.__str__()

    fn size(self, lib: DLHandle) raises -> Int:
        var status = Status(lib)
        var size = call_dylib_func[Int](
            lib, Self.GetTensorMapSizeFnName, self, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return size

    fn free(self, lib: DLHandle):
        """
        Free the AsyncTensorMap ptr.
        """
        call_dylib_func(lib, Self.FreeAsyncTensorMapFnName, self)
