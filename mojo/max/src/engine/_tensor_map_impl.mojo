# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._dtypes import *
from ._tensor_spec_impl import *
from memory.unsafe import bitcast
from tensor import Tensor, TensorShape
from ._tensor_impl import *
from ._tensor_spec_impl import *
from ._value_impl import CValue
from ._context import *
from .session import InferenceSession


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
        let status = Status(lib)
        let tensor = call_dylib_func[CTensor](
            lib, Self.GetTensorByNameFromFnName, self, name, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return tensor

    fn get_value_by_name(self, name: CString, lib: DLHandle) raises -> CValue:
        let status = Status(lib)
        let value = call_dylib_func[CValue](
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
        let status = Status(lib)
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
        let status = Status(lib)
        call_dylib_func(
            lib,
            Self.BorrowValueIntoFnName,
            self,
            name._as_ptr(),
            ptr,
            status.borrow_ptr(),
        )
        _ = name
        if status:
            raise status.__str__()

    fn size(self, borrowed lib: DLHandle) raises -> Int:
        let status = Status(lib)
        let size = call_dylib_func[Int](
            lib, Self.GetTensorMapSizeFnName, self, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return size

    fn free(self, borrowed lib: DLHandle):
        """
        Free the AsyncTensorMap ptr.
        """
        call_dylib_func(lib, Self.FreeAsyncTensorMapFnName, self)
