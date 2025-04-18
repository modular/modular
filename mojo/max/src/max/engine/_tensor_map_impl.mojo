# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import alignof, sizeof
from sys.ffi import DLHandle, external_call

from max._utils import CString, call_dylib_func
from memory import UnsafePointer

from ._status import Status
from ._tensor_impl import CTensor
from ._value_impl import CValue


fn _destroy_pointee_wrapper[T: AnyType](ptr: UnsafePointer[T]):
    ptr.destroy_pointee()


@value
@register_passable("trivial")
struct CTensorMap:
    """Represents AsyncTensorMap ptr from Engine."""

    var ptr: UnsafePointer[NoneType]

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn get_tensor_by_name(self, name: String, lib: DLHandle) raises -> CTensor:
        var status = Status(lib)
        var tensor = call_dylib_func[CTensor](
            lib,
            "M_getTensorByNameFrom",
            self,
            name.unsafe_cstr_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()
        return tensor

    fn get_value_by_name(self, name: String, lib: DLHandle) raises -> CValue:
        var status = Status(lib)
        var value = call_dylib_func[CValue](
            lib,
            "M_getValueByNameFrom",
            self,
            name.unsafe_cstr_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()
        return value

    fn borrow_tensor_by_name(
        self,
        ptr: UnsafePointer[NoneType],
        spec: EngineTensorSpec,
        lib: DLHandle,
    ) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            "M_borrowTensorInto",
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
        ptr: UnsafePointer[NoneType],
        lib: DLHandle,
    ) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            "M_borrowValueInto",
            self,
            name.unsafe_ptr(),
            ptr,
            status.borrow_ptr(),
        )
        _ = name
        if status:
            raise status.__str__()

    fn move_mojo_value_by_name[
        T: Movable
    ](self, name: String, owned val: T, lib: DLHandle,) raises:
        """Create a new MojoValue object and store in the tensormap.

        Parameters:
            T: Type of the mojo object.

        Arguments:
            name: Name of the entry in the tensormap.
            val: mojo object stored in the map as a MojoValue.
            lib: dlhandle for the lib
        """

        # Allocate buffer and move val.
        var value_destructor = _destroy_pointee_wrapper[T]
        var data_ptr = external_call[
            "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[T]
        ](sizeof[T](), alignof[T]())
        data_ptr.init_pointee_move(val^)

        # Store the data_ptr and destructor into an AnyAsyncValue.
        var status = Status(lib)
        call_dylib_func(
            lib,
            "M_moveMojoValueInto",
            self,
            name.unsafe_ptr(),
            data_ptr,
            value_destructor,
            status.borrow_ptr(),
        )
        _ = name
        if status:
            raise String(status)

    fn keys(
        self, size_ptr: UnsafePointer[Int64], lib: DLHandle
    ) -> UnsafePointer[CString]:
        return call_dylib_func[UnsafePointer[CString]](
            lib, "M_tensorMapKeys", self, size_ptr
        )

    fn size(self, lib: DLHandle) raises -> Int:
        var status = Status(lib)
        var size = call_dylib_func[Int](
            lib, "M_getTensorMapSize", self, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return size

    fn copy(self, lib: DLHandle) -> CTensorMap:
        """
        Copies the AsyncTensorMap ptr. Increases underlying refcount.
        """
        return call_dylib_func[CTensorMap, CTensorMap](
            lib,
            "M_copyAsyncTensorMap",
            self,
        )

    fn free(self, lib: DLHandle):
        """
        Free the AsyncTensorMap ptr.
        """
        call_dylib_func(lib, "M_freeAsyncTensorMap", self)
