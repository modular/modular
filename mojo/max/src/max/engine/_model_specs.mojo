# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.string import StaticString
from sys.ffi import DLHandle

from max._utils import CString, call_dylib_func, exchange
from memory import UnsafePointer

from ._compilation import CCompiledModel
from ._status import Status


@value
@register_passable("trivial")
struct CTensorNameArray:
    """Mojo representation of Engine's TensorArray pointer.
    This doesn't free the memory on destruction.
    """

    var ptr: UnsafePointer[NoneType]

    alias FreeTensorNameArrayFnName = "M_freeTensorNameArray"
    alias GetTensorNameAtFnName = "M_getTensorNameAt"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn get_name_at(self, idx: Int, lib: DLHandle) raises -> String:
        if not self.ptr:
            raise "failed to get tensor name"
        var name = call_dylib_func[CString](
            lib, Self.GetTensorNameAtFnName, self, idx
        )
        return name.__str__()

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeTensorNameArrayFnName, self)


struct TensorNamesIterator(Sized):
    var ptr: CTensorNameArray
    var current: Int
    var length: Int
    var lib: DLHandle

    fn __init__(out self, ptr: CTensorNameArray, length: Int, lib: DLHandle):
        self.ptr = ptr
        self.current = 0
        self.length = length
        self.lib = lib

    fn __next__(mut self) raises -> String:
        var next = self.ptr.get_name_at(self.current, self.lib)
        self.current += 1
        return next

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        if self.current == self.length:
            return 0
        return 1


struct TensorNames(Sized):
    var ptr: CTensorNameArray
    var lib: DLHandle
    var length: Int

    fn __init__(
        out self,
        fn_name: String,
        ptr: CCompiledModel,
        length: Int,
        lib: DLHandle,
    ):
        var status = Status(lib)
        self.ptr = call_dylib_func[CTensorNameArray](
            lib,
            StaticString(ptr=fn_name.unsafe_ptr(), length=len(fn_name)),
            ptr,
            status.borrow_ptr(),
        )
        if status:
            print(status.__str__())
            self.ptr = UnsafePointer[NoneType]()
        self.length = length
        self.lib = lib

    fn __moveinit__(out self, owned existing: Self):
        self.ptr = exchange[CTensorNameArray](
            existing.ptr, UnsafePointer[NoneType]()
        )
        self.length = existing.length
        self.lib = existing.lib

    fn __getitem__(self, idx: Int) raises -> String:
        return self.ptr.get_name_at(idx, self.lib)

    fn __len__(self) -> Int:
        return self.length

    fn __del__(owned self):
        self.ptr.free(self.lib)


struct InputTensorNames(Sized):
    """Collection of model input names."""

    var names: TensorNames

    alias GetInputTensorNamesFnName = "M_getInputNames"

    fn __init__(
        out self,
        ptr: CCompiledModel,
        length: Int,
        lib: DLHandle,
    ):
        self.names = TensorNames(
            Self.GetInputTensorNamesFnName, ptr, length, lib
        )

    fn __moveinit__(out self, owned existing: Self):
        self.names = existing.names^

    fn __getitem__(self, idx: Int) raises -> String:
        return self.names[idx]

    fn __len__(self) -> Int:
        return len(self.names)


struct OutputTensorNames(Sized):
    """Collection of model output names."""

    var names: TensorNames

    alias GetOutputTensorNamesFnName = "M_getOutputNames"

    fn __init__(
        out self,
        ptr: CCompiledModel,
        length: Int,
        lib: DLHandle,
    ):
        self.names = TensorNames(
            Self.GetOutputTensorNamesFnName, ptr, length, lib
        )

    fn __moveinit__(out self, owned existing: Self):
        self.names = existing.names^

    fn __getitem__(self, idx: Int) raises -> String:
        return self.names[idx]

    fn __len__(self) -> Int:
        return len(self.names)
