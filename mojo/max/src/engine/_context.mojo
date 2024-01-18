# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from utils._optional import Optional
from ._utils import *
from ._status import Status
from sys.param_env import is_defined

alias MODULAR_RELEASE_PACKAGE_BUILD = is_defined[
    "MODULAR_RELEASE_PACKAGE_BUILD"
]()


@value
@register_passable("trivial")
struct CRuntimeConfig:
    var ptr: DTypePointer[DType.invalid]

    alias FreeRuntimeConfigFnName = "M_freeRuntimeConfig"
    alias SetNumThreadsFnName = "M_setNumThreads"
    alias SetDeviceFnName = "M_setDevice"

    fn set_num_threads(self, lib: DLHandle, num_threads: Int):
        call_dylib_func(lib, Self.SetNumThreadsFnName, self.ptr, num_threads)

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeConfigFnName, self)

    fn set_device(self, borrowed lib: DLHandle, device: StringRef):
        call_dylib_func(lib, Self.SetDeviceFnName, self, device.data, 0)


struct RuntimeConfig:
    var ptr: CRuntimeConfig
    var lib: DLHandle

    alias NewRuntimeConfigFnName = "M_newRuntimeConfig"

    fn __init__(
        inout self,
        lib: DLHandle,
        device: StringRef = "cpu",
        num_threads: Optional[Int] = None,
    ):
        self.ptr = call_dylib_func[CRuntimeConfig](
            lib, Self.NewRuntimeConfigFnName
        )
        if num_threads:
            self.ptr.set_num_threads(lib, num_threads.value())
        self.lib = lib

        @parameter
        if MODULAR_RELEASE_PACKAGE_BUILD:
            if device != "cpu":
                print(
                    "The device",
                    device,
                    "is not valid. The device must be set to 'cpu'.",
                )
            return
        else:
            if device == "cuda":
                self.ptr.set_device(self.lib, device)

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CRuntimeConfig](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeConfig:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)


@value
@register_passable("trivial")
struct CRuntimeContext:
    var ptr: DTypePointer[DType.invalid]

    alias FreeRuntimeContextFnName = "M_freeRuntimeContext"

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeContextFnName, self)


struct RuntimeContext:
    var ptr: CRuntimeContext
    var lib: DLHandle

    alias NewRuntimeContextFnName = "M_newRuntimeContext"

    fn __init__(inout self, owned config: RuntimeConfig, lib: DLHandle):
        let status = Status(lib)
        self.ptr = call_dylib_func[CRuntimeContext](
            lib,
            Self.NewRuntimeContextFnName,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            print(status.__str__())
            self.ptr = DTypePointer[DType.invalid]()
        _ = config ^
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CRuntimeContext](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeContext:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
