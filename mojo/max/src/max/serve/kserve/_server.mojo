# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe server types."""

from memory import UnsafePointer
from sys.ffi import DLHandle
from utils import StringRef

from max.engine._compilation import CCompiledModel
from max._utils import call_dylib_func, exchange

from .._batch import CBatch

# ===----------------------------------------------------------------------=== #
# CGRPCServer
# ===----------------------------------------------------------------------=== #


struct CGRPCServer:
    var _lib: DLHandle
    var _ptr: UnsafePointer[NoneType]

    alias _NewFnName = "M_newKServeServer"
    alias _FreeFnName = "M_freeKServeServer"

    alias _InitFnName = "M_kserveInit"
    alias _RunFnName = "M_kserveRun"
    alias _StopFnName = "M_kserveStopServer"
    alias _SignalStopFnName = "M_kserveSignalStopServer"

    alias _PopReadyFnName = "M_kservePopReady"

    fn __init__(inout self, lib: DLHandle, address: StringRef):
        self._lib = lib
        self._ptr = UnsafePointer[NoneType]()
        call_dylib_func(
            self._lib,
            Self._NewFnName,
            address,
            UnsafePointer.address_of(self._ptr),
        )

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[UnsafePointer[NoneType]](
            existing._ptr, UnsafePointer[NoneType]()
        )

    fn __del__(owned self):
        call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

    fn init(inout self, models: List[CCompiledModel]):
        call_dylib_func(
            self._lib, Self._InitFnName, self._ptr, models.data, len(models)
        )

    fn run(inout self):
        call_dylib_func(self._lib, Self._RunFnName, self._ptr)

    fn stop(inout self):
        call_dylib_func(self._lib, Self._StopFnName, self._ptr)

    fn signal_stop(inout self):
        call_dylib_func(self._lib, Self._SignalStopFnName, self._ptr)

    async fn pop_ready(inout self, inout batch: CBatch):
        await batch.load[Self._PopReadyFnName](self._ptr)
