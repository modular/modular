# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from collections.optional import Optional
from max.engine import InferenceSession
from max.engine._status import Status
from max.engine._context import RuntimeContext, CRuntimeContext
from sys.param_env import is_defined
from max.engine._utils import (
    call_dylib_func,
    handle_from_config,
)


struct Counter:
    var ptr: DTypePointer[DType.invalid]
    var lib: DLHandle
    # alias AddCounterVal = "M_addUInt64_Counter"
    alias AddCounterVal = "M_addDouble_Counter"

    # var lib: DLHandle
    fn __init__(
        inout self: Self, lib: DLHandle, ptr: DTypePointer[DType.invalid]
    ):
        self.ptr = ptr
        self.lib = lib

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn __copyinit__(inout self: Self, existing: Self):
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn add(inout self: Self, val: Float64):
        call_dylib_func[DTypePointer[DType.invalid]](
            self.lib, Self.AddCounterVal, self.ptr, val
        )


# TODO rashid - add a trait which can be implemented here
struct PrometheusMetricsEndPoint:
    var ptr: DTypePointer[DType.invalid]
    var lib: DLHandle

    fn __init__(inout self: Self, end_point: String):
        self.lib = handle_from_config("serving", ".serve_lib")
        self.ptr = DTypePointer[DType.invalid]()
        var endpoint_ref = end_point._strref_dangerous()
        self.ptr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib,
            "M_createCustomMetricsPrometheus",
            endpoint_ref.data,
        )
        end_point._strref_keepalive()

    fn __del__(owned self: Self):
        call_dylib_func[DTypePointer[DType.invalid]](
            self.lib,
            "M_freeCustomMetricsPrometheus",
            self.ptr,
        )


struct TelemetryContext:
    var context: DTypePointer[DType.invalid]
    var lib: DLHandle
    alias CreateUIntCounterFnName = "M_createUInt64Counter"
    alias CreateDoubleCounterFnName = "M_createDoubleCounter"
    alias FlushTelemetryContextFnName = "M_flushTelemetryContext"

    fn __init__(inout self, session: InferenceSession):
        self.context = session._ptr[].context.ptr.ptr
        self.lib = session._ptr[].context.lib

    fn __copyinit__(inout self, existing: TelemetryContext):
        self.context = existing.context
        self.lib = existing.lib

    fn borrow_ptr(self) -> DTypePointer[DType.invalid]:
        return self.context

    fn flush(self) -> NoneType:
        call_dylib_func(
            self.lib, Self.FlushTelemetryContextFnName, self.context
        )

    fn initCustomMetricsPrometheusEndpoint(
        self: Self, end_point: PrometheusMetricsEndPoint
    ) -> Bool:
        return call_dylib_func[Bool](
            self.lib, "M_initUserMetricsReader", self.context, end_point.ptr
        )

    fn create_counter(
        inout self: Self, name: String, desc: String, unit: String
    ) -> Counter:
        var name_ref = name._strref_dangerous()
        var desc_ref = desc._strref_dangerous()
        var unit_ref = unit._strref_dangerous()
        var ctr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib,
            Self.CreateDoubleCounterFnName,
            self.context.address,
            name_ref.data,
            desc_ref.data,
            unit_ref.data,
        )
        name._strref_keepalive()
        desc._strref_keepalive()
        unit._strref_keepalive()
        var res = Counter(self.lib, ctr)
        return res
