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
    """A user created counter."""

    var ptr: DTypePointer[DType.invalid]
    var lib: DLHandle
    # alias AddCounterVal = "M_addUInt64_Counter"
    alias AddCounterVal = "M_addDouble_Counter"

    # var lib: DLHandle
    fn __init__(
        inout self: Self, lib: DLHandle, ptr: DTypePointer[DType.invalid]
    ):
        """Creates a user defined counter.

        Args:
            lib: Handle to serving library.
            ptr: A pointer to the underlying counter instance.
        """
        self.ptr = ptr
        self.lib = lib

    fn __moveinit__(inout self: Self, owned existing: Self):
        """Move initializes a counter instance from an existing instance.

        Args:
            existing: The existing counter instanace.
        """
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn __copyinit__(inout self: Self, existing: Self):
        """[summary].

        Args:
            existing: [description].
        """
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn add(inout self: Self, val: Float64):
        """Increments the counter with a given value.

        Args:
            val: [description].
        """
        call_dylib_func[DTypePointer[DType.invalid]](
            self.lib, Self.AddCounterVal, self.ptr, val
        )


# TODO rashid - add a trait which can be implemented here
struct PrometheusMetricsEndPoint:
    """Represents a prometheus custom metrics reader.

    A prometheus metric reader can be used to export user defined
    custom metrics."""

    var ptr: DTypePointer[DType.invalid]
    var lib: DLHandle

    fn __init__(inout self: Self, end_point: String):
        """Initializes a prometheus custom metrics exporter at the specified
        end-point.

        Args:
            end_point: The end-point url to be used to query the custom metrics, for instance 'localhost:9464'.
        """
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
        """Destroys the prometheus end-point."""
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

    fn flush(self) -> NoneType:
        call_dylib_func(
            self.lib, Self.FlushTelemetryContextFnName, self.context
        )

    fn init_custom_metrics_prometheus_endpoint(
        self: Self, end_point: PrometheusMetricsEndPoint
    ) -> Bool:
        """Initializes the custom metrics end-point with a prometheus reader.

        Args:
            end_point: A PrometheusEndpoint instance to be used for custom metrics.

        Returns:
            True if the end-point was successfully set.
        """
        return call_dylib_func[Bool](
            self.lib, "M_initUserMetricsReader", self.context, end_point.ptr
        )

    fn create_counter(
        inout self: Self, name: String, desc: String, unit: String
    ) -> Counter:
        """Creates a custom counter.

        Args:
            name: The name of the counter as it will appear in the logs.
            desc: A description of the counter.
            unit: The unit of the counter.

        Returns:
            A Counter instance that can be used to update the values.
        """

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
