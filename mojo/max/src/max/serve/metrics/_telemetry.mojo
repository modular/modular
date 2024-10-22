# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from sys.ffi import DLHandle
from collections import Optional, Dict
from max.engine import InferenceSession
from max.engine._status import Status
from max.engine._context import RuntimeContext, CRuntimeContext
from sys.param_env import is_defined
from max._utils import call_dylib_func, handle_from_config
from utils import StringRef


trait Instrument(Movable):
    fn __init__(inout self: Self, lib: DLHandle, ptr: UnsafePointer[NoneType]):
        ...

    @staticmethod
    fn _get_create_fn_name() -> StringRef:
        ...

    @staticmethod
    fn _get_delete_fn_name() -> StringRef:
        ...


@value
struct Counter[T: DType](Instrument):
    var ptr: UnsafePointer[NoneType]
    var lib: DLHandle

    @staticmethod
    fn _get_create_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_createDoubleCounter"
        else:
            return "M_createUInt64Counter"

    @staticmethod
    fn _get_delete_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_deleteDouble_Counter"
        else:
            return "M_deleteUInt64_Counter"

    @staticmethod
    fn _get_add_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_addDouble_Counter"
        else:
            return "M_addUInt64_Counter"

    fn __init__(inout self: Self, lib: DLHandle, ptr: UnsafePointer[NoneType]):
        """Creates a user defined counter.

        Args:
            lib: Handle to serving library.
            ptr: A pointer to the underlying counter instance.
        """
        constrained[
            T == DType.float64 or T == DType.uint64,
            "Type must be uint64 or float64.",
        ]()
        self.ptr = ptr
        self.lib = lib

    fn add[type: AnyTrivialRegType](inout self: Self, val: type):
        """Increments the counter with a given value.

        Args:
            val: [description].
        """
        call_dylib_func[UnsafePointer[NoneType]](
            self.lib, Self._get_add_fn_name(), self.ptr, val
        )

    fn __del__(owned self: Self):
        call_dylib_func[UnsafePointer[NoneType]](
            self.lib, Self._get_delete_fn_name(), self.ptr
        )


struct Histogram[T: DType](Instrument):
    var ptr: UnsafePointer[NoneType]
    var lib: DLHandle

    @staticmethod
    fn _get_create_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_createDoubleHistogram"
        else:
            return "M_createUInt64Histogram"

    @staticmethod
    fn _get_delete_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_deleteDouble_Histogram"
        else:
            return "M_deleteUInt64_Histogram"

    @staticmethod
    fn _get_record_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_recordDoubleHistogramVal"
        else:
            return "M_recordUintHistogramVal"

    fn __init__(inout self: Self, lib: DLHandle, ptr: UnsafePointer[NoneType]):
        constrained[
            T == DType.float64 or T == DType.uint64,
            "Type must be uint64 or float64.",
        ]()
        self.ptr = ptr
        self.lib = lib

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn __copyinit__(inout self: Self, existing: Self):
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn record[type: AnyTrivialRegType](inout self: Self, val: type):
        call_dylib_func[UnsafePointer[NoneType]](
            self.lib, Self._get_record_fn_name(), self.ptr, val
        )

    fn __del__(owned self: Self):
        call_dylib_func[UnsafePointer[NoneType]](
            self.lib, Self._get_delete_fn_name(), self.ptr
        )


struct Gauge[T: DType](Instrument):
    var ptr: UnsafePointer[NoneType]
    var lib: DLHandle

    @staticmethod
    fn _get_create_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_createDoubleGauge"
        else:
            return "M_createInt64Gauge"

    @staticmethod
    fn _get_delete_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_deleteDouble_Gauge"
        else:
            return "M_deleteInt64_Gauge"

    @staticmethod
    fn _get_add_fn_name() -> StringRef:
        if T == DType.float64:
            return "M_addDouble_Gauge"
        else:
            return "M_addInt64_Gauge"

    fn __init__(inout self: Self, lib: DLHandle, ptr: UnsafePointer[NoneType]):
        """Creates a user defined counter.

        Args:
            lib: Handle to serving library.
            ptr: A pointer to the underlying counter instance.
        """
        constrained[
            T == DType.float64 or T == DType.int64,
            "Type must be int64 or float64.",
        ]()
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

    fn add[type: AnyTrivialRegType](inout self: Self, val: type):
        """Increments the gauge with a given value.

        Args:
            val: [description].
        """

        call_dylib_func[UnsafePointer[NoneType]](
            self.lib, Self._get_add_fn_name(), self.ptr, val
        )

    fn __del__(owned self: Self):
        call_dylib_func[UnsafePointer[NoneType]](
            self.lib, Self._get_delete_fn_name(), self.ptr
        )


# TODO rashid - add a trait which can be implemented here
struct PrometheusMetricsEndPoint:
    """Represents a prometheus custom metrics reader.

    A prometheus metric reader can be used to export user defined
    custom metrics."""

    var ptr: UnsafePointer[NoneType]
    var lib: DLHandle

    fn __init__(inout self: Self, end_point: String):
        """Initializes a prometheus custom metrics exporter at the specified
        end-point.

        Args:
            end_point: The end-point url to be used to query the custom metrics, for instance 'localhost:9464'.
        """
        self.lib = handle_from_config("serving", ".serve_lib")
        self.ptr = UnsafePointer[NoneType]()
        call_dylib_func(
            self.lib,
            "M_createCustomMetricsPrometheus",
            end_point.unsafe_ptr(),
            UnsafePointer.address_of(self.ptr),
        )
        end_point._strref_keepalive()

    fn __moveinit__(inout self: Self, owned existing: Self):
        """Initializes a prometheus custom metrics exporter at the specified
        end-point.

        Args:
            existing: TODO.
        """
        self.lib = existing.lib
        self.ptr = existing.ptr

    fn __del__(owned self: Self):
        """Destroys the prometheus end-point."""
        call_dylib_func(
            self.lib,
            "M_freeCustomMetricsPrometheus",
            self.ptr,
        )


struct TelemetryContext:
    var context: UnsafePointer[NoneType]
    var lib: DLHandle

    alias FlushTelemetryContextFnName = "M_flushTelemetryContext"
    alias InitUserMetricsReader = "M_initUserMetricsReader"
    alias ClearUserMetricsReader = "M_clearUserMetricsReader"

    fn __init__(inout self, session: InferenceSession):
        self.context = session._ptr[].context.ptr.ptr
        self.lib = session._ptr[].context.lib

    fn __copyinit__(inout self, existing: TelemetryContext):
        self.context = existing.context
        self.lib = existing.lib

    fn flush(self) -> None:
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
            self.lib, Self.InitUserMetricsReader, self.context, end_point.ptr
        )

    fn clearCustomMetricsPrometheusEndpoint(self: Self) -> Bool:
        return call_dylib_func[Bool](
            self.lib, Self.ClearUserMetricsReader, self.context
        )

    fn create_counter[
        T: DType
    ](
        inout self: Self,
        name: String,
        desc: String,
        unit: String,
        attributes: Dict[String, String] = Dict[String, String](),
    ) -> Counter[T]:
        constrained[
            T == DType.float64 or T == DType.uint64,
            "Type must be uint64 or float64.",
        ]()
        """Creates a custom counter.

        Args:
            name: The name of the counter as it will appear in the logs.
            desc: A description of the counter.
            unit: The unit of the counter.

        Returns:
            A Counter instance that can be used to update the values.
        """
        var attrib_count = 0
        var addresses = List[UnsafePointer[UInt8]]()
        for k in attributes.items():
            addresses.append(k[].key.unsafe_ptr())
            addresses.append(k[].value.unsafe_ptr())

        attrib_count = len(addresses)
        var res: Counter[T]

        var ctr = call_dylib_func[UnsafePointer[NoneType]](
            self.lib,
            Counter[T]._get_create_fn_name(),
            self.context.address,
            name.unsafe_ptr(),
            desc.unsafe_ptr(),
            unit.unsafe_ptr(),
            attrib_count,
            addresses.data,
        )
        res = Counter[T](self.lib, ctr)

        return res

    fn create_gauge[
        T: DType
    ](
        inout self: Self,
        name: String,
        desc: String,
        unit: String,
        attributes: Dict[String, String] = Dict[String, String](),
    ) -> Gauge[T]:
        constrained[
            T == DType.float64 or T == DType.int64,
            "Type must be int64 or float64.",
        ]()
        """Creates a custom gauge.

        Args:
            name: The name of the gauge as it will appear in the logs.
            desc: A description of the gauge.
            unit: The unit of the gauge.

        Returns:
            A Gauge instance that can be used to update the values.
        """
        var attrib_count = 0
        var addresses = List[UnsafePointer[UInt8]]()
        for k in attributes.items():
            addresses.append(k[].key.unsafe_ptr())
            addresses.append(k[].value.unsafe_ptr())

        attrib_count = len(addresses)
        var res: Gauge[T]

        var ctr = call_dylib_func[UnsafePointer[NoneType]](
            self.lib,
            Gauge[T]._get_create_fn_name(),
            self.context.address,
            name.unsafe_ptr(),
            desc.unsafe_ptr(),
            unit.unsafe_ptr(),
            attrib_count,
            addresses.data,
        )
        res = Gauge[T](self.lib, ctr)

        return res

    fn create_histogram[
        T: DType
    ](
        inout self: Self,
        name: String,
        desc: String,
        unit: String,
        attributes: Dict[String, String] = Dict[String, String](),
    ) -> Histogram[T]:
        constrained[
            T == DType.float64 or T == DType.uint64,
            "Type must be uint64 or float64.",
        ]()
        var attrib_count = 0
        var addresses = List[UnsafePointer[UInt8]]()
        for k in attributes.items():
            addresses.append(k[].key.unsafe_ptr())
            addresses.append(k[].value.unsafe_ptr())

        attrib_count = len(addresses)
        var res: Histogram[T]

        var ctr = call_dylib_func[UnsafePointer[NoneType]](
            self.lib,
            Histogram[T]._get_create_fn_name(),
            self.context.address,
            name.unsafe_ptr(),
            desc.unsafe_ptr(),
            unit.unsafe_ptr(),
            attrib_count,
            addresses.data,
        )
        res = Histogram[T](self.lib, ctr)

        return res

    fn create_instrument[
        T: Instrument
    ](
        inout self: Self,
        name: String,
        desc: String,
        unit: String,
        attributes: Dict[String, String] = Dict[String, String](),
    ) -> T:
        var attrib_count = 0
        var addresses = List[UnsafePointer[UInt8]]()
        for k in attributes.items():
            addresses.append(k[].key.unsafe_ptr())
            addresses.append(k[].value.unsafe_ptr())

        attrib_count = len(addresses)

        var ctr = call_dylib_func[UnsafePointer[NoneType]](
            self.lib,
            T._get_create_fn_name(),
            self.context.address,
            name.unsafe_ptr(),
            desc.unsafe_ptr(),
            unit.unsafe_ptr(),
            attrib_count,
            addresses.data,
        )
        var res = T(self.lib, ctr)

        return res^
