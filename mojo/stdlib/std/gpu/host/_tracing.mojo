# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from os import abort
from pathlib import Path
from sys import (
    has_accelerator,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from ffi import _get_dylib_function as _ffi_get_dylib_function
from ffi import _Global, OwnedDLHandle, _try_find_dylib
from sys.param_env import env_get_int

from utils.variant import Variant

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_NVTX_LIBRARY_PATHS: List[Path] = [
    "libnvToolsExt.so",
    "/usr/local/cuda/lib64/libnvToolsExt.so",
    "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1",
]
comptime ROCTX_LIBRARY_PATHS: List[Path] = [
    "librocprofiler-sdk-roctx.so",
    "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
]

comptime LIBRARY_PATHS = CUDA_NVTX_LIBRARY_PATHS if has_nvidia_gpu_accelerator() else ROCTX_LIBRARY_PATHS


comptime _TraceType_OTHER = 0
comptime _TraceType_ASYNCRT = 1
comptime _TraceType_MEM = 2
comptime _TraceType_KERNEL = 3
comptime _TraceType_MAX = 4


@always_inline
fn _setup_category(
    name_category: fn(UInt32, UnsafePointer[UInt8, ImmutAnyOrigin]) -> NoneType,
    value: Int,
    name: StaticString,
):
    name_category(UInt32(value), name.unsafe_ptr())


fn _setup_categories(
    name_category: fn(UInt32, UnsafePointer[UInt8, ImmutAnyOrigin]) -> NoneType
):
    _setup_category(name_category, _TraceType_OTHER, "Other")
    _setup_category(name_category, _TraceType_ASYNCRT, "AsyncRT")
    _setup_category(name_category, _TraceType_MEM, "Memory")
    _setup_category(name_category, _TraceType_KERNEL, "Kernel")
    _setup_category(name_category, _TraceType_MAX, "Max")


fn _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the GPU Tracing libraries. Please make sure that "
            "the library path is correctly set in one of the following paths ["
        ),
        ", ".join(materialize[LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime GPU_TRACING_LIBRARY = _Global[
    "GPU_TRACING_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]()


fn _init_dylib() -> OwnedDLHandle:
    comptime if _is_disabled():
        abort("cannot load dylib when disabled")

    try:
        var dylib = _try_find_dylib["GPU tracing library"](
            materialize[LIBRARY_PATHS]()
        )

        comptime if has_nvidia_gpu_accelerator():
            _setup_categories(
                dylib._handle.get_function[
                    fn(UInt32, UnsafePointer[UInt8, ImmutAnyOrigin]) -> NoneType
                ]("nvtxNameCategoryA")
            )

        return dylib^
    except e:
        return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: __TypeOfAllTypes
]() raises -> result_type:
    return _ffi_get_dylib_function[
        GPU_TRACING_LIBRARY,
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#


comptime RangeID = UInt64
comptime EventPayload = UInt64
comptime NVTXVersion = 2


@fieldwise_init
struct Color(Intable, TrivialRegisterPassable):
    var _value: Int

    comptime FORMAT = 1  # ARGB
    comptime MODULAR_PURPLE = Self(0xB5BAF5)
    comptime BLUE = Self(0x0000FF)
    comptime GREEN = Self(0x008000)
    comptime ORANGE = Self(0xFFA500)
    comptime PURPLE = Self(0x800080)
    comptime RED = Self(0xFF0000)
    comptime WHITE = Self(0xFFFFFF)
    comptime YELLOW = Self(0xFFFF00)

    fn __init__(out self, colorname: StaticString):
        """Initialize Color from a StaticString color name.

        Args:
            colorname: The name of the color to use.
        """
        if colorname == "modular_purple":
            self = Color.MODULAR_PURPLE
        elif colorname == "blue":
            self = Color.BLUE
        elif colorname == "green":
            self = Color.GREEN
        elif colorname == "orange":
            self = Color.ORANGE
        elif colorname == "purple":
            self = Color.PURPLE
        elif colorname == "red":
            self = Color.RED
        elif colorname == "white":
            self = Color.WHITE
        elif colorname == "yellow":
            self = Color.YELLOW
        else:
            # Default to MODULAR_PURPLE for unknown color names
            self = Color.MODULAR_PURPLE

    fn __int__(self) -> Int:
        return self._value


@fieldwise_init
struct _C_EventAttributes(TrivialRegisterPassable):
    var version: UInt16
    """Version flag of the structure."""

    var size: UInt16
    """Size of the structure."""

    var category: UInt32
    """ID of the category the event is assigned to."""

    var color_type: Int32
    """Color type specified in this attribute structure."""

    var color: UInt32
    """Color assigned to this event."""

    var payload_type: Int32
    """Payload type specified in this attribute structure."""

    var _reserved: Int32
    """Reserved."""

    var event_payload: EventPayload
    """Payload assigned to this event."""

    var message_type: Int32
    """Message type specified in this attribute structure."""

    var message: UnsafePointer[UInt8, ImmutAnyOrigin]
    """Message assigned to this attribute structure."""


@always_inline
fn color_from_category(category: Int) -> Color:
    if category == _TraceType_MAX:
        return Color.MODULAR_PURPLE
    if category == _TraceType_KERNEL:
        return Color.GREEN
    if category == _TraceType_ASYNCRT:
        return Color.ORANGE
    if category == _TraceType_MEM:
        return Color.RED
    return Color.PURPLE


struct EventAttributes(TrivialRegisterPassable):
    var _value: _C_EventAttributes

    @always_inline
    fn __init__(
        out self,
        *,
        message: String = "",
        category: Int = _TraceType_MAX,
        color: Optional[Color] = None,
    ):
        comptime ASCII = 1
        var resolved_color: Color
        if color:
            resolved_color = color.value()
        else:
            resolved_color = color_from_category(category)
        self._value = _C_EventAttributes(
            version=NVTXVersion,
            size=UInt16(size_of[_C_EventAttributes]()),
            category=UInt32(category),
            color_type=Color.FORMAT,
            color=UInt32(Int(resolved_color)),
            payload_type=0,
            _reserved=0,
            event_payload=0,
            message_type=ASCII,
            message=message.unsafe_ptr(),
        )


struct _dylib_function[fn_name: StaticString, type: __TypeOfAllTypes](
    TrivialRegisterPassable
):
    comptime fn_type = Self.type

    @staticmethod
    fn load() raises -> Self.type:
        return _get_dylib_function[Self.fn_name, Self.type]()


# ===-----------------------------------------------------------------------===#
# NVTX Bindings
# ===-----------------------------------------------------------------------===#

# NVTX_DECLSPEC void NVTX_API nvtxMarkEx(const nvtxEventAttributes_t* eventAttrib);
comptime _nvtxMarkEx = _dylib_function[
    "nvtxMarkEx",
    fn(UnsafePointer[_C_EventAttributes, ImmutAnyOrigin]) -> NoneType,
]

# NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartEx(const nvtxEventAttributes_t* eventAttrib);
comptime _nvtxRangeStartEx = _dylib_function[
    "nvtxRangeStartEx",
    fn(UnsafePointer[_C_EventAttributes, ImmutAnyOrigin]) -> RangeID,
]

# NVTX_DECLSPEC void NVTX_API nvtxRangeEnd(nvtxRangeId_t id);
comptime _nvtxRangeEnd = _dylib_function[
    "nvtxRangeEnd", fn(RangeID) -> NoneType
]

# NVTX_DECLSPEC int NVTX_API nvtxRangePushEx(const nvtxEventAttributes_t* eventAttrib);
comptime _nvtxRangePushEx = _dylib_function[
    "nvtxRangePushEx",
    fn(UnsafePointer[_C_EventAttributes, ImmutAnyOrigin]) -> Int32,
]

# NVTX_DECLSPEC int NVTX_API nvtxRangePop(void);
comptime _nvtxRangePop = _dylib_function["nvtxRangePop", fn() -> Int32]


# ===-----------------------------------------------------------------------===#
# ROCTX Bindings
# ===-----------------------------------------------------------------------===#

# ROCTX_API void roctxMarkA(const char* message) ROCTX_VERSION_4_1;
comptime _roctxMarkA = _dylib_function[
    "roctxMarkA", fn(UnsafePointer[UInt8, ImmutAnyOrigin]) -> NoneType
]

# ROCTX_API int roctxRangePushA(const char* message) ROCTX_VERSION_4_1;
comptime _roctxRangePushA = _dylib_function[
    "roctxRangePushA", fn(UnsafePointer[UInt8, ImmutAnyOrigin]) -> Int32
]

# ROCTX_API int roctxRangePop() ROCTX_VERSION_4_1;
comptime _roctxRangePop = _dylib_function["roctxRangePop", fn() -> Int32]
# ROCTX_API roctx_range_id_t roctxRangeStartA(const char* message)
comptime _roctxRangeStartA = _dylib_function[
    "roctxRangeStartA", fn(UnsafePointer[UInt8, ImmutAnyOrigin]) -> RangeID
]

# ROCTX_API void roctxRangeStop(roctx_range_id_t id) ROCTX_VERSION_4_1;
comptime _roctxRangeStop = _dylib_function[
    "roctxRangeStop", fn(RangeID) -> NoneType
]

# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


struct _Mark:
    var _fn: Variant[_nvtxMarkEx.fn_type, _roctxMarkA.fn_type]

    fn __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxMarkEx.load()
        else:
            self._fn = _roctxMarkA.load()

    fn __call__(self, val: UnsafePointer[_C_EventAttributes, ImmutAnyOrigin]):
        comptime assert has_nvidia_gpu_accelerator()
        self._fn[_nvtxMarkEx.fn_type](val)

    fn __call__(self, val: UnsafePointer[UInt8, ImmutAnyOrigin]):
        comptime assert has_amd_gpu_accelerator()
        self._fn[_roctxMarkA.fn_type](val)


struct _RangeStart:
    var _fn: Variant[_nvtxRangeStartEx.fn_type, _roctxRangeStartA.fn_type]

    fn __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangeStartEx.load()
        else:
            self._fn = _roctxRangeStartA.load()

    fn __call__(
        self, val: UnsafePointer[_C_EventAttributes, ImmutAnyOrigin]
    ) -> RangeID:
        comptime assert has_nvidia_gpu_accelerator()
        return self._fn[_nvtxRangeStartEx.fn_type](val)

    fn __call__(self, val: UnsafePointer[UInt8, ImmutAnyOrigin]) -> RangeID:
        comptime assert has_amd_gpu_accelerator()
        return self._fn[_roctxRangeStartA.fn_type](val)


struct _RangeEnd:
    var _fn: fn(RangeID) -> NoneType

    fn __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangeEnd.load()
        else:
            self._fn = _roctxRangeStop.load()

    fn __call__(self, val: RangeID):
        self._fn(val)


struct _RangePush:
    var _fn: Variant[_nvtxRangePushEx.fn_type, _roctxRangePushA.fn_type]

    fn __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangePushEx.load()
        else:
            self._fn = _roctxRangePushA.load()

    fn __call__(
        self, val: UnsafePointer[_C_EventAttributes, ImmutAnyOrigin]
    ) -> Int32:
        comptime assert has_nvidia_gpu_accelerator()
        return self._fn[_nvtxRangePushEx.fn_type](val)

    fn __call__(self, val: UnsafePointer[UInt8, ImmutAnyOrigin]) -> Int32:
        comptime assert has_amd_gpu_accelerator()
        return self._fn[_roctxRangePushA.fn_type](val)


struct _RangePop:
    var _fn: _nvtxRangePop.fn_type

    fn __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangePop.load()
        else:
            self._fn = _roctxRangePop.load()

    fn __call__(self) -> Int32:
        return self._fn()


# ===-----------------------------------------------------------------------===#
# Functions
# ===-----------------------------------------------------------------------===#


fn _is_enabled_details() -> Bool:
    return (
        has_accelerator()
        and env_get_int["MODULAR_ENABLE_GPU_PROFILING_DETAILED", 0]() == 1
    )


fn _is_enabled() -> Bool:
    return has_accelerator() and (
        env_get_int["MODULAR_ENABLE_GPU_PROFILING", 0]() == 1
        or _is_enabled_details()
    )


fn _is_disabled() -> Bool:
    return not _is_enabled()


@always_inline
fn _start_range(
    *,
    message: String = "",
    category: Int = _TraceType_MAX,
    color: Optional[Color] = None,
) raises -> RangeID:
    comptime if _is_disabled():
        return 0

    comptime if has_nvidia_gpu_accelerator():
        var info = EventAttributes(
            message=message, color=color, category=category
        )
        return _RangeStart()(UnsafePointer(to=info._value))
    else:
        return _RangeStart()(message.unsafe_ptr())


@always_inline
fn _end_range(id: RangeID) raises:
    comptime if _is_disabled():
        return
    _RangeEnd()(id)


@always_inline
fn _mark(
    *,
    message: String = "",
    color: Optional[Color] = None,
    category: Int = _TraceType_MAX,
) raises:
    comptime if _is_disabled():
        return

    comptime if has_nvidia_gpu_accelerator():
        var info = EventAttributes(
            message=message, color=color, category=category
        )
        _Mark()(UnsafePointer(to=info._value))
    else:
        _Mark()(message.unsafe_ptr())


struct Range:
    var _info: EventAttributes
    var _id: RangeID

    var _start_fn: _RangeStart
    var _end_fn: _RangeEnd

    fn __init__(
        out self,
        *,
        message: String = "",
        color: Optional[Color] = None,
        category: Int = _TraceType_MAX,
    ) raises:
        comptime assert _is_enabled(), "GPU tracing must be enabled"
        self._info = EventAttributes(
            message=message, color=color, category=category
        )
        self._id = 0
        self._start_fn = _RangeStart()
        self._end_fn = _RangeEnd()

    @always_inline
    fn __enter__(mut self):
        comptime if has_nvidia_gpu_accelerator():
            self._id = self._start_fn(UnsafePointer(to=self._info._value))
        else:
            self._id = self._start_fn(self._info._value.message)

    @always_inline
    fn __exit__(self):
        self._end_fn(self._id)

    @always_inline
    fn id(self) -> RangeID:
        return self._id

    @staticmethod
    @always_inline
    fn mark(
        *,
        message: String = "",
        color: Optional[Color] = None,
        category: Int = _TraceType_MAX,
    ) raises:
        _mark(message=message, color=color)


struct RangeStack:
    var _info: EventAttributes

    var _push_fn: _RangePush
    var _pop_fn: _RangePop

    fn __init__(
        out self,
        *,
        message: String = "",
        color: Optional[Color] = None,
        category: Int = _TraceType_MAX,
    ) raises:
        comptime assert _is_enabled(), "GPU tracing must be enabled"
        self._info = EventAttributes(
            message=message, color=color, category=category
        )
        self._push_fn = _RangePush()
        self._pop_fn = _RangePop()

    @always_inline
    fn __enter__(mut self):
        comptime if has_nvidia_gpu_accelerator():
            _ = self._push_fn(UnsafePointer(to=self._info._value))
        else:
            _ = self._push_fn(self._info._value.message)

    @always_inline
    fn __exit__(self):
        _ = self._pop_fn()
