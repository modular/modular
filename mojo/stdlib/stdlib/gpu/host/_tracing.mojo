# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.string import StaticString
from os import abort
from pathlib import Path
from sys import (
    has_accelerator,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    sizeof,
)
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle, _try_find_dylib
from sys.param_env import env_get_int

from memory import UnsafePointer, stack_allocation

from utils.variant import Variant

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_NVTX_LIBRARY_PATHS = List[Path](
    "libnvToolsExt.so",
    "/usr/local/cuda/lib64/libnvToolsExt.so",
    "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1",
)
alias ROCTX_LIBRARY_PATHS = List[Path](
    "librocprofiler-sdk-roctx.so",
    "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
)

alias LIBRARY_PATHS = CUDA_NVTX_LIBRARY_PATHS if has_nvidia_gpu_accelerator() else ROCTX_LIBRARY_PATHS


alias _TraceType_OTHER = 0
alias _TraceType_ASYNCRT = 1
alias _TraceType_MEM = 2
alias _TraceType_KERNEL = 3
alias _TraceType_MAX = 4


@always_inline
fn _setup_category(
    name_category: fn (UInt32, UnsafePointer[UInt8]) -> NoneType,
    value: Int,
    name: StaticString,
):
    name_category(value, name.unsafe_ptr())


fn _setup_categories(
    name_category: fn (UInt32, UnsafePointer[UInt8]) -> NoneType
):
    _setup_category(name_category, _TraceType_OTHER, "Other")
    _setup_category(name_category, _TraceType_ASYNCRT, "AsyncRT")
    _setup_category(name_category, _TraceType_MEM, "Memory")
    _setup_category(name_category, _TraceType_KERNEL, "Kernel")
    _setup_category(name_category, _TraceType_MAX, "Max")


alias GPU_TRACING_LIBRARY = _Global[
    "GPU_TRACING_LIBRARY", _OwnedDLHandle, _init_dylib
]()


fn _init_dylib() -> _OwnedDLHandle:
    @parameter
    if _is_disabled():
        return abort[_OwnedDLHandle]("cannot load dylib when disabled")

    try:
        var dylib = _try_find_dylib["GPU tracing library"](LIBRARY_PATHS)

        @parameter
        if has_nvidia_gpu_accelerator():
            _setup_categories(
                dylib._handle.get_function[
                    fn (UInt32, UnsafePointer[UInt8]) -> NoneType
                ]("nvtxNameCategoryA")
            )

        return dylib^
    except e:
        var msg = String(e, "\n")

        @parameter
        if has_nvidia_gpu_accelerator():
            msg += " please install the cuda toolkit. "
            msg += "In apt-get this can be done with "
            msg += (
                "`sudo apt-get -y install cuda-toolkit-XX-Y`. Where XX and Y "
            )
            msg += "are the major and minor versions."
        else:
            msg += " please install ROCprofiler."

        return abort[_OwnedDLHandle](msg)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        GPU_TRACING_LIBRARY,
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#


alias RangeID = UInt64
alias EventPayload = UInt64
alias NVTXVersion = 2


@fieldwise_init
@register_passable("trivial")
struct Color(Intable):
    var _value: Int

    alias FORMAT = 1  # ARGB
    alias MODULAR_MAX = Self(0xB5BAF5)
    alias BLUE = Self(0x0000FF)
    alias GREEN = Self(0x008000)
    alias ORANGE = Self(0xFFA500)
    alias PURPLE = Self(0x800080)
    alias RED = Self(0xFF0000)
    alias WHITE = Self(0xFFFFFF)
    alias YELLOW = Self(0xFFFF00)

    fn __int__(self) -> Int:
        return self._value


@fieldwise_init
@register_passable("trivial")
struct _C_EventAttributes:
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

    var message: UnsafePointer[UInt8]
    """Message assigned to this attribute structure."""


@always_inline
fn color_from_category(category: Int) -> Color:
    if category == _TraceType_MAX:
        return Color.MODULAR_MAX
    if category == _TraceType_KERNEL:
        return Color.GREEN
    if category == _TraceType_ASYNCRT:
        return Color.ORANGE
    if category == _TraceType_MEM:
        return Color.RED
    return Color.PURPLE


@register_passable("trivial")
struct EventAttributes:
    var _value: _C_EventAttributes

    @always_inline
    fn __init__(
        out self,
        *,
        message: String = "",
        category: Int = _TraceType_MAX,
        color: Optional[Color] = None,
    ):
        alias ASCII = 1
        var resolved_color: Color
        if color:
            resolved_color = color.value()
        else:
            resolved_color = color_from_category(category)
        self._value = _C_EventAttributes(
            version=NVTXVersion,
            size=sizeof[_C_EventAttributes](),
            category=category,
            color_type=Color.FORMAT,
            color=Int(resolved_color),
            payload_type=0,
            _reserved=0,
            event_payload=0,
            message_type=ASCII,
            message=message.unsafe_ptr(),
        )


@register_passable("trivial")
struct _dylib_function[fn_name: StaticString, type: AnyTrivialRegType]:
    alias fn_type = type

    @staticmethod
    fn load() -> type:
        return _get_dylib_function[fn_name, type]()


# ===-----------------------------------------------------------------------===#
# NVTX Bindings
# ===-----------------------------------------------------------------------===#

# NVTX_DECLSPEC void NVTX_API nvtxMarkEx(const nvtxEventAttributes_t* eventAttrib);
alias _nvtxMarkEx = _dylib_function[
    "nvtxMarkEx", fn (UnsafePointer[_C_EventAttributes]) -> NoneType
]

# NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartEx(const nvtxEventAttributes_t* eventAttrib);
alias _nvtxRangeStartEx = _dylib_function[
    "nvtxRangeStartEx", fn (UnsafePointer[_C_EventAttributes]) -> RangeID
]

# NVTX_DECLSPEC void NVTX_API nvtxRangeEnd(nvtxRangeId_t id);
alias _nvtxRangeEnd = _dylib_function["nvtxRangeEnd", fn (RangeID) -> NoneType]

# NVTX_DECLSPEC int NVTX_API nvtxRangePushEx(const nvtxEventAttributes_t* eventAttrib);
alias _nvtxRangePushEx = _dylib_function[
    "nvtxRangePushEx", fn (UnsafePointer[_C_EventAttributes]) -> Int32
]

# NVTX_DECLSPEC int NVTX_API nvtxRangePop(void);
alias _nvtxRangePop = _dylib_function["nvtxRangePop", fn () -> Int32]


# ===-----------------------------------------------------------------------===#
# ROCTX Bindings
# ===-----------------------------------------------------------------------===#

# ROCTX_API void roctxMarkA(const char* message) ROCTX_VERSION_4_1;
alias _roctxMarkA = _dylib_function[
    "roctxMarkA", fn (UnsafePointer[UInt8]) -> NoneType
]

# ROCTX_API int roctxRangePushA(const char* message) ROCTX_VERSION_4_1;
alias _roctxRangePushA = _dylib_function[
    "roctxRangePushA", fn (UnsafePointer[UInt8]) -> Int32
]

# ROCTX_API int roctxRangePop() ROCTX_VERSION_4_1;
alias _roctxRangePop = _dylib_function["roctxRangePop", fn () -> Int32]
# ROCTX_API roctx_range_id_t roctxRangeStartA(const char* message)
alias _roctxRangeStartA = _dylib_function[
    "roctxRangeStartA", fn (UnsafePointer[UInt8]) -> RangeID
]

# ROCTX_API void roctxRangeStop(roctx_range_id_t id) ROCTX_VERSION_4_1;
alias _roctxRangeStop = _dylib_function[
    "roctxRangeStop", fn (RangeID) -> NoneType
]

# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


struct _Mark:
    var _fn: Variant[_nvtxMarkEx.fn_type, _roctxMarkA.fn_type]

    fn __init__(out self):
        @parameter
        if has_nvidia_gpu_accelerator():
            self._fn = _nvtxMarkEx.load()
        else:
            self._fn = _roctxMarkA.load()

    fn __call__(self, val: UnsafePointer[_C_EventAttributes]):
        constrained[has_nvidia_gpu_accelerator()]()
        self._fn[_nvtxMarkEx.fn_type](val)

    fn __call__(self, val: UnsafePointer[UInt8]):
        constrained[has_amd_gpu_accelerator()]()
        self._fn[_roctxMarkA.fn_type](val)


struct _RangeStart:
    var _fn: Variant[_nvtxRangeStartEx.fn_type, _roctxRangeStartA.fn_type]

    fn __init__(out self):
        @parameter
        if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangeStartEx.load()
        else:
            self._fn = _roctxRangeStartA.load()

    fn __call__(self, val: UnsafePointer[_C_EventAttributes]) -> RangeID:
        constrained[has_nvidia_gpu_accelerator()]()
        return self._fn[_nvtxRangeStartEx.fn_type](val)

    fn __call__(self, val: UnsafePointer[UInt8]) -> RangeID:
        constrained[has_amd_gpu_accelerator()]()
        return self._fn[_roctxRangeStartA.fn_type](val)


struct _RangeEnd:
    var _fn: fn (RangeID) -> NoneType

    fn __init__(out self):
        @parameter
        if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangeEnd.load()
        else:
            self._fn = _roctxRangeStop.load()

    fn __call__(self, val: RangeID):
        self._fn(val)


struct _RangePush:
    var _fn: Variant[_nvtxRangePushEx.fn_type, _roctxRangePushA.fn_type]

    fn __init__(out self):
        @parameter
        if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangePushEx.load()
        else:
            self._fn = _roctxRangePushA.load()

    fn __call__(self, val: UnsafePointer[_C_EventAttributes]) -> Int32:
        constrained[has_nvidia_gpu_accelerator()]()
        return self._fn[_nvtxRangePushEx.fn_type](val)

    fn __call__(self, val: UnsafePointer[UInt8]) -> Int32:
        constrained[has_amd_gpu_accelerator()]()
        return self._fn[_roctxRangePushA.fn_type](val)


struct _RangePop:
    var _fn: _nvtxRangePop.fn_type

    fn __init__(out self):
        @parameter
        if has_nvidia_gpu_accelerator():
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
) -> RangeID:
    @parameter
    if _is_disabled():
        return 0

    @parameter
    if has_nvidia_gpu_accelerator():
        var info = EventAttributes(
            message=message, color=color, category=category
        )
        return _RangeStart()(UnsafePointer(to=info._value))
    else:
        return _RangeStart()(message.unsafe_ptr())


@always_inline
fn _end_range(id: RangeID):
    @parameter
    if _is_disabled():
        return
    _RangeEnd()(id)


@always_inline
fn _mark(
    *,
    message: String = "",
    color: Optional[Color] = None,
    category: Int = _TraceType_MAX,
):
    @parameter
    if _is_disabled():
        return

    @parameter
    if has_nvidia_gpu_accelerator():
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
    ):
        constrained[_is_enabled(), "GPU tracing must be enabled"]()
        self._info = EventAttributes(
            message=message, color=color, category=category
        )
        self._id = 0
        self._start_fn = _RangeStart()
        self._end_fn = _RangeEnd()

    @always_inline
    fn __enter__(mut self):
        @parameter
        if has_nvidia_gpu_accelerator():
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
    ):
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
    ):
        constrained[_is_enabled(), "GPU tracing must be enabled"]()
        self._info = EventAttributes(
            message=message, color=color, category=category
        )
        self._push_fn = _RangePush()
        self._pop_fn = _RangePop()

    @always_inline
    fn __enter__(mut self):
        @parameter
        if has_nvidia_gpu_accelerator():
            _ = self._push_fn(UnsafePointer(to=self._info._value))
        else:
            _ = self._push_fn(self._info._value.message)

    @always_inline
    fn __exit__(self):
        _ = self._pop_fn()
