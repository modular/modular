# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List, Optional
from os import abort
from pathlib import Path
from sys.ffi import c_char, DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from memory import UnsafePointer, stack_allocation
from sys.param_env import env_get_int

# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#

alias CUDA_NVTX_LIBRARY_PATH = "/usr/local/cuda/lib64/libnvToolsExt.so"


alias _TraceType_OTHER = 0
alias _TraceType_ASYNCRT = 1
alias _TraceType_MEM = 2
alias _TraceType_KERNEL = 3
alias _TraceType_MAX = 4


@always_inline
fn _setup_category(
    name_category: fn (UInt32, UnsafePointer[UInt8]) -> NoneType,
    value: Int,
    name: StringLiteral,
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


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    @parameter
    if _is_disabled():
        return UnsafePointer[NoneType]()

    if not Path(CUDA_NVTX_LIBRARY_PATH).exists():
        return abort[UnsafePointer[NoneType]](
            "the CUDA NVTX library was not found at " + CUDA_NVTX_LIBRARY_PATH
        )
    var ptr = UnsafePointer[DLHandle].alloc(1)
    ptr.init_pointee_move(DLHandle(CUDA_NVTX_LIBRARY_PATH))
    _setup_categories(
        ptr[].get_function[fn (UInt32, UnsafePointer[UInt8]) -> NoneType](
            "nvtxNameCategoryA"
        )
    )
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    @parameter
    if _is_disabled():
        return None

    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        "CUDA_NVTX_LIBRARY",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


# ===----------------------------------------------------------------------===#
# Types
# ===----------------------------------------------------------------------===#


alias RangeID = UInt64
alias EventPayload = UInt64
alias NVTXVersion = 2


@value
struct Color:
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


@value
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


@register_passable
struct EventAttributes:
    var _value: _C_EventAttributes

    @always_inline
    fn __init__(
        inout self,
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
            color=int(resolved_color),
            payload_type=0,
            _reserved=0,
            event_payload=0,
            message_type=ASCII,
            message=message.unsafe_ptr(),
        )


# ===----------------------------------------------------------------------===#
# Binding
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct _dylib_function[fn_name: StringLiteral, type: AnyTrivialRegType]:
    @staticmethod
    fn load() -> type:
        return _get_dylib_function[fn_name, type]()


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


# ===----------------------------------------------------------------------===#
# Functions
# ===----------------------------------------------------------------------===#


fn _is_enabled() -> Bool:
    return env_get_int["KERNEL_E2E_GPU_PROFILING", 0]() == 1


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
    var info = EventAttributes(message=message, color=color, category=category)
    return _nvtxRangeStartEx.load()(UnsafePointer.address_of(info._value))


@always_inline
fn _end_range(id: RangeID):
    @parameter
    if _is_disabled():
        return
    _nvtxRangeEnd.load()(int(id))


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
    var info = EventAttributes(message=message, color=color, category=category)
    _nvtxMarkEx.load()(UnsafePointer.address_of(info._value))


struct Range:
    var _info: EventAttributes
    var _id: RangeID

    var _start_fn: fn (UnsafePointer[_C_EventAttributes]) -> RangeID
    var _end_fn: fn (RangeID) -> NoneType

    fn __init__(
        inout self,
        *,
        message: String = "",
        color: Optional[Color] = None,
        category: Int = _TraceType_MAX,
    ):
        constrained[_is_enabled(), "nvtx must be enabled"]()
        self._info = EventAttributes(
            message=message, color=color, category=category
        )
        self._id = 0
        self._start_fn = _nvtxRangeStartEx.load()
        self._end_fn = _nvtxRangeEnd.load()

    @always_inline
    fn __enter__(inout self):
        self._id = self._start_fn(UnsafePointer.address_of(self._info._value))

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

    var _push_fn: fn (UnsafePointer[_C_EventAttributes]) -> Int32
    var _pop_fn: fn () -> Int32

    fn __init__(
        inout self,
        *,
        message: String = "",
        color: Optional[Color] = None,
        category: Int = _TraceType_MAX,
    ):
        constrained[_is_enabled(), "nvtx must be enabled"]()
        self._info = EventAttributes(
            message=message, color=color, category=category
        )
        self._push_fn = _nvtxRangePushEx.load()
        self._pop_fn = _nvtxRangePop.load()

    @always_inline
    fn __enter__(inout self):
        _ = self._push_fn(UnsafePointer.address_of(self._info._value))

    @always_inline
    fn __exit__(self):
        _ = self._pop_fn()
