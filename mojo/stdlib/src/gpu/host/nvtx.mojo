# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List, Optional
from os import abort
from pathlib import Path
from sys.ffi import C_char, DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from memory import UnsafePointer, stack_allocation

# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#

alias CUDA_NVTX_LIBRARY_PATH = "/usr/local/cuda/lib64/libnvToolsExt.so"


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    if not Path(CUDA_NVTX_LIBRARY_PATH).exists():
        return abort[UnsafePointer[NoneType]](
            "the CUDA NVTX library was not found at " + CUDA_NVTX_LIBRARY_PATH
        )
    var ptr = UnsafePointer[DLHandle].alloc(1)
    ptr.init_pointee_move(DLHandle(CUDA_NVTX_LIBRARY_PATH))
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
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
alias NVTXVersion = 3


@value
@register_passable("trivial")
struct _C_EventAttributes:
    var version: UInt64
    """Version flag of the structure."""

    var size: UInt16
    """Size of the structure."""

    var category: UInt32
    """ID of the category the event is assigned to."""

    var color_type: UInt32
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


struct EventAttributes:
    var _value: _C_EventAttributes

    fn __init__(
        inout self,
        *,
        message: String = "",
        color: Int = 0xFF880000,
    ):
        alias ARGB = 1
        alias ASCII = 1
        self._value = _C_EventAttributes(
            version=NVTXVersion,
            size=sizeof[_C_EventAttributes](),
            category=0,
            color_type=ARGB,
            color=color,
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


# ===----------------------------------------------------------------------===#
# Functions
# ===----------------------------------------------------------------------===#


struct Range:
    var _info: _C_EventAttributes
    var _id: RangeID

    var _start_fn: fn (UnsafePointer[_C_EventAttributes]) -> RangeID
    var _end_fn: fn (RangeID) -> NoneType

    fn __init__(inout self, info: EventAttributes):
        self._info = info._value
        self._id = 0
        self._start_fn = _nvtxRangeStartEx.load()
        self._end_fn = _nvtxRangeEnd.load()

    @always_inline
    fn __enter__(inout self):
        self._id = self._start_fn(UnsafePointer.address_of(self._info))

    @always_inline
    fn __exit__(self):
        self._end_fn(self._id)


fn mark(info: EventAttributes):
    _nvtxMarkEx.load()(UnsafePointer.address_of(info._value))
