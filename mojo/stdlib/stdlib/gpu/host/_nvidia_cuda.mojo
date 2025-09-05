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

from sys import external_call, size_of
from sys.ffi import c_uint, c_int

from gpu._utils import to_llvm_ptr
from gpu.host import DeviceContext, DeviceStream, DeviceFunction
from gpu.host.device_context import (
    _CharPtr,
    _checked,
    _DeviceBufferPtr,
    _DeviceContextPtr,
    _DeviceStreamPtr,
    _DeviceFunctionPtr,
)
from memory import stack_allocation
from memory.unsafe import bitcast

from utils import IndexList, StaticTuple


struct _CUctx_st:
    pass


struct _CUstream_st:
    pass


struct _CUmod_st:
    pass


struct _CUevent_st:
    pass


alias CUcontext = UnsafePointer[_CUctx_st]
alias CUstream = UnsafePointer[_CUstream_st]
alias CUmodule = UnsafePointer[_CUmod_st]
alias CUevent = UnsafePointer[_CUevent_st]


# Accessor function to get access to the underlying CUcontext from a abstract DeviceContext.
# Use `var cuda_ctx: CUcontext = CUDA(ctx)` where ctx is a `DeviceContext` to get access to the underlying CUcontext.
@always_inline
fn CUDA(ctx: DeviceContext) raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_context(CUcontext *result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_cuda_context",
            _CharPtr,
            UnsafePointer[CUcontext],
            _DeviceContextPtr,
        ](
            UnsafePointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUstream from a abstract DeviceStream.
# Use `var cuda_stream: CUstream = CUDA(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying CUstream.
@always_inline
fn CUDA(stream: DeviceStream) raises -> CUstream:
    var result = CUstream()
    # const char *AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_cuda_stream",
            _CharPtr,
            UnsafePointer[CUstream],
            _DeviceStreamPtr,
        ](
            UnsafePointer(to=result),
            stream._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUmodule from a DeviceFunction.
@always_inline
fn CUDA_MODULE(func: DeviceFunction) raises -> CUmodule:
    var result = CUmodule()
    # const char *AsyncRT_DeviceFunction_cuda_module(CUmodule *result, const DeviceFunction *func)
    _checked(
        external_call[
            "AsyncRT_DeviceFunction_cuda_module",
            _CharPtr,
            UnsafePointer[CUmodule],
            _DeviceFunctionPtr,
        ](
            UnsafePointer(to=result),
            func._handle,
        )
    )
    return result


fn CUDA_get_current_context() raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_current_context(CUcontext *result)
    _checked(
        external_call["AsyncRT_DeviceContext_cuda_current_context", _CharPtr,](
            UnsafePointer(to=result),
        )
    )
    return result


# ===----------------------------------------------------------------------=== #
# TMA
# ===----------------------------------------------------------------------=== #


@fieldwise_init("implicit")
@register_passable("trivial")
struct TensorMapDataType:
    var _value: Int32

    alias UINT8 = Self(0)
    alias UINT16 = Self(1)
    alias UINT32 = Self(2)
    alias INT32 = Self(3)
    alias UINT64 = Self(4)
    alias INT64 = Self(5)
    alias FLOAT16 = Self(6)
    alias FLOAT32 = Self(7)
    alias FLOAT64 = Self(8)
    alias BFLOAT16 = Self(9)
    alias FLOAT32_FTZ = Self(10)
    alias TFLOAT32 = Self(11)
    alias TFLOAT32_FTZ = Self(12)

    @staticmethod
    fn from_dtype[dtype: DType]() -> Self:
        constrained[
            dtype in (DType.float32, DType.bfloat16, DType.float8_e4m3fn),
            "Unsupported dtype",
        ]()

        @parameter
        if dtype is DType.float32:
            return Self.FLOAT32
        elif dtype is DType.float8_e4m3fn:
            return Self.UINT8
        else:
            return Self.BFLOAT16


@fieldwise_init("implicit")
@register_passable("trivial")
struct TensorMapInterleave:
    var _value: Int32

    alias INTERLEAVE_NONE = Self(0)
    alias INTERLEAVE_16B = Self(1)
    alias INTERLEAVE_32B = Self(2)


@fieldwise_init("implicit")
@register_passable("trivial")
struct TensorMapSwizzle(
    EqualityComparable,
    ImplicitlyCopyable,
    Intable,
    Movable,
    Stringable,
    Writable,
):
    var _value: Int32

    alias SWIZZLE_NONE = Self(0)
    alias SWIZZLE_32B = Self(1)
    alias SWIZZLE_64B = Self(2)
    alias SWIZZLE_128B = Self(3)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return Int(self._value)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @always_inline
    fn bytes(self) -> Int:
        return Int((2**self._value) * 16)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        if self._value == 1:
            writer.write("32B swizzle")
        elif self._value == 2:
            writer.write("64B swizzle")
        elif self._value == 3:
            writer.write("128B swizzle")
        elif self._value == 0:
            writer.write("no swizzle")
        else:
            writer.write("invalid swizzle")


@fieldwise_init("implicit")
@register_passable("trivial")
struct TensorMapL2Promotion:
    var _value: Int32

    alias NONE = Self(0)
    alias L2_64B = Self(1)
    alias L2_128B = Self(2)
    alias L2_256B = Self(3)


@fieldwise_init("implicit")
@register_passable("trivial")
struct TensorMapFloatOOBFill:
    var _value: Int32

    alias NONE = Self(0)
    alias NAN_REQUEST_ZERO_FMA = Self(1)


# The TMA descriptor is a 128-byte opaque object filled by the driver API.
# It should be 64-byte aligned both on the host and the device (if passed to constant memory).
struct TMADescriptor(ImplicitlyCopyable):
    var data: StaticTuple[UInt8, 128]

    @always_inline
    fn __init__(out self):
        self.data = StaticTuple[UInt8, 128]()

    @always_inline
    fn __copyinit__(out self, other: Self):
        self.data = other.data


fn prefetch_tma_descriptor(desc_ptr: OpaquePointer):
    __mlir_op.`nvvm.prefetch`[tensormap = __mlir_attr.unit](
        to_llvm_ptr(desc_ptr),
    )


@always_inline
fn create_tma_descriptor[
    dtype: DType,
    rank: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    global_buf: DeviceBuffer[dtype],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    shared_mem_shape: IndexList[rank],
) raises -> TMADescriptor:
    """Create a tensor map descriptor object representing tiled memory region.
    """
    # Enforces host-side alignment
    var tma_descriptor = stack_allocation[1, TMADescriptor, alignment=64]()[0]
    var tensor_map_ptr = UnsafePointer(to=tma_descriptor).bitcast[NoneType]()

    # NOTE: These are initialized in the comptime loop below.
    var global_dim_arg = InlineArray[Int64, rank](uninitialized=True)
    var global_strides_arg = InlineArray[Int64, rank](uninitialized=True)
    var box_dim_arg = InlineArray[Int32, rank](uninitialized=True)
    var element_stride_arg = InlineArray[Int32, rank](fill=1)

    @parameter
    for i in range(rank):
        global_dim_arg[i] = global_shape[rank - i - 1]
        global_strides_arg[i] = global_strides[rank - i - 1] * size_of[dtype]()
        box_dim_arg[i] = shared_mem_shape[rank - i - 1]

    debug_assert(
        global_strides_arg[0] == size_of[dtype](),
        "TMA GMEM should be row-major, global stride",
        " at dim 0 should be size_of[dtype](): ",
        size_of[dtype](),
        " but is: ",
        global_strides_arg[0],
    )
    # const char *AsyncRT_cuda_tensorMapEncodeTiled(
    #     void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    #     const DeviceBuffer *globalAddress, const uint64_t *globalDim,
    #     const uint64_t *globalStrides, const uint32_t *boxDim,
    #     const uint32_t *elementStrides, int32_t interleave, int32_t swizzle,
    #     int32_t l2Promotion, int32_t oobFill) {
    _checked(
        external_call[
            "AsyncRT_cuda_tensorMapEncodeTiled",
            _CharPtr,
            OpaquePointer,  # tensorMap
            Int32,  # tensorDataType
            Int32,  # tensorRank
            _DeviceBufferPtr,  #  globalAddress
            UnsafePointer[Int64],  # globalDim
            UnsafePointer[Int64],  # globalStrides
            UnsafePointer[Int32],  # boxDim
            UnsafePointer[Int32],  # elementStrides
            Int32,  # interleave
            Int32,  # swizzle
            Int32,  # l2Promotion
            Int32,  # oobFill
        ](
            tensor_map_ptr,
            TensorMapDataType.from_dtype[dtype]()._value,
            rank,
            global_buf._handle,
            global_dim_arg.unsafe_ptr(),
            # global_strides_arg[0] is implicitly size_of[dtype]()
            global_strides_arg.unsafe_ptr() + 1,
            box_dim_arg.unsafe_ptr(),
            element_stride_arg.unsafe_ptr(),
            TensorMapInterleave.INTERLEAVE_NONE._value,
            swizzle_mode._value,
            TensorMapL2Promotion.NONE._value,
            TensorMapFloatOOBFill.NONE._value,
        )
    )
    return tma_descriptor
