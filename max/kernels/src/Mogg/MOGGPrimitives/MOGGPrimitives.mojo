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

from std.logger import Logger
from std.math import fma
from std.ffi import external_call, c_size_t
from std.sys import size_of, align_of

from compiler_internal import StaticTensorSpec
from std.collections import InlineArray
from std.gpu.host import DeviceBuffer
from std.gpu.host.device_context import _DeviceContextPtr
from std.gpu.host.info import is_cpu, is_gpu
from layout import (
    Coord,
    Idx,
    IntTuple,
    TensorLayout,
    TileTensor,
    row_major,
)
from std.memory import memcpy
from std.memory.unsafe_pointer import unsafe_cast

from nn.concat import concat
from register import register_internal
from std.runtime.asyncrt import DeviceContextPtr
from tensor import (
    IOSpec,
    ManagedTensorSlice,
)
from tensor.io_spec import IO
from tensor.managed_tensor_slice import DynamicTensor, get_kernel_simd_width

from std.utils import Index, IndexList, StaticTuple

from .buffer_plan import BufferPlanState, BufferPlanStats

comptime MutByteBuffer = DynamicTensor[DType.int8, 1]
comptime ImmutByteBuffer = DynamicTensor[DType.int8, 1]
comptime logger = Logger()

# ===-----------------------------------------------------------------------===#
# Helper Structures
# ===-----------------------------------------------------------------------===#


# TODO: This struct should be deleted. Mojo and C++ should always communicate
# with pointers. If the Mojo wants to do something with this object, we should
# just create a C++ function for it. For the time being, this is safe because of
# the `constrained` and `static_assert` we added to ensure the type has the
# right byte size.
struct StateContext(TrivialRegisterPassable):
    """Defines a StateContext structure which holds a ptr to context and has accessors that go to external calls
    This is currently meant as a mojo-side container for GML::StateContext."""

    var num_slots: Int
    var ctx_ptr: OpaquePointer[MutAnyOrigin]

    @always_inline
    def __init__(
        out self, num_slots: Int, ctx_ptr: OpaquePointer[MutAnyOrigin]
    ):
        self.num_slots = num_slots
        self.ctx_ptr = ctx_ptr

        comptime assert size_of[StateContext]() == 16, (
            "Expecting StateContext to be 16 bytes wide, to match the C++"
            " equivalent"
        )

    @always_inline
    def __getitem__(self, index: Int) -> OpaquePointer[MutAnyOrigin]:
        assert 0 <= index < self.num_slots, "index must be within bounds"
        return external_call[
            "MGP_RT_GetContextPayloadPtr",
            OpaquePointer[MutAnyOrigin],
        ](index, self.ctx_ptr)


def pack_string_res(
    str_ptr: UnsafePointer[Byte, ImmutAnyOrigin], str_len: Int
) raises -> String:
    var span = Span(ptr=str_ptr, length=str_len)
    # We can not free the resource ptr embedded in MEF, create a copy
    return String(StringSlice(from_utf8=span))


# ===-----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===-----------------------------------------------------------------------===#


@no_inline
def create_index_async(value: Int, async_ptr: OpaquePointer[MutAnyOrigin]):
    external_call["MGP_RT_CreateAsync_ssizet", NoneType](value, async_ptr)


@no_inline
@export
def create_si64_async(value: Int64, async_ptr: OpaquePointer[MutAnyOrigin]):
    external_call["MGP_RT_CreateAsync_int64t", NoneType](value, async_ptr)


@no_inline
def create_i1_async(
    value: Bool,
    async_ptr: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_CreateAsync_bool", NoneType](value, async_ptr)


@no_inline
def create_buffer_ref_async(
    buffer: MutByteBuffer,
    async_ptr: OpaquePointer[MutAnyOrigin],
    call_ctx: DeviceContextPtr,
):
    external_call["MGP_RT_CreateAsyncDeviceBufferRef", NoneType](
        buffer.unsafe_ptr(), buffer.size(), async_ptr, call_ctx._handle
    )


@no_inline
def create_tensor_spec_async[
    spec_rank: Int
](spec: IndexList[spec_rank], async_ptr: OpaquePointer[MutAnyOrigin],):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var storage = InlineArray[Int, spec_rank](uninitialized=True)

    comptime for i in range(spec_rank):
        storage[i] = spec[i]

    external_call["MGP_RT_CreateAsyncTensorShape", NoneType](
        storage.unsafe_ptr(), spec_rank, async_ptr
    )


@export
def empty_destructor(ptr: UnsafePointer[UInt8, MutExternalOrigin]):
    pass


@no_inline
def unpack_device_ctx(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> DeviceContextPtr:
    var ptr = external_call[
        "MGP_RT_UnpackDeviceContext",
        _DeviceContextPtr[mut=True],
    ](async_ptr)

    return DeviceContextPtr(ptr)


@no_inline
def unpack_buffer_ref(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> MutByteBuffer:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "MGP_RT_GetDataFromBuffer",
        OpaquePointer[MutAnyOrigin],
    ](async_ptr, UnsafePointer(to=size))
    var shape = IndexList[1](Int(size))
    return MutByteBuffer(data_ptr.bitcast[Int8](), shape)


@no_inline
def unpack_tensor[
    buffer_rank: Int,
    tensor_rank: Int,
    dtype: DType,
](tensor_async_ptr: OpaquePointer[MutAnyOrigin]) -> DynamicTensor[
    dtype, buffer_rank
]:
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a DynamicTensor<[1]>
    comptime assert tensor_rank == buffer_rank or (
        tensor_rank == 0 and buffer_rank == 1
    )
    var shapes = IndexList[buffer_rank]()
    var buffer_ptr = external_call[
        "MGP_RT_GetShapeAndDataFromTensor",
        OpaquePointer[MutAnyOrigin],
    ](
        UnsafePointer(to=shapes.data),
        tensor_async_ptr,
    )

    comptime if tensor_rank == 0:
        shapes[0] = 1

    return DynamicTensor[dtype, buffer_rank](
        buffer_ptr.bitcast[Scalar[dtype]](), shapes
    )


@no_inline
def unpack_tensor_spec[
    spec_rank: Int
](async_ptr: OpaquePointer[MutAnyOrigin]) -> IndexList[spec_rank]:
    var storage = InlineArray[Int, spec_rank](uninitialized=True)
    external_call[
        "MGP_RT_GetTensorShapeFromAsync",
        NoneType,
    ](storage.unsafe_ptr(), spec_rank, async_ptr)
    var shape = IndexList[spec_rank]()

    comptime for i in range(spec_rank):
        shape[i] = storage[i]

    return shape


@always_inline
def get_buffer_data(
    buffer: MutByteBuffer,
) -> UnsafePointer[Int8, MutAnyOrigin]:
    return buffer.unsafe_ptr()


# ===-----------------------------------------------------------------------===#
# MGP Tensor Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.tensor.create")
@no_inline
def mgp_tensor_create[
    spec_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](
    buffer: MutByteBuffer,
    spec: IndexList[spec_rank],
) -> DynamicTensor[
    dtype, buffer_rank
]:
    comptime if spec_rank == 0:
        # We promote scalar tensor to tensor<[1]>
        comptime assert buffer_rank == 1
        return DynamicTensor[dtype, buffer_rank](
            buffer.unsafe_ptr().bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](IndexList[1](1)),
        )
    else:
        comptime assert spec_rank == buffer_rank
        return DynamicTensor[dtype, buffer_rank](
            buffer.unsafe_ptr().bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](spec),
        )


@register_internal("mgp.tensor.extract.tensor_spec")
@no_inline
def mgp_tensor_extract_tensor_spec[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](buffer: DynamicTensor[dtype, buffer_rank]) -> IndexList[tensor_rank]:
    comptime if tensor_rank == 0:
        comptime assert buffer_rank == 1
        return rebind[IndexList[tensor_rank]](IndexList[0]())
    else:
        comptime assert buffer_rank == tensor_rank
        return rebind[IndexList[tensor_rank]](buffer.shape().canonicalize())


@register_internal("mgp.tensor.extract.buffer")
@no_inline
def mgp_tensor_extract_buffer[
    buffer_rank: Int,
    dtype: DType,
](buffer: DynamicTensor[dtype, buffer_rank]) -> MutByteBuffer:
    # Unwrap the tensor into a size-less buffer pointer.
    return MutByteBuffer(
        buffer.unsafe_ptr[DType.int8](), IndexList[1](buffer.spec().bytecount())
    )


@register_internal("mgp.tensor.slice")
@no_inline
def mgp_tensor_slice[
    rank: Int,
    dtype: DType,
](
    input: DynamicTensor[dtype, rank],
    output_spec: IndexList[rank],
    start: DynamicTensor[DType.int64, 1],
) -> DynamicTensor[dtype, rank]:
    var input_shape = input.shape()

    # Find k: the first non-size-1 input dimension (the sliced dimension).
    var k = rank
    for i in range(rank):
        if input_shape[i] != 1:
            k = i
            break

    # Compute stride_k = product of input dims strictly after k.
    var stride_k = 1
    for i in range(k + 1, rank):
        stride_k *= input_shape[i]

    # start is a 1-element vector holding the scalar start value for
    # dimension k.  (mogg.slice scalars are rank-0 in MO but are lowered to
    # rank-1 DynamicTensors of size 1 by TensorCreateOp::emitMojo.)
    var start_k = Int(start.unsafe_ptr()[0]) if k < rank else 0

    # Compute the offset, normalizing negative start values.
    if start_k >= 0:
        return DynamicTensor[dtype, rank](
            input.unsafe_ptr() + start_k * stride_k, output_spec
        )
    else:
        var dim_k = input_shape[k]
        var normalized = max(0, dim_k + start_k)
        return DynamicTensor[dtype, rank](
            input.unsafe_ptr() + normalized * stride_k, output_spec
        )


# ===-----------------------------------------------------------------------===#
# MGP Buffer Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.buffer.alloc")
@no_inline
def mgp_buffer_alloc(
    byte_size: Int, dev_context: DeviceContextPtr
) raises -> MutByteBuffer:
    # Default to alignment of 0 which means kPreferredMemoryAlignment if cRawAlign is kUnknownSize (SizeUtils.h).
    # alias alignment = 0 if bRawAlign == UInt64.MAX else Int(bRawAlign)

    # This primitive has a byte-size input, so always assume a byte format
    var shape = IndexList[1](byte_size)
    var buf = dev_context[].enqueue_create_buffer[DType.int8](byte_size)
    return MutByteBuffer(buf^.take_ptr(), shape)


@register_internal("mgp.buffer.constant")
@export
def mgp_buffer_constant(
    resource_ptr: OpaquePointer[MutAnyOrigin],
    resource_bytecount: Int,
) -> MutByteBuffer:
    # Should we keep the alignment? It seems that the static alignment is
    # dropped in the kernels anyway.
    return MutByteBuffer(
        resource_ptr.bitcast[Int8](), IndexList[1](resource_bytecount)
    )


@no_inline
def fill_buffer[dtype: DType](buf: MutByteBuffer, *vals: Int):
    var ptr = buf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var offset: Int = 0
    for val in vals:
        ptr.store(offset, Scalar[dtype](val))
        offset += 1


@register_internal("mgp.buffer.set_with_index")
@no_inline
def mgp_buffer_set_with_index[
    bDevice: StaticString
](buffer: MutByteBuffer, *vals: Int) raises:
    assert is_cpu[bDevice](), "set_with_index can only work on cpu buffers"
    var bufSize = buffer.size()
    var numArgs = len(vals)
    assert (
        bufSize % numArgs == 0
    ), "buffer size not divisible by number of index args"

    var elSize = bufSize // numArgs
    if elSize == 4:
        fill_buffer[DType.int32](buffer, *vals)
    elif elSize == 8:
        fill_buffer[DType.int64](buffer, *vals)
    else:
        raise Error("unsupported element size")


@register_internal("mgp.buffer.to_bool")
@no_inline
def mgp_buffer_to_bool[bDevice: StaticString](buffer: ImmutByteBuffer) -> Bool:
    assert is_cpu[bDevice](), "to_bool can only work on cpu buffers"
    var bufSize = buffer.size()
    assert bufSize == 1, "buffer size must be a size of 1"
    return buffer.unsafe_ptr()[0] != 0


@register_internal("mgp.buffer.to_index")
@no_inline
def mgp_buffer_to_index(
    buffer: ImmutByteBuffer,
) raises -> Int:
    var bufSize = buffer.size()
    if bufSize == 4:
        return Int(buffer.unsafe_ptr().bitcast[Int32]()[0])
    if bufSize == 8:
        return Int(buffer.unsafe_ptr().bitcast[Int64]()[0])

    raise Error(
        "mgp.buffer.to_index must be called on either a 4- or 8-byte buffer"
    )


@register_internal("mgp.buffer.slice")
@no_inline
def mgp_buffer_slice(
    buffer: MutByteBuffer, offset: Int, size: Int
) -> MutByteBuffer:
    return MutByteBuffer(buffer.unsafe_ptr() + offset, Index(size))


@register_internal("mgp.buffer.bulk_slice")
@no_inline
def mgp_buffer_bulk_slice[
    N: Int,
    //,
](
    base: MutByteBuffer,
    offsets: InlineArray[Int, N],
    sizes: InlineArray[Int, N],
) -> InlineArray[MutByteBuffer, N]:
    """Bulk slice: produce N non-overlapping sub-buffers from a pool buffer.

    Parameters:
        N: Number of slices.

    Args:
        base: The pool buffer.
        offsets: Byte offset of each slice within the pool.
        sizes: Byte size of each slice.

    Returns:
        An InlineArray of N MutByteBuffer views into the pool.
    """
    var result = InlineArray[MutByteBuffer, N](uninitialized=True)

    for i in range(N):
        result[i] = mgp_buffer_slice(base, offsets[i], sizes[i])
    return result


@register_internal("mgp.buffer.plan")
@no_inline
def mgp_buffer_plan[
    num_static_sizes: Int,
    num_runtime_sizes: Int,
    //,
    alignments: InlineArray[Int, num_static_sizes + num_runtime_sizes],
    can_share: InlineArray[
        Int,
        (num_static_sizes + num_runtime_sizes)
        * (num_static_sizes + num_runtime_sizes),
    ],
    static_sizes: InlineArray[Int, num_static_sizes],
](runtime_sizes: InlineArray[Int, num_runtime_sizes]) -> Tuple[
    Int, InlineArray[Int, num_static_sizes + num_runtime_sizes]
]:
    """Runtime memory planning for buffers.

    Given static and runtime size information along with a sharing matrix for
    allocations, returns the high watermark size and offsets for each
    allocation.

    The allocations are ordered as: [static_sizes..., runtime_sizes...]
    where the first num_static_sizes allocations have compile-time known sizes,
    and the remaining num_runtime_sizes allocations have runtime sizes.

    can_share is a flat NxN matrix (row-major) where can_share[i*N+j]=1 iff
    allocations i and j have non-overlapping lifetimes and can therefore
    occupy the same memory slot. N = num_static_sizes + num_runtime_sizes.

    Parameters:
        num_static_sizes: Number of allocations with static sizes.
        num_runtime_sizes: Number of allocations with runtime sizes.
        alignments: Alignment requirements for each allocation.
        can_share: NxN sharing matrix (row-major, 0/1 values).
        static_sizes: Compile-time known sizes for first num_static_sizes allocations.

    Args:
        runtime_sizes: Runtime sizes for last num_runtime_sizes allocations.

    Returns:
        A tuple containing:
        - highWatermark: Total memory required.
        - offsets: Offsets for each allocation (static_sizes first, then runtime_sizes).
    """

    @parameter
    def compute_static_allocations(
        out result: BufferPlanState[
            alignments,
            can_share,
        ],
    ):
        result = {}
        result.allocate_greedy(static_sizes)

    comptime state = compute_static_allocations()

    # If all sizes are static, then we can avoid materializing the allocator
    # state.
    comptime if num_runtime_sizes == 0:
        comptime stats = state.stats()
        logger.debug(stats)

        comptime results = state.take_results()
        return results
    else:
        var runtime_state = materialize[state]()
        runtime_state.allocate_greedy[start=num_static_sizes](runtime_sizes)

        logger.debug(runtime_state.stats())
        return runtime_state^.take_results()


@register_internal("mgp.buffer.concat")
@no_inline
def mgp_buffer_concat[
    bDevice: StaticString
](
    output: MutByteBuffer,
    inputs: StaticTuple[MutByteBuffer, ...],
    call_ctx: DeviceContextPtr,
) raises:
    var output_lt = TileTensor(
        output.unsafe_ptr(),
        row_major(Coord(Idx(output.size()))),
    )
    var input_tensors = StaticTuple[_, inputs.size](
        TileTensor(
            inputs[0].unsafe_ptr(), row_major(Coord(Idx(inputs[0].size())))
        )
        .as_any_origin()
        .as_immut()
    )
    for i in range(1, len(inputs)):
        input_tensors[i] = (
            TileTensor(
                inputs[i].unsafe_ptr(), row_major(Coord(Idx(inputs[i].size())))
            )
            .as_any_origin()
            .as_immut()
        )
    concat[DType.int8, bDevice, None](
        output_lt, 0, input_tensors, context=call_ctx
    )


@register_internal("mgp.buffer.device_to_host")
@no_inline
def mgp_buffer_device_to_host[
    cOtherDevice: StaticString,
    dHostDevice: StaticString,
](
    dev_buf: MutByteBuffer,
    host_buf: MutByteBuffer,
    dev_ctx: DeviceContextPtr,
) raises:
    comptime if is_cpu[dHostDevice]() and is_gpu[cOtherDevice]():
        dev_ctx[].enqueue_copy[DType.int8](
            host_buf.unsafe_ptr(),
            dev_buf.to_device_buffer(dev_ctx[]),
        )
    else:
        raise Error("mgp.buffer.device_to_host must be scheduled on gpu device")


@register_internal("mgp.buffer.device_to_device")
@no_inline
def mgp_buffer_device_to_device[
    cSrcDevice: StaticString,
    dDstDevice: StaticString,
](
    src_buf: MutByteBuffer,
    dst_buf: MutByteBuffer,
    src_dev_ctx: DeviceContextPtr,
    dst_dev_ctx: DeviceContextPtr,
) raises:
    comptime if is_gpu[cSrcDevice]() and is_gpu[dDstDevice]():
        dst_dev_ctx[].enqueue_copy[DType.int8](
            dst_buf.to_device_buffer(dst_dev_ctx[]),
            src_buf.to_device_buffer(src_dev_ctx[]),
        )
    elif is_cpu[cSrcDevice]() and is_cpu[dDstDevice]():
        memcpy(
            dest=dst_buf.unsafe_ptr(),
            src=src_buf.unsafe_ptr(),
            count=src_buf.size(),
        )
    else:
        raise Error(
            "mgp.buffer.device_to_device can be scheduled between same device"
            " dtypes (cpu-cpu) or (gpu-gpu)"
        )


@register_internal("mgp.buffer.host_to_device")
@no_inline
def mgp_buffer_host_to_device[
    cHostDevice: StaticString,
    dOtherDevice: StaticString,
](
    host_buf: MutByteBuffer,
    dev_buf: MutByteBuffer,
    dev_ctx: DeviceContextPtr,
) raises:
    comptime if is_gpu[dOtherDevice]() and is_cpu[cHostDevice]():
        dev_ctx[].enqueue_copy[DType.int8](
            dev_buf.to_device_buffer(dev_ctx[]),
            host_buf.unsafe_ptr(),
        )
    else:
        raise Error("mgp.buffer.host_to_device must be scheduled on gpu device")


@register_internal("mgp.int.cache")
@no_inline
def mgp_int_cache[bIntSlot: UInt64](ctx: StateContextRef, value: Int):
    external_call["MGP_RT_SetCachedInt", NoneType](Int(bIntSlot), ctx, value)


@register_internal("mgp.int.get_cached")
@no_inline
def mgp_int_get_cached(ctx: StateContextRef, buffer_slot: Int) -> Int:
    return external_call["MGP_RT_GetCachedInt", Int](
        buffer_slot,
        ctx,
    )


@register_internal("mgp.buffer.get_size")
@no_inline
def mgp_buffer_get_size(
    buf: ImmutByteBuffer,
) -> Int:
    return buf.size()


# ===-----------------------------------------------------------------------===#
# MGP Tensor Spec Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.tensor_spec.create")
@no_inline
def mgp_tensor_spec_create[
    aRawDims: IntTuple,
    aRawDimsRank: Int,
](*runtimeDims: Int) -> IndexList[aRawDimsRank]:
    var shape = IndexList[aRawDimsRank]()
    var runtimeIndex = 0
    # Update Shape with runtime elements.
    # Negative values in aRawDims indicate dynamic dimensions.
    comptime for i in range(aRawDimsRank):
        if Int(aRawDims[i]) >= 0:
            shape[i] = Int(aRawDims[i])
        else:
            shape[i] = runtimeDims[runtimeIndex]
            runtimeIndex += 1
    return shape


@register_internal("mgp.tensor_spec.get_dim")
@no_inline
def mgp_tensor_spec_get_dim[
    spec_rank: Int, axis: UInt64
](spec: IndexList[spec_rank]) -> Int:
    comptime assert axis < UInt64(
        spec_rank
    ), "axis for get_dim must be less than rank of TensorSpec"
    return spec[Int(axis)]


# ===-----------------------------------------------------------------------===#
# MGP Device Context Primitives
# ===-----------------------------------------------------------------------===#


@export
def mgp_device_context_destroy(dev_ctx: DeviceContextPtr):
    # DeviceContext is refcounted, we don't need to explicitly destroy it
    pass


@register_internal("mgp.sync")
@no_inline
def mgp_sync(ctx: StateContext, dev_ctx: DeviceContextPtr) raises:
    dev_ctx[].synchronize()


@register_internal("mgp.debug.print")
@no_inline
def mgp_debug_print[
    aDebugString: StaticString,
    bLabel: StaticString,
](ctx: StateContext,) raises:
    var prefix = String()
    if bLabel:
        prefix = "[" + bLabel + "] "
    print(prefix + aDebugString)


@register_internal("mgp.debug.print.int")
@no_inline
def mgp_debug_print_int[
    aLabel: StaticString,
](ctx: StateContext, value: Int):
    var prefix = String()
    if aLabel:
        prefix = "[" + aLabel + "] "
    print(prefix + String(value))


@register_internal("mgp.debug.tensor.print")
@no_inline
def mgp_debug_tensor_print[
    spec_rank: Int,
    dtype: DType,
](
    buffer: ImmutByteBuffer,
    shape: IndexList[spec_rank],
    label_ptr: UnsafePointer[Byte, ImmutAnyOrigin],
    label_len: Int,
) raises:
    external_call["MGP_RT_DebugTensorPrint", NoneType](
        label_ptr,
        c_size_t(label_len),
        dtype,
        UnsafePointer(to=shape.data),
        spec_rank,
        buffer.unsafe_ptr(),
        buffer.size(),
    )


# ===----------------------------------------------------------------------===#
# Additional expected primitives
# ===-----------------------------------------------------------------------===#


@always_inline
def get_simd_width_for_dtypes[
    dtypes: StaticTuple[DType, _], target: StaticString
]() -> Int:
    comptime assert dtypes.size > 0

    var width = get_kernel_simd_width[dtypes[0], target]()

    comptime for i in range(dtypes.size - 1):
        width = max(get_kernel_simd_width[dtypes[i + 1], target](), width)

    return width


# TODO: this should take IOSpec as a param -- will require graph compiler changes
# Used by the graph compiler to construct tensors from MGP repr. of tensor
@always_inline
def to_managed_tensor_slice[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    shape: UnsafePointer[Int, ImmutAnyOrigin],
) -> ManagedTensorSlice[
    io_spec=IOSpec[mut, input](),
    static_spec=StaticTensorSpec[dtype, rank, ...].get_unknown(),
]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    comptime for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        shape_tuple[i] = shape_ptr[i]
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return {data, shape_tuple, stride_tuple}


# Extract a scalar from a managed tensor slice.
@always_inline
def _get_scalar_from_managed_tensor_slice[
    dtype: DType,
](tensor: ManagedTensorSlice[dtype=dtype, ...]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    # This is used instead of [0] since __getitem__ for `ManagedTesnorSlice`
    # does not work with `register_internal` out of the box.
    return tensor.load[width=1](IndexList[1](0))


# ===-----------------------------------------------------------------------===#
# Opaque Test Primitives
# ===-----------------------------------------------------------------------===#


struct MyInt(Movable):
    var val: Int

    def __init__(out self, val: Int):
        self.val = val

    def __init__(out self, *, deinit take: MyInt):
        print("MyInt.__moveinit__", take.val)
        self.val = take.val

    def __del__(deinit self):
        print("MyInt.__del__", self.val)


@register_internal("testfuse.my_int.from_index")
@no_inline
def test_my_int_from_index(x: Int) -> MyInt:
    return MyInt(x)


@register_internal("testfuse.my_int.square")
@no_inline
def test_my_int_square(x: MyInt) -> MyInt:
    return MyInt(x.val * x.val)


@register_internal("testfuse.my_int.to_index")
@no_inline
def test_my_int_to_index(x: MyInt) -> Int:
    return x.val


struct MyIntReg2(ImplicitlyCopyable, RegisterPassable):
    var val: Int

    def __init__(out self, val: Int):
        self.val = val

    def __del__(deinit self):
        print("MyIntReg2.__del__", self.val)


@register_internal("testfuse.my_int_reg2.from_index")
@no_inline
def test_my_int_reg2_from_index(x: Int) -> MyIntReg2:
    return MyIntReg2(x)


@register_internal("testfuse.my_int_reg2.square")
@no_inline
def test_my_int_reg2_square(x: MyIntReg2) -> MyIntReg2:
    return MyIntReg2(x.val * x.val)


@register_internal("testfuse.my_int_reg2.to_index")
@no_inline
def test_my_int_reg2_to_index(x: MyIntReg2) -> Int:
    return x.val


# ===-----------------------------------------------------------------------===#
# Mojo generation hooks
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# Mojo-C++ interop aliases
# ===-----------------------------------------------------------------------===#

# The purpose of these aliases is to make it easier to visually parse the
# interop. There is only one rule: Do not use types, always use OpaquePointer.
# This saves us from having to statically assert that a certain type has a
# specific byte size.

# AnyAsyncValueRef is a C++ struct. The runtime passes a reference to it.
# Therefore, we alias it to OpaquePointer which will have the same bitwidth as
# C++'s pointers.
comptime AnyAsyncValueRefPtr = OpaquePointer[MutAnyOrigin]

# TensorBufferRef is a C++ struct. Primitives should always manipulate a
# reference to it. Therefore, it is modeled here as an OpaquePointer.
comptime TensorBufferRefPtr = OpaquePointer[MutAnyOrigin]

# StateContext is a C++ struct. Primitives should always manipulate a reference
# to it. Therefore, it is modeled here as an OpaquePointer.
comptime StateContextRef = OpaquePointer[MutAnyOrigin]


# ===-----------------------------------------------------------------------===#
# MOGG primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mogg.as_scalar")
@always_inline
def mogg_as_scalar(tensor: ManagedTensorSlice) -> Scalar[tensor.dtype]:
    return _get_scalar_from_managed_tensor_slice(tensor)


@register_internal("mogg.async.__del__")
@no_inline
def mogg_async_del(
    async_ptr: UnsafePointer[AnyAsyncValueRefPtr, MutAnyOrigin], size: Int
):
    """
    Decrement the AnyAsyncValueRef. Typically called at the end of a kernel for
    all input and output operands.
    """
    external_call["MGP_RT_DestructAsyncRefs", NoneType](size, async_ptr, False)


@register_internal("mogg.async.unpack")
@no_inline
def mogg_async_unpack[
    T: TrivialRegisterPassable
](async_ptr: AnyAsyncValueRefPtr) -> T:
    """
    Returns the value stored in the AnyAsyncValueRef.
    """
    var ptr = external_call[
        "MGP_RT_GetValueFromAsync", OpaquePointer[MutAnyOrigin]
    ](async_ptr).bitcast[T]()

    return UnsafePointer[T, MutAnyOrigin].__getitem__(ptr, 0)


struct MoggAsyncPackHelper:
    """
    Helper struct for packing various data types into an asynchronous context
    for MOGG operations. Provides constructor overloads for different supported
    types.
    """

    def __init__(out self, data: Int, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs an integer value into the asynchronous context.
        Calls create_index_async to handle the packing.
        """
        create_index_async(data, async_ptr)

    def __init__(out self, data: Int64, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs a 64-bit integer value into the asynchronous context.
        Calls create_si64_async to handle the packing.
        """
        create_si64_async(data, async_ptr)

    def __init__(out self, data: Bool, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs a boolean value into the asynchronous context.
        Calls create_i1_async to handle the packing.
        """
        create_i1_async(data, async_ptr)

    def __init__[
        spec_rank: Int
    ](out self, data: IndexList[spec_rank], async_ptr: AnyAsyncValueRefPtr):
        """
        Packs an IndexList of specified rank into the asynchronous context.
        Calls create_tensor_spec_async to handle the packing.
        """
        create_tensor_spec_async(data, async_ptr)

    def __init__(
        out self,
        data: MutByteBuffer,
        device_ctx_ptr: DeviceContextPtr,
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """
        Packs a MutByteBuffer into the asynchronous context.
        Calls create_buffer_ref_async to handle the packing.
        """
        create_buffer_ref_async(data, async_ptr, device_ctx_ptr)

    def __init__(
        out self,
        var data: Some[Movable & ImplicitlyDestructible],
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """
        Packs a generic Movable value into the asynchronous context.
        Used for opaque types like SIMDPair.
        """
        comptime Type = type_of(data)

        # MGP_RT_CreateOwnedAsyncMojoValue expects a type erased destructor
        @always_inline("nodebug")
        def erased_destructor(ptr: UnsafePointer[UInt8, MutExternalOrigin]):
            ptr.bitcast[Type]().destroy_pointee()

        var dst_ptr = external_call[
            "MGP_RT_MojoValueAllocateBuffer",
            UnsafePointer[UInt8, MutExternalOrigin],
        ](size_of[Type](), align_of[Type]())

        dst_ptr.bitcast[Type]().init_pointee_move(data^)

        external_call["MGP_RT_CreateOwnedAsyncMojoValue", NoneType](
            dst_ptr,
            erased_destructor,
            async_ptr,
        )


@register_internal("mogg.async.pack")
@no_inline
def mogg_async_pack(pack_helper: MoggAsyncPackHelper):
    """
    Packs asynchronous data using the provided MoggAsyncPackHelper.

    This function serves as an entry point for packing data into an asynchronous
    reference. The actual packing logic is handled by the MoggAsyncPackHelper struct,
    which provides specialized constructors for different data types. This function
    itself is a no-op and exists to satisfy the internal registration mechanism.
    """
    return


@no_inline
def mogg_async_pack_borrow[
    buffer_rank: Int,
    dtype: DType,
    //,
    spec_rank: Int,
    is_tensor: Bool,
](
    borrower: AnyAsyncValueRefPtr,
    buffer: DynamicTensor[dtype, buffer_rank],
    mem: Optional[TensorBufferRefPtr],
):
    """
    Borrows an async value. This differs from `mogg.async.pack` which assigns a
    value to the given async value in that it's a simple refcount increment.
    """

    comptime if is_tensor:
        var shape = buffer.shape()
        external_call["MGP_RT_TensorBorrowV2", NoneType](
            borrower,
            buffer.unsafe_ptr(),
            buffer.bytecount(),
            spec_rank,
            UnsafePointer(to=shape.data),
            dtype,
            mem,
        )
    else:
        external_call["MGP_RT_BufferBorrowV2", NoneType](
            borrower, buffer.unsafe_ptr(), buffer.size(), mem
        )


@no_inline
def mogg_async_pack_borrow[
    spec_rank: Int,  # unused
    is_tensor: Bool,  # unused
](
    borrower: AnyAsyncValueRefPtr,
    buffer: TensorBufferRefPtr,
    mem: Optional[TensorBufferRefPtr],
):
    """
    Borrows an async value. This differs from `mogg.async.pack` which assigns a
    value to the given async value in that it's a simple refcount increment.
    """
    external_call["MGP_RT_BufferBorrowForTensorRef", NoneType](
        borrower, buffer, mem
    )


@register_internal("mogg.tensor.__init__")
@always_inline
def mogg_tensor_init[
    dtype: DType,
    rank: Int,
    mut: Bool,
    input: IO,
    static_layout: TensorLayout,
    alignment: Int,
    exclusive: Bool,
](
    ptr: OpaquePointer[MutAnyOrigin], shape: IndexList[rank]
) -> ManagedTensorSlice[
    io_spec=IOSpec[mut, input](),
    static_spec=StaticTensorSpec[
        dtype,
        rank,
        static_layout=static_layout,
    ](
        alignment,
        AddressSpace.GENERIC,
        exclusive,
    ),
]:
    """
    Helper for constructing a ManagedTensorSlice.
    """
    return {ptr.bitcast[Scalar[dtype]](), shape}


@register_internal("mogg.async.ready")
@no_inline
def mogg_async_ready(async_ptr: AnyAsyncValueRefPtr):
    """
    Marks the chain as ready.
    """
    external_call["MGP_RT_CreateAsync_chain", NoneType](async_ptr)


@register_internal("mogg.async.check_task_error")
@no_inline
def mogg_async_check_task_error(mut error: Optional[Error]) raises:
    """Raises the captured error from an async task, if present.

    Raises:
        If an error was captured from the async task.
    """
    if error:
        raise error.take()


@register_internal("mogg.async.error")
@no_inline
def mogg_async_error(
    async_ptr: AnyAsyncValueRefPtr,
    err: Error,
    source_notes: String = "",
):
    """Indicates to the C++ runtime that the kernel has failed.

    When source_notes is non-empty it is prepended to the error message.
    The "Source Traceback:" header is included by the compiler only when
    actual Python tracebacks are present (see buildNotesString in MOGGOps.cpp).
    See GEX-2678.
    """
    var error_message = String(err)
    if source_notes:
        error_message = "\n" + source_notes + "\n\n" + error_message
    external_call["MGP_RT_AsyncRT_CreateAsync_Error", NoneType](
        async_ptr,
        error_message.as_c_string_slice().unsafe_ptr(),
        error_message.byte_length(),
    )


@register_internal("mogg.raise")
@no_inline
def mogg_format_kernel_error(
    kernel_name: String,
    error: Error,
    fusion_info: String = "",
    traceback: String = "",
) -> Error:
    """Format a kernel error with context (name, fusion info, source traceback).

    Called from MOGG ABI stub except handlers. The formatted error is re-raised
    and eventually caught by the outer MGP region's except handler.
    """
    var msg = (
        String('An error occurred in kernel named "')
        + kernel_name
        + '":\n'
        + String(error)
    )
    if fusion_info:
        msg += "\n\nFusion info:\n" + fusion_info
    if traceback:
        msg += "\n\nSource Traceback:\n" + traceback
    return Error(msg)


@register_internal("mogg.format_region_error")
@no_inline
def mogg_format_region_error(
    region_name: String,
    error: Error,
) -> Error:
    """Format a region error with the entry point name prefix.

    Called from MGP ABI stub except handlers after catching a kernel error.
    """
    return Error(
        String('An error occurred in kernel entry point named "')
        + region_name
        + '":\n'
        + String(error)
    )


@register_internal("mogg.tensor.reshape")
@always_inline
def reshape_contiguous_buffer[
    static_layout: TensorLayout, new_rank: Int
](
    buffer: ManagedTensorSlice,
    shape: IndexList[new_rank],
) -> ManagedTensorSlice[
    io_spec=buffer.io_spec,
    static_spec=StaticTensorSpec[
        buffer.dtype,
        new_rank,
        static_layout=static_layout,
    ](
        1,
        AddressSpace.GENERIC,
        True,
    ),
]:
    """
    Constructs a new ManagedTensorSlice with a new shape and static spec.
    """
    return {buffer._ptr, shape}


# ===-----------------------------------------------------------------------===#
# MGP primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.buffer.get_cached")
@no_inline
def mgp_buffer_get_cached(
    ctx: StateContextRef,
    buffer_slot: Int,
) -> Tuple[MutByteBuffer, TensorBufferRefPtr]:
    """
    Get a reference to the cached tensor.
    """
    var buffer_size: UInt64 = 0
    var buffer_data = Optional[OpaquePointer[MutAnyOrigin]]()

    var buffer_ref = external_call[
        "TMP_MGP_RT_GetCachedBuffer", TensorBufferRefPtr
    ](
        buffer_slot,
        ctx,
        UnsafePointer(to=buffer_size),
        UnsafePointer(to=buffer_data),
    )

    var buffer = MutByteBuffer(
        buffer_data.unsafe_value().bitcast[Int8](),
        Index(buffer_size),
    )
    var res = Tuple[MutByteBuffer, TensorBufferRefPtr](buffer, buffer_ref)

    return res


@register_internal("mgp.buffer.remove_cached")
@no_inline
def mgp_buffer_remove_cached(ctx: StateContextRef, buffer_slot: Int):
    external_call["TMP_MGP_RT_RemoveCachedBuffer", NoneType](buffer_slot, ctx)


@register_internal("mgp.assert")
@no_inline
def mgp_assert(
    cond: Bool, msg_ptr: UnsafePointer[Byte, ImmutAnyOrigin], msg_len: Int
) raises:
    """
    Raises an error when the input condition is not true.
    """
    if not cond:
        raise Error(pack_string_res(msg_ptr, msg_len))


def all_zeros(indices: IndexList) -> Bool:
    comptime for i in range(indices.size):
        if indices[i] != 0:
            return False
    return True


def get_buffer_mem_storage_handle(
    buffer: OpaquePointer[MutAnyOrigin],
    type: Int,
    memStorageHandle: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_GetBufferMemStorageHandle", NoneType](
        buffer, type, memStorageHandle
    )


# ===----------------------------------------------------------------------===#
# Affine view kernels
# ===----------------------------------------------------------------------===#


@register_internal("mo.split_dim")
@always_inline
def split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].
    # In this case, the axis = 0 and the new_shape_dim = 8.

    comptime for i in range(rank + 1):
        comptime if i == axis:
            out[i] = indices[axis] // new_shape_dim
        elif i == axis + 1:
            out[i] = indices[axis] % new_shape_dim
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("mo.merge_dim")
@always_inline
def merge_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], old_shape_dim: Int) -> IndexList[rank - 1]:
    var out = IndexList[rank - 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [5, 8, 30, 2] and we reshape it to [40, 30, 2].
    # If we are accessing the index [2, 5, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [21, 16, 1].
    # Or [2 * 8 + 5, 16, 1].
    # In this case, the axis = 0 and the old_shape_dim = 8.

    comptime for i in range(rank - 1):
        comptime if i == axis:
            out[i] = fma(indices[i], old_shape_dim, indices[i + 1])
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i + 1]

    return out


@register_internal("mo.add_singleton_dim")
@always_inline
def insert_index[
    rank: Int, axis: Int, value: Int
](indices: IndexList[rank]) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    comptime for i in range(rank + 1):
        comptime if i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]
        else:
            out[i] = value

    return out


# ===----------------------------------------------------------------------===#
# POP operations
# ===----------------------------------------------------------------------===#


@register_internal("pop.select")
@always_inline
def select[
    T: TrivialRegisterPassable
](cond: Bool, true_case: T, false_case: T) -> T:
    if cond:
        return true_case

    return false_case


@register_internal("pop.simd.select")
@always_inline
def simd_select[
    T: TrivialRegisterPassable
](cond: Bool, true_case: T, false_case: T) -> T:
    return select(cond, true_case, false_case)
