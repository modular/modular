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

from memory import (
    LegacyOpaquePointer as OpaquePointer,
    LegacyUnsafePointer as UnsafePointer,
)
from sys import has_amd_gpu_accelerator, size_of
from pathlib import Path
from sys.ffi import _get_global_or_null, external_call
from sys.ffi import _find_dylib
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import OwnedDLHandle, _Global
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host._amdgpu_hip import HIP
from gpu.host._nvidia_cuda import CUDA
from comm.allreduce import MAX_GPUS

alias ncclComm_t = OpaquePointer


@fieldwise_init
@register_passable("trivial")
struct ncclResult_t(EqualityComparable, Writable):
    var _value: Int32
    alias ncclSuccess = Self(0)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("ncclResult_t(", Int(self._value), ")")


@fieldwise_init
@register_passable("trivial")
struct ncclRedOp_t:
    var _value: Int32
    alias ncclSum = Self(0)

    fn __init__(out self, value: Int):
        self._value = value


@fieldwise_init
@register_passable("trivial")
struct ncclDataType_t:
    var _value: Int32
    alias ncclFloat16 = Self(6)
    alias ncclFloat32 = Self(7)
    alias ncclBfloat16 = Self(9)

    fn __init__(out self, value: Int):
        self._value = value


alias RCCL_LIBRARY_PATHS = List[Path](
    "librccl.so",
    "librccl.so.1",
    "/opt/rocm/lib/librccl.so",
    "/opt/rocm/lib/librccl.so.1",
)


alias NCCL_LIBRARY_PATHS = List[Path](
    "libnccl.so",
    "libnccl.so.2",
    "/usr/lib/x86_64-linux-gnu/libnccl.so",
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
)


# Unified CCL loader (selects RCCL/NCCL at compile time)
fn _init_ccl_dylib() -> OwnedDLHandle:
    @parameter
    if has_amd_gpu_accelerator():
        return _find_dylib["RCCL"](materialize[RCCL_LIBRARY_PATHS]())
    else:
        return _find_dylib["NCCL"](materialize[NCCL_LIBRARY_PATHS]())


alias CCL_LIBRARY = _Global["CCL_LIBRARY", _init_ccl_dylib]


@always_inline
fn _get_ccl_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() raises -> result_type:
    return _ffi_get_dylib_function[CCL_LIBRARY(), func_name, result_type]()


# Common function signatures for CCL APIs (shared by RCCL/NCCL)
alias CCLAllReduceFn = fn (
    OpaquePointer,
    OpaquePointer,
    Int,
    ncclDataType_t,
    ncclRedOp_t,
    ncclComm_t,
    OpaquePointer,
) -> ncclResult_t

alias CCLAllGatherFn = fn (
    OpaquePointer,
    OpaquePointer,
    Int,
    ncclDataType_t,
    ncclComm_t,
    OpaquePointer,
) -> ncclResult_t


# Paired wrappers grouped RCCl/NCCL for comparison
fn ncclGroupStart() raises -> ncclResult_t:
    return _get_ccl_function["ncclGroupStart", fn () -> ncclResult_t]()()


fn ncclGroupEnd() raises -> ncclResult_t:
    return _get_ccl_function["ncclGroupEnd", fn () -> ncclResult_t]()()


fn ncclCommInitAll(
    comms: UnsafePointer[ncclComm_t], ndev: Int, devlist: UnsafePointer[Int32]
) raises -> ncclResult_t:
    return _get_ccl_function[
        "ncclCommInitAll",
        fn (
            UnsafePointer[ncclComm_t], Int, UnsafePointer[Int32]
        ) -> ncclResult_t,
    ]()(comms, ndev, devlist)


@always_inline
fn _ccl_allreduce(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function["ncclAllReduce", CCLAllReduceFn]()(
        sendbuff, recvbuff, count, datatype, op, comm, stream_ptr
    )


# === AllGather binding (unified) ===
@always_inline
fn _ccl_allgather(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function["ncclAllGather", CCLAllGatherFn]()(
        sendbuff, recvbuff, count, datatype, comm, stream_ptr
    )


@always_inline
fn _ccl_stream_ptr(ctx: DeviceContext) raises -> OpaquePointer:
    @parameter
    if has_amd_gpu_accelerator():
        return HIP(ctx.stream()).bitcast[NoneType]()
    else:
        return CUDA(ctx.stream()).bitcast[NoneType]()


@fieldwise_init
struct Communicators(ImplicitlyCopyable, Movable):
    var ngpus: Int
    var comms: InlineArray[ncclComm_t, MAX_GPUS]


fn _dtype_to_ccl[dtype: DType]() raises -> ncclDataType_t:
    @parameter
    if dtype is DType.float32:
        return ncclDataType_t.ncclFloat32
    elif dtype is DType.bfloat16:
        return ncclDataType_t.ncclBfloat16
    elif dtype is DType.float16:
        return ncclDataType_t.ncclFloat16

    raise Error("vendor_ccl: dtype not supported: ", dtype)


@always_inline
fn _check_ccl_ok(status: ncclResult_t) raises:
    if status != ncclResult_t.ncclSuccess:
        raise Error("CCL call failed with status ", Int(status._value))


fn _get_global_comms(ngpus: Int) raises -> Communicators:
    var NAME = String("COMM_VENDOR_CCL_", ngpus)
    if global_ptr := _get_global_or_null(NAME).bitcast[Communicators]():
        return global_ptr[]

    if ngpus > MAX_GPUS:
        raise Error("too many GPUs for CCL")

    var comms = InlineArray[ncclComm_t, MAX_GPUS](fill={})
    var devlist = InlineArray[Int32, MAX_GPUS](fill={})
    for i in range(ngpus):
        devlist[i] = Int32(i)

    _check_ccl_ok(
        ncclCommInitAll(comms.unsafe_ptr(), ngpus, devlist.unsafe_ptr())
    )

    var c = Communicators(ngpus=ngpus, comms=comms)
    var ptr = UnsafePointer[Communicators].alloc(1)
    ptr.init_pointee_move(c)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(NAME), ptr.bitcast[NoneType]()
    )
    return ptr[]


@parameter
fn allreduce[
    dtype: DType,
    rank: Int,
    ngpus: Int,
](
    inputs: InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus],
    outputs: InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus],
    list_of_ctx: List[DeviceContext],
) raises:
    if ngpus < 1:
        raise Error("ngpus must be >= 1")
    if ngpus > MAX_GPUS:
        raise Error("too many GPUs")
    if len(list_of_ctx) != ngpus:
        raise Error("ctx count must match ngpus")

    var count = inputs[0].num_elements()
    var dtype_rccl = _dtype_to_ccl[dtype]()
    var op = ncclRedOp_t.ncclSum
    var comms = _get_global_comms(ngpus)

    _check_ccl_ok(ncclGroupStart())

    for i in range(ngpus):
        with list_of_ctx[i].push_context():
            _check_ccl_ok(
                _ccl_allreduce(
                    inputs[i].data.bitcast[NoneType](),
                    outputs[i].data.bitcast[NoneType](),
                    count,
                    dtype_rccl,
                    op,
                    comms.comms[i],
                    list_of_ctx[i],
                )
            )

    _check_ccl_ok(ncclGroupEnd())


@parameter
fn _is_ccl_symbol_available[name: StaticString]() -> Bool:
    # Resolve a CCL symbol by name from the appropriate vendor DSO.
    # We intentionally cast to a trivial signature and do not call it.
    try:
        _ = _get_ccl_function[name, fn () -> ncclResult_t]()
        return True
    except:
        return False


fn is_allreduce_available() -> Bool:
    return _is_ccl_symbol_available["ncclAllReduce"]()


fn is_allgather_available() -> Bool:
    return _is_ccl_symbol_available["ncclAllGather"]()


@parameter
fn allgather[
    dtype: DType,
    rank: Int,
    ngpus: Int,
](
    inputs: InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus],
    outputs: InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus * ngpus],
    list_of_ctx: List[DeviceContext],
) raises:
    if ngpus < 1:
        raise Error("ngpus must be >= 1")
    if ngpus > MAX_GPUS:
        raise Error("too many GPUs")
    if len(list_of_ctx) != ngpus:
        raise Error("ctx count must match ngpus")

    var count = inputs[0].num_elements()
    for i in range(ngpus):
        if inputs[i].num_elements() != count:
            raise Error("vendor_ccl allgather requires equal per-rank counts")

    var dtype_nccl = _dtype_to_ccl[dtype]()
    var comms = _get_global_comms(ngpus)

    var recv_tmp = List[DeviceBuffer[dtype]](capacity=ngpus)
    for i in range(ngpus):
        recv_tmp.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](ngpus * count)
        )

    _check_ccl_ok(ncclGroupStart())

    for i in range(ngpus):
        with list_of_ctx[i].push_context():
            _check_ccl_ok(
                _ccl_allgather(
                    inputs[i].data.bitcast[NoneType](),
                    recv_tmp[i].unsafe_ptr().bitcast[NoneType](),
                    count,
                    dtype_nccl,
                    comms.comms[i],
                    list_of_ctx[i],
                )
            )

    _check_ccl_ok(ncclGroupEnd())

    for dev in range(ngpus):
        var ctx = list_of_ctx[dev]
        for src in range(ngpus):
            var src_off = src * count
            var out_idx = dev * ngpus + src
            var dest_db = DeviceBuffer[dtype](
                ctx, outputs[out_idx].data, count, owning=False
            )
            var src_db = DeviceBuffer[dtype](
                ctx, recv_tmp[dev].unsafe_ptr() + src_off, count, owning=False
            )
            # API takes (dst, src)
            ctx.enqueue_copy(dest_db, src_db)
