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

from std.sys import has_amd_gpu_accelerator
from std.pathlib import Path
from std.ffi import _get_global_or_null, external_call
from std.ffi import _find_dylib
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import OwnedDLHandle, _Global
from std.collections.optional import Optional
from buffer import NDBuffer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host._amdgpu_hip import HIP
from std.gpu.host._nvidia_cuda import CUDA
from comm import MAX_GPUS, Signal
from comm.allreduce import elementwise_epilogue_type
from std.gpu.primitives.grid_controls import PDLLevel

# Safety: don't use `ExternalOrigin` here as that will turn of extending the
# lifetime of any Mojo structs that pass an internal pointer to the functions
# below.
comptime ncclComm_t = OpaquePointer[AnyOrigin[mut=True]]


@fieldwise_init
struct ncclResult_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime ncclSuccess = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ncclResult_t(", Int(self._value), ")")


@fieldwise_init
struct ncclRedOp_t(TrivialRegisterPassable):
    var _value: Int32
    comptime ncclSum = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)


@fieldwise_init
struct ncclDataType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime ncclFloat16 = Self(6)
    comptime ncclFloat32 = Self(7)
    comptime ncclBfloat16 = Self(9)

    def __init__(out self, value: Int):
        self._value = Int32(value)


comptime RCCL_LIBRARY_PATHS: List[Path] = [
    "librccl.so",
    "librccl.so.1",
    "/opt/rocm/lib/librccl.so",
    "/opt/rocm/lib/librccl.so.1",
]


comptime NCCL_LIBRARY_PATHS: List[Path] = [
    "libnccl.so",
    "libnccl.so.2",
    "/usr/lib/x86_64-linux-gnu/libnccl.so",
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
]


# Unified CCL loader (selects RCCL/NCCL at compile time)
def _init_ccl_dylib() -> OwnedDLHandle:
    comptime if has_amd_gpu_accelerator():
        return _find_dylib["RCCL"](materialize[RCCL_LIBRARY_PATHS]())
    else:
        return _find_dylib["NCCL"](materialize[NCCL_LIBRARY_PATHS]())


comptime CCL_LIBRARY = _Global["CCL_LIBRARY", _init_ccl_dylib]


@always_inline
def _get_ccl_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[CCL_LIBRARY(), func_name, result_type]()


# Paired wrappers grouped RCCl/NCCL for comparison
struct _Group:
    def __init__(out self):
        pass

    def __enter__(self) raises:
        _check_ccl_ok(
            _get_ccl_function["ncclGroupStart", fn() -> ncclResult_t]()()
        )

    def __exit__(self) raises:
        _check_ccl_ok(
            _get_ccl_function["ncclGroupEnd", fn() -> ncclResult_t]()()
        )


def group() -> _Group:
    return _Group()


def ncclCommInitAll(
    comms: UnsafePointer[ncclComm_t, _],
    ndev: Int,
    devlist: UnsafePointer[Int32, _],
) raises -> ncclResult_t:
    return _get_ccl_function[
        "ncclCommInitAll",
        fn(type_of(comms), Int, type_of(devlist)) -> ncclResult_t,
    ]()(comms, ndev, devlist)


@always_inline
def _ccl_allreduce(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function[
        "ncclAllReduce",
        fn(
            type_of(sendbuff),
            type_of(recvbuff),
            Int,
            ncclDataType_t,
            ncclRedOp_t,
            ncclComm_t,
            type_of(stream_ptr),
        ) -> ncclResult_t,
    ]()(sendbuff, recvbuff, count, datatype, op, comm, stream_ptr)


# === AllGather binding (unified) ===
@always_inline
def _ccl_allgather(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function[
        "ncclAllGather",
        fn(
            type_of(sendbuff),
            type_of(recvbuff),
            Int,
            ncclDataType_t,
            ncclComm_t,
            type_of(stream_ptr),
        ) -> ncclResult_t,
    ]()(sendbuff, recvbuff, count, datatype, comm, stream_ptr)


# === Broadcast binding (unified) ===
@always_inline
def _ccl_broadcast(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    root: Int,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function[
        "ncclBroadcast",
        fn(
            type_of(sendbuff),
            type_of(recvbuff),
            Int,
            ncclDataType_t,
            Int,
            ncclComm_t,
            type_of(stream_ptr),
        ) -> ncclResult_t,
    ]()(
        sendbuff,
        recvbuff,
        count,
        datatype,
        root,
        comm,
        stream_ptr,
    )


@always_inline
def _ccl_stream_ptr(
    ctx: DeviceContext,
) raises -> OpaquePointer[ExternalOrigin[mut=True]]:
    comptime if has_amd_gpu_accelerator():
        return HIP(ctx.stream()).bitcast[NoneType]()
    else:
        return CUDA(ctx.stream()).bitcast[NoneType]()


@fieldwise_init
struct Communicators(ImplicitlyCopyable):
    var ngpus: Int
    var comms: InlineArray[ncclComm_t, MAX_GPUS]

    def __init__(out self, *, copy: Self):
        self.ngpus = copy.ngpus
        self.comms = copy.comms.copy()


def _dtype_to_ccl[dtype: DType]() raises -> ncclDataType_t:
    comptime if dtype == DType.float32:
        return ncclDataType_t.ncclFloat32
    elif dtype == DType.bfloat16:
        return ncclDataType_t.ncclBfloat16
    elif dtype == DType.float16:
        return ncclDataType_t.ncclFloat16

    raise Error("vendor_ccl: dtype not supported: ", dtype)


@always_inline
def _check_ccl_ok(status: ncclResult_t) raises:
    if status != ncclResult_t.ncclSuccess:
        raise Error("CCL call failed with status ", Int(status._value))


def _get_global_comms(ngpus: Int) raises -> Communicators:
    var NAME = String(t"COMM_VENDOR_CCL_{ngpus}")
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

    var c = Communicators(ngpus=ngpus, comms=comms.copy())

    var ptr = alloc[Communicators](1)
    ptr.init_pointee_move(c)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(NAME), ptr.bitcast[NoneType]()
    )
    return ptr[]


def init_comms(ngpus: Int) raises:
    """Pre-initialize NCCL/RCCL communicators.

    Must be called from a single thread before using allreduce
    from multiple threads. This ensures thread-safe initialization since
    ncclCommInitAll is not designed for concurrent calls.
    """
    _ = _get_global_comms(ngpus)


@parameter
def allreduce[
    dtype: DType,
    input_origin: Origin[mut=False],
    rank_sigs_origin: Origin[mut=True],
    //,
    rank: Int,
    ngpus: Int,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    use_multimem: Bool = False,
](
    input_buffers: InlineArray[
        NDBuffer[rank=rank, dtype, input_origin], 1 if use_multimem else ngpus
    ],
    output_buffer: NDBuffer[rank=rank, dtype, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, rank_sigs_origin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-GPU allreduce for use in multi-threaded contexts.

    Currently requires prior single-threaded call to init_comms, as thread-safe
    version not yet implemented.
    """
    comptime assert (
        not output_lambda
    ), "vendor_ccl allreduce does not support output epilogue lambdas yet"
    comptime assert (
        not use_multimem
    ), "vendor_ccl allreduce does not support multimem path"
    # Determine this device's rank from its context id.
    var device_rank = Int(ctx.id())
    var count = input_buffers[0].num_elements()
    var dtype_ccl = _dtype_to_ccl[dtype]()
    var op = ncclRedOp_t.ncclSum
    var comms = _get_global_comms(ngpus)

    var input_buffer = input_buffers[0] if use_multimem else input_buffers[
        device_rank
    ]

    _check_ccl_ok(
        _ccl_allreduce(
            input_buffer.data.bitcast[NoneType](),
            output_buffer.data.bitcast[NoneType](),
            count,
            dtype_ccl,
            op,
            comms.comms[device_rank],
            ctx,
        )
    )


@parameter
def _is_ccl_symbol_available[name: StaticString]() -> Bool:
    # Resolve a CCL symbol by name from the appropriate vendor DSO.
    # We intentionally cast to a trivial signature and do not call it.
    try:
        _ = _get_ccl_function[name, fn() -> ncclResult_t]()
        return True
    except:
        return False


def is_allreduce_available() -> Bool:
    return _is_ccl_symbol_available["ncclAllReduce"]()


def is_allgather_available() -> Bool:
    return _is_ccl_symbol_available["ncclAllGather"]()


def is_broadcast_available() -> Bool:
    return _is_ccl_symbol_available["ncclBroadcast"]()


@parameter
def allgather[
    dtype: DType,
    inputs_origin: Origin[mut=False],
    outputs_origin: Origin[mut=True],
    //,
    rank: Int,
    ngpus: Int,
](
    inputs: InlineArray[NDBuffer[rank=rank, dtype, inputs_origin], ngpus],
    outputs: InlineArray[
        NDBuffer[rank=rank, dtype, outputs_origin], ngpus * ngpus
    ],
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

    with group():
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


@parameter
def broadcast[
    dtype: DType,
    rank: Int,
    input_origin: Origin[mut=False],
    output_origin: Origin[mut=True],
    //,
    ngpus: Int,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    input_buffer: NDBuffer[rank=rank, dtype, input_origin],
    output_buffer: NDBuffer[rank=rank, dtype, output_origin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    root: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-GPU broadcast for use in multi-threaded contexts.

    Currently requires prior single-threaded call to init_comms, as thread-safe
    version not yet implemented.
    """
    comptime assert (
        not use_multimem
    ), "vendor_ccl broadcast does not support multimem path"
    # Determine this device's rank from its context id.
    var device_rank = Int(ctx.id())
    var count = output_buffer.num_elements()
    var dtype_ccl = _dtype_to_ccl[dtype]()
    var comms = _get_global_comms(ngpus)

    _check_ccl_ok(
        _ccl_broadcast(
            input_buffer.data.bitcast[NoneType](),
            output_buffer.data.bitcast[NoneType](),
            count,
            dtype_ccl,
            root,
            comms.comms[device_rank],
            ctx,
        )
    )
