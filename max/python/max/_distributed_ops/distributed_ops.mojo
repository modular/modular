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

from std.collections import InlineArray
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.gpu.host import DeviceContext, DeviceContextList
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder


from comm import MAX_GPUS, Signal
from comm.broadcast import broadcast
from comm.device_collective import _launch_device_collective
from layout import Idx, TileTensor, row_major


@export
def PyInit_distributed_ops() abi("C") -> PythonObject:
    """Creates a Python module with distributed-ops bindings."""
    try:
        var b = PythonModuleBuilder("distributed_ops")
        b.def_function[broadcast_kernel]("broadcast_kernel")
        return b.finalize()
    except e:
        abort(t"failed to create distributed_ops bindings module: {e}")


def broadcast_kernel(
    input_data_ptr: PythonObject,
    output_buffers: PythonObject,
    signal_data_ptrs: PythonObject,
    num_bytes: PythonObject,
    ngpus: PythonObject,
    root: PythonObject,
) raises -> PythonObject:
    """Broadcasts ``num_bytes`` bytes from rank ``root`` to every rank's output.

    The transfer is byte-oriented; callers can broadcast any dtype by passing
    the buffer's total byte count and trusting the broadcast kernel's
    uint8 SIMD path (which matches a float32 SIMD path for memory-bound
    transfers).
    """
    var ngpus_v = Int(py=ngpus)

    comptime for n in range(2, MAX_GPUS + 1):
        if ngpus_v == n:
            _do_broadcast[n](
                input_data_ptr,
                output_buffers,
                signal_data_ptrs,
                num_bytes,
                root,
            )
            return Python.none()

    raise Error(
        t"distributed_broadcast: ngpus={ngpus_v} must be in [2, {MAX_GPUS}]"
    )


@parameter
def _do_broadcast[
    ngpus: Int
](
    input_data_ptr: PythonObject,
    output_buffers: PythonObject,
    signal_data_ptrs: PythonObject,
    num_bytes: PythonObject,
    root: PythonObject,
) raises:
    var n = Int(py=num_bytes)
    var root_v = Int(py=root)
    var in_addr = Int(py=input_data_ptr)

    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    for i in range(ngpus):
        var sig_addr = Int(py=signal_data_ptrs[i])
        rank_sigs[i] = UnsafePointer[Signal, MutAnyOrigin](
            unsafe_from_address=sig_addr
        )

    var in_ptr = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](
        unsafe_from_address=in_addr
    )
    var in_tile = TileTensor(in_ptr, row_major(n)).as_immut()

    var out_ptrs = InlineArray[
        UnsafePointer[Scalar[DType.uint8], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var ctx_array = InlineArray[DeviceContext, ngpus](uninitialized=True)
    for i in range(ngpus):
        var buf = output_buffers[i]
        var out_addr = Int(py=buf._data_ptr())
        var ctx_addr = Int(py=buf.device._device_context_ptr())
        out_ptrs[i] = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=out_addr
        )
        # init_pointee_move prevents DeviceContext.__del__ from dropping a
        # refcount, so assigning into the uninitialized slot would destroy it
        (ctx_array.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](unsafe_from_address=ctx_addr)
            )
        )
    var dev_ctxs = DeviceContextList[ngpus](ctx_array^)

    @always_inline
    def launch_broadcast[
        index: Int
    ]() raises {
        read in_tile,
        read rank_sigs,
        read out_ptrs,
        read dev_ctxs,
        read n,
        read root_v,
    }:
        var out_tile = TileTensor(out_ptrs[index], row_major(n))
        # use_multimem=False: the multicast-store path needs an SM90+ build
        # target, i.e. per-arch .so variants of the shared library.
        broadcast[ngpus, use_multimem=False](
            in_tile, out_tile, rank_sigs, dev_ctxs[index], root_v
        )

    # Release the GIL during the blocking tg.wait() so other Python threads
    # aren't stalled while the enqueues run on worker threads. Pass a copy:
    # launch_broadcast borrows dev_ctxs, which the call below moves.
    with GILReleased(Python()):
        _launch_device_collective[ngpus](
            launch_broadcast, DeviceContextList[ngpus](copy=dev_ctxs)
        )
