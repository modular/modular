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
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime.asyncrt import DeviceContextPtr

from comm import MAX_GPUS, Signal
from comm.broadcast import broadcast
from layout import Idx, TileTensor, row_major


@export
def PyInit_distributed_ops() -> PythonObject:
    """Creates a Python module with distributed-ops bindings."""
    try:
        var b = PythonModuleBuilder("distributed_ops")
        b.def_function[broadcast_kernel]("broadcast_kernel")
        return b.finalize()
    except e:
        abort(t"failed to create distributed_ops bindings module: {e}")


@export
def broadcast_kernel(
    input_data_ptr: PythonObject,
    output_data_ptrs: PythonObject,
    signal_data_ptrs: PythonObject,
    device_context_ptrs: PythonObject,
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
                output_data_ptrs,
                signal_data_ptrs,
                device_context_ptrs,
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
    output_data_ptrs: PythonObject,
    signal_data_ptrs: PythonObject,
    device_context_ptrs: PythonObject,
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
    var in_tile = TileTensor(in_ptr, row_major(Idx(n))).as_immut()

    for i in range(ngpus):
        var out_addr = Int(py=output_data_ptrs[i])
        var ctx_addr = Int(py=device_context_ptrs[i])
        var ctx = DeviceContextPtr(
            OpaquePointer[MutExternalOrigin](unsafe_from_address=ctx_addr)
        ).get_device_context()
        var out_ptr = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=out_addr
        )
        var out_tile = TileTensor(out_ptr, row_major(Idx(n)))

        # Matches the graph-side ``mo.distributed.broadcast`` which also runs
        # without multimem. Enabling the multimem (multicast-store) path needs
        # an SM90+ build target for the shared library, which would require
        # per-arch ``.so`` variants.
        broadcast[ngpus, use_multimem=False](
            in_tile, out_tile, rank_sigs, ctx, root_v
        )
