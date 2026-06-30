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

"""Mojo extension for batched KV-cache block H2D and D2H copies.

``copy_h2d`` handles the full H2D pipeline for one ``(dst, src)`` block pair:

1. Enqueue one async DMA per device buffer on its aux stream.
2. Insert a device-side barrier: ``main_ctx.enqueue_wait_for(aux_ctx)`` for
   each unit, so the main stream will not advance until the DMA completes.
3. For replicated (MLA) units: fan-out via a peer broadcast on the main stream
   of every participating device.

``DeviceContext`` objects are built once from pre-extracted integer addresses
(before the GIL is released) and reused across all loop iterations.  The GIL
is then released for the entire GPU-work region.

``copy_d2h`` handles all D2H copies in a single batched call.
"""

from std.collections import InlineArray
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.gpu.host import DeviceBuffer, DeviceContext, DeviceContextList
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from comm import MAX_GPUS, Signal
from comm.broadcast import broadcast
from comm.device_collective import _launch_device_collective
from layout import TileTensor, row_major

# The number of KV-cache units (sharded MHA layers + replicated MLA units) is
# NOT bounded by the GPU count, so per-unit data is pre-extracted into
# heap-backed ``List``s rather than fixed-size ``InlineArray``s. Only arrays
# indexed purely by GPU rank are sized with ``MAX_GPUS``.


@export
def PyInit_block_offload_ops() abi("C") -> PythonObject:
    """Creates a Python module exposing batched H2D/D2H copy kernels."""
    try:
        var b = PythonModuleBuilder("block_offload_ops")
        b.def_function[copy_h2d]("copy_h2d")
        b.def_function[copy_d2h]("copy_d2h")
        return b.finalize()
    except e:
        abort(t"failed to create block_offload_ops bindings: {e}")


def copy_h2d(
    host_info: PythonObject,
    aux_info: PythonObject,
    dsts: PythonObject,
    srcs: PythonObject,
    wait_main_ctx_ptrs: PythonObject,
    bcast_info: PythonObject,
) raises -> PythonObject:
    """Enqueue H2D, stream wait, and peer broadcast for all ``(dst, src)`` pairs.

    Loops over every pair inside Mojo with the GIL released, so a single
    Python call covers the full request batch.

    For each pair:
    1. **H2D DMA** (aux stream): one DMA per entry in ``aux_info[0]``.
    2. **Stream wait** (device-side): ``main_ctx.enqueue_wait_for(aux_ctx)``
       per entry in ``wait_main_ctx_ptrs``.  Pass ``[]`` for non-replicated.
    3. **Broadcast** (main stream): fan the block out to all peers.
       Skipped when ``bcast_info`` is empty.

    All Python values are extracted into native Mojo arrays before the GIL is
    released.  ``DeviceContext`` objects are built once and reused across all
    pairs and phases.

    Args:
        host_info:           ``[host_buf_ptr, host_stride]``.
        aux_info:            ``[[buf_ptrs], [strides], [ctx_ptrs]]`` — one
                             entry per device buffer, each bound to its aux
                             stream (from ``DeviceStream._device_context_ptr()``).
        dsts:                ``[n]`` destination device block IDs.
        srcs:                ``[n]`` source host block IDs.
        wait_main_ctx_ptrs:  ``[num_units]`` main-stream DeviceContext pointers
                             for the stream-wait barrier.  ``[]`` = skip.
        bcast_info:          ``[[sig_ptrs], [out_ptrs], [out_strides],
                             [main_ctx_ptrs]]`` for replicated (MLA) units,
                             or ``[]`` to skip the broadcast entirely.
                             ``out_ptrs`` is ``[num_units][ngpus]`` nested.
    """
    # ------------------------------------------------------------------ #
    # Pre-extraction: materialise all Python ints into native Mojo arrays. #
    # ------------------------------------------------------------------ #
    var host_base = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](
        unsafe_from_address=Int(py=host_info[0])
    )
    var host_stride_v = Int(py=host_info[1])

    var aux_buf_ptrs = aux_info[0]
    var aux_strides = aux_info[1]
    var aux_ctx_ptrs = aux_info[2]
    var num_aux_v = len(aux_buf_ptrs)
    var num_wait_v = len(wait_main_ctx_ptrs)
    var n_v = len(dsts)

    var aux_buf_arr = List[Int](capacity=num_aux_v)
    var aux_stride_arr = List[Int](capacity=num_aux_v)
    var aux_ctx_addr_arr = List[Int](capacity=num_aux_v)
    var wait_ctx_addr_arr = List[Int](capacity=num_wait_v)
    for j in range(num_aux_v):
        aux_buf_arr.append(Int(py=aux_buf_ptrs[j]))
        aux_stride_arr.append(Int(py=aux_strides[j]))
        aux_ctx_addr_arr.append(Int(py=aux_ctx_ptrs[j]))
    for j in range(num_wait_v):
        wait_ctx_addr_arr.append(Int(py=wait_main_ctx_ptrs[j]))

    # sig_arr / main_ctx_addr_arr are indexed by GPU rank, so MAX_GPUS is a
    # correct bound. out_stride_arr / out_addr_arr are per-unit, so they use
    # heap-backed Lists; out_addr_arr is flat row-major as [num_units][ngpus].
    var sig_arr = InlineArray[Int, MAX_GPUS](uninitialized=True)
    var main_ctx_addr_arr = InlineArray[Int, MAX_GPUS](uninitialized=True)
    var out_stride_arr = List[Int]()
    var out_addr_arr = List[Int]()
    var num_bcast_v = 0
    var ngpus_v = 0
    if len(bcast_info) > 0:
        var bcast_signal_ptrs = bcast_info[0]
        var bcast_out_ptrs = bcast_info[1]
        var bcast_out_strides = bcast_info[2]
        var bcast_main_ctx_ptrs = bcast_info[3]
        num_bcast_v = len(bcast_out_ptrs)
        ngpus_v = len(bcast_main_ctx_ptrs)
        out_stride_arr = List[Int](capacity=num_bcast_v)
        out_addr_arr = List[Int](capacity=num_bcast_v * ngpus_v)
        for i in range(ngpus_v):
            sig_arr[i] = Int(py=bcast_signal_ptrs[i])
            main_ctx_addr_arr[i] = Int(py=bcast_main_ctx_ptrs[i])
        for u in range(num_bcast_v):
            out_stride_arr.append(Int(py=bcast_out_strides[u]))
            var row = bcast_out_ptrs[u]
            for i in range(ngpus_v):
                out_addr_arr.append(Int(py=row[i]))

    # Block IDs: List (heap-backed) since n_v can be large; RAII-freed on every
    # exit path, including if a later conversion or enqueue raises.
    var dst_arr = List[Int](capacity=n_v)
    var src_arr = List[Int](capacity=n_v)
    for i in range(n_v):
        dst_arr.append(Int(py=dsts[i]))
        src_arr.append(Int(py=srcs[i]))

    # ------------------------------------------------------------------ #
    # Build DeviceContext objects once from the extracted addresses.       #
    # Done here, with the GIL still held, but reused inside the             #
    # GIL-released region below; building them touches no Python objects.   #
    # ------------------------------------------------------------------ #
    var aux_ctxs = List[DeviceContext](capacity=num_aux_v)
    for j in range(num_aux_v):
        aux_ctxs.append(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=aux_ctx_addr_arr[j]
                )
            )
        )
    var wait_main_ctxs = List[DeviceContext](capacity=num_wait_v)
    for j in range(num_wait_v):
        wait_main_ctxs.append(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=wait_ctx_addr_arr[j]
                )
            )
        )

    # ------------------------------------------------------------------ #
    # GPU-work region: GIL released for the entire batch.                 #
    # ------------------------------------------------------------------ #
    with GILReleased(Python()):
        for i in range(n_v):
            var dst_v = dst_arr[i]
            var src_v = src_arr[i]

            # Phase 1: async H2D DMAs on aux streams.
            var host_off = 0
            for j in range(num_aux_v):
                var stride = aux_stride_arr[j]
                var dev_base = UnsafePointer[
                    Scalar[DType.uint8], MutUntrackedOrigin
                ](unsafe_from_address=aux_buf_arr[j])
                var dev_buf = DeviceBuffer[DType.uint8](
                    aux_ctxs[j],
                    dev_base + dst_v * stride,
                    stride,
                    owning=False,
                )
                aux_ctxs[j].enqueue_copy(
                    dev_buf, host_base + src_v * host_stride_v + host_off
                )
                host_off += stride

            # Phase 2: device-side stream barriers.
            for j in range(num_wait_v):
                wait_main_ctxs[j].enqueue_wait_for(aux_ctxs[j])

            # Phase 3: peer broadcast for replicated (MLA) units.
            if num_bcast_v > 0:
                comptime for k in range(2, MAX_GPUS + 1):
                    if ngpus_v == k:
                        _do_broadcast_units[k](
                            dst_v,
                            num_bcast_v,
                            sig_arr,
                            out_addr_arr,
                            out_stride_arr,
                            main_ctx_addr_arr,
                        )

    # dst_arr / src_arr / aux_ctxs / wait_main_ctxs are Lists: their elements
    # and backing storage are released automatically when they go out of scope.

    return Python.none()


@parameter
def _do_broadcast_units[
    ngpus: Int
](
    dst: Int,
    num_units: Int,
    sig_arr: InlineArray[Int, MAX_GPUS],
    out_addr_arr: List[Int],
    out_stride_arr: List[Int],
    main_ctx_addr_arr: InlineArray[Int, MAX_GPUS],
) raises:
    """Broadcast block ``dst`` from root to all peers for each replicated unit.

    Called with the GIL already released.  All inputs are native Mojo ints.
    The ``DeviceContextList`` is built once and shallow-copied for each unit's
    ``_launch_device_collective`` call.
    """
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    for i in range(ngpus):
        rank_sigs[i] = UnsafePointer[Signal, MutAnyOrigin](
            unsafe_from_address=sig_arr[i]
        )

    # Build the DeviceContextList once; copy it for each unit's launch.
    var ctx_array = InlineArray[DeviceContext, ngpus](uninitialized=True)
    for i in range(ngpus):
        (ctx_array.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=main_ctx_addr_arr[i]
                )
            )
        )
    var dev_ctxs = DeviceContextList[ngpus](ctx_array^)

    for u in range(num_units):
        var stride = out_stride_arr[u]
        var unit_ptrs = InlineArray[
            UnsafePointer[Scalar[DType.uint8], MutAnyOrigin], ngpus
        ](uninitialized=True)
        for i in range(ngpus):
            unit_ptrs[i] = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](
                unsafe_from_address=out_addr_arr[u * ngpus + i] + dst * stride
            )
        var in_tile = TileTensor(unit_ptrs[0], row_major(stride)).as_immut()

        @always_inline
        def launch_broadcast[
            index: Int
        ]() raises {
            read in_tile,
            read rank_sigs,
            read unit_ptrs,
            read dev_ctxs,
            read stride,
        }:
            var out_tile = TileTensor(unit_ptrs[index], row_major(stride))
            broadcast[ngpus, use_multimem=False](
                in_tile, out_tile, rank_sigs, dev_ctxs[index], 0
            )

        _launch_device_collective[ngpus](
            launch_broadcast, DeviceContextList[ngpus](copy=dev_ctxs)
        )


def copy_d2h(
    host_buf_ptr: PythonObject,
    host_stride: PythonObject,
    device_buf_ptrs: PythonObject,
    device_strides: PythonObject,
    device_ctx_ptrs: PythonObject,
    dst_ids: PythonObject,
    src_ids: PythonObject,
) raises -> PythonObject:
    """Enqueue async D2H copies for all block pairs across all device buffers.

    For each ``(dst, src)`` pair and device buffer ``j``:

        host_buf[dst, offset_j : offset_j + strides[j]] ← device_bufs[j][src, :]

    ``DeviceContext`` objects are built once and reused across all block pairs;
    the GIL is released for the full enqueue region.

    Args:
        host_buf_ptr:     Integer address of the pinned host buffer.
        host_stride:      Row stride of the host buffer in bytes.
        device_buf_ptrs:  ``[num_units]`` integer device-buffer base addresses.
        device_strides:   ``[num_units]`` bytes-per-block per buffer.
        device_ctx_ptrs:  ``[num_units]`` aux-stream DeviceContext pointers.
        dst_ids:          Destination host block IDs.
        src_ids:          Source device block IDs.
    """
    var host_base = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](
        unsafe_from_address=Int(py=host_buf_ptr)
    )
    var host_stride_v = Int(py=host_stride)
    var num_bufs_v = len(device_buf_ptrs)
    var n_v = len(dst_ids)

    var buf_arr = List[Int](capacity=num_bufs_v)
    var stride_arr = List[Int](capacity=num_bufs_v)
    var ctx_addr_arr = List[Int](capacity=num_bufs_v)
    for j in range(num_bufs_v):
        buf_arr.append(Int(py=device_buf_ptrs[j]))
        stride_arr.append(Int(py=device_strides[j]))
        ctx_addr_arr.append(Int(py=device_ctx_ptrs[j]))

    # Block IDs: List (heap-backed) since n_v can be large; RAII-freed on every
    # exit path, including if a later conversion or enqueue raises.
    var dst_arr = List[Int](capacity=n_v)
    var src_arr = List[Int](capacity=n_v)
    for i in range(n_v):
        dst_arr.append(Int(py=dst_ids[i]))
        src_arr.append(Int(py=src_ids[i]))

    # Build one DeviceContext per buffer; reused for all n_v block pairs.
    var dev_ctxs = List[DeviceContext](capacity=num_bufs_v)
    for j in range(num_bufs_v):
        dev_ctxs.append(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=ctx_addr_arr[j]
                )
            )
        )

    with GILReleased(Python()):
        for i in range(n_v):
            var host_off = 0
            for j in range(num_bufs_v):
                var stride = stride_arr[j]
                var dev_base = UnsafePointer[
                    Scalar[DType.uint8], MutUntrackedOrigin
                ](unsafe_from_address=buf_arr[j])
                var dev_buf = DeviceBuffer[DType.uint8](
                    dev_ctxs[j],
                    dev_base + src_arr[i] * stride,
                    stride,
                    owning=False,
                )
                dev_ctxs[j].enqueue_copy(
                    host_base + dst_arr[i] * host_stride_v + host_off,
                    dev_buf,
                )
                host_off += stride

    # dev_ctxs / dst_arr / src_arr are Lists: their elements and backing storage
    # are released automatically when they go out of scope.

    return Python.none()
