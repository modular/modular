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
"""Tests the P2P all-to-all against a host reference built from coordinates.

The reference never uses the kernel's run arithmetic: for every output
coordinate it finds the source rank from the concat axis and reads that
rank's input at the coordinate the split axis maps it to. Inputs carry
their device and flat index, exactly representable in float32, and outputs
start poisoned, so a misplaced or unwritten element fails.
"""

from std.collections import Array
from std.math.uutils import ualign_down
from std.sys import size_of

from comm import Signal
from comm.alltoall import all_to_all
from comm.sync import enable_p2p, init_signal_buffer
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.testing import assert_equal, assert_true


def _product(shape: List[Int], start: Int = 0) -> Int:
    var n = 1
    for d in range(start, len(shape)):
        n *= shape[d]
    return n


def _flat(coord: List[Int], shape: List[Int]) -> Int:
    var f = 0
    for d in range(len(shape)):
        f = f * shape[d] + coord[d]
    return f


def _coord(flat: Int, shape: List[Int]) -> List[Int]:
    var coord = List[Int](length=len(shape), fill=0)
    var rest = flat
    for d in reversed(range(len(shape))):
        coord[d] = rest % shape[d]
        rest //= shape[d]
    return coord^


def _value[dtype: DType](device: Int, index: Int) -> Scalar[dtype]:
    comptime if dtype == .bfloat16:
        # bfloat16 holds integers up to 256 exactly.
        return Scalar[dtype]((device * 37 + index * 11) % 251)
    else:
        # Unique and exact in float32 up to 2M elements per device.
        return Scalar[dtype](index * 8 + device)


def all_to_all_test[
    dtype: DType, ngpus: Int, group_size: Int = ngpus
](
    list_of_ctx: List[DeviceContext[]],
    in_shape: List[Int],
    split_axis: Int,
    concat_axis: Int,
) raises -> None:
    print(
        t"  all_to_all {dtype} ngpus={ngpus} group={group_size}"
        t" split={split_axis} concat={concat_axis} shape={in_shape}"
    )
    var n = group_size
    var out_shape = in_shape.copy()
    out_shape[split_axis] //= n
    out_shape[concat_axis] *= n
    var num_elements = _product(in_shape)
    var in_inner = _product(in_shape, split_axis) // n
    var out_inner = _product(out_shape, concat_axis) // n

    var in_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var host_inputs = List[HostBuffer[dtype]](capacity=ngpus)

    for i in range(ngpus):
        ref ctx = list_of_ctx[i]
        var alloc = max(num_elements, 1)
        in_bufs.append(ctx.create_buffer_sync[dtype](alloc))
        out_bufs.append(ctx.create_buffer_sync[dtype](alloc))
        var host = ctx.enqueue_create_host_buffer[dtype](alloc)
        ctx.synchronize()
        for j in range(num_elements):
            host[j] = _value[dtype](i, j)
        ctx.enqueue_copy(in_bufs[i], host)
        ctx.enqueue_memset[dtype](out_bufs[i], val=Scalar[dtype](-1))
        signal_buffers.append(ctx.create_buffer_sync[.uint8](size_of[Signal]()))
        init_signal_buffer(signal_buffers[i], ctx)
        host_inputs.append(host^)
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    var in_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> ImmPointer[
            Scalar[dtype], ImmutAnyOrigin
        ]: rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            in_bufs[i].unsafe_ptr()
        )
    )

    for i in range(ngpus):
        all_to_all[ngpus=ngpus, group_size=group_size](
            in_ptrs,
            out_bufs[i].unsafe_ptr().as_unsafe_any_origin(),
            num_elements,
            in_inner,
            out_inner,
            rank_sigs,
            list_of_ctx[i],
            i,
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var chunk_concat = out_shape[concat_axis] // n
    var chunk_split = in_shape[split_axis] // n
    for i in range(ngpus):
        ref ctx = list_of_ctx[i]
        var group_start = ualign_down(i, group_size)
        var loc_rank = i - group_start
        var host_out = ctx.enqueue_create_host_buffer[dtype](
            max(num_elements, 1)
        )
        ctx.enqueue_copy(host_out, out_bufs[i])
        ctx.synchronize()
        for f in range(num_elements):
            var coord = _coord(f, out_shape)
            var source = coord[concat_axis] // chunk_concat
            coord[concat_axis] %= chunk_concat
            coord[split_axis] += loc_rank * chunk_split
            var src_index = _flat(coord, in_shape)
            var expected = _value[dtype](group_start + source, src_index)
            if host_out[f] != expected:
                print(
                    t"    mismatch on device {i} at output {f}: got"
                    t" {host_out[f]}, expected {expected} (source"
                    t" {group_start + source}, index {src_index})"
                )
                assert_equal(host_out[f], expected)
    _ = host_inputs^


def _ctxs(count: Int) raises -> List[DeviceContext[]]:
    var ctx = List[DeviceContext[]]()
    for i in range(count):
        ctx.append(DeviceContext(device_id=i))
    return ctx^


def main() raises -> None:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    var ctx2 = _ctxs(2)
    # all_to_all_single: split and concatenate the leading axis.
    all_to_all_test[.float32, ngpus=2](ctx2, [6, 5, 8], 0, 0)
    # The DCP merge: heads split and gathered on the same axis.
    all_to_all_test[.float32, ngpus=2](ctx2, [3, 8, 12], 1, 1)
    # A sharding transpose: split heads, gather tokens.
    all_to_all_test[.float32, ngpus=2](ctx2, [3, 8, 12], 1, 0)
    # The innermost axis, with runs too short to vectorize.
    all_to_all_test[.float32, ngpus=2](ctx2, [4, 3, 10], 2, 0)
    # Odd runs on both sides take the scalar path.
    all_to_all_test[.bfloat16, ngpus=2](ctx2, [3, 2, 515], 1, 1)
    all_to_all_test[.bfloat16, ngpus=2](ctx2, [7, 16, 516], 1, 1)
    # No tokens: every rank returns without a barrier.
    all_to_all_test[.bfloat16, ngpus=2](ctx2, [0, 16, 516], 1, 1)

    if DeviceContext.number_of_devices() >= 4:
        var ctx4 = _ctxs(4)
        all_to_all_test[.float32, ngpus=4](ctx4, [5, 8, 64], 1, 1)
        all_to_all_test[.float32, ngpus=4](ctx4, [8, 4, 6], 0, 2)
        all_to_all_test[.bfloat16, ngpus=4](ctx4, [33, 64, 515], 1, 1)
        # Two independent groups of two.
        all_to_all_test[.float32, ngpus=4, group_size=2](ctx4, [5, 8, 64], 1, 1)

    if DeviceContext.number_of_devices() >= 8:
        var ctx8 = _ctxs(8)
        # GLM-5.3 under DCP8 at batch 32: 64 heads, 512 latent + 3 LSE lanes.
        all_to_all_test[.bfloat16, ngpus=8](ctx8, [32, 64, 515], 1, 1)
        all_to_all_test[.float32, ngpus=8](ctx8, [16, 64, 520], 1, 1)
        all_to_all_test[.float32, ngpus=8](ctx8, [64, 9, 7], 0, 0)
        # Two groups of four, as a (dp, tp) mesh runs it.
        all_to_all_test[.float32, ngpus=8, group_size=4](
            ctx8, [6, 16, 40], 1, 1
        )
