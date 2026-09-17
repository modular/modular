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
"""Destination-shard invariance oracle for the reduce-scatter sum.

A reduce-scatter element's sum must be a function of its `ngpus` input
values alone -- never of which destination shard (row range) the element
landed in. The kernel once rotated its peer-accumulation order by the
destination rank to stagger NVLink reads, which is a legal reassociation but
makes byte-identical rows summed at different row indices produce different
floats (one bf16 ULP in the model, amplified downstream into diverging
output.

Oracle: every rank's input carries ONE row pattern repeated down all rows,
and the patterns are built so the accumulation order is observable despite
the single final rounding to the narrow input dtype. Every output row on
every shard must then be bit-identical -- the summands are the same for
every (destination rank, row) pair, so any differing output is an
accumulation-order dependence. Each output row is also checked against a
host-side canonical-order (peer 0, 1, 2, ...) reference, which pins the
order itself, not just its shard-invariance.

The value pattern uses near-cancellation across a wide magnitude gap, the
one shape that survives the narrow output dtype: peer 0 contributes 2^24,
peer 1 a small integer, peer 2 cancels peer 0, peer 3 a small integer.
Canonical order folds the small terms into 2^24 first (lost at f32 ULP 2),
then cancels to ~0; an order starting at peer 1 keeps the small terms
exactly and they survive the cancellation. Distinct orders give sums that
differ by whole integers -- far above the bf16 rounding step.

Fails on the rotated order, passes on the canonical one. Two peers cannot
expose rotation (a two-term sum is order-independent), so the 2-GPU cases
are consistency checks for the ragged partition, and the 4-GPU cases gate.
"""

from std.sys import simd_width_of, size_of
from std.itertools import product
from std.utils.coord import _coerce_dynamic

from layout import Coord, TileTensor, row_major
from layout.coord import DynamicCoord
from std.collections import Array, Optional
from comm import Signal, MAX_GPUS
from comm.sync import enable_p2p, init_signal_buffer
from comm.reducescatter import reducescatter, ReduceScatterConfig
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer, get_gpu_target
from std.testing import assert_true
from std.utils.numerics import get_accum_type

comptime test_dtypes = (DType.bfloat16, DType.float32)
comptime test_gpu_counts = (2, 4)

# Row counts: one that divides evenly and one that does not (the ragged
# partition puts remainder rows on low ranks and shifts later rows' ranks).
comptime test_rows = (8, 10)

# Cancellation pair magnitude. 2^24 is where f32's mantissa ends, so adding
# a small integer to it rounds the integer away; below it nothing is lost.
comptime _CANCEL_MAG = 16777216.0


@inline(.always)
def _input_value[dtype: DType](gpu_rank: Int, col: Int) -> Scalar[dtype]:
    """One rank's contribution at a column, identical in every row.

    The wobble varies per column so the pattern is not degenerate; it stays
    in the small-integer band, where the order-induced differences (also
    small integers) are exactly representable in the output dtype.
    """
    var v = Scalar[DType.float32](0)
    if gpu_rank == 0:
        v = Scalar[DType.float32](_CANCEL_MAG)
    elif gpu_rank == 1:
        v = Scalar[DType.float32]((col % 3) + 1)
    elif gpu_rank == 2:
        v = Scalar[DType.float32](-_CANCEL_MAG)
    elif gpu_rank == 3:
        v = Scalar[DType.float32](-(col % 2))
    return v.cast[dtype]()


def shard_invariance_test[
    dtype: DType,
    ngpus: Int,
    rows: Int,
    cols: Int,
](list_of_ctx: List[DeviceContext]) raises:
    """Runs the collective and gates the bit-exactness oracle."""
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime accum_t = get_accum_type[dtype]()

    print(
        String(
            "====reducescatter-shard-invariance-",
            dtype,
            "-",
            ngpus,
            "gpus-rows",
            rows,
            "-cols",
            cols,
        )
    )

    var num_elements = rows * cols
    var config = ReduceScatterConfig[dtype, ngpus](rows, cols, 0)

    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[HostBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)

    for gpu_idx in range(ngpus):
        in_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](num_elements)
        )
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                config.rank_num_elements(gpu_idx)
            )
        )
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[gpu_idx], list_of_ctx[gpu_idx])

        var h = list_of_ctx[gpu_idx].enqueue_create_host_buffer[dtype](
            num_elements
        )
        for r in range(rows):
            for c in range(cols):
                # Same pattern in every row: cross-row equality is the oracle.
                h[r * cols + c] = _input_value[dtype](gpu_idx, c)
        list_of_ctx[gpu_idx].enqueue_copy(in_bufs_list[gpu_idx], h)
        host_in.append(h^)

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Host-side canonical reference: one row of the expected output,
    # accumulated as peer 0, 1, 2, ... in the accumulate type, with the same
    # single final rounding to `dtype` the kernel applies.
    var ref_row = List[Scalar[dtype]](length=cols, fill=Scalar[dtype](0))

    for c in range(cols):
        var accum = Scalar[accum_t](0)
        for k in range(ngpus):
            accum += Scalar[accum_t](_input_value[dtype](k, c))
        ref_row[c] = accum.cast[dtype]()

    comptime InputTileType = type_of(
        TileTensor[mut=False](
            in_bufs_list[0].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(rows, cols)),
        )
    )
    var in_bufs = Array[InputTileType, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: InputTileType(
            in_bufs_list[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(rows, cols)),
        )
    )
    comptime shape_type = DynamicCoord[.int, 2]
    comptime OutputTileType = type_of(
        TileTensor[mut=True](
            out_bufs_list[0].unsafe_ptr().as_unsafe_any_origin(),
            row_major(shape_type()),
        )
    )
    var out_bufs = Array[OutputTileType, ngpus](uninitialized=True)
    for i in range(ngpus):
        var output_shape = shape_type()
        output_shape[0] = _coerce_dynamic[output_shape.element_types[0]](
            config.rank_units(i)
        )
        output_shape[1] = _coerce_dynamic[output_shape.element_types[1]](cols)
        out_bufs[i] = OutputTileType(
            out_bufs_list[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(output_shape),
        )

    for i in range(ngpus):
        reducescatter[dtype=dtype, ngpus=ngpus, axis=0](
            in_bufs,
            out_bufs,
            rank_sigs,
            list_of_ctx[i],
            my_rank=Optional[Int](i),
        )

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Gate 1: every output row on every shard is bit-identical to the
    # canonical reference row. Under the rotated order, a rank whose
    # rotation starts at peer 1 or 2 keeps the small terms through the
    # cancellation and disagrees with the canonical sum by a whole integer.
    var checked = 0
    for gpu_idx in range(ngpus):
        var out_len = config.rank_num_elements(gpu_idx)
        var result_host = list_of_ctx[gpu_idx].enqueue_create_host_buffer[
            dtype
        ](out_len)
        list_of_ctx[gpu_idx].enqueue_copy(result_host, out_bufs_list[gpu_idx])
        list_of_ctx[gpu_idx].synchronize()

        var my_rows = config.rank_units(gpu_idx)
        for r in range(my_rows):
            for c in range(cols):
                assert_true(
                    result_host[r * cols + c] == ref_row[c],
                    msg=String(
                        "destination-shard dependence: gpu ",
                        gpu_idx,
                        " local row ",
                        r,
                        " col ",
                        c,
                        " got ",
                        result_host[r * cols + c],
                        ", canonical order gives ",
                        ref_row[c],
                    ),
                )
                checked += 1
        _ = result_host^

    # Every (rank, local row) pair was compared to the same canonical
    # reference, so cross-row and cross-shard equality both hold transitively.
    print(String("  checked ", checked, " elements, all bit-identical"))
    _ = host_in^


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    var list_of_ctx = List[DeviceContext](capacity=MAX_GPUS)
    for i in range(DeviceContext.number_of_devices()):
        list_of_ctx.append(DeviceContext(i))

    comptime for dtype_idx, ngpus_idx, rows_idx in product(
        range(len(test_dtypes)),
        range(len(test_gpu_counts)),
        range(len(test_rows)),
    ):
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime ngpus = rebind[Int](test_gpu_counts[ngpus_idx])
        comptime rows = rebind[Int](test_rows[rows_idx])
        if ngpus <= DeviceContext.number_of_devices():
            shard_invariance_test[dtype, ngpus, rows, 512](list_of_ctx)
    print("All reduce-scatter shard-invariance tests passed!")
