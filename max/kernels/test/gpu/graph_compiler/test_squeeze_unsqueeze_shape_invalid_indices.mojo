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
"""Checks that the GPU `mo.squeeze_shape` / `mo.unsqueeze_shape` kernels never
write outside their output when given duplicate or out-of-range indices.

Built with assertions off (see BUILD.bazel), so the invalid indices reach the
kernels' bounds guards instead of tripping the index asserts. The canary slots
on both sides of the output must stay untouched, and the whole padded buffer
must match the CPU path, so duplicates cannot make GPU threads race on a slot.
"""

from std.testing import TestSuite, assert_equal

from builtin_kernels.gather_scatter import SqueezeShape, UnsqueezeShape
from extensibility import IOSpec, ManagedTensorSlice, StaticTensorSpec
from max.gpu.host import DeviceContext

comptime shape_dtype = DType.int64
comptime CANARY = -7
comptime PAD = 4
comptime NUM_PATTERNS = 5


def _invalid_indices(pattern: Int, count: Int, index_rank: Int) -> List[Int]:
    """Returns `count` indices that are duplicated or out of `index_rank`."""
    var indices = List[Int]()
    for k in range(count):
        match pattern:
            case 0:
                # All point at dim 0.
                indices.append(0)
            case 1:
                # The last dim, spelled positively and negatively.
                indices.append(index_rank - 1 if k % 2 == 0 else -1)
            case 2:
                indices.append(index_rank + k)
            case 3:
                indices.append(-index_rank - 1 - k)
            case _:
                # One valid index, then far out of range in both directions.
                indices.append(
                    0 if k == 0 else (1000000 if k % 2 == 1 else -1000000)
                )
    return indices^


def run_case[
    squeeze: Bool, indices_type: DType
](ctx: DeviceContext, rank: Int, indices: List[Int]) raises:
    var num_indices = len(indices)
    var out_rank = rank - num_indices if squeeze else rank + num_indices

    var in_host = ctx.enqueue_create_host_buffer[shape_dtype](rank + 1)
    var idx_host = ctx.enqueue_create_host_buffer[indices_type](num_indices + 1)
    var out_host = ctx.enqueue_create_host_buffer[shape_dtype](
        out_rank + 2 * PAD
    )
    ctx.synchronize()
    for d in range(rank):
        in_host[d] = Scalar[shape_dtype](10 + d)
    for k in range(num_indices):
        idx_host[k] = Scalar[indices_type](indices[k])

    var in_dev = ctx.enqueue_create_buffer[shape_dtype](rank + 1)
    var idx_dev = ctx.enqueue_create_buffer[indices_type](num_indices + 1)
    var out_dev = ctx.enqueue_create_buffer[shape_dtype](out_rank + 2 * PAD)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    out_dev.enqueue_fill(CANARY)

    comptime in_spec = StaticTensorSpec[shape_dtype, 1].get_unknown()
    comptime idx_spec = StaticTensorSpec[indices_type, 1].get_unknown()
    var input = ManagedTensorSlice[io_spec=IOSpec.Input, static_spec=in_spec](
        in_dev.unsafe_ptr(), (rank,)
    )
    var shape_indices = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=idx_spec
    ](idx_dev.unsafe_ptr(), (num_indices,))
    # The output starts `PAD` slots into the buffer so a write at a negative
    # position lands on a canary.
    var output = ManagedTensorSlice[io_spec=IOSpec.Output, static_spec=in_spec](
        out_dev.unsafe_ptr().unsafe_offset(PAD), (out_rank,)
    )

    comptime if squeeze:
        SqueezeShape.execute[
            target="gpu", dtype=shape_dtype, indices_type=indices_type
        ](output, input, shape_indices, ctx)
    else:
        UnsqueezeShape.execute[
            target="gpu", dtype=shape_dtype, indices_type=indices_type
        ](output, input, shape_indices, ctx)

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var ref_host = ctx.enqueue_create_host_buffer[shape_dtype](
        out_rank + 2 * PAD
    )
    ctx.synchronize()
    for t in range(out_rank + 2 * PAD):
        ref_host[t] = CANARY
    var ref_input = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=in_spec
    ](in_host.unsafe_ptr(), (rank,))
    var ref_indices = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=idx_spec
    ](idx_host.unsafe_ptr(), (num_indices,))
    var ref_output = ManagedTensorSlice[
        io_spec=IOSpec.Output, static_spec=in_spec
    ](ref_host.unsafe_ptr().unsafe_offset(PAD), (out_rank,))
    comptime if squeeze:
        SqueezeShape.execute[
            target="cpu", dtype=shape_dtype, indices_type=indices_type
        ](ref_output, ref_input, ref_indices, ctx)
    else:
        UnsqueezeShape.execute[
            target="cpu", dtype=shape_dtype, indices_type=indices_type
        ](ref_output, ref_input, ref_indices, ctx)

    var label = String(t"squeeze={squeeze} rank={rank} n={num_indices}")
    for t in range(PAD):
        assert_equal(Int(out_host[t]), CANARY, String(t"{label} head[{t}]"))
        assert_equal(
            Int(out_host[PAD + out_rank + t]),
            CANARY,
            String(t"{label} tail[{t}]"),
        )
    for t in range(out_rank + 2 * PAD):
        assert_equal(
            Int(out_host[t]), Int(ref_host[t]), String(t"{label} vs cpu [{t}]")
        )


def run_all[squeeze: Bool, indices_type: DType](ctx: DeviceContext) raises:
    for rank in range(1, 9):
        for count in range(1, 5):
            if squeeze and count > rank:
                continue
            var index_rank = rank if squeeze else rank + count
            for pattern in range(NUM_PATTERNS):
                run_case[squeeze, indices_type](
                    ctx, rank, _invalid_indices(pattern, count, index_rank)
                )


def test_squeeze_shape_invalid_indices_stay_in_bounds() raises:
    var ctx = DeviceContext()
    run_all[True, DType.int64](ctx)
    run_all[True, DType.int32](ctx)


def test_unsqueeze_shape_invalid_indices_stay_in_bounds() raises:
    var ctx = DeviceContext()
    run_all[False, DType.int64](ctx)
    run_all[False, DType.int32](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
