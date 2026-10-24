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
"""GPU half of the empty-reduce-axis regression, for argmax — the nastier
failure mode.

`algorithm.gpu.rowwise.launch` used to guard on `num_rows == 0 or
row_size == 0` and return before writing anything. That guard is wrong
for a reduce-shaped body (one output per row): an empty *axis* still
leaves `product(shape \\ axis)` outputs that must be written with the
monoid identity.

Nothing faulted along the way, which is why this stayed hidden. `num_rows`
came from `flattened_length() // row_size`, and Mojo's `//` inserts a
zero-guard, so the `0 // 0` yielded `0` rather than trapping — the same
`0` a shape with genuinely no rows produces. The guard could not tell the
two apart and took the silent branch.

On GPU, skipping the write is worse than on CPU (`test_reduce_empty_axis.mojo`
in `test/algorithm/`, which only needs to show an unwritten output):
output buffers are recycled across launches, so the value left behind is
whatever the *previous* launch on that buffer wrote — a plausible,
non-obviously-wrong number that also changes run to run. Each test here
reproduces that exact shape: launch once with a real (nonzero) axis so
the output buffer holds a real, distinguishable result, then relaunch on
the *same* output buffer with an empty axis and require the stale value
to have been replaced by the identity, not merely absent from a fresh
buffer.

Ambiguous-monoid note (`max`/`argmax`): `numpy.amax` raises on an empty
axis; this codebase does not special-case it. Every `ReduceOp` monoid's
`__init__` is a required, unconditional identity (`ReduceMax`:
`min_finite`; `ArgMax`: `Int64.MAX`, clamped to index `0` by
`_in_range_index` — the same clamp `test_arg_reduce_nan.mojo` already
exercises for an all-NaN row, since "no candidate ever won" covers both
cases identically), and `rowwise.launch` runs one generic code path for
every monoid. Special-casing max/min/argmax/argmin to raise instead would
mean a second code path (a host-side shape check ahead of dispatch) that
`sum`/`product`/`mean` don't need — more surface for exactly the kind of
skip-the-write bug this file guards against, for a case (an
empty-batch-dependent shape reaching a raise mid-request) that is worse
in a serving context than a defined sentinel value. So: same identity
rule for every monoid, on both backends.

Sum, max, min and mean compile to generated rowwise kernels, so their
empty-axis cases live in the graph-level property test,
`//max/tests/integration/graph:test_rowwise_reductions_fuzz`. This file keeps
argmax, which still runs its hand-written kernel, and reuses one output
buffer on purpose, which a graph-level test can't.

Tier coverage: at `row_size == 0` an inner-axis reduce always lands on the
warp tier (`row_size <= warp_max` holds trivially), so this file exercises
`_WarpKernel` only.
"""

from std.sys import align_of

from builtin_kernels.rowwise_reductions import reduce_argmax
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def _test_reduce_argmax_stale_buffer_gpu[dtype: DType = DType.float32]() raises:
    """Same pattern for `argmax`. A real run picks a real winning index
    (not `0`), so a stale-buffer bug can't hide behind the identity index
    also being `0`."""
    with DeviceContext() as ctx:
        comptime out_dtype = DType.int64
        comptime num_rows = 2
        comptime real_axis_size = 4
        # Deliberately not `0`, so `out_h` reading back `0` after run 1
        # would itself be a (different) bug, not a false pass for run 2.
        comptime winner_idx = 2

        var in_h = ctx.enqueue_create_host_buffer[dtype](
            num_rows * real_axis_size
        )
        var out_h = ctx.enqueue_create_host_buffer[out_dtype](num_rows)
        ctx.synchronize()
        for i in range(num_rows * real_axis_size):
            in_h[i] = 1
        for r in range(num_rows):
            in_h[r * real_axis_size + winner_idx] = 5

        var in_d = ctx.enqueue_create_buffer[dtype](num_rows * real_axis_size)
        var out_d = ctx.enqueue_create_buffer[out_dtype](num_rows)
        ctx.enqueue_copy(in_d, in_h)

        var in_buf = TileTensor(in_d, row_major((num_rows, real_axis_size)))
        var out_ptr = out_d.unsafe_ptr()

        @inline(.always)
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
            return in_buf.load[
                width=width, alignment=alignment * align_of[dtype]()
            ](coords)

        @inline(.always)
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[out_dtype, width]) {var out_ptr}:
            # The output is one value per row, so index `out_ptr` by row.
            out_ptr.unsafe_store[width=width](Int(coords[0]), val)

        # Run 1: real winner at `winner_idx`.
        reduce_argmax[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(num_rows, real_axis_size), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Int64(winner_idx))
        assert_equal(out_h[1], Int64(winner_idx))

        # Run 2: SAME `out_d`, empty axis. Must become `0` (the
        # `_in_range_index`-clamped identity), not the stale `winner_idx`.
        reduce_argmax[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(num_rows, 0), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Int64(0))
        assert_equal(out_h[1], Int64(0))

        _ = in_buf
        _ = in_d^
        _ = out_d^


def test_reduce_argmax_stale_buffer_gpu() raises:
    _test_reduce_argmax_stale_buffer_gpu()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
