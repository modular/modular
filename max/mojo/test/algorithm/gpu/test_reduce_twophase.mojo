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

from max.algorithm.backend.gpu.reduction import reduce_launch
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite

from std.utils import IndexList, StaticTuple

# The failure is intermittent (a single call often passes), so every shape
# runs this many times back to back with nothing else enqueued.
comptime TRIALS = 12


def reduce_add[
    dtype: DType, width: SIMDLength, reduction_idx: Int
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return x + y


def run_repeated_sums(ctx: DeviceContext, rows: Int, n: Int) raises:
    """Sums `rows` rows of `n` ones along the row axis `TRIALS` times and
    checks that every result is exactly `n`."""
    print("rows", rows, "n", n)
    var data = ctx.enqueue_create_buffer[.float32](rows * n)
    var out = ctx.enqueue_create_buffer[.float32](TRIALS * rows)
    data.enqueue_fill(1)
    out.enqueue_fill(-1)  # Sentinel: an unwritten slot fails below.
    ctx.synchronize()
    var x = data.unsafe_ptr()
    var y = out.unsafe_ptr()

    for trial in range(TRIALS):

        @__copy_capture(x, n)
        @__parameter
        def input_fn[
            dtype: DType, width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[dtype, width]:
            return x.unsafe_load[width=width](coords[0] * n + coords[1]).cast[
                dtype
            ]()

        @__copy_capture(y, trial, rows)
        @__parameter
        def output_fn[
            dtype: DType, width: SIMDLength, rank: Int
        ](coords: IndexList[rank], val: StaticTuple[SIMD[dtype, width], 1]):
            y.unsafe_store[width=1](
                trial * rows + coords[0], val[0][0].cast[.float32]()
            )

        reduce_launch[
            1, input_fn, output_fn, reduce_add, 2, DType.float32, reduce_dim=1
        ](IndexList[2](rows, n), StaticTuple[Float32, 1](0), ctx)

    with out.map_to_host() as host:
        for i in range(TRIALS * rows):
            assert_equal(host[i], Float32(n))


def test_twophase_repeated_sums() raises:
    """Regression test for the two-phase (under-saturated) reduction's
    cross-block handshake.

    `twophase_reduce_kernel` has each block publish a partial and increment a
    per-row counter; the block that observes the final count reduces every
    partial. On Apple GPUs `Atomic.fetch_add`'s default ordering is relaxed,
    so without an explicit fence the last block read stale partials and
    dropped whole blocks' contributions, in single-row sums and in all but
    the last row of a multi-row reduction.
    """
    with DeviceContext() as ctx:
        # Single-row sums with many blocks per row.
        for n in [32768, 1 << 20]:
            run_repeated_sums(ctx, 1, n)
        # Several rows with many blocks each: every row has its own counter.
        run_repeated_sums(ctx, 3, 4096)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
