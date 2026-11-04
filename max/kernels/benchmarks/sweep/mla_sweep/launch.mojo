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
"""Launch-side helpers: split-K forcing, top-k lengths, stamps, PDL spin."""

from std.os import getenv, setenv
from std.time import global_perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import launch_dependent_grids
from nn.attention.gpu.nvidia.sm100.mla_stamp import (
    MLA_STAMP,
    STAMP_MAX_CTAS,
    STAMP_SLOTS,
    STAMP_WARPS,
)


def force_split(
    ctx: DeviceContext,
    scalars: DeviceBuffer[DType.int64, ...],
    batch_size: Int,
    seq_len: Int,
    num_partitions: Int,
) raises:
    """Overwrites a decode's dispatch scalars with a forced split-K count.

    A forced count must reach both the grid, through `num_partitions_in`,
    and the kernel, which reads it from these scalars.
    """
    var host = ctx.enqueue_create_host_buffer[DType.int64](3)
    ctx.synchronize()
    host[0] = Int64(batch_size)
    host[1] = Int64(seq_len)
    host[2] = Int64(num_partitions)
    ctx.enqueue_copy(
        DeviceBuffer[DType.int64](ctx, scalars.unsafe_ptr(), 3, owning=False),
        host,
    )
    ctx.synchronize()


def fill_topk_lengths[
    dtype: DType
](
    ctx: DeviceContext,
    dst: DeviceBuffer[dtype, ...],
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    top_k: Int,
) raises:
    """Writes each query token's valid pick count, `min(top_k, cache_len +
    s + 1)` at position `s`, as the indexer reports it."""
    var tokens = batch_size * seq_len
    var host = ctx.enqueue_create_host_buffer[dtype](tokens)
    ctx.synchronize()
    for g in range(tokens):
        host[g] = Scalar[dtype](min(top_k, cache_len + g % seq_len + 1))
    ctx.enqueue_copy(
        DeviceBuffer[dtype](ctx, dst.unsafe_ptr(), tokens, owning=False), host
    )
    ctx.synchronize()


def early_trigger_spin(delay_ns: UInt64):
    """Triggers the PDL successor at entry, then spins for `delay_ns`.

    Launched before a decode, it makes the decode's predecessor trigger early,
    as production's does, so the decode's entry wait has something to cost.
    The bench loop's own predecessor, the previous combine, triggers only by
    completing.
    """
    launch_dependent_grids()
    var t0 = global_perf_counter_ns()
    while global_perf_counter_ns() - t0 < delay_ns:
        pass


def dump_stamps[
    LaunchFn: def(DeviceContext[]) raises -> None
](ctx: DeviceContext[], launch: LaunchFn, name: String) raises:
    """Writes the kernel's per-warp clock stamps under `-D MLA_STAMP=1`.

    Writes `MLA_STAMP_DUMPS` files (default 1), each the stamps of the last of
    `MLA_STAMP_LAUNCHES` launches (default 3; every launch overwrites the
    same rows), so a dump can come from a warm launch rather than the
    process's first. Files land in the test's undeclared outputs as
    `stamps_<name>_d<i>.bin`. Does nothing without `MLA_STAMP`.
    """
    comptime if MLA_STAMP:
        comptime n_words = STAMP_MAX_CTAS * STAMP_WARPS * STAMP_SLOTS
        var launches = atol(getenv("MLA_STAMP_LAUNCHES", "3"))
        var dumps = atol(getenv("MLA_STAMP_DUMPS", "1"))
        var stamps = ctx.enqueue_create_buffer[DType.uint64](n_words)
        for d in range(dumps):
            stamps.enqueue_fill(0)
            _ = setenv("MLA_STAMP_ADDR", String(Int(stamps.unsafe_ptr())))
            for _ in range(launches):
                launch(ctx)
            ctx.synchronize()
            _ = setenv("MLA_STAMP_ADDR", "0")
            var path = String(
                getenv("TEST_UNDECLARED_OUTPUTS_DIR", "."),
                "/stamps_",
                name,
                "_d",
                d,
                ".bin",
            )
            with stamps.map_to_host() as host:
                var bytes = Span(
                    unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[Byte](),
                    length=n_words * 8,
                )
                with open(path, "w") as f:
                    f.write_bytes(bytes)
            print("wrote", path)
