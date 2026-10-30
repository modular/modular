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
"""PDL ordering test for the dynamic scaled FP8 quantize kernels.

The quantizers launch with programmatic dependent launch (PDL), so they can
become resident while the kernel before them is still running, and they
release their own dependents before they are done (`OVERLAP_AT_BEGINNING`, the
default) or when they exit (`ON`). Both levels and both directions are checked
with adversarial neighbors:

- `late_producer` releases its dependents at once, sleeps, and only then
  publishes the activations (and, for the row-bounded launch, the device-side
  live row count) over zeroed buffers. A quantizer that reads either before its
  PDL wait sees zeros: an all-zero scale and output, or an empty row range.
- `consumer` waits on the quantizer and copies its outputs and scales. A
  quantizer that lets its dependents read before its stores are visible is
  caught the same way.

The PDL chain must be bit-identical to a run of the same kernel on inputs that
were published first, since PDL reorders loads and never arithmetic. The cases
cover both launch shapes of the kernel: a group narrower than a warp's reach
(a warp quantizes several groups) and a per-token group (a block per group),
each unbounded and row-bounded, the three-launch per-tensor path, and the
batched variant.

`check_release_timing` separately checks the release point itself for the two
row-bounded launches: a probe queued behind a long-running quantizer sits
resident for the whole kernel at `OVERLAP_AT_BEGINNING` and for next to nothing
at `ON`.
"""

from std.math import ceildiv
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.testing import assert_equal, assert_true
from std.time import global_perf_counter_ns, sleep

from layout import Idx, TileTensor, row_major
from layout._fillers import random
from linalg.fp8_quantization import (
    batched_quantize_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    quantize_tensor_dynamic_scaled_fp8,
)
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

comptime out_dtype = DType.float8_e4m3fn
comptime in_dtype = DType.bfloat16
comptime scales_dtype = DType.float32


def late_producer[
    dtype: DType
](
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int32,
    src_limit: MutPointer[UInt32, MutAnyOrigin],
    dst_limit: MutPointer[UInt32, MutAnyOrigin],
):
    """Releases dependents at once, then publishes the activations late."""
    launch_dependent_grids()
    sleep(0.01)

    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(length), stride):
        dst[i] = src[i]
    if tid == 0:
        dst_limit[0] = src_limit[0]


def consumer[
    dtype: DType
](
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int32,
):
    """Copies a quantizer output after waiting on it."""
    wait_on_dependent_grids()

    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(length), stride):
        dst[i] = src[i]


def overlap_probe(waited_ns: MutPointer[UInt64, MutAnyOrigin]):
    """Records how long this grid was resident before its predecessor finished.
    """
    var start = global_perf_counter_ns()
    wait_on_dependent_grids()
    var waited = global_perf_counter_ns() - start
    if thread_idx.x == 0:
        waited_ns[0] = waited


def check_equal[
    dtype: DType
](
    got: MutPointer[Scalar[dtype], _],
    want: MutPointer[Scalar[dtype], _],
    length: Int,
    what: StaticString,
) raises:
    # Compare encodings, not values: a float compare would equate +0.0 with
    # -0.0 and hide a differing sign bit.
    var got_bytes = got.unsafe_bitcast[UInt8]()
    var want_bytes = want.unsafe_bitcast[UInt8]()
    var mismatch = 0
    for i in range(length * size_of[dtype]()):
        if got_bytes[i] != want_bytes[i]:
            mismatch += 1
    assert_equal(
        mismatch,
        0,
        msg=String(
            what,
            ": ",
            mismatch,
            " of ",
            length * size_of[dtype](),
            " bytes differ from the",
            " run on already-published inputs",
        ),
    )


def run_case[
    pdl_level: PDLLevel,
    group_size_or_per_token: Int,
    K: Int,
    bounded: Bool,
    per_tensor: Bool = False,
](ctx: DeviceContext, rows: Int) raises:
    comptime group_size = K if group_size_or_per_token == -1 else group_size_or_per_token
    comptime num_groups = K // group_size
    print(
        "fp8 quantize PDL: level=",
        pdl_level._level,
        " per_tensor=",
        per_tensor,
        " rows=",
        rows,
        " K=",
        K,
        " group=",
        group_size,
        " bounded=",
        bounded,
        sep="",
    )

    var in_elems = rows * K
    var out_elems = rows * K
    var sc_elems = num_groups * rows
    var live_rows = rows // 2 + 1 if bounded else rows

    var in_host = unsafe_alloc[Scalar[in_dtype]](in_elems)
    random(TileTensor(in_host, row_major(rows, Idx[K])), -1.0, 1.0)
    var limit_host = unsafe_alloc[UInt32](1)
    limit_host[0] = UInt32(live_rows)
    var zero_limit_host = unsafe_alloc[UInt32](1)
    zero_limit_host[0] = 0

    var src_in = ctx.enqueue_create_buffer[in_dtype](in_elems)
    var dst_in = ctx.enqueue_create_buffer[in_dtype](in_elems)
    var src_limit = ctx.enqueue_create_buffer[.uint32](1)
    var dst_limit = ctx.enqueue_create_buffer[.uint32](1)
    ctx.enqueue_copy(src_in, in_host)
    ctx.enqueue_copy(src_limit, limit_host)

    var ref_out = ctx.enqueue_create_buffer[out_dtype](out_elems)
    var ref_sc = ctx.enqueue_create_buffer[scales_dtype](sc_elems)
    var out = ctx.enqueue_create_buffer[out_dtype](out_elems)
    var sc = ctx.enqueue_create_buffer[scales_dtype](sc_elems)
    var out_copy = ctx.enqueue_create_buffer[out_dtype](out_elems)
    var sc_copy = ctx.enqueue_create_buffer[scales_dtype](sc_elems)
    ctx.enqueue_memset[out_dtype](ref_out, 0)
    ctx.enqueue_memset[scales_dtype](ref_sc, 0)

    var src_t = TileTensor(src_in, row_major(rows, Idx[K]))
    var dst_t = TileTensor(dst_in, row_major(rows, Idx[K]))
    var ref_out_t = TileTensor(ref_out, row_major(rows, Idx[K]))
    var ref_sc_t = TileTensor(ref_sc, row_major(Idx[num_groups], rows))
    var out_t = TileTensor(out, row_major(rows, Idx[K]))
    var sc_t = TileTensor(sc, row_major(Idx[num_groups], rows))

    @always_inline
    def src_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var src_t} -> SIMD[in_dtype, width]:
        return src_t.load[width=width, alignment=alignment]((row, col))

    @always_inline
    def dst_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var dst_t} -> SIMD[in_dtype, width]:
        return dst_t.load[width=width, alignment=alignment]((row, col))

    # Reference: the activations and the row count are in place before the
    # quantizer runs, and PDL is off so the reference cannot depend on the
    # release behavior under test.
    comptime if per_tensor:
        quantize_tensor_dynamic_scaled_fp8[
            in_dtype=in_dtype,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=K,
            pdl_level=PDLLevel.OFF,
        ](src_fn, ref_out_t, ref_sc_t, 1200.0, ctx, rows)
    else:
        quantize_dynamic_scaled_fp8[
            in_dtype=in_dtype,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=K,
            pdl_level=PDLLevel.OFF,
            row_bounded=bounded,
        ](
            src_fn,
            ref_out_t,
            ref_sc_t,
            1200.0,
            ctx,
            rows,
            row_limit=OptionalPointer[UInt32, ImmUntrackedOrigin](
                src_limit.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .as_imm()
            ),
        )
    ctx.synchronize()

    var want_out = unsafe_alloc[Scalar[out_dtype]](out_elems)
    var want_sc = unsafe_alloc[Scalar[scales_dtype]](sc_elems)
    var got_out = unsafe_alloc[Scalar[out_dtype]](out_elems)
    var got_sc = unsafe_alloc[Scalar[scales_dtype]](sc_elems)
    ctx.enqueue_copy(want_out, ref_out)
    ctx.enqueue_copy(want_sc, ref_sc)
    ctx.synchronize()

    for it in range(20):
        ctx.enqueue_memset[in_dtype](dst_in, 0)
        ctx.enqueue_copy(dst_limit, zero_limit_host)
        ctx.enqueue_memset[out_dtype](out, 0)
        ctx.enqueue_memset[scales_dtype](sc, 0)
        ctx.enqueue_memset[out_dtype](out_copy, 0)
        ctx.enqueue_memset[scales_dtype](sc_copy, 0)

        ctx.enqueue_function[late_producer[in_dtype]](
            src_in,
            dst_in,
            Int32(in_elems),
            src_limit,
            dst_limit,
            grid_dim=4,
            block_dim=128,
        )
        comptime if per_tensor:
            quantize_tensor_dynamic_scaled_fp8[
                in_dtype=in_dtype,
                group_size_or_per_token=group_size_or_per_token,
                num_cols=K,
                pdl_level=pdl_level,
            ](dst_fn, out_t, sc_t, 1200.0, ctx, rows)
        else:
            quantize_dynamic_scaled_fp8[
                in_dtype=in_dtype,
                group_size_or_per_token=group_size_or_per_token,
                num_cols=K,
                pdl_level=pdl_level,
                row_bounded=bounded,
            ](
                dst_fn,
                out_t,
                sc_t,
                1200.0,
                ctx,
                rows,
                row_limit=OptionalPointer[UInt32, ImmUntrackedOrigin](
                    dst_limit.unsafe_ptr()
                    .unsafe_origin_cast[MutUntrackedOrigin]()
                    .as_imm()
                ),
            )
        # One consumer per chain, alternating between the outputs and the
        # scales: a second consumer queued behind the first would wait on
        # that consumer, not on the quantizer.
        if it % 2 == 0:
            ctx.enqueue_function[consumer[out_dtype]](
                out,
                out_copy,
                Int32(out_elems),
                grid_dim=ceildiv(out_elems, 256),
                block_dim=256,
                attributes=pdl_launch_attributes(pdl_level),
            )
            ctx.enqueue_copy(got_out, out_copy)
        else:
            ctx.enqueue_function[consumer[scales_dtype]](
                sc,
                sc_copy,
                Int32(sc_elems),
                grid_dim=ceildiv(sc_elems, 256),
                block_dim=256,
                attributes=pdl_launch_attributes(pdl_level),
            )
            ctx.enqueue_copy(got_sc, sc_copy)
        ctx.synchronize()

        if it % 2 == 0:
            check_equal(got_out, want_out, out_elems, "quantized values")
        else:
            check_equal(got_sc, want_sc, sc_elems, "scales")

    in_host.unsafe_free()
    limit_host.unsafe_free()
    zero_limit_host.unsafe_free()
    want_out.unsafe_free()
    want_sc.unsafe_free()
    got_out.unsafe_free()
    got_sc.unsafe_free()


def run_batched[
    pdl_level: PDLLevel, group_size_or_per_token: Int, K: Int
](ctx: DeviceContext, batch: Int, rows: Int) raises:
    comptime group_size = K if group_size_or_per_token == -1 else group_size_or_per_token
    comptime num_groups = K // group_size
    print(
        "batched fp8 quantize PDL: level=",
        pdl_level._level,
        " batch=",
        batch,
        " rows=",
        rows,
        " K=",
        K,
        " group=",
        group_size,
        sep="",
    )

    var in_elems = batch * rows * K
    var sc_elems = batch * num_groups * rows

    var in_host = unsafe_alloc[Scalar[in_dtype]](in_elems)
    random(TileTensor(in_host, row_major(batch, rows, Idx[K])), -1.0, 1.0)
    var dummy_limit = ctx.enqueue_create_buffer[.uint32](1)
    ctx.enqueue_memset[.uint32](dummy_limit, 0)

    var src_in = ctx.enqueue_create_buffer[in_dtype](in_elems)
    var dst_in = ctx.enqueue_create_buffer[in_dtype](in_elems)
    ctx.enqueue_copy(src_in, in_host)

    var ref_out = ctx.enqueue_create_buffer[out_dtype](in_elems)
    var ref_sc = ctx.enqueue_create_buffer[scales_dtype](sc_elems)
    var out = ctx.enqueue_create_buffer[out_dtype](in_elems)
    var sc = ctx.enqueue_create_buffer[scales_dtype](sc_elems)
    var out_copy = ctx.enqueue_create_buffer[out_dtype](in_elems)
    var sc_copy = ctx.enqueue_create_buffer[scales_dtype](sc_elems)
    ctx.enqueue_memset[out_dtype](ref_out, 0)
    ctx.enqueue_memset[scales_dtype](ref_sc, 0)

    var src_t = TileTensor(src_in, row_major(batch, rows, Idx[K]))
    var dst_t = TileTensor(dst_in, row_major(batch, rows, Idx[K]))
    var ref_out_t = TileTensor(ref_out, row_major(batch, rows, Idx[K]))
    var ref_sc_t = TileTensor(ref_sc, row_major(batch, Idx[num_groups], rows))
    var out_t = TileTensor(out, row_major(batch, rows, Idx[K]))
    var sc_t = TileTensor(sc, row_major(batch, Idx[num_groups], rows))

    @always_inline
    def src_fn[
        width: Int, alignment: Int
    ](b: Int, row: Int, col: Int) {var src_t} -> SIMD[in_dtype, width]:
        return src_t.load[width=width, alignment=alignment]((b, row, col))

    @always_inline
    def dst_fn[
        width: Int, alignment: Int
    ](b: Int, row: Int, col: Int) {var dst_t} -> SIMD[in_dtype, width]:
        return dst_t.load[width=width, alignment=alignment]((b, row, col))

    batched_quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=K,
        pdl_level=PDLLevel.OFF,
    ](src_fn, ref_out_t, ref_sc_t, 1200.0, ctx, rows, batch)
    ctx.synchronize()

    var want_out = unsafe_alloc[Scalar[out_dtype]](in_elems)
    var want_sc = unsafe_alloc[Scalar[scales_dtype]](sc_elems)
    var got_out = unsafe_alloc[Scalar[out_dtype]](in_elems)
    var got_sc = unsafe_alloc[Scalar[scales_dtype]](sc_elems)
    ctx.enqueue_copy(want_out, ref_out)
    ctx.enqueue_copy(want_sc, ref_sc)
    ctx.synchronize()

    for it in range(20):
        ctx.enqueue_memset[in_dtype](dst_in, 0)
        ctx.enqueue_memset[out_dtype](out, 0)
        ctx.enqueue_memset[scales_dtype](sc, 0)
        ctx.enqueue_memset[out_dtype](out_copy, 0)
        ctx.enqueue_memset[scales_dtype](sc_copy, 0)

        ctx.enqueue_function[late_producer[in_dtype]](
            src_in,
            dst_in,
            Int32(in_elems),
            dummy_limit,
            dummy_limit,
            grid_dim=4,
            block_dim=128,
        )
        batched_quantize_dynamic_scaled_fp8[
            in_dtype=in_dtype,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=K,
            pdl_level=pdl_level,
        ](dst_fn, out_t, sc_t, 1200.0, ctx, rows, batch)
        if it % 2 == 0:
            ctx.enqueue_function[consumer[out_dtype]](
                out,
                out_copy,
                Int32(in_elems),
                grid_dim=ceildiv(in_elems, 256),
                block_dim=256,
                attributes=pdl_launch_attributes(pdl_level),
            )
            ctx.enqueue_copy(got_out, out_copy)
        else:
            ctx.enqueue_function[consumer[scales_dtype]](
                sc,
                sc_copy,
                Int32(sc_elems),
                grid_dim=ceildiv(sc_elems, 256),
                block_dim=256,
                attributes=pdl_launch_attributes(pdl_level),
            )
            ctx.enqueue_copy(got_sc, sc_copy)
        ctx.synchronize()

        if it % 2 == 0:
            check_equal(got_out, want_out, in_elems, "quantized values")
        else:
            check_equal(got_sc, want_sc, sc_elems, "scales")

    in_host.unsafe_free()
    want_out.unsafe_free()
    want_sc.unsafe_free()
    got_out.unsafe_free()
    got_sc.unsafe_free()


def longest_residency_ns[
    pdl_level: PDLLevel, group_size_or_per_token: Int, K: Int
](ctx: DeviceContext, rows: Int) raises -> UInt64:
    """Returns the longest time a one-block probe sat resident behind a
    long-running row-bounded quantizer before the quantizer completed.

    The row-bounded launch is capped at a fraction of a wave of blocks that
    grid-stride over `rows`, so every block starts early and runs for the whole
    kernel. A dependent can therefore turn resident while the quantizer is still
    working only if the quantizer released it at entry; one released at exit
    starts after the quantizer is done and waits for next to nothing.
    """
    var in_elems = rows * K
    comptime num_groups = K // (
        K if group_size_or_per_token == -1 else group_size_or_per_token
    )
    var src_in = ctx.enqueue_create_buffer[in_dtype](in_elems)
    var out = ctx.enqueue_create_buffer[out_dtype](in_elems)
    var sc = ctx.enqueue_create_buffer[scales_dtype](num_groups * rows)
    var limit = ctx.enqueue_create_buffer[.uint32](1)
    var waited = ctx.enqueue_create_buffer[.uint64](1)
    var limit_host = unsafe_alloc[UInt32](1)
    limit_host[0] = UInt32(rows)
    var waited_host = unsafe_alloc[UInt64](1)
    ctx.enqueue_memset[in_dtype](src_in, 0)
    ctx.enqueue_copy(limit, limit_host)

    var src_t = TileTensor(src_in, row_major(rows, Idx[K]))
    var out_t = TileTensor(out, row_major(rows, Idx[K]))
    var sc_t = TileTensor(sc, row_major(Idx[num_groups], rows))

    @always_inline
    def src_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var src_t} -> SIMD[in_dtype, width]:
        return src_t.load[width=width, alignment=alignment]((row, col))

    var longest = UInt64(0)
    # PDL allows early scheduling but does not guarantee it, so take the best
    # of a few launches; the first one also loads the kernels.
    for _ in range(6):
        ctx.enqueue_memset[.uint64](waited, 0)
        quantize_dynamic_scaled_fp8[
            in_dtype=in_dtype,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=K,
            pdl_level=pdl_level,
            row_bounded=True,
        ](
            src_fn,
            out_t,
            sc_t,
            1200.0,
            ctx,
            rows,
            row_limit=OptionalPointer[UInt32, ImmUntrackedOrigin](
                limit.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .as_imm()
            ),
        )
        ctx.enqueue_function[overlap_probe](
            waited,
            grid_dim=1,
            block_dim=32,
            attributes=pdl_launch_attributes(pdl_level),
        )
        ctx.enqueue_copy(waited_host, waited)
        ctx.synchronize()
        longest = max(longest, waited_host[0])

    limit_host.unsafe_free()
    waited_host.unsafe_free()
    return longest


def check_release_timing[
    group_size_or_per_token: Int, K: Int
](ctx: DeviceContext, rows: Int) raises:
    var early = longest_residency_ns[
        PDLLevel.OVERLAP_AT_BEGINNING, group_size_or_per_token, K
    ](ctx, rows)
    var at_exit = longest_residency_ns[PDLLevel.ON, group_size_or_per_token, K](
        ctx, rows
    )
    print(
        "fp8 quantize release timing: group=",
        group_size_or_per_token,
        " K=",
        K,
        " rows=",
        rows,
        " resident before completion: entry=",
        early,
        " ns, exit=",
        at_exit,
        " ns",
        sep="",
    )
    assert_true(
        early > 4 * at_exit + 10_000,
        msg="OVERLAP_AT_BEGINNING did not release dependents at kernel entry",
    )


def run_level[pdl_level: PDLLevel](ctx: DeviceContext) raises:
    # 128-wide groups: a warp quantizes several groups (a ragged tail of
    # rows leaves part of the last warp idle).
    run_case[pdl_level, 128, 2048, False](ctx, 1)
    run_case[pdl_level, 128, 2048, False](ctx, 37)
    run_case[pdl_level, 128, 2048, False](ctx, 515)
    run_case[pdl_level, 128, 2048, True](ctx, 37)
    run_case[pdl_level, 128, 2048, True](ctx, 515)
    # One group per token: a block per (row, group).
    run_case[pdl_level, -1, 1024, False](ctx, 1)
    run_case[pdl_level, -1, 1024, False](ctx, 37)
    run_case[pdl_level, -1, 1024, True](ctx, 37)
    run_case[pdl_level, -1, 1024, True](ctx, 515)
    # Per-tensor scale: three chained launches (group maxes, reduce, quantize).
    run_case[pdl_level, 128, 2048, False, per_tensor=True](ctx, 37)
    run_case[pdl_level, 128, 2048, False, per_tensor=True](ctx, 515)
    run_batched[pdl_level, 128, 512](ctx, 3, 5)
    run_batched[pdl_level, 128, 512](ctx, 16, 33)


def main() raises:
    with DeviceContext() as ctx:
        # `ON` releases dependents at exit, `OVERLAP_AT_BEGINNING` (the
        # default) right after the wait. Both give bit-identical values, so
        # `run_level` checks that the wait and the release are safe at each
        # level, not that the level reaches the kernel. `check_release_timing`
        # does, for the two row-bounded launches. The per-tensor and batched
        # kernels launch one short block per row or group, so a dependent
        # barely overlaps them and their wiring is read from the generated
        # code (`griddepcontrol.launch_dependents` follows the wait at the
        # default level and ends the kernel at `ON`).
        run_level[PDLLevel.ON](ctx)
        run_level[PDLLevel.OVERLAP_AT_BEGINNING](ctx)
        check_release_timing[128, 128](ctx, 1 << 22)
        check_release_timing[-1, 1024](ctx, 1 << 19)
        print("PASSED")
