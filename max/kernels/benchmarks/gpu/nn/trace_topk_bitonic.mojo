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
"""Per-row phase attribution for the indexer top-k radix select.

Stamps `global_perf_counter_ns` at every phase boundary of the select into a
per-row trace buffer and prints it, so the kernel's time splits into "round `r`
row scan", "round `r` split", "append" and "sort" instead of one opaque total.
Reads the same hashed input as `xcheck_topk_bitonic`, so a trace and a
correctness dump describe the same run.

Traces whichever kernel the decode dispatch would pick for the shape --
`histsel_resident_topk` for a row that fits in registers, `histsel_topk` with
prefetch and narrow refine digits otherwise. Tracing a configuration the
dispatch never launches attributes phases that nothing runs.

    trace_topk_bitonic --rows=8192 --N=76000 --K=2048 --levels=17
"""

from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, FuncAttribute
from internal_utils import arg_parse
from layout import TileTensor, row_major
from structured_kernels.trace_buf import GmemTrace

from nn.topk_bitonic import (
    HSEL_TRACE_EVENTS,
    _histsel_resident_kernel,
    _histsel_topk_kernel,
    _HSEL_RANK_BITS,
    _HSEL_RES_BLOCK,
    _HSEL_RES_MAX,
    _HSEL_SEL_CAP,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    _PTOPK_TOTAL,
)


def _hash32(a: UInt32, b: UInt32) -> UInt32:
    """Wrapping-uint32 mixer, reproducible in CUDA C and NumPy."""
    var h = a * UInt32(0x9E3779B1) + b * UInt32(0x85EBCA77)
    h ^= h >> 16
    h = h * UInt32(0x7FEB352D)
    h ^= h >> 15
    h = h * UInt32(0x846CA68B)
    h ^= h >> 16
    return h


def _launch(
    ctx: DeviceContext,
    scores_t: TileTensor[DType.float32, ...],
    idxs_t: TileTensor[DType.int32, ...],
    trace_ptr: UnsafePointer[UInt64, MutUntrackedOrigin],
    N: Int,
    K: Int,
    rows: Int,
) raises:
    if N <= _HSEL_RES_MAX and N < K + K // 2:
        ctx.enqueue_function[
            _histsel_resident_kernel[
                GmemTrace, enable_trace=True, bin_digit=True
            ]
        ](
            rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
                scores_t.ptr
            ),
            rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
                idxs_t.ptr
            ),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_HSEL_RES_BLOCK,
        )
        return

    if N <= _HSEL_RES_MAX:
        ctx.enqueue_function[
            _histsel_resident_kernel[GmemTrace, enable_trace=True]
        ](
            rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
                scores_t.ptr
            ),
            rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
                idxs_t.ptr
            ),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_HSEL_RES_BLOCK,
        )
        return

    ctx.enqueue_function[
        _histsel_topk_kernel[
            GmemTrace,
            enable_trace=True,
            prefetch=True,
            tail_bits=_HSEL_TAIL_BITS,
            rank_bits=_HSEL_RANK_BITS,
            rank_slots=True,
            sel_cap=_HSEL_SEL_CAP,
        ]
    ](
        rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
            scores_t.ptr
        ),
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](idxs_t.ptr),
        Int32(N),
        Int32(K),
        GmemTrace(trace_ptr),
        grid_dim=rows,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=_HSEL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(_HSEL_SMEM_BYTES)
        ),
    )


def main() raises:
    var rows = arg_parse("rows", 48)
    var N = arg_parse("N", 107228)
    var K = arg_parse("K", 2048)
    var levels = arg_parse("levels", 17)

    with DeviceContext() as ctx:
        var scores_buf = ctx.enqueue_create_buffer[DType.float32](rows * N)
        var idxs_buf = ctx.enqueue_create_buffer[DType.int32](rows * K)
        var trace_buf = ctx.enqueue_create_buffer[DType.uint64](
            rows * HSEL_TRACE_EVENTS
        )

        with scores_buf.map_to_host() as h:
            for r in range(rows):
                var nk = N - Int(
                    _hash32(UInt32(r), UInt32(0xABCD1234))
                    % UInt32(max(1, N // 8))
                )
                for c in range(N):
                    if c < nk:
                        var v = _hash32(UInt32(r), UInt32(c)) % UInt32(levels)
                        h[r * N + c] = Float32(Int(v))
                    else:
                        h[r * N + c] = Float32(-3.0e38)
        ctx.enqueue_memset(trace_buf, 0)

        var scores_t = TileTensor(scores_buf, row_major(rows, N))
        var idxs_t = TileTensor(idxs_buf, row_major(rows, K))
        ctx.synchronize()

        var trace_ptr = rebind[UnsafePointer[UInt64, MutUntrackedOrigin]](
            trace_buf.unsafe_ptr()
        )

        # Warm up before the traced launch: a cold launch's first row scan
        # measures instruction-cache and page-table misses, not the kernel.
        for _ in range(20):
            _launch(ctx, scores_t, idxs_t, trace_ptr, N, K, rows)
        ctx.synchronize()

        var t0 = perf_counter_ns()
        var iters = arg_parse("iters", 20)
        for _ in range(iters):
            _launch(ctx, scores_t, idxs_t, trace_ptr, N, K, rows)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        print("TIME_MS", Float64(t1 - t0) / Float64(iters) / 1.0e6)

        print("TRACE", rows, N, K, levels, HSEL_TRACE_EVENTS)
        with trace_buf.map_to_host() as h:
            for r in range(rows):
                var line = String("T ")
                line += String(r)
                for e in range(HSEL_TRACE_EVENTS):
                    line += " "
                    line += String(h[r * HSEL_TRACE_EVENTS + e])
                print(line)
        print("TRACEDONE")

        _ = scores_buf
        _ = idxs_buf
        _ = trace_buf
