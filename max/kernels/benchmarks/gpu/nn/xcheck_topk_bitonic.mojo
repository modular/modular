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
"""Dumps `persistent_topk_block_split` output for cross-implementation checking.

Not a benchmark: it generates its scores from an integer hash so that a CUDA and
a NumPy harness can reproduce the identical fp32 bits, runs the kernel, and
prints the selected indices. The comparison itself lives outside (compare_topk.py).

Values are small non-negative integers cast to fp32 (exactly representable, so
no cross-language rounding can differ) drawn from `levels` distinct levels:
`levels=17` reproduces the tie plateaus of fp8-quantized indexer scores;
`levels` above `rows*N` makes the top-k set unique **in fp32**, which is the
only regime where two exact implementations must agree element for element.

`--dtype=bfloat16` reads the same generated values through a bf16 buffer, which
turns `levels` into a dial that separates two effects rather than confusing
them. Integers up to 256 are exact in bf16, so at `levels=17` the two dtypes
hold IDENTICAL values and must select identically -- a difference there is a bug
in the read path, not a precision effect. Above 256 they diverge by rounding
alone: at `levels=1000003` bf16 snaps values near 1e6 to multiples of 1024,
collapsing ~1000 fp32 levels into one plateau, so no `levels` however large
makes the bf16 set unique.

    xcheck_topk_bitonic --rows=48 --N=107228 --K=2048 --levels=17
    xcheck_topk_bitonic --rows=48 --N=107228 --K=2048 --levels=17 \\
        --dtype=bfloat16 --deterministic=True
"""

from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import TileTensor, row_major

from nn.topk_bitonic import persistent_topk_block_split


def _hash32(a: UInt32, b: UInt32) -> UInt32:
    """Wrapping-uint32 mixer, reproducible in CUDA C and NumPy."""
    var h = a * UInt32(0x9E3779B1) + b * UInt32(0x85EBCA77)
    h ^= h >> 16
    h = h * UInt32(0x7FEB352D)
    h ^= h >> 15
    h = h * UInt32(0x846CA68B)
    h ^= h >> 16
    return h


def _launch[
    in_dtype: DType, deterministic: Bool
](
    ctx: DeviceContext,
    scores_t: TileTensor[in_dtype, ...],
    idxs_t: TileTensor[.int32, ...],
    N: Int,
    K: Int,
    rows: Int,
) raises:
    persistent_topk_block_split[deterministic=deterministic](
        ctx,
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](scores_t.ptr),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
        N,
        K,
        rows,
    )


def _run[
    in_dtype: DType, deterministic: Bool
](
    ctx: DeviceContext,
    rows: Int,
    N: Int,
    K: Int,
    levels: Int,
    dump: Bool,
    iters: Int,
) raises:
    var scores_buf = ctx.enqueue_create_buffer[in_dtype](rows * N)
    var idxs_buf = ctx.enqueue_create_buffer[.int32](rows * K)

    # Row lengths and a per-row checksum let the peer harnesses prove they
    # built the same input before any output is compared.
    var num_keys = List[Int]()
    var checksum = List[UInt64]()
    with scores_buf.map_to_host() as h:
        for r in range(rows):
            var nk = N - Int(
                _hash32(UInt32(r), UInt32(0xABCD1234)) % UInt32(max(1, N // 8))
            )
            num_keys.append(nk)
            var cs = UInt64(0)
            for c in range(N):
                if c < nk:
                    var v = _hash32(UInt32(r), UInt32(c)) % UInt32(levels)
                    h[r * N + c] = Float32(Int(v)).cast[in_dtype]()
                    cs += UInt64(Int(v))
                else:
                    h[r * N + c] = Float32(-3.0e38).cast[in_dtype]()
            checksum.append(cs)

    var scores_t = TileTensor(scores_buf, row_major(rows, N))
    var idxs_t = TileTensor(idxs_buf, row_major(rows, K))
    ctx.synchronize()

    _launch[in_dtype, deterministic](ctx, scores_t, idxs_t, N, K, rows)
    ctx.synchronize()

    # Time on this exact input so the MAX/vLLM comparison is not just
    # distribution-matched but bit-identical.
    for _ in range(20):
        _launch[in_dtype, deterministic](ctx, scores_t, idxs_t, N, K, rows)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        _launch[in_dtype, deterministic](ctx, scores_t, idxs_t, N, K, rows)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("TIME_MS", Float64(t1 - t0) / Float64(iters) / 1.0e6)

    if dump:
        print("XCHECK", rows, N, K, levels)
        for r in range(rows):
            print("LEN", r, num_keys[r], checksum[r])
        with idxs_buf.map_to_host() as h:
            for r in range(rows):
                var line = String("IDX ")
                line += String(r)
                for j in range(K):
                    line += " "
                    line += String(Int(h[r * K + j]))
                print(line)
        print("XCHECKDONE")

    _ = scores_buf
    _ = idxs_buf


def main() raises:
    var rows = arg_parse("rows", 48)
    var N = arg_parse("N", 107228)
    var K = arg_parse("K", 2048)
    var levels = arg_parse("levels", 17)
    # The prefill shapes dump ~17M indices; off, this times only.
    var dump = arg_parse("dump", True)
    var iters = arg_parse("iters", 100)
    var dtype = arg_parse("dtype", String("float32"))
    # `deterministic=False` is production's contract and admits churn between
    # runs; `True` makes each arm reproducible, which is what lets a bf16-vs-f32
    # diff attribute a difference to rounding rather than to a race.
    var deterministic = arg_parse("deterministic", True)

    if dtype not in ["float32", "bfloat16"]:
        raise Error("unknown dtype: ", dtype)

    with DeviceContext() as ctx:
        if dtype == "bfloat16":
            if deterministic:
                _run[DType.bfloat16, True](ctx, rows, N, K, levels, dump, iters)
            else:
                _run[DType.bfloat16, False](
                    ctx, rows, N, K, levels, dump, iters
                )
        elif deterministic:
            _run[DType.float32, True](ctx, rows, N, K, levels, dump, iters)
        else:
            _run[DType.float32, False](ctx, rows, N, K, levels, dump, iters)
