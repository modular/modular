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
"""Greedy (effective top_k == 1) rows of `fused_token_sampling_gpu`.

A k=1 draw must return a row maximum whatever the temperature, top-p or
min-p. The only freedom is which maximum when several tie: the rejection
sampler draws among them with the row's seed, so on most targets the test
accepts any tied maximum. On Apple, greedy rows take an argmax, and the test
pins its rules exactly: the lowest tied index wins (`numpy.argmax`), NaN never
wins, and an all-NaN or all -inf row returns 0. Rows with k > 1 in the same
batch must still land inside their top-k set.
"""

from std.random import random_float64, seed
from std.sys.info import has_apple_gpu_accelerator
from std.utils.numerics import inf, nan, neg_inf

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.topk import fused_token_sampling_gpu

comptime LARGE_VOCAB = 163840

comptime FILL_RANDOM = 0
comptime FILL_TIES = 1
comptime FILL_ALL_EQUAL = 2
comptime FILL_NON_FINITE = 3
comptime FILL_ALL_NAN = 4
comptime FILL_ALL_NEG_INF = 5
comptime FILL_NAN_FIRST = 6


def _is_non_finite_fill(kind: Int) -> Bool:
    return (
        kind == FILL_NON_FINITE
        or kind == FILL_ALL_NAN
        or kind == FILL_ALL_NEG_INF
        or kind == FILL_NAN_FIRST
    )


def _fill_row(
    rows: TileTensor[mut=True, ...], b: Int, kind: Int, tie_idxs: List[Int]
):
    comptime assert rows.flat_rank == 2
    comptime dtype = rows.dtype
    var n = Int(rows.dim[1]())
    for i in range(n):
        if kind == FILL_ALL_EQUAL:
            rows[b, i] = Scalar[dtype](1.5)
        elif kind == FILL_ALL_NAN:
            rows[b, i] = nan[dtype]()
        elif kind == FILL_ALL_NEG_INF:
            rows[b, i] = neg_inf[dtype]()
        else:
            # Integer multiples of 1/32 in [-8, 8) are exact in bf16, so
            # distinct values never collapse after the kernel's exp.
            rows[b, i] = (Float64(Int(random_float64(-256, 256))) / 32).cast[
                dtype
            ]()
    if kind == FILL_TIES:
        for t in range(len(tie_idxs)):
            if tie_idxs[t] < n:
                rows[b, tie_idxs[t]] = Scalar[dtype](9.0)
    elif kind == FILL_NON_FINITE:
        for i in range(0, n, 3):
            rows[b, i] = neg_inf[dtype]()
        for i in range(1, n, 5):
            rows[b, i] = nan[dtype]()
        if n > 4:
            rows[b, n - 2] = Scalar[dtype](12.0)
            rows[b, 2] = inf[dtype]() if n > 100 else Scalar[dtype](-3.0)
    elif kind == FILL_NAN_FIRST:
        rows[b, 0] = nan[dtype]()


def _first_argmax(rows: TileTensor[...], b: Int) -> Int:
    """Strict scan from -inf: NaN never wins and ties keep the lowest index."""
    comptime assert rows.flat_rank == 2
    var best = neg_inf[DType.float32]()
    var best_idx = 0
    for i in range(Int(rows.dim[1]())):
        var v = rows[b, i][0].cast[DType.float32]()
        if v > best:
            best = v
            best_idx = i
    return best_idx


def _num_greater(rows: TileTensor[...], b: Int, idx: Int) -> Int:
    comptime assert rows.flat_rank == 2
    var pivot = rows[b, idx][0].cast[DType.float32]()
    var count = 0
    for i in range(Int(rows.dim[1]())):
        if rows[b, i][0].cast[DType.float32]() > pivot:
            count += 1
    return count


def _check[
    dtype: DType
](
    ctx: DeviceContext,
    label: String,
    vocab: Int,
    kinds: List[Int],
    ks: List[Int],
    max_k: Int,
    temperature: Float32 = 1.0,
    top_p: Float32 = 1.0,
    min_p: Float32 = 0.0,
    tie_idxs: List[Int] = [],
) raises:
    var batch = len(kinds)
    var in_host = ctx.enqueue_create_host_buffer[dtype](batch * vocab)
    var k_host = ctx.enqueue_create_host_buffer[DType.int64](batch)
    var temp_host = ctx.enqueue_create_host_buffer[DType.float32](batch)
    var top_p_host = ctx.enqueue_create_host_buffer[DType.float32](batch)
    var min_p_host = ctx.enqueue_create_host_buffer[DType.float32](batch)
    var seed_host = ctx.enqueue_create_host_buffer[DType.uint64](batch)
    var out_host = ctx.enqueue_create_host_buffer[DType.int64](batch)
    ctx.synchronize()

    var in_rows = TileTensor(in_host, row_major(batch, vocab))
    for b in range(batch):
        _fill_row(in_rows, b, kinds[b], tie_idxs)
        k_host[b] = Int64(ks[b])
        temp_host[b] = temperature
        top_p_host[b] = top_p
        min_p_host[b] = min_p
        seed_host[b] = UInt64(1234 + 77 * b)

    var in_dev = ctx.enqueue_create_buffer[dtype](batch * vocab)
    var k_dev = ctx.enqueue_create_buffer[DType.int64](batch)
    var temp_dev = ctx.enqueue_create_buffer[DType.float32](batch)
    var top_p_dev = ctx.enqueue_create_buffer[DType.float32](batch)
    var min_p_dev = ctx.enqueue_create_buffer[DType.float32](batch)
    var seed_dev = ctx.enqueue_create_buffer[DType.uint64](batch)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](batch)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(min_p_dev, min_p_host)
    ctx.enqueue_copy(seed_dev, seed_host)
    ctx.enqueue_memset(out_dev, -7)

    var min_top_p = top_p
    fused_token_sampling_gpu(
        ctx,
        max_k,
        min_top_p,
        TileTensor(in_dev, row_major(batch, vocab)),
        TileTensor(out_dev, row_major(batch, 1)),
        k=TileTensor(k_dev, row_major(batch)).as_unsafe_any_origin().as_imm(),
        temperature=TileTensor(temp_dev, row_major(batch))
        .as_unsafe_any_origin()
        .as_imm(),
        top_p=TileTensor(top_p_dev, row_major(batch))
        .as_unsafe_any_origin()
        .as_imm(),
        min_p=TileTensor(min_p_dev, row_major(batch))
        .as_unsafe_any_origin()
        .as_imm(),
        seed=TileTensor(seed_dev, row_major(batch))
        .as_unsafe_any_origin()
        .as_imm(),
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for b in range(batch):
        var got = Int(out_host[b])
        var desc = String(
            label,
            ": dtype=",
            dtype,
            " vocab=",
            vocab,
            " row=",
            b,
            " k=",
            ks[b],
            " max_k=",
            max_k,
            " got=",
            got,
        )
        if got < 0 or got >= vocab:
            raise Error(desc + " is out of range")
        var k = ks[b]
        if k == -1:
            k = max_k
        if k != 1:
            if _num_greater(in_rows, b, got) >= k:
                raise Error(desc + " is outside the top-k set")
            continue
        var expected = _first_argmax(in_rows, b)
        comptime if has_apple_gpu_accelerator():
            if got != expected:
                raise Error(
                    desc + " expected the first argmax " + String(expected)
                )
        else:
            # The sampler's softmax cannot rank a non-finite row.
            if _is_non_finite_fill(kinds[b]):
                continue
            if in_rows[b, got][0] != in_rows[b, expected][0]:
                raise Error(desc + " is not a row maximum")

    _ = in_dev^
    _ = k_dev^
    _ = temp_dev^
    _ = top_p_dev^
    _ = min_p_dev^
    _ = seed_dev^
    _ = out_dev^


def _run_dtype[dtype: DType](ctx: DeviceContext) raises:
    var r = FILL_RANDOM
    var t = FILL_TIES

    # One planted maximum: the sampler and the argmax must agree exactly.
    _check[dtype](
        ctx, "unique-max", LARGE_VOCAB, [t, t], [1, 1], 1, tie_idxs=[123457]
    )
    _check[dtype](ctx, "large-vocab-batch1", LARGE_VOCAB, [r], [1], 1)
    _check[dtype](
        ctx, "large-vocab-batch3", LARGE_VOCAB, [r, r, r], [1, 1, 1], 1
    )

    # Ties straddling a vector, far apart in the row, and at both ends. In
    # bf16 the random rows above already tie at their maximum as well.
    _check[dtype](
        ctx, "tie-in-vector", LARGE_VOCAB, [t], [1], 1, tie_idxs=[8003, 8005]
    )
    _check[dtype](
        ctx,
        "tie-far-apart",
        LARGE_VOCAB,
        [t, t],
        [1, 1],
        1,
        tie_idxs=[163000, 81920, 1000],
    )
    _check[dtype](
        ctx, "tie-ends", LARGE_VOCAB, [t], [1], 1, tie_idxs=[LARGE_VOCAB - 1, 0]
    )
    _check[dtype](ctx, "all-equal", LARGE_VOCAB, [FILL_ALL_EQUAL], [1], 1)

    # Sampling knobs cannot move a k=1 draw off the maximum.
    _check[dtype](
        ctx,
        "temp-0",
        LARGE_VOCAB,
        [t],
        [1],
        1,
        temperature=0.0,
        tie_idxs=[77, 7],
    )
    _check[dtype](
        ctx,
        "temp-knobs",
        LARGE_VOCAB,
        [r, t],
        [1, 1],
        1,
        temperature=1.7,
        top_p=0.3,
        min_p=0.5,
        tie_idxs=[4096, 4095],
    )

    # With max_k == 1, a row asking for top_k == -1 resolves to k == 1.
    _check[dtype](ctx, "k-minus-one", LARGE_VOCAB, [r, r], [-1, 1], 1)

    # Greedy rows next to sampling rows in one launch.
    _check[dtype](
        ctx,
        "mixed-batch",
        LARGE_VOCAB,
        [t, r, t],
        [1, 20, 1],
        20,
        tie_idxs=[99999, 12],
    )

    # Short and unaligned rows take narrower vector widths.
    _check[dtype](ctx, "odd-vocab", 32003, [r, t], [1, 1], 1, tie_idxs=[31999])
    _check[dtype](ctx, "tiny-vocab", 7, [r, r], [1, 1], 1)
    _check[dtype](ctx, "vocab-1", 1, [r], [1], 1)

    # NaN never wins, even at index 0, and a row with nothing above -inf
    # returns 0.
    var nf = FILL_NON_FINITE
    var all_nan = FILL_ALL_NAN
    var all_neg_inf = FILL_ALL_NEG_INF
    var nan_first = FILL_NAN_FIRST
    _check[dtype](ctx, "non-finite", LARGE_VOCAB, [nf], [1], 1)
    _check[dtype](ctx, "all-nan", 4096, [all_nan], [1], 1)
    _check[dtype](ctx, "all-neg-inf", 4096, [all_neg_inf], [1], 1)
    _check[dtype](ctx, "nan-first", 4096, [nan_first], [1], 1)
    _check[dtype](
        ctx,
        "non-finite-batch",
        LARGE_VOCAB,
        [all_nan, r, all_neg_inf, nan_first],
        [1, 1, 1, 1],
        1,
    )


def main() raises:
    seed(20260925)
    with DeviceContext() as ctx:
        _run_dtype[DType.float32](ctx)
        _run_dtype[DType.bfloat16](ctx)
    print("OK")
