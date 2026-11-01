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
"""`argmax_with_bias_gpu` against an argmax of the materialized float32 sum.

Each row's answer must be exactly the argmax of
`float32(logits[:, position, :]) + float32(bias)`: the lowest tied index wins,
NaN never wins, and a row with nothing above -inf reports 0. Covers aligned
and unaligned widths, every position, and 16-bit and float32 inputs.
"""

from std.random import random_float64, seed
from std.utils.numerics import inf, nan, neg_inf

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.argmaxmin_gpu import argmax_with_bias_gpu

comptime STATIC_BOTH = 1
comptime STATIC_LOGITS = 2
comptime STATIC_BIAS = 3

comptime FILL_RANDOM = 0
comptime FILL_TIES = 1
comptime FILL_NON_FINITE = 2
comptime FILL_ALL_NAN = 3
comptime FILL_ALL_NEG_INF = 4


def _check[
    dtype: DType, bias_dtype: DType, static_layout: Int = 0
](
    ctx: DeviceContext,
    label: String,
    rows: Int,
    positions: Int,
    vocab: Int,
    kinds: List[Int],
) raises:
    var n_logits = rows * positions * vocab
    var logits_host = ctx.enqueue_create_host_buffer[dtype](n_logits)
    var bias_host = ctx.enqueue_create_host_buffer[bias_dtype](rows * vocab)
    var sum_host = ctx.enqueue_create_host_buffer[DType.float32](rows * vocab)
    var got_host = ctx.enqueue_create_host_buffer[DType.int64](rows)
    ctx.synchronize()

    # Multiples of 1/32 in [-8, 8) are exact in bf16, and the sums tie often.
    for i in range(n_logits):
        logits_host[i] = (Float64(Int(random_float64(-256, 256))) / 32).cast[
            dtype
        ]()
    for r in range(rows):
        for i in range(vocab):
            var b = r * vocab + i
            bias_host[b] = (Float64(Int(random_float64(-64, 64))) / 32).cast[
                bias_dtype
            ]()
            var kind = kinds[r % len(kinds)]
            if kind == FILL_ALL_NAN:
                bias_host[b] = nan[bias_dtype]()
            elif kind == FILL_ALL_NEG_INF:
                bias_host[b] = neg_inf[bias_dtype]()
            elif kind == FILL_NON_FINITE:
                if i % 3 == 0:
                    bias_host[b] = nan[bias_dtype]()
                elif i % 7 == 0:
                    bias_host[b] = neg_inf[bias_dtype]()
        var kind = kinds[r % len(kinds)]
        if kind == FILL_TIES and vocab > 8:
            # Plant equal winning sums at three spots, highest index first.
            for p in range(positions):
                for idx in [vocab - 1, vocab // 2 + 1, 5]:
                    logits_host[(r * positions + p) * vocab + idx] = Scalar[
                        dtype
                    ](20.0)
                    bias_host[r * vocab + idx] = Scalar[bias_dtype](1.0)
        if kind == FILL_NON_FINITE and vocab > 8:
            for p in range(positions):
                logits_host[(r * positions + p) * vocab + 0] = inf[dtype]()

    var logits_dev = ctx.enqueue_create_buffer[dtype](n_logits)
    var bias_dev = ctx.enqueue_create_buffer[bias_dtype](rows * vocab)
    var got_dev = ctx.enqueue_create_buffer[DType.int64](rows)
    ctx.enqueue_copy(logits_dev, logits_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    for position in range(positions):
        # Reference: materialize the float32 sum on the host.
        for r in range(rows):
            for i in range(vocab):
                sum_host[r * vocab + i] = (
                    logits_host[(r * positions + position) * vocab + i].cast[
                        DType.float32
                    ]()
                    + bias_host[r * vocab + i].cast[DType.float32]()
                )
        ctx.enqueue_memset(got_dev, -7)
        comptime if static_layout == STATIC_BOTH:
            argmax_with_bias_gpu(
                ctx,
                TileTensor(logits_dev, row_major(rows, Idx[7], Idx[131072])),
                TileTensor(bias_dev, row_major(rows, Idx[131072])),
                position,
                TileTensor(got_dev, row_major(rows)),
            )
        elif static_layout == STATIC_LOGITS:
            argmax_with_bias_gpu(
                ctx,
                TileTensor(logits_dev, row_major(rows, Idx[7], Idx[131072])),
                TileTensor(bias_dev, row_major(rows, vocab)),
                position,
                TileTensor(got_dev, row_major(rows)),
            )
        elif static_layout == STATIC_BIAS:
            argmax_with_bias_gpu(
                ctx,
                TileTensor(logits_dev, row_major(rows, positions, vocab)),
                TileTensor(bias_dev, row_major(rows, Idx[131072])),
                position,
                TileTensor(got_dev, row_major(rows)),
            )
        else:
            argmax_with_bias_gpu(
                ctx,
                TileTensor(logits_dev, row_major(rows, positions, vocab)),
                TileTensor(bias_dev, row_major(rows, vocab)),
                position,
                TileTensor(got_dev, row_major(rows)),
            )
        ctx.enqueue_copy(got_host, got_dev)
        ctx.synchronize()

        for r in range(rows):
            # Host scan: strict from -inf, so NaN never wins and ties keep the
            # lowest index.
            var best = neg_inf[DType.float32]()
            var best_idx = 0
            for i in range(vocab):
                var v = sum_host[r * vocab + i]
                if v > best:
                    best = v
                    best_idx = i
            var got = Int(got_host[r])
            var desc = String(
                label,
                ": dtype=",
                dtype,
                " bias=",
                bias_dtype,
                " vocab=",
                vocab,
                " position=",
                position,
                " row=",
                r,
                " got=",
                got,
            )
            if got != best_idx:
                raise Error(desc + " expected " + String(best_idx))

    _ = logits_dev^
    _ = bias_dev^
    _ = got_dev^


def _run[dtype: DType, bias_dtype: DType](ctx: DeviceContext) raises:
    var r = FILL_RANDOM
    var t = FILL_TIES
    # The DSpark Markov head shape: 7 positions over a 131072 vocabulary.
    _check[dtype, bias_dtype](ctx, "markov", 3, 7, 131072, [r, t, r])
    # Production layouts keep the draft width and vocabulary static.
    _check[dtype, bias_dtype, STATIC_BOTH](
        ctx, "static-markov", 16, 7, 131072, [r, t]
    )
    # FP32 uses a smaller, already-launchable block at a single row.
    _check[dtype, bias_dtype, STATIC_BOTH](
        ctx, "static-single", 1, 7, 131072, [r, t]
    )
    # Each input can independently carry a static vocabulary.
    _check[dtype, bias_dtype, STATIC_LOGITS](
        ctx, "static-logits", 16, 7, 131072, [r, t]
    )
    _check[dtype, bias_dtype, STATIC_BIAS](
        ctx, "static-bias", 16, 7, 131072, [r, t]
    )
    # Odd widths take the unaligned loads and the ragged tail.
    _check[dtype, bias_dtype](ctx, "odd-vocab", 2, 3, 32003, [t, r])
    _check[dtype, bias_dtype](ctx, "tiny-vocab", 3, 2, 7, [r])
    _check[dtype, bias_dtype](ctx, "vocab-1", 2, 2, 1, [r])
    # More rows than SMs: one split per row.
    _check[dtype, bias_dtype](ctx, "many-rows", 300, 2, 4096, [r, t])
    _check[dtype, bias_dtype](
        ctx,
        "non-finite",
        4,
        2,
        16384,
        [FILL_NON_FINITE, FILL_ALL_NAN, FILL_ALL_NEG_INF, r],
    )


def main() raises:
    seed(20261015)
    with DeviceContext() as ctx:
        _run[DType.bfloat16, DType.bfloat16](ctx)
        _run[DType.float32, DType.float32](ctx)
        _run[DType.bfloat16, DType.float32](ctx)
        _run[DType.float32, DType.bfloat16](ctx)
        _run[DType.float16, DType.float16](ctx)
    print("OK")
