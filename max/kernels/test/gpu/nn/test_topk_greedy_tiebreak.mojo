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

"""Greedy sampling breaks exact ties at the row maximum by lowest index.

Temperature 0 reaches the sampler as `top_k = 1` with the temperature left at
0 and a per-request seed. The rejection sampler accepts any token tied at the
maximum, so without a tie-break the seed decides which one is emitted. Every
row here carries a different seed, and every row must return the lowest tied
index. Vocab sizes sit below and above one block's pass width
(1024 threads x 8 lanes), and the tied indices straddle vector and block
boundaries.
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.topk import fused_token_sampling_gpu
from std.testing import assert_equal

comptime IN_TYPE = DType.bfloat16
comptime IDX_TYPE = DType.int64
comptime NUM_SEEDS = 100
comptime TIE_LOGIT = 5.0


def check_greedy(
    ctx: DeviceContext, vocab: Int, var tied: List[Int], expected: Int
) raises:
    var in_host = ctx.enqueue_create_host_buffer[IN_TYPE](NUM_SEEDS * vocab)
    for r in range(NUM_SEEDS):
        for i in range(vocab):
            in_host[r * vocab + i] = Scalar[IN_TYPE](
                Float32((i * 37) % 101) * 0.01 - 2.0
            )
        for t in tied:
            in_host[r * vocab + t] = Scalar[IN_TYPE](TIE_LOGIT)

    var k_host = ctx.enqueue_create_host_buffer[IDX_TYPE](NUM_SEEDS)
    var temp_host = ctx.enqueue_create_host_buffer[.float32](NUM_SEEDS)
    var seed_host = ctx.enqueue_create_host_buffer[.uint64](NUM_SEEDS)
    for r in range(NUM_SEEDS):
        k_host[r] = 1
        temp_host[r] = 0.0
        seed_host[r] = UInt64(r) * 0x9E3779B97F4A7C15 + 1

    var in_dev = ctx.enqueue_create_buffer[IN_TYPE](NUM_SEEDS * vocab)
    var k_dev = ctx.enqueue_create_buffer[IDX_TYPE](NUM_SEEDS)
    var temp_dev = ctx.enqueue_create_buffer[.float32](NUM_SEEDS)
    var seed_dev = ctx.enqueue_create_buffer[.uint64](NUM_SEEDS)
    var out_dev = ctx.enqueue_create_buffer[IDX_TYPE](NUM_SEEDS)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(seed_dev, seed_host)

    fused_token_sampling_gpu(
        ctx,
        1,
        Float32(1.0),
        TileTensor(in_dev, row_major(NUM_SEEDS, vocab)),
        TileTensor(out_dev, row_major(NUM_SEEDS, 1)),
        k=TileTensor(k_dev, row_major(NUM_SEEDS))
        .as_unsafe_any_origin()
        .as_imm(),
        temperature=TileTensor(temp_dev, row_major(NUM_SEEDS))
        .as_unsafe_any_origin()
        .as_imm(),
        seed=TileTensor(seed_dev, row_major(NUM_SEEDS))
        .as_unsafe_any_origin()
        .as_imm(),
    )

    var out_host = ctx.enqueue_create_host_buffer[IDX_TYPE](NUM_SEEDS)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var hits = 0
    for r in range(NUM_SEEDS):
        if Int(out_host[r]) == expected:
            hits += 1
    print(
        "vocab",
        vocab,
        "tied",
        String(tied),
        "->",
        hits,
        "of",
        NUM_SEEDS,
        "seeds return",
        expected,
    )
    for r in range(NUM_SEEDS):
        assert_equal(
            Int(out_host[r]),
            expected,
            String(t"vocab {vocab} tied {String(tied)} seed row {r}"),
        )


def main() raises:
    with DeviceContext() as ctx:
        # Below one block pass: one iteration covers the row.
        check_greedy(ctx, 1000, [517, 3], 3)
        check_greedy(ctx, 1000, [999, 4, 3], 3)
        check_greedy(ctx, 1000, [517], 517)
        # Above it: the tied tokens land in different block passes.
        check_greedy(ctx, 151936, [70001, 9000], 9000)
        check_greedy(ctx, 151936, [151935, 70001, 9000], 9000)
        check_greedy(ctx, 151936, [70001], 70001)
        print("PASS")
