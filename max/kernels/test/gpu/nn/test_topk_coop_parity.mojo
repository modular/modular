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
"""Parity gate: splitting a row across blocks must not change the draw.

`CoopRow` splits one row's vocabulary, its softmax and its rejection search
across `coop_size` blocks, then folds the partial results back in rank order.
Rank order is token order, so a group follows the same pivot sequence as a
single block and accepts the same set of token values; a combine that folded
out of order, a publish slot read before its peer wrote it, or a barrier that
lets a block run ahead all show up here as a different answer.

The reference arm is the same launcher with `num_coop_blocks=1`, which takes the
whole row in one block. Both arms see identical logits, per-row seeds and
sampler settings, so a mismatch is the split's, not the input's.

Two data sets, because the two kernels agree to different degrees on them:

- Distinct logits (`fine_logit`, `float32`) settle the accepted cutoff on a
  value exactly one token holds, and then the split must select the *same
  token*. This is the case that would catch a broken reduction.
- Coarsely quantized logits (`coarse_logit`, `bfloat16`) hold hundreds of
  tokens per distinct value, so the accepted cutoff falls on a tie class. Both
  arms still have to accept the same *value* -- the same tie class -- but which
  member of the class the walk lands on moves with the last bit of the row's
  mass, which a sliced sum rounds differently from a whole-row sum. Selecting a
  different token from the same class is a legal draw; accepting a token of a
  different value is not.

Covered shapes: the ones that make a split worthwhile (batch one with a wide
vocabulary), `k=1` (which bypasses the search for the row's argmax) and `k>1`,
rows that ask for different `k` in one launch, batch sizes that do not divide
the group size, a vocabulary whose width is not a multiple of the vector width,
and the non-deterministic and deterministic draw paths.
"""

from max.gpu.host import DeviceAttribute, DeviceContext, HostBuffer
from std.math import gcd
from layout import TileTensor, row_major
from std.testing import assert_true

from nn.sampling.topk_fi import coop_group_size, topk_topp_sampling_from_prob

comptime DTYPE = DType.bfloat16
comptime IDX = DType.int64
# The launcher's default block size, and what production launches.
comptime BLOCK = 1024

# The vocabularies the sampler runs widest on, from the released model configs.
comptime D_LLAMA = 128256
comptime D_GEMMA = 256000


def fine_logit(row: Int, col: Int) -> Float32:
    """Distinct logits: 32 bits of hash per token, so no two of a row's 128k
    values round to the same `float32`."""
    var h = (col * 2654435761 + row * 40503) % 4_294_967_296
    return Float32(Float64(h) / 4_294_967_296.0 * 12.0 - 6.0)


def coarse_logit(row: Int, col: Int) -> Float32:
    """Quantized logits: a million steps across twelve decades rounds, in
    `bfloat16`, to a few hundred distinct values per row."""
    var h = (col * 2654435761 + row * 97) % 1_000_003
    return Float32(Float64(h) / 1_000_003.0 * 12.0 - 6.0)


def fill_logits[
    dtype: DType
](ctx: DeviceContext, rows: Int, d: Int, coarse: Bool) raises -> HostBuffer[
    dtype
]:
    var host = ctx.enqueue_create_host_buffer[dtype](rows * d)
    for row in range(rows):
        for col in range(d):
            var v = coarse_logit(row, col) if coarse else fine_logit(row, col)
            host[row * d + col] = v.cast[dtype]()
    return host^


def sample_tokens[
    dtype: DType = DTYPE
](
    ctx: DeviceContext,
    logits_host: HostBuffer[dtype],
    rows: Int,
    d: Int,
    ks: List[Int],
    top_p: Float32,
    temperature: Float32,
    num_coop_blocks: Int,
    deterministic: Bool = False,
    rng_offset: UInt64 = 0,
) raises -> List[Int]:
    """Runs one launch and returns the token it drew for each row."""
    var logits_dev = ctx.enqueue_create_buffer[dtype](rows * d)
    ctx.enqueue_copy(logits_dev, logits_host)

    var tokens_dev = ctx.enqueue_create_buffer[IDX](rows)

    var temp_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[IDX](rows)
    var seed_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    for row in range(rows):
        temp_host[row] = temperature
        top_p_host[row] = top_p
        top_k_host[row] = Scalar[IDX](ks[row])
        seed_host[row] = UInt64(1234 + row)

    var temp_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[IDX](rows)
    var seed_dev = ctx.enqueue_create_buffer[.uint64](rows)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.enqueue_copy(seed_dev, seed_host)

    topk_topp_sampling_from_prob[from_logits=True](
        ctx,
        TileTensor(logits_dev, row_major(rows, d)),
        TileTensor(tokens_dev, row_major(rows)),
        d,
        deterministic=deterministic,
        rng_seed=TileTensor(seed_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_imm(),
        rng_offset=rng_offset,
        top_k_arr=TileTensor(top_k_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_imm(),
        top_p_arr=TileTensor(top_p_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_imm(),
        temperature=TileTensor(temp_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_imm(),
        num_coop_blocks=num_coop_blocks,
    )

    var tokens_host = ctx.enqueue_create_host_buffer[IDX](rows)
    ctx.enqueue_copy(tokens_host, tokens_dev)
    ctx.synchronize()

    var tokens = List[Int]()
    for row in range(rows):
        tokens.append(Int(tokens_host[row]))

    _ = logits_dev^
    _ = tokens_dev^
    _ = temp_dev^
    _ = top_p_dev^
    _ = top_k_dev^
    _ = seed_dev^

    return tokens^


def check_split_parity[
    dtype: DType = DTYPE, coarse: Bool = False
](
    ctx: DeviceContext,
    name: String,
    rows: Int,
    d: Int,
    ks: List[Int],
    *,
    top_p: Float32 = 0.95,
    temperature: Float32 = 1.0,
    requested: Int = 0,
    deterministic: Bool = False,
    rounds: Int = 2,
) raises:
    """Compares one block per row against a split row, over several draws."""
    var vec_size = gcd(8, d)
    var n_vec = d // vec_size
    var group = coop_group_size(ctx, rows, BLOCK, n_vec, requested)

    # A device with multiprocessors to spare has to split: a launch that
    # silently stayed at one block per row would make this whole test pass for
    # nothing.
    var mp_count = ctx.device().get_attribute(
        DeviceAttribute.MULTIPROCESSOR_COUNT
    )
    var can_split = rows * 2 <= mp_count and n_vec >= 2 * BLOCK
    if can_split:
        assert_true(
            group > 1,
            String(
                t"{name}: a device with ",
                mp_count,
                t" multiprocessors must split ",
                rows,
                t" rows, got a group of ",
                group,
            ),
        )

    var logits_host = fill_logits[dtype](ctx, rows, d, coarse)

    for draw in range(rounds):
        var offset = UInt64(draw) * 7919
        var single = sample_tokens[dtype](
            ctx,
            logits_host,
            rows,
            d,
            ks,
            top_p,
            temperature,
            1,
            deterministic=deterministic,
            rng_offset=offset,
        )
        var split = sample_tokens[dtype](
            ctx,
            logits_host,
            rows,
            d,
            ks,
            top_p,
            temperature,
            requested,
            deterministic=deterministic,
            rng_offset=offset,
        )

        for row in range(rows):
            assert_true(
                single[row] >= 0 and single[row] < d,
                String(
                    t"{name}: one block drew ",
                    single[row],
                    t" outside [0, ",
                    d,
                    t") on row ",
                    row,
                ),
            )
            assert_true(
                split[row] >= 0 and split[row] < d,
                String(
                    t"{name}: a split row drew ",
                    split[row],
                    t" outside [0, ",
                    d,
                    t") on row ",
                    row,
                ),
            )
            var same_value = (
                logits_host[row * d + split[row]]
                == logits_host[row * d + single[row]]
            )
            # A tie class accepts either token, but the accepted *value* has to
            # match, and on tie-free logits so does the token itself.
            comptime if coarse:
                assert_true(
                    same_value,
                    String(
                        t"{name}: row ",
                        row,
                        t" (k=",
                        ks[row],
                        t", group=",
                        group,
                        t", offset=",
                        offset,
                        t") accepted a different logit: split drew ",
                        split[row],
                        t" (",
                        logits_host[row * d + split[row]],
                        t") but one block drew ",
                        single[row],
                        t" (",
                        logits_host[row * d + single[row]],
                        t")",
                    ),
                )
            else:
                assert_true(
                    split[row] == single[row],
                    String(
                        t"{name}: row ",
                        row,
                        t" (k=",
                        ks[row],
                        t", group=",
                        group,
                        t", offset=",
                        offset,
                        t") drew ",
                        split[row],
                        t" from a split row but ",
                        single[row],
                        t" from a single block",
                    ),
                )

    print(
        String(
            t"{name}: rows=",
            rows,
            t" d=",
            d,
            t" group=",
            group,
            t" k=",
            ks[0],
            t"..",
            ks[rows - 1],
            t" ok",
        )
    )

    _ = logits_host^


def uniform_ks(rows: Int, k: Int) -> List[Int]:
    return List[Int](length=rows, fill=k)


def mixed_ks(rows: Int) -> List[Int]:
    """Ragged per-row work: a mixture-of-experts step draws one token per
    expert, and the rows arrive with different k. Every block of a group reads
    the same row, so the split stays legal."""
    var ks = List[Int]()
    for row in range(rows):
        ks.append(1 if row % 3 == 0 else 8 * (1 + row % 4))
    return ks^


def main() raises:
    with DeviceContext() as ctx:
        # Batch one is where a split pays: one block leaves the rest of the
        # device idle. k=1 takes the argmax, k>1 the rejection search.
        check_split_parity[dtype=DType.float32](
            ctx, "b1 k1 llama", 1, D_LLAMA, uniform_ks(1, 1)
        )
        check_split_parity[dtype=DType.float32](
            ctx, "b1 k50 llama", 1, D_LLAMA, uniform_ks(1, 50)
        )
        check_split_parity[dtype=DType.float32](
            ctx, "b1 k1 gemma", 1, D_GEMMA, uniform_ks(1, 1)
        )
        check_split_parity[dtype=DType.float32](
            ctx, "b1 k50 gemma", 1, D_GEMMA, uniform_ks(1, 50)
        )

        # Wider batches shrink the group the device can keep resident before
        # they reach one block per row, so the middle of the range is covered
        # too.
        check_split_parity[dtype=DType.float32](
            ctx, "b8 k50 llama", 8, D_LLAMA, uniform_ks(8, 50)
        )
        check_split_parity[dtype=DType.float32](
            ctx, "b8 mixed gemma", 8, D_GEMMA, mixed_ks(8)
        )
        check_split_parity[dtype=DType.float32](
            ctx, "b32 k50 llama", 32, D_LLAMA, uniform_ks(32, 50)
        )

        # Ragged shapes: the batch is not a multiple of the group, and one
        # vocabulary width leaves a tail that the slice bounds have to clamp.
        check_split_parity[dtype=DType.float32](
            ctx, "b5 mixed gemma", 5, D_GEMMA, mixed_ks(5)
        )
        check_split_parity[dtype=DType.float32](
            ctx, "b3 mixed odd vocab", 3, 100003, mixed_ks(3)
        )

        # Pinning is a hint: one block per row must not split, and a count the
        # device cannot keep resident must fall back rather than ask for a
        # group that would spin forever.
        var one_row_vectors = D_LLAMA // gcd(8, D_LLAMA)
        assert_true(
            coop_group_size(ctx, 1, BLOCK, one_row_vectors, 1) == 1,
            "pinning one block per row must not split the row",
        )
        assert_true(
            coop_group_size(ctx, 1, BLOCK, one_row_vectors, 4096)
            == coop_group_size(ctx, 1, BLOCK, one_row_vectors, 0),
            "a pinned group the device cannot keep resident must fall back",
        )

        # Pinning is how the profile compares group sizes; each one must agree
        # with the single block, not only the group the device would pick.
        for requested in [2, 4, 8]:
            check_split_parity[dtype=DType.float32](
                ctx,
                String(t"b1 k50 pinned ", requested),
                1,
                D_LLAMA,
                uniform_ks(1, 50),
                requested=requested,
            )

        # The deterministic draw path skips the RNG state, so it must agree too.
        check_split_parity[dtype=DType.float32](
            ctx,
            "b8 k50 deterministic",
            8,
            D_LLAMA,
            uniform_ks(8, 50),
            deterministic=True,
        )

        # The production dtype: bf16 logits leave a few hundred distinct values
        # per row, so the accepted cutoff sits on a tie class.
        check_split_parity[coarse=True](
            ctx, "b1 k50 bf16 ties", 1, D_LLAMA, uniform_ks(1, 50)
        )
        check_split_parity[coarse=True](
            ctx, "b8 k50 bf16 ties", 8, D_LLAMA, uniform_ks(8, 50)
        )
        check_split_parity[coarse=True](
            ctx, "b8 mixed bf16 ties", 8, D_GEMMA, mixed_ks(8)
        )
