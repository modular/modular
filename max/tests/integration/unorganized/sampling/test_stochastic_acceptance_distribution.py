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
"""Output-distribution test for ``stochastic_acceptance_sampler``.

Speculative decoding is only lossless if the token committed at each draft
position is marginally distributed as the target softmax, regardless of how
the draft proposes. This test drives the sampler with a fixed LLM-shaped
target distribution and drafts drawn uniformly from the target's top-5
tokens, then checks the empirical committed-token frequencies against the
target probabilities.

Why this discriminates: let ``q`` be the draft proposal and ``p`` the
(truncated) target distribution. The sampler draws ``x ~ p`` at each
position and accepts the draft ``d`` iff ``d == x``; the committed token is
``x`` either way, so the committed marginal is exactly ``p`` for any ``q``:

    P(commit = x) = P(sample = x) * (q(x) + sum_{d != x} q(d)) = p(x)

The previous coin-flip acceptance with full-target recovery skewed the
marginal to ``p(x) * (q(x) + 1 - E_q[p])``: with drafts uniform over the
top-5 (``q = 0.2`` there, ``0`` elsewhere) and ~0.78 of target mass on the
top-5, top-5 tokens were inflated ~4% and the tail deflated ~16% relative —
many sigma at the sample sizes below, so the chi-square and tail-mass
checks failed under that scheme (see CENG-970).
"""

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    gumbel_argmax_from_probs,
    gumbel_argmax_from_residual,
)
from max.nn._graph_ops.sampling import stochastic_acceptance_sampler

VOCAB_SIZE = 1024
NUM_STEPS = 4
BATCH_SIZE = 512
NUM_TRIALS = 64
TOP_TOKEN_COUNT = 5

# One narrow graph pins exact draws: small enough that every accepted and
# recovered token in a run can be written down, wide enough that the
# vocabulary does not collapse into the kernel's ragged tail.
NARROW_VOCAB_SIZE = 64
NARROW_BATCH_SIZE = 4
NARROW_NUM_STEPS = 3
NARROW_SEED = 0x5EED1234


def _compile_recovery_graph(
    session: InferenceSession, fuse: bool, vocab_size: int
) -> Model:
    """Compiles the residual draw, both ways, over the same inputs.

    ``fuse=True`` is the shipped op, which forms ``max(p - q, 0)`` inside the
    sampling kernel. ``fuse=False`` is the graph it replaced: a one-hot select,
    a subtract, a clamp, and the sampling pass reading the residual back. They
    draw from the same per-row seed, so equal token ids at a fixed seed is the
    statement that the fusion changed nothing.
    """
    device = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.float32, ["rows", vocab_size], device=device),
        TensorType(DType.float32, ["rows", vocab_size], device=device),
        TensorType(DType.int64, ["rows"], device=device),
        TensorType(DType.uint64, ["rows"], device=device),
    ]
    with Graph(
        "residual_recovery" if fuse else "residual_recovery_materialized",
        input_types=graph_inputs,
    ) as graph:
        probs, proposal, fallback, seed = graph.inputs
        if fuse:
            out = gumbel_argmax_from_residual(
                probs.tensor,
                proposal.tensor,
                fallback=fallback.tensor,
                seed=seed.tensor,
            )
        else:
            has_dist = fallback.tensor < 0
            # `probs` is the graph input, whose width is the `vocab_size`
            # dimension this helper was compiled with.
            onehot = ops.range(
                0, vocab_size, 1, device=device, dtype=DType.int64
            ) == ops.unsqueeze(fallback.tensor, axis=-1)
            draft_mass = ops.where(
                ops.unsqueeze(has_dist, axis=-1),
                proposal.tensor,
                onehot.cast(DType.float32),
            )
            out = gumbel_argmax_from_probs(
                ops.relu(probs.tensor - draft_mass), seed=seed.tensor
            )
        graph.output(out)
    return session.load(graph)


@pytest.fixture(scope="module")
def acceptance_sampler(session: InferenceSession) -> Model:
    """Compile the stochastic acceptance sampler once for the module."""
    device_ref = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device_ref),
    ]
    with Graph(
        "stochastic_acceptance_distribution", input_types=graph_inputs
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            seed,
        ) = graph.inputs
        graph.output(
            *stochastic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                temperature=temperature.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
            )
        )
    return session.load(graph)


def _compile_sampled_sampler(
    session: InferenceSession, vocab_size: int
) -> Model:
    """Compiles the sampled-draft-proposal acceptance sampler.

    ``vocab_size`` is the width of ``draft_probs_full``'s last dimension and
    the value the graph verifies against; the target logits stay a
    ``[rows, vocab_size]`` table like the model's own output.
    """
    device_ref = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device_ref),
        TensorType(
            DType.float32,
            ["batch_size", "num_steps", vocab_size],
            device=device_ref,
        ),
    ]
    with Graph(
        "stochastic_acceptance_distribution_sampled", input_types=graph_inputs
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            seed,
            draft_probs_full,
        ) = graph.inputs
        graph.output(
            *stochastic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                temperature=temperature.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
                draft_proposal="sampled",
                draft_probs_full=draft_probs_full.tensor,
                vocab_size=vocab_size,
            )
        )
    return session.load(graph)


@pytest.fixture(scope="module")
def sampled_acceptance_sampler(session: InferenceSession) -> Model:
    """Compile the sampled-draft-proposal acceptance sampler for the module."""
    return _compile_sampled_sampler(session, VOCAB_SIZE)


@pytest.fixture(scope="module")
def narrow_sampled_acceptance_sampler(session: InferenceSession) -> Model:
    """The same graph over a vocabulary small enough to pin by hand."""
    return _compile_sampled_sampler(session, NARROW_VOCAB_SIZE)


def _make_target_probs(
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Builds an LLM-shaped target distribution over ``VOCAB_SIZE`` tokens.

    A sharp head (5 tokens holding 78% of the mass) over a long lognormal
    tail, mimicking a typical LLM next-token distribution. Returns the
    probability vector and the top-5 token ids (descending probability).
    """
    head_probs = np.array([0.28, 0.20, 0.13, 0.10, 0.07])
    top_idx = rng.choice(VOCAB_SIZE, size=TOP_TOKEN_COUNT, replace=False)

    tail = np.exp(rng.normal(0.0, 1.0, VOCAB_SIZE))
    tail[top_idx] = 0.0
    tail *= (1.0 - head_probs.sum()) / tail.sum()

    probs = tail
    probs[top_idx] = head_probs
    return probs, top_idx.astype(np.int64)


def _noisy_draft_distribution(
    rng: np.random.Generator,
    logits_row: npt.NDArray[np.float32],
    low: float = -0.3,
    high: float = 0.3,
) -> npt.NDArray[np.float64]:
    """A full-support draft distribution perturbing the target's logits.

    Softmaxed with full support everywhere, so ``q`` is positive everywhere
    and the rejection-sampling exactness identity applies regardless of the
    draft's own distribution.
    """
    draft_logits_row = logits_row * (
        1.0 + rng.uniform(low, high, logits_row.shape[0])
    )
    draft_e = np.exp(draft_logits_row - draft_logits_row.max())
    return draft_e / draft_e.sum()


def test_recovered_tokens_respect_top_p(
    session: InferenceSession, acceptance_sampler: Model
) -> None:
    """Recovered tokens must stay inside the request's top-p nucleus.

    Drives the sampler with drafts drawn from the deep tail (target prob
    ~1e-4, so every draft position is rejected) and ``top_p=0.4``, whose
    nucleus under the LLM-shaped target is the top-2 tokens (0.28 + 0.20
    crosses 0.4). Every committed recovered token must then be one of the
    top-3 tokens (top-3 allows for either boundary-inclusion convention).
    A recovered path that samples the full target distribution commits a
    tail token ~22% of the time, so with hundreds of rejected rows the
    membership check fails immediately.
    """
    device = session.devices[0]
    rng = np.random.default_rng(1)

    target_probs, top_idx = _make_target_probs(rng)
    logits_row = np.log(target_probs).astype(np.float32)
    logits_np = np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)

    temperature = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(
        device
    )
    top_k = Buffer.from_numpy(np.full(BATCH_SIZE, -1, dtype=np.int64)).to(
        device
    )
    max_k = Buffer.from_numpy(np.array(-1, dtype=np.int64))
    top_p = Buffer.from_numpy(np.full(BATCH_SIZE, 0.4, dtype=np.float32)).to(
        device
    )
    min_top_p = Buffer.from_numpy(np.array(0.4, dtype=np.float32))

    # Draft only tail tokens so acceptance probability is ~1e-4 and the
    # first position is rejected on essentially every row.
    tail_tokens = np.setdiff1d(np.arange(VOCAB_SIZE), top_idx)

    committed: list[npt.NDArray[np.int64]] = []
    for _ in range(4):
        seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        draft_np = rng.choice(tail_tokens, size=(BATCH_SIZE, NUM_STEPS)).astype(
            np.int64
        )
        first_rejected, recovered, _bonus = acceptance_sampler(
            Buffer.from_dlpack(draft_np).to(device),
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([seed], dtype=np.uint64)).to(device),
        )
        assert isinstance(first_rejected, Buffer)
        assert isinstance(recovered, Buffer)
        fri_np = first_rejected.to_numpy().reshape(BATCH_SIZE)
        recovered_np = recovered.to_numpy()
        rejected_rows = np.nonzero(fri_np < NUM_STEPS)[0]
        committed.append(recovered_np[rejected_rows, fri_np[rejected_rows]])

    all_tokens = np.concatenate(committed)
    assert len(all_tokens) > BATCH_SIZE  # rejection path was exercised
    nucleus = set(top_idx[:3].tolist())
    outside = [int(t) for t in all_tokens if int(t) not in nucleus]
    assert not outside, (
        f"{len(outside)}/{len(all_tokens)} recovered tokens fell outside "
        f"the top_p=0.4 nucleus {sorted(nucleus)}; sample: {outside[:10]}"
    )


def test_stochastic_acceptance_output_distribution(
    session: InferenceSession, acceptance_sampler: Model
) -> None:
    """The committed token at each draft position must be ``~ softmax(target)``."""
    device = session.devices[0]
    rng = np.random.default_rng(0)

    target_probs, top_idx = _make_target_probs(rng)
    # softmax(log(p) / temperature) == p at temperature 1.0, so ``target_probs``
    # is the reference distribution for the committed tokens.
    logits_row = np.log(target_probs).astype(np.float32)
    # Every draft and bonus position sees the same fixed target logits.
    logits_np = np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)

    temperature = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(
        device
    )
    top_k = Buffer.from_numpy(np.full(BATCH_SIZE, -1, dtype=np.int64)).to(
        device
    )
    max_k = Buffer.from_numpy(np.array(-1, dtype=np.int64))
    top_p = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(device)
    min_top_p = Buffer.from_numpy(np.array(1.0, dtype=np.float32))

    committed: list[npt.NDArray[np.int64]] = []
    step_range = np.arange(NUM_STEPS)
    for _ in range(NUM_TRIALS):
        # The sampler's RNG seed is a single [1] uint64 per execute
        # (ops.random.SeedType); per-row variation comes from the elementwise
        # RNG. Draw a random seed per trial from the int64 range.
        seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        draft_np = rng.choice(top_idx, size=(BATCH_SIZE, NUM_STEPS)).astype(
            np.int64
        )
        first_rejected, recovered, _bonus = acceptance_sampler(
            Buffer.from_dlpack(draft_np).to(device),
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([seed], dtype=np.uint64)).to(device),
        )
        assert isinstance(first_rejected, Buffer)
        assert isinstance(recovered, Buffer)
        fri_np = first_rejected.to_numpy().reshape(BATCH_SIZE)
        recovered_np = recovered.to_numpy()

        # The tokens a speculative decode commits: the draft token at every
        # accepted position, then the recovered token at the first rejection.
        # (Bonus tokens go through topk_fused_sampling — a separate path —
        # and are excluded.)
        committed.append(draft_np[step_range[None, :] < fri_np[:, None]])
        rejected_rows = np.nonzero(fri_np < NUM_STEPS)[0]
        committed.append(recovered_np[rejected_rows, fri_np[rejected_rows]])

    all_tokens = np.concatenate(committed)
    counts = np.bincount(all_tokens, minlength=VOCAB_SIZE).astype(np.float64)
    total = counts.sum()
    # E[acceptance] = mean top-5 prob ~ 0.156, so ~1.18 committed tokens per
    # row: ~38k samples. Sanity-check both accept and reject paths were hit.
    assert total > BATCH_SIZE * NUM_TRIALS
    assert len(np.setdiff1d(all_tokens, top_idx)) > 0

    # Tail mass check: target-only recovery deflates the tail frequency by
    # a factor (1 - E_q[p]) ~ 0.84, i.e. from 0.22 to ~0.186 — over 14 sigma
    # at ~38k samples. The residual sampler is exact; noise is ~0.002.
    tail_mass = 1.0 - target_probs[top_idx].sum()
    tail_freq = 1.0 - counts[top_idx].sum() / total
    assert abs(tail_freq - tail_mass) < 0.015, (
        f"committed-token tail mass {tail_freq:.4f} deviates from target "
        f"{tail_mass:.4f}: recovered tokens are not distribution-preserving"
    )

    # Chi-square over 6 buckets (each top-5 token + aggregated tail).
    # dof = 5; the 0.999 critical value is ~20.5, threshold 30 adds flake
    # margin. Target-only recovery lands around ~270.
    observed = np.append(counts[top_idx], total - counts[top_idx].sum())
    expected = np.append(target_probs[top_idx], tail_mass) * total
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    assert chi2 < 30.0, (
        f"chi-square {chi2:.1f} over top-5 + tail buckets exceeds 30: "
        f"observed freq {observed / total}, expected {expected / total}"
    )


def test_argmax_vs_sampled_committed_distribution_match(
    session: InferenceSession,
    acceptance_sampler: Model,
    sampled_acceptance_sampler: Model,
) -> None:
    """``draft_proposal="argmax"`` and ``"sampled"`` must commit the same
    distribution.

    For any full-support draft distribution, rejection sampling commits the
    target distribution regardless of the draft's own distribution -- so the
    two modes can only differ in accept/reject rate, never in what ends up
    committed. Each mode is also checked against the analytic target
    directly, so a bug shared by both verdict functions can't hide behind
    the two modes merely agreeing with each other.

    Both modes' ``recovered`` output is already the committed token at
    every position, independent of what happened at earlier positions in
    the same row (each position samples from a fused kernel with a distinct
    per-row/per-position seed) -- so the whole ``[batch, num_steps]`` array
    is used directly, unlike the reconstruction above that stops at the
    first rejection to mirror an actual decode step.
    """
    device = session.devices[0]
    rng = np.random.default_rng(2)

    target_probs, top_idx = _make_target_probs(rng)
    logits_row = np.log(target_probs).astype(np.float32)
    logits_np = np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)
    draft_probs_row = _noisy_draft_distribution(rng, logits_row)

    temperature = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(
        device
    )
    top_k = Buffer.from_numpy(np.full(BATCH_SIZE, -1, dtype=np.int64)).to(
        device
    )
    max_k = Buffer.from_numpy(np.array(-1, dtype=np.int64))
    top_p = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(device)
    min_top_p = Buffer.from_numpy(np.array(1.0, dtype=np.float32))

    committed_argmax: list[npt.NDArray[np.int64]] = []
    committed_sampled: list[npt.NDArray[np.int64]] = []
    for _ in range(NUM_TRIALS):
        # Both modes are driven from the same draft draws: sharing the draw
        # removes an unnecessary asymmetry between the two setups rather
        # than corrupting either one.
        draft_np = rng.choice(
            VOCAB_SIZE, size=(BATCH_SIZE, NUM_STEPS), p=draft_probs_row
        ).astype(np.int64)
        draft_tokens = Buffer.from_dlpack(draft_np).to(device)

        argmax_seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        _, recovered_argmax, _ = acceptance_sampler(
            draft_tokens,
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([argmax_seed], dtype=np.uint64)).to(
                device
            ),
        )
        assert isinstance(recovered_argmax, Buffer)
        committed_argmax.append(recovered_argmax.to_numpy().reshape(-1))

        draft_probs_full_np = np.tile(
            draft_probs_row.astype(np.float32), (BATCH_SIZE, NUM_STEPS, 1)
        )
        sampled_seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        _, recovered_sampled, _ = sampled_acceptance_sampler(
            draft_tokens,
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([sampled_seed], dtype=np.uint64)).to(
                device
            ),
            Buffer.from_dlpack(draft_probs_full_np).to(device),
        )
        assert isinstance(recovered_sampled, Buffer)
        committed_sampled.append(recovered_sampled.to_numpy().reshape(-1))

    all_argmax = np.concatenate(committed_argmax)
    all_sampled = np.concatenate(committed_sampled)
    counts_argmax = np.bincount(all_argmax, minlength=VOCAB_SIZE).astype(
        np.float64
    )
    counts_sampled = np.bincount(all_sampled, minlength=VOCAB_SIZE).astype(
        np.float64
    )
    total_argmax = counts_argmax.sum()
    total_sampled = counts_sampled.sum()

    tail_mass = 1.0 - target_probs[top_idx].sum()
    tail_freq_argmax = 1.0 - counts_argmax[top_idx].sum() / total_argmax
    tail_freq_sampled = 1.0 - counts_sampled[top_idx].sum() / total_sampled
    assert abs(tail_freq_argmax - tail_mass) < 0.015, (
        f"argmax committed-token tail mass {tail_freq_argmax:.4f} deviates "
        f"from target {tail_mass:.4f}"
    )
    assert abs(tail_freq_sampled - tail_mass) < 0.015, (
        f"sampled committed-token tail mass {tail_freq_sampled:.4f} "
        f"deviates from target {tail_mass:.4f}"
    )

    # Each mode vs. the analytic target independently, before diffing them
    # against each other: a bug shared by both verdict functions would make
    # the two modes agree while both are wrong, which a pure cross-mode diff
    # can't see.
    observed_argmax = np.append(
        counts_argmax[top_idx], total_argmax - counts_argmax[top_idx].sum()
    )
    expected_argmax = np.append(target_probs[top_idx], tail_mass) * total_argmax
    chi2_argmax = float(
        np.sum((observed_argmax - expected_argmax) ** 2 / expected_argmax)
    )
    assert chi2_argmax < 30.0, (
        f"argmax chi-square {chi2_argmax:.1f} over top-5 + tail buckets "
        f"exceeds 30: observed freq {observed_argmax / total_argmax}, "
        f"expected {expected_argmax / total_argmax}"
    )

    observed_sampled = np.append(
        counts_sampled[top_idx], total_sampled - counts_sampled[top_idx].sum()
    )
    expected_sampled = (
        np.append(target_probs[top_idx], tail_mass) * total_sampled
    )
    chi2_sampled = float(
        np.sum((observed_sampled - expected_sampled) ** 2 / expected_sampled)
    )
    assert chi2_sampled < 30.0, (
        f"sampled chi-square {chi2_sampled:.1f} over top-5 + tail buckets "
        f"exceeds 30: observed freq {observed_sampled / total_sampled}, "
        f"expected {expected_sampled / total_sampled}"
    )

    # The actual ask: diff the two modes' committed marginals directly via a
    # two-sample chi-square-of-homogeneity over the same buckets.
    pooled = observed_argmax + observed_sampled
    grand_total = total_argmax + total_sampled
    expected_a = pooled * total_argmax / grand_total
    expected_b = pooled * total_sampled / grand_total
    chi2_cross = float(
        np.sum((observed_argmax - expected_a) ** 2 / expected_a)
        + np.sum((observed_sampled - expected_b) ** 2 / expected_b)
    )
    assert chi2_cross < 30.0, (
        f"argmax vs sampled cross-mode chi-square {chi2_cross:.1f} over "
        f"top-5 + tail buckets exceeds 30: argmax freq "
        f"{observed_argmax / total_argmax}, sampled freq "
        f"{observed_sampled / total_sampled}"
    )
    assert abs(tail_freq_argmax - tail_freq_sampled) < 0.02, (
        f"argmax tail freq {tail_freq_argmax:.4f} vs sampled tail freq "
        f"{tail_freq_sampled:.4f} diverge by more than 0.02"
    )


def _narrow_probabilities() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """A sharp-head target and a proposal that overlaps only its head.

    The proposal puts all of its mass on the target's most likely token and on
    a token the target barely ranks, so the residual ``max(p - q, 0)`` is the
    target with its head removed: a sampler that skips the subtract leaves
    token 0 in, and one that skips the clamp hits a negative lobe there.
    """
    target = np.zeros(NARROW_VOCAB_SIZE, dtype=np.float64)
    target[0] = 0.30
    target[1] = 0.20
    target[2] = 0.12
    target[3] = 0.08
    tail = np.arange(4, NARROW_VOCAB_SIZE - 1, dtype=np.float64)
    tail = tail / tail.sum()
    target[4 : NARROW_VOCAB_SIZE - 1] = tail * (
        1.0 - target[:4].sum()
    )  # token 63 keeps exactly zero mass
    proposal = np.zeros(NARROW_VOCAB_SIZE, dtype=np.float64)
    proposal[0] = 0.45
    proposal[40] = 0.55
    return target, proposal


def _narrow_draft_tokens() -> npt.NDArray[np.int64]:
    """The drafted token at each (request, position).

    Two positions take the sampler's one-hot branch, where it subtracts the
    drafted token instead of a proposal row: (0, 0) carries no proposal at
    all, and (1, 1) drafts a token the proposal gives no mass, so
    ``has_dist`` is false there too. Both draft token 63, which has no target
    mass, so the accept test rejects them with certainty and they never
    commit their draft -- which is what lets the one-hot behaviour be
    asserted exactly rather than statistically.
    """
    draft = np.full((NARROW_BATCH_SIZE, NARROW_NUM_STEPS), 40, dtype=np.int64)
    draft[0, 0] = 63
    draft[1, 1] = 63
    return draft


def _narrow_inputs(
    device: Device,
) -> tuple[
    Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer
]:
    """Deterministic inputs for the narrow graph.

    Row 0 of the draft distribution is all zeros, which is how a scheduler
    says it had no distribution for that row: the sampler then subtracts a
    one-hot at the drafted token instead of the proposal.
    """
    target, proposal = _narrow_probabilities()
    logits_row = np.log(
        target, where=target > 0, out=np.full_like(target, -1e9)
    )
    logits_np = np.tile(
        logits_row.astype(np.float32),
        (NARROW_BATCH_SIZE * (NARROW_NUM_STEPS + 1), 1),
    )
    draft_np = _narrow_draft_tokens()
    draft_probs_full = np.tile(
        proposal.astype(np.float32),
        (NARROW_BATCH_SIZE, NARROW_NUM_STEPS, 1),
    )
    draft_probs_full[0, 0, :] = 0.0

    def _device(array: npt.NDArray[np.generic]) -> Buffer:
        return Buffer.from_dlpack(np.ascontiguousarray(array)).to(device)

    return (
        _device(draft_np),
        _device(logits_np),
        _device(np.ones(NARROW_BATCH_SIZE, dtype=np.float32)),
        _device(np.full(NARROW_BATCH_SIZE, -1, dtype=np.int64)),
        Buffer.from_numpy(np.array(-1, dtype=np.int64)),
        _device(np.ones(NARROW_BATCH_SIZE, dtype=np.float32)),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        _device(np.array([NARROW_SEED], dtype=np.uint64)),
        _device(draft_probs_full),
    )


def test_residual_op_matches_materialized_reference(
    session: InferenceSession,
) -> None:
    """The fused residual draw is the materialized one, token for token.

    Both graphs draw with the same per-row seed, over the same target and
    proposal rows, and the rows taken by the no-distribution branch (a
    non-negative `fallback` id) sit in the same launch as the rest. Nothing
    but the arithmetic and the order of the noise can separate them, so this
    is the check that the noise still starts where it started, and that a
    fallback row still loses exactly its drafted token.
    """
    fused = _compile_recovery_graph(
        session, fuse=True, vocab_size=NARROW_VOCAB_SIZE
    )
    materialized = _compile_recovery_graph(
        session, fuse=False, vocab_size=NARROW_VOCAB_SIZE
    )
    device = session.devices[0]
    target, proposal = _narrow_probabilities()

    rows = 512
    proposal_np = np.tile(proposal.astype(np.float32), (rows, 1)).reshape(
        rows, NARROW_VOCAB_SIZE
    )
    fallback_np = np.full(rows, -1, dtype=np.int64)
    # Every third row carries no proposal distribution, which is the one-hot
    # branch; its id points at a token with residual mass so the branch is
    # observable in the draw.
    fallback_np[::3] = 40
    seeds_np = np.arange(rows, dtype=np.uint64) * np.uint64(7919) + np.uint64(
        NARROW_SEED
    )

    probs_buf = Buffer.from_dlpack(
        np.ascontiguousarray(np.tile(target.astype(np.float32), (rows, 1)))
    ).to(device)
    proposal_buf = Buffer.from_dlpack(np.ascontiguousarray(proposal_np)).to(
        device
    )
    fallback_buf = Buffer.from_dlpack(np.ascontiguousarray(fallback_np)).to(
        device
    )
    seed_buf = Buffer.from_dlpack(np.ascontiguousarray(seeds_np)).to(device)

    fused_tokens = fused(probs_buf, proposal_buf, fallback_buf, seed_buf)[0]
    reference_tokens = materialized(
        probs_buf, proposal_buf, fallback_buf, seed_buf
    )[0]
    assert isinstance(fused_tokens, Buffer)
    assert isinstance(reference_tokens, Buffer)

    observed = fused_tokens.to_numpy().reshape(-1)
    expected = reference_tokens.to_numpy().reshape(-1)
    mismatched = np.flatnonzero(observed != expected)
    assert mismatched.size == 0, (
        f"{mismatched.size} of {rows} rows drew differently: "
        f"fused {observed[mismatched[:8]]} != reference "
        f"{expected[mismatched[:8]]} at rows {mismatched[:8]}"
    )


def test_narrow_recovered_tokens_avoid_the_subtracted_head(
    session: InferenceSession, narrow_sampled_acceptance_sampler: Model
) -> None:
    """Recovered tokens follow ``max(p - q, 0)``, not ``p``.

    The proposal zeroes the target's head, so the residual has no mass at
    token 0 -- and none at token 63, which has no target mass to begin with.
    The draft positions whose proposal gives the drafted token no mass take
    the sampler's one-hot branch instead, and their residual is ``p`` with
    the drafted token removed; those positions keep token 0's mass, so they
    are counted separately and checked against their own rule rather than
    folded into this histogram.
    """
    device = session.devices[0]
    inputs = list(_narrow_inputs(device))
    draft = _narrow_draft_tokens()
    target, proposal = _narrow_probabilities()
    # The sampler's own `has_dist`: a position whose proposal gives the
    # drafted token no mass subtracts a one-hot instead of the proposal.
    onehot_mask = proposal[draft] == 0.0
    onehot_positions = int(onehot_mask.sum())
    assert onehot_positions == 2, (
        f"the fixture must exercise the one-hot branch on two positions, got"
        f" {onehot_positions}"
    )

    counts = np.zeros(NARROW_VOCAB_SIZE, dtype=np.int64)
    onehot_draws = 0
    onehot_kept_draft = 0
    for trial in range(64):
        inputs[7] = Buffer.from_dlpack(
            np.ascontiguousarray(
                np.array([NARROW_SEED + trial * 7919], dtype=np.uint64)
            )
        ).to(device)
        _, recovered, _ = narrow_sampled_acceptance_sampler(*inputs)
        assert isinstance(recovered, Buffer)
        tokens = recovered.to_numpy().reshape(
            NARROW_BATCH_SIZE, NARROW_NUM_STEPS
        )
        for b in range(NARROW_BATCH_SIZE):
            for step in range(NARROW_NUM_STEPS):
                token = int(tokens[b, step])
                if onehot_mask[b, step]:
                    onehot_draws += 1
                    onehot_kept_draft += int(token == int(draft[b, step]))
                else:
                    counts[token] += 1

    # A one-hot row removes exactly the drafted token, and that token has no
    # target mass, so the row can never commit it: rejected with certainty,
    # and the residual it draws from has zero mass there.
    assert onehot_draws == 64 * onehot_positions
    assert onehot_kept_draft == 0, (
        f"a one-hot row committed its own drafted token"
        f" {onehot_kept_draft} times: the one-hot was not subtracted"
    )

    total = int(counts.sum())
    assert total == 64 * (
        NARROW_BATCH_SIZE * NARROW_NUM_STEPS - onehot_positions
    )
    assert counts[0] == 0, (
        f"token 0 kept {counts[0]} draws of {total}: the proposal's 0.45 mass"
        " there was not subtracted"
    )
    assert counts[NARROW_VOCAB_SIZE - 1] == 0, (
        f"token {NARROW_VOCAB_SIZE - 1} has no residual mass but drew"
        f" {counts[NARROW_VOCAB_SIZE - 1]} times"
    )

    residual = np.maximum(target - proposal, 0.0)
    expected = residual / residual.sum()
    observed = counts[1:4].astype(np.float64) / total
    wanted = expected[1:4]
    # Five binomial standard deviations of the widest bucket, plus slack for
    # the handful of accepted non-zero draft tokens.
    tolerance = 5.0 * np.sqrt(wanted[0] * (1.0 - wanted[0]) / total) + 0.01
    assert np.all(np.abs(observed - wanted) < tolerance), (
        f"head frequencies {observed} deviate from residual {wanted}"
    )
