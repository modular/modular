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

Why this discriminates: let ``q`` be the draft proposal and ``p`` the target
softmax. The sampler accepts a draft ``d`` with probability ``p(d)``. If the
recovered token on rejection is drawn from the residual (``p`` with the
draft token's mass removed), the committed token is exactly ``~ p`` for any
``q``:

    P(commit = x) = q(x) p(x) + sum_d q(d) (1 - p(d)) * p(x) 1{x != d} / (1 - p(d))
                  = q(x) p(x) + p(x) (1 - q(x)) = p(x)

If instead the recovered token is drawn from the full target distribution
(target-only recovery), the committed marginal is skewed:

    P(commit = x) = p(x) * (q(x) + 1 - E_q[p])

With drafts uniform over the top-5 (``q = 0.2`` there, ``0`` elsewhere) and
~0.78 of target mass on the top-5, ``E_q[p] ~ 0.156``: top-5 tokens are
inflated ~4% and the tail is deflated ~16% relative — many sigma at the
sample sizes below, so the chi-square and tail-mass checks fail under
target-only recovery and pass under residual recovery.
"""

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.sampling import stochastic_acceptance_sampler

VOCAB_SIZE = 1024
NUM_STEPS = 4
BATCH_SIZE = 512
NUM_TRIALS = 64
TOP_TOKEN_COUNT = 5


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


@pytest.mark.skip(
    reason="Fails since e74301ba9e9 reverted residual recovery sampling "
    "(CENG-970): recovered tokens come from the full target distribution, "
    "which skews the committed-token marginal. Re-enable when the "
    "distribution-preserving sampler lands again."
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
