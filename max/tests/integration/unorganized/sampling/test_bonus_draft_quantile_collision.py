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
"""What does a shared key between the bonus and a draft proposal actually cost?

Before the bonus draw carried a domain tag it ran off the bare per-execute
seed, which is the key a sampled draft proposal's step 0 draws with. Both
lower to an inverse-CDF walk over the vocabulary in index order, so one
uniform drove both walks and the two tokens agreed far more often than two
independent draws would -- and on an all-accepted step both are committed.

The cost is not a constant: it depends on how alike the two distributions
are, and it is bounded by neither. This measures the agreement rate directly,
against a baseline that is *measured rather than assumed* -- the same graph
with the tag in place, whose keys are independent by construction -- across a
sweep from identical to unrelated distributions.

Each row carries its own seed and its own pair of distributions, so a batch is
a batch of independent trials. ``top_p`` sits just under 1 to hold the draw on
the dual-pivot route production takes, whose RNG counter is the row index, and
not the Gumbel route a fully untruncated draw would select.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import topk_fused_sampling_with_dist
from max.nn.sampling import rejection_sampler, stochastic_acceptance_sampler
from max.nn.sampling.rejection_sampler import _draft_step_seed

VOCAB_SIZE = 512
ROWS = 512
EXECUTES = 4

# Mixing weight toward an unrelated distribution: 0.0 is identical draft and
# target, 1.0 is unrelated. Spans the range because the cost is a function of
# the separation; measured M3 traces sit at the unrelated end.
BLENDS = (0.0, 0.1, 0.25, 0.5, 1.0)

# Just under 1: keeps `min_top_p != 1.0` so the fused sampler takes the
# dual-pivot route, while truncating a negligible tail.
_TOP_P = 0.999


def _build(session: InferenceSession) -> Model:
    """Draft proposal and verdict in one graph, wired as the arch files wire them."""
    d = DeviceRef.from_device(session.devices[0])
    input_types = [
        TensorType(DType.float32, ["batch_size", VOCAB_SIZE], device=d),
        TensorType(DType.float32, ["total_output_len", VOCAB_SIZE], device=d),
        TensorType(DType.uint64, ["batch_size"], device=d),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.int64, ["batch_size"], device=d),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
    ]
    with Graph("bonus_draft_collision", input_types=input_types) as graph:
        dl, tl, seed, temp, tk, mk, tp, mtp = (i.tensor for i in graph.inputs)
        # Draft step 0, exactly as `unified_mtp_minimax_m3.py` issues it.
        step_tokens, step_dist = topk_fused_sampling_with_dist(
            dl.rebind(["batch_size", VOCAB_SIZE]),
            top_k=tk,
            temperature=temp,
            top_p=tp,
            seed=_draft_step_seed(seed, 0),
        )
        # The verdict binds ``num_steps`` symbolically off the target logits,
        # so the single draft position has to be rebound onto that dim rather
        # than left as a static 1, or the two disagree inside the sampler.
        draft_tokens = ops.rebind(
            ops.unsqueeze(
                ops.rebind(step_tokens.reshape([-1]), ["batch_size"]), axis=-1
            ),
            ["batch_size", "num_steps"],
        )
        _, _, bonus = stochastic_acceptance_sampler(
            draft_tokens=draft_tokens,
            target_logits=tl,
            temperature=temp,
            top_k=tk,
            max_k=mk,
            top_p=tp,
            min_top_p=mtp,
            seed=seed,
            draft_proposal="sampled",
            draft_probs_full=ops.rebind(
                ops.unsqueeze(
                    ops.rebind(step_dist, ["batch_size", VOCAB_SIZE]), axis=1
                ),
                ["batch_size", "num_steps", VOCAB_SIZE],
            ),
            vocab_size=VOCAB_SIZE,
        )
        graph.output(draft_tokens, bonus)
    return session.load(graph)


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def tagged(session: InferenceSession) -> Model:
    """The shipping derivation: the bonus draw carries its own domain."""
    return _build(session)


@pytest.fixture(scope="module")
def untagged(
    session: InferenceSession, monkeypatch_module: pytest.MonkeyPatch
) -> Model:
    """The pre-tag derivation: the bonus draw rides the bare per-execute seed."""
    monkeypatch_module.setattr(
        rejection_sampler,
        "_bonus_seed_rows",
        lambda seed, batch_size, device: seed,
    )
    return _build(session)


def _pairs(
    blend: float, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]:
    """Per-row draft and target distributions, blended toward independence."""

    def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    p = _softmax(rng.normal(0, 2.0, (ROWS, VOCAB_SIZE)))
    other = _softmax(rng.normal(0, 2.0, (ROWS, VOCAB_SIZE)))
    q = (1.0 - blend) * p + blend * other
    tvd = float((0.5 * np.abs(p - q).sum(axis=-1)).mean())
    return np.log(q).astype(np.float32), np.log(p).astype(np.float32), tvd


def _agreement(
    model: Model, session: InferenceSession, blend: float, seed0: int
) -> float:
    """Fraction of rows whose bonus token equals their own step-0 draft token."""
    device = session.devices[0]
    rng = np.random.default_rng(seed0)
    agree = total = 0
    for ex in range(EXECUTES):
        draft_logits, p_logits, _ = _pairs(blend, rng)
        # [batch * (num_steps + 1), vocab]: slot 0 verifies, slot 1 is the bonus.
        target = np.empty((ROWS * 2, VOCAB_SIZE), dtype=np.float32)
        target[0::2] = rng.normal(0, 2.0, (ROWS, VOCAB_SIZE)).astype(np.float32)
        target[1::2] = p_logits
        seeds = np.arange(ROWS, dtype=np.uint64) + np.uint64(seed0 + ex * ROWS)

        dt, bonus = model(
            Buffer.from_dlpack(draft_logits).to(device),
            Buffer.from_dlpack(target).to(device),
            Buffer.from_numpy(seeds).to(device),
            Buffer.from_numpy(np.ones(ROWS, np.float32)).to(device),
            Buffer.from_numpy(np.full(ROWS, -1, np.int64)).to(device),
            Buffer.from_numpy(np.array(-1, np.int64)),
            Buffer.from_numpy(np.full(ROWS, _TOP_P, np.float32)).to(device),
            Buffer.from_numpy(np.array(_TOP_P, np.float32)),
        )
        assert isinstance(dt, Buffer) and isinstance(bonus, Buffer)
        agree += int(
            (dt.to_numpy().reshape(-1) == bonus.to_numpy().reshape(-1)).sum()
        )
        total += ROWS
    return agree / total


@pytest.mark.parametrize("blend", BLENDS)
def test_tag_removes_the_shared_quantile(
    session: InferenceSession,
    tagged: Model,
    untagged: Model,
    blend: float,
) -> None:
    """The tagged draw agrees at chance; the untagged one far above it.

    The tagged rate is the baseline: same graph, same distributions, keys
    separated only by the domain. Anything the untagged rate carries above it
    is the shared quantile. The coupling is real in the code, but what it
    costs depends on how alike the two distributions are, and the bonus slot
    and the draft's step 0 sit one position apart rather than on the same
    one. On measured MiniMax-M3 traces the two are near-disjoint (median TVD
    1.00 over 699 samples), and the realized agreement there is nil.
    """
    base = _agreement(tagged, session, blend, seed0=1_000_003)
    collided = _agreement(untagged, session, blend, seed0=1_000_003)
    n = ROWS * EXECUTES
    # 4 sigma on the tagged rate, so "above chance" is not noise.
    band = 4.0 * float(np.sqrt(max(base, 1e-9) * (1 - base) / n))

    print(
        f"\n[blend={blend:.2f}] tagged={base:.4f} untagged={collided:.4f} "
        f"excess={collided - base:+.4f} "
        f"ratio={collided / max(base, 1e-9):.1f}x band={band:.4f}"
    )

    if blend == 1.0:
        # Unrelated distributions share no quantile worth speaking of; this is
        # the negative control that keeps the sweep from reading as an artifact
        # of the measurement rather than of the distributions.
        assert collided - base < 0.05, (
            f"unrelated distributions still agreed {collided:.4f} vs a "
            f"{base:.4f} baseline -- the excess is not coming from the "
            "distributions overlapping"
        )
    else:
        assert collided > base + band, (
            f"blend={blend}: untagged agreement {collided:.4f} is not above "
            f"the tagged baseline {base:.4f} (4-sigma band {band:.4f}); the "
            "shared-key coupling did not reproduce"
        )
