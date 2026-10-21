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
"""Guards two claims about ``stochastic_acceptance_sampler``'s implicit RNG,
for ``draft_proposal="sampled"``.

1. FIXED here. The verdict's accept coin (``set_seed`` + ``uniform``) lowers
   to the same zero-counter ``std.random.philox.Random`` as the draft's own
   token draw (``topk_topp_sampling_from_prob``'s ``Random(seed=seed_val)``),
   so seeding it with the bare per-execute seed handed batch row 0 the same
   float for both. ``test_verdict_coin_equals_raw_philox_draw`` pins that
   op-level identity, and
   ``test_verdict_domain_tag_is_off_the_draft_and_recovery_lattices`` pins
   the tag that keeps the surviving implicit stream off the draft and
   recovery keys.
2. FIXED here. ``set_seed`` always reads index 0 of a rank-1 seed tensor, so
   while the accept coin came off that implicit stream, every row's draw was
   keyed off row 0's seed alone and changing only row 0's seed changed every
   other row's committed stream. The coin is now an explicitly keyed per-row
   draw, so the sampler takes the whole rank-1 seed and no row's verdict
   depends on another row's seed. (A row's own batch position still reaches
   the draw, through the row term in its key and through the sampling
   kernel's per-row RNG counter; that is a separate concern and not what
   these tests claim.) That is what
   ``test_row_verdict_ignores_another_rows_seed`` pins, with
   ``test_row_verdict_follows_its_own_seed`` as the control that keeps it
   from passing vacuously.

The op-level lemma in claim 1 still holds for the implicit stream, which
other callers reach; the sampled verdict's coin simply no longer rides it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.sampling import stochastic_acceptance_sampler
from max.nn.sampling.rejection_sampler import (
    _SEED_DOMAIN_RECOVERY,
    _SEED_DOMAIN_VERDICT,
    _SEED_GOLDEN_GAMMA,
)

# --- Philox4x32-10, reimplemented from
# oss/modular/mojo/stdlib/std/random/philox.mojo's `Random` struct, so this
# comparison is against a from-scratch reading of the algorithm rather than
# a call into the same code under test. ---

BATCH_SIZE = 2
NUM_STEPS = 2
VOCAB_SIZE = 64

_K_PHILOX_SA = np.uint32(0xD2511F53)
_K_PHILOX_SB = np.uint32(0xCD9E8D57)
_K_PHILOX_10 = np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint32)


def _mulhilow(a: np.uint32, b: np.uint32) -> tuple[np.uint32, np.uint32]:
    prod = np.uint64(a) * np.uint64(b)
    return (
        np.uint32(prod & np.uint64(0xFFFFFFFF)),
        np.uint32(prod >> np.uint64(32)),
    )


def _single_round(
    counter: npt.NDArray[np.uint32], key: npt.NDArray[np.uint32]
) -> npt.NDArray[np.uint32]:
    lo1, hi1 = _mulhilow(_K_PHILOX_SB, counter[2])
    lo0, hi0 = _mulhilow(_K_PHILOX_SA, counter[0])
    return np.array(
        [hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0],
        dtype=np.uint32,
    )


def _philox_step_uniform(
    seed: int, offset: int = 0, subsequence: int = 0, rounds: int = 10
) -> npt.NDArray[np.float32]:
    """Reimplements ``Random(seed=seed, offset=offset).step_uniform()``."""
    key = np.array(
        [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF], dtype=np.uint32
    )
    counter = np.array(
        [
            offset & 0xFFFFFFFF,
            (offset >> 32) & 0xFFFFFFFF,
            subsequence & 0xFFFFFFFF,
            (subsequence >> 32) & 0xFFFFFFFF,
        ],
        dtype=np.uint32,
    )
    for _ in range(rounds - 1):
        counter = _single_round(counter, key)
        key = key + _K_PHILOX_10
    raw = _single_round(counter, key)
    scale = np.float32(4.6566127342e-10)
    return (raw & np.uint32(0x7FFFFFFF)).astype(np.float32) * scale


@pytest.fixture(scope="module")
def verdict_coin_graph(session: InferenceSession) -> Model:
    """One draw off the implicit stream, at the offset a batch-level draw uses.

    ``ops.random.set_seed(seed) ; ops.random.uniform(TensorType(..., [1]))``
    is what any batch-level implicit draw in ``rejection_sampler.py`` lowers
    to at flat position 0, since a size-1 output has row-major flat offset 0
    for its only element. The sampled verdict's accept coin used to be
    exactly this draw; it is now keyed per row instead, but the lemma still
    describes every remaining implicit consumer.
    """
    d = DeviceRef.from_device(session.devices[0])
    with Graph("verdict_coin", input_types=[ops.random.SeedType(d)]) as graph:
        (seed,) = graph.inputs
        ops.random.set_seed(seed.tensor)
        out = ops.random.uniform(TensorType(DType.float32, [1], device=d))
        graph.output(out)
    return session.load(graph)


@pytest.mark.parametrize(
    "seed_int",
    [123456789, 42, 0xDEADBEEF, 2**33 + 7, 999999999999],
)
def test_verdict_coin_equals_raw_philox_draw(
    session: InferenceSession,
    verdict_coin_graph: Model,
    seed_int: int,
) -> None:
    """``set_seed(S)`` then ``uniform`` is bit-identical to ``Random(seed=S)``.

    The op-level lemma the fix rests on: whatever seed reaches
    ``ops.random.set_seed``, the first implicit draw off it is exactly
    ``Random(seed=that_seed, offset=0)`` -- the same primitive
    ``topk_topp_sampling_from_prob`` uses for the draft's own token draw
    (``topk_fi.mojo``'s ``PHASE 3`` block, also offset/subsequence 0) -- the
    identity that made the pre-fix collision exact rather than approximate,
    and the reason every batch-level stream here carries a domain tag.
    """
    device = session.devices[0]
    seed_buf = Buffer.from_numpy(np.array([seed_int], dtype=np.uint64)).to(
        device
    )
    (coin,) = verdict_coin_graph(seed_buf)
    assert isinstance(coin, Buffer)
    graph_val = float(coin.to_numpy()[0])
    philox_val = float(_philox_step_uniform(seed_int, offset=0)[0])
    assert graph_val == philox_val, (
        f"seed={seed_int}: verdict coin {graph_val!r} != raw Philox draw "
        f"{philox_val!r} -- if this ever fails, the Philox reimplementation "
        "or the kernel's offset/subsequence defaults have drifted, not the "
        "collision claim"
    )


def test_verdict_domain_tag_is_off_the_draft_and_recovery_lattices() -> None:
    """The verdict domain tag cannot be reached by any other seed family.

    Every seed family in ``rejection_sampler.py`` walks the golden gamma off
    the same per-execute base: a sampled draft proposal's step ``s`` uses
    ``seed + s * gamma``, and residual recovery's row ``b`` uses ``seed +
    _SEED_DOMAIN_RECOVERY + b * gamma``. The verdict's single key,
    ``seed + _SEED_DOMAIN_VERDICT``, therefore separates from both families
    only if the tag is not itself a lattice point -- so this checks the
    arithmetic instead of assuming it. The bound below is far past any
    reachable step count or batch size; the exact distances are ~7.7e18 and
    ~1.3e19 gamma-steps.
    """
    modulus = 1 << 64
    reachable = 1 << 20

    assert _SEED_DOMAIN_VERDICT % 2 == 1, (
        "a domain tag must be odd, like every other seed constant here"
    )
    assert _SEED_DOMAIN_VERDICT != _SEED_DOMAIN_RECOVERY

    # The gamma is odd, so it is invertible mod 2**64: dividing a key by it
    # gives the exact index that would produce that key, for every index at
    # once rather than a sampled prefix.
    gamma_inverse = pow(_SEED_GOLDEN_GAMMA, -1, modulus)
    draft_step = (_SEED_DOMAIN_VERDICT * gamma_inverse) % modulus
    recovery_row = (
        (_SEED_DOMAIN_VERDICT - _SEED_DOMAIN_RECOVERY) * gamma_inverse
    ) % modulus

    assert draft_step > reachable, (
        f"the verdict key is a draft proposal's key at step {draft_step}"
    )
    assert recovery_row > reachable, (
        f"the verdict key is a recovery row's key at row {recovery_row}"
    )


def _build_sampled_verdict(session: InferenceSession) -> Model:
    """Mirrors ``eagle3_unified.py``'s stochastic/sampled acceptance call."""
    d = DeviceRef.from_device(session.devices[0])
    # Named (dynamic) dims, matching test_sampled_draft_q_calibration.py's
    # `sampled_verdict` fixture: `_reshape_target_logits` rebinds against
    # ``Dim("batch_size")`` / ``Dim("num_steps")`` symbolically, so a graph
    # input with a concrete static shape does not unify with it.
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=d),
        TensorType(DType.float32, ["total_output_len", "vocab_size"], device=d),
        TensorType(
            DType.float32,
            ["batch_size", "num_steps", "vocab_size"],
            device=d,
        ),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.int64, ["batch_size"], device=d),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        # The per-row [batch_size] seed tensor `overlap_text_generation.py`
        # builds, passed through whole, as `eagle3_unified.py` now does.
        TensorType(DType.uint64, ["batch_size"], device=d),
    ]
    with Graph("sampled_verdict_slot0", input_types=graph_inputs) as graph:
        (dt, tl, dpf, temp, tk, mk, tp, mtp, seed) = graph.inputs
        graph.output(
            *stochastic_acceptance_sampler(
                draft_tokens=dt.tensor,
                target_logits=tl.tensor,
                temperature=temp.tensor,
                top_k=tk.tensor,
                max_k=mk.tensor,
                top_p=tp.tensor,
                min_top_p=mtp.tensor,
                seed=seed.tensor,
                draft_proposal="sampled",
                draft_probs_full=dpf.tensor,
                vocab_size=VOCAB_SIZE,
            )
        )
    return session.load(graph)


@pytest.fixture(scope="module")
def sampled_verdict(session: InferenceSession) -> Model:
    return _build_sampled_verdict(session)


# Row 1's accept decision is one bit per execute, so a handful of executions
# is enough to show it never moves with row 0's seed, and enough to show it
# does move with its own.
_PROBE_SEEDS = tuple(range(1, 49))


def _row_accept_bits(
    model: Model, session: InferenceSession, seeds: npt.NDArray[np.uint64]
) -> list[bool]:
    """Returns whether each row accepted its single draft token.

    ``q = 1`` (a one-hot draft distribution over the drafted token) and
    ``p_target = 0.5`` (target logits putting half the mass on the drafted
    token and half on one other, everything else ~1e-22 so no top-k/top-p
    rule can shift the split) make each row's verdict a fair coin. With one
    draft step, ``first_rejected_idx`` is 0 when the row rejected and 1 --
    ``num_steps``, the "all accepted" sentinel -- when it accepted, so each
    row yields exactly one bit of its own coin.
    """
    device = session.devices[0]
    batch = len(seeds)
    logits_row = np.full(VOCAB_SIZE, -50.0, dtype=np.float32)
    logits_row[0] = 0.0
    logits_row[1] = 0.0
    # [batch * (num_steps + 1), vocab]
    logits_np = np.tile(logits_row, (batch * 2, 1))
    draft_tokens_np = np.zeros((batch, 1), dtype=np.int64)
    draft_probs_np = np.zeros((batch, 1, VOCAB_SIZE), dtype=np.float32)
    draft_probs_np[:, 0, 0] = 1.0

    fri, _, _ = model(
        Buffer.from_dlpack(draft_tokens_np).to(device),
        Buffer.from_dlpack(logits_np).to(device),
        Buffer.from_dlpack(draft_probs_np).to(device),
        Buffer.from_numpy(np.ones(batch, np.float32)).to(device),
        Buffer.from_numpy(np.full(batch, -1, np.int64)).to(device),
        Buffer.from_numpy(np.array(-1, np.int64)),
        Buffer.from_numpy(np.ones(batch, np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, np.float32)),
        Buffer.from_numpy(seeds).to(device),
    )
    assert isinstance(fri, Buffer)
    return [bool(v == 1) for v in fri.to_numpy()]


def test_row_verdict_ignores_another_rows_seed(
    session: InferenceSession, sampled_verdict: Model
) -> None:
    """Row 1's verdict does not move when only row 0's seed changes.

    The form the leak took: one request's generated-token count drove every
    co-resident request's accept decisions, so a request's output depended on
    who it shared a batch with. Row 1's seed is held fixed while row 0's
    sweeps, and row 1's bit must not budge.
    """
    row1_seed = np.uint64(0x5EED_1)
    bits = {
        int(probe): _row_accept_bits(
            sampled_verdict,
            session,
            np.array([probe, row1_seed], dtype=np.uint64),
        )[1]
        for probe in _PROBE_SEEDS
    }

    distinct = set(bits.values())
    assert len(distinct) == 1, (
        "row 1's verdict changed with row 0's seed alone, at probes "
        f"{sorted(k for k, v in bits.items() if v != bits[_PROBE_SEEDS[0]])}"
        " -- the verdict is keyed off another row's seed again"
    )


def test_row_verdict_follows_its_own_seed(
    session: InferenceSession, sampled_verdict: Model
) -> None:
    """Row 1's verdict does move when its own seed changes.

    Without this, :func:`test_row_verdict_ignores_another_rows_seed` would
    pass just as well against a verdict that ignored every seed, or one whose
    coin was pinned to a constant.
    """
    row0_seed = np.uint64(0x5EED_0)
    bits = [
        _row_accept_bits(
            sampled_verdict,
            session,
            np.array([row0_seed, probe], dtype=np.uint64),
        )[1]
        for probe in _PROBE_SEEDS
    ]

    assert len(set(bits)) == 2, (
        f"row 1 accepted on {sum(bits)} of {len(bits)} of its own seeds: its "
        "coin is not keyed on its seed at all, so the companion test proves "
        "nothing"
    )
