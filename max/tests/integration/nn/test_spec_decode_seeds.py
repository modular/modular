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
"""Seed-family separation for sampled speculative decoding.

A request's per-execute seed is its own seed plus its generated-token count
plus a fixed hash of its request id (:func:`request_row_seed`), so it advances
by however many tokens the last iteration committed -- 1 to ``K + 1`` under
speculation, rather than always 1. The id term is fixed per request, so it
shifts a request's whole lattice without changing any of the distances below.
Three invariants keep the sampling streams apart under that advance:

1. No draft step key repeats across iterations, for any commit count a
   speculative iteration can produce.
2. No residual-recovery key ever equals a draft-proposal key, in either
   adjacent-iteration ordering.
3. No two seed families meet anywhere, which the accept coin and the bonus
   token now need in their own right: both are keyed per row, so each spans a
   lattice of keys rather than the single key a batch-level draw occupied.

The offsets come from the production helpers, so a call site reverting to
consecutive integers changes what these assert.
"""

from __future__ import annotations

import itertools

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.nn.sampling.rejection_sampler import (
    _SEED_DOMAIN_BONUS,
    _SEED_DOMAIN_COIN,
    _SEED_DOMAIN_RECOVERY,
    _SEED_DOMAIN_VERDICT,
    _SEED_GOLDEN_GAMMA,
    _recovery_row_offset,
    _recovery_seed_rows,
    _seed_offset,
)

_U64 = 1 << 64

# Draft width in production recipes; the bound on how far one iteration can
# advance the base seed is num_speculative_tokens + 1 (drafts plus the bonus).
_MAX_DRAFT_STEPS = 8
_MAX_COMMITTED = _MAX_DRAFT_STEPS + 1
_MAX_BATCH = 64

_DRAFT_OFFSETS = [_seed_offset(step) for step in range(_MAX_DRAFT_STEPS)]
_RECOVERY_OFFSETS = [_recovery_row_offset(row) for row in range(_MAX_BATCH)]


def test_draft_offsets_are_distinct() -> None:
    assert len(set(_DRAFT_OFFSETS)) == len(_DRAFT_OFFSETS)


def test_step_zero_leaves_the_base_seed_alone() -> None:
    """Why ``_draft_step_seed`` may return the seed unchanged at step 0."""
    assert _seed_offset(0) == 0


def test_draft_offsets_survive_the_token_count_advance() -> None:
    """No step key repeats once the base seed advances by a commit count.

    This is the collision the spacing exists to remove: with consecutive
    integers, an iteration committing ``c`` tokens puts the next iteration's
    step ``s`` on this iteration's step ``s + c``.
    """
    draft = set(_DRAFT_OFFSETS)
    for committed in range(1, _MAX_COMMITTED + 1):
        for step, offset in enumerate(_DRAFT_OFFSETS):
            key = (committed + offset) % _U64
            assert key not in draft, (
                f"draft step {step} at commit advance {committed} reuses the "
                f"key of another step"
            )


def test_recovery_and_draft_offsets_never_meet() -> None:
    """Recovery and proposal draws must not share a key, in either ordering.

    Row ``b``'s recovery draw and draft step ``b``'s proposal draw hang off
    the same base seed. Walking the same small integers puts them on the same
    key whenever ``b == step``, which is what the recovery domain tag removes.
    Both adjacent-iteration orderings matter: a recovery key can be carried
    forward onto a later draft key, and a draft key onto a later recovery key.
    """
    draft = set(_DRAFT_OFFSETS)
    recovery = set(_RECOVERY_OFFSETS)
    assert not (draft & recovery)

    for committed in range(1, _MAX_COMMITTED + 1):
        for row, offset in enumerate(_RECOVERY_OFFSETS):
            assert (committed + offset) % _U64 not in draft, (
                f"recovery row {row} at commit advance {committed} shares a "
                f"key with a draft step"
            )
        for step, offset in enumerate(_DRAFT_OFFSETS):
            assert (committed + offset) % _U64 not in recovery, (
                f"draft step {step} at commit advance {committed} shares a "
                f"key with a recovery row"
            )


def test_recovery_offsets_are_distinct_per_row() -> None:
    assert len(set(_RECOVERY_OFFSETS)) == len(_RECOVERY_OFFSETS)


def test_gamma_is_odd() -> None:
    """Odd multipliers are bijective mod 2**64, so distinct steps stay distinct."""
    assert _SEED_GOLDEN_GAMMA % 2 == 1


# One request per row, three draft positions each: enough rows to see spacing
# and enough positions to see a key shared across a request's positions.
_RECOVERY_BATCH = 4
_RECOVERY_STEPS = 3


def _recovery_rows(session: InferenceSession, base: np.ndarray) -> np.ndarray:
    """Runs ``_recovery_seed_rows`` over a seed shaped like ``base``.

    Static dims throughout: a shared seed carries no batch dim of its own, so
    a symbolic one would have nothing in the graph to bind it.
    """
    device = DeviceRef.CPU()
    with Graph(
        f"recovery_seed_rows_{base.size}",
        input_types=[TensorType(DType.uint64, [base.size], device=device)],
    ) as graph:
        seed = graph.inputs[0].tensor
        graph.output(
            _recovery_seed_rows(
                seed, Dim(_RECOVERY_BATCH), Dim(_RECOVERY_STEPS), device
            )
        )
    return session.load(graph)(Buffer.from_numpy(base))[0].to_numpy()


def test_recovery_rows_of_a_per_row_seed_carry_no_row_term(
    session: InferenceSession,
) -> None:
    """A per-row seed's recovery key is the tag alone -- no batch position.

    The row index is a physical batch slot, so spending it here would make a
    request's recovered token depend on where it landed in the batch. A
    per-row seed is already distinct per request, so the tag is all the
    separation the family needs.
    """
    base = np.arange(_RECOVERY_BATCH, dtype=np.uint64) * np.uint64(1_000_000)
    rows = _recovery_rows(session, base)

    expected = np.array(
        [
            (int(base[row]) + _recovery_row_offset(0)) % _U64
            for row in range(_RECOVERY_BATCH)
            for _ in range(_RECOVERY_STEPS)
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(rows, expected)


def test_recovery_rows_of_a_shared_seed_still_spend_the_row_offsets(
    session: InferenceSession,
) -> None:
    """A shared seed must keep its row spacing, or every row recovers alike.

    The graph-level ``SeedType`` input is one key for the whole batch, with no
    per-request term to tell the rows apart. The arithmetic above pins the
    offsets; this pins that the path still spends them.
    """
    base = np.array([12345], dtype=np.uint64)
    rows = _recovery_rows(session, base)

    expected = np.array(
        [
            (int(base[0]) + _recovery_row_offset(row)) % _U64
            for row in range(_RECOVERY_BATCH)
            for _ in range(_RECOVERY_STEPS)
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(rows, expected)


# Every family walks the golden gamma off the same per-execute base, so a
# family is a lattice ``domain + i * gamma`` and the draft proposal's is the
# untagged one at domain 0.
_DOMAIN_LATTICES = {
    "draft proposal": 0,
    "residual recovery": _SEED_DOMAIN_RECOVERY,
    "verdict stream": _SEED_DOMAIN_VERDICT,
    "bonus token": _SEED_DOMAIN_BONUS,
    "accept coin": _SEED_DOMAIN_COIN,
}

# Far past any index a family can reach: draft steps are bounded by K, and
# recovery, bonus and coin rows by the batch times at most K positions.
_REACHABLE = 1 << 20


def _lattice_gap(delta: int) -> int:
    """Returns the gamma-step distance from one lattice to another.

    The gamma is odd, hence invertible mod 2**64, so dividing an offset by it
    gives the exact index that would produce it -- every index at once, rather
    than a sampled prefix. Taken as a signed distance, so a lattice sitting
    just *below* another is not read as being ~2**64 steps away.
    """
    index = (delta * pow(_SEED_GOLDEN_GAMMA, -1, _U64)) % _U64
    return min(index, _U64 - index)


def test_domain_tags_are_distinct() -> None:
    """Two families sharing a tag would share every key they ever draw.

    Distinctness is the requirement; oddness deliberately is not. The gamma is
    odd, so ``i * gamma`` alternates parity and every family's lattice already
    covers both -- an even tag separates exactly as well as an odd one, and
    :func:`test_no_two_seed_families_meet_under_any_commit_advance` is what
    establishes that they never meet. (The *gamma* must still be odd, to stay
    invertible mod 2**64; :func:`test_gamma_is_odd` covers that.)
    """
    tags = [d for d in _DOMAIN_LATTICES.values() if d != 0]
    assert len(set(tags)) == len(tags), "two families share a domain tag"


def test_no_two_seed_families_meet_under_any_commit_advance() -> None:
    """No key of any family equals a key of another, at any reachable index.

    Two lattices ``d1 + i * gamma`` and ``d2 + j * gamma`` collide exactly
    when ``(d1 - d2)`` is itself a gamma multiple, so this checks the
    arithmetic rather than sampling index pairs. The commit advance is folded
    in because the next iteration's base is this one's plus the committed
    count, which shifts one family's lattice against the other's.
    """
    for (name_a, dom_a), (name_b, dom_b) in itertools.combinations(
        _DOMAIN_LATTICES.items(), 2
    ):
        for committed in range(_MAX_COMMITTED + 1):
            gap = _lattice_gap((dom_a - dom_b + committed) % _U64)
            assert gap > _REACHABLE, (
                f"{name_a} and {name_b} share a key at a gamma-step distance "
                f"of {gap} under a commit advance of {committed}"
            )
