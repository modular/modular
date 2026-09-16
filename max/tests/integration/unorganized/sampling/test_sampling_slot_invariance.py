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
"""A request's sampled token must not depend on its physical batch slot.

``topk_fused_sampling`` is the general sampler -- ordinary decode as well as
speculative verification -- and its dual-pivot route used to mix the batch
slot into the Philox counter, so a request that moved slots drew a different
token from an unchanged seed. The counter term is gone; the per-row seed,
which carries a stable hash of the request id, is now the only thing keying a
draw.

``test_token_follows_the_request_not_the_slot`` is the claim. The two controls
are what keep it from passing vacuously and what guard the regression the
change could have introduced:

- A different seed at the same slot must move the token, or the claim would
  hold just as well against a sampler that ignored the seed entirely.
- Two requests handed the same ``sampling_params.seed`` at the same token
  count must still draw independently. The slot term used to be what pulled
  them apart on the dual-pivot route, and on the other two routes nothing ever
  did -- the request id is now the only separator, on all three.

Everything is parametrised over the three routes
``fused_token_sampling_gpu`` forks into, since the bug lived in exactly one of
them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import topk_fused_sampling
from max.pipelines.context import SamplingParams, TextContext, TokenBuffer
from max.pipelines.request import RequestID
from max.pipelines.sampling import request_row_seed

BATCH_SIZE = 2
VOCAB_SIZE = 256

# Equal mass on the first `_SUPPORT` tokens and nothing anywhere else, so every
# route below draws near-uniformly over a support wide enough that two
# independent draws usually disagree.
_SUPPORT = 16

# Deterministic request ids: the mixing hash is fixed, so an unlucky draw fails
# every run rather than one in ten.
_REQUEST_IDS = [f"slot-invariance-{i}" for i in range(24)]

# `fused_token_sampling_gpu` forks on (max_k, min_top_p): `-1`/`1.0` takes the
# fused Gumbel kernel, a `max_k` of 10 or more the dual-pivot search that
# production runs, and a smaller one the heap. Gumbel and heap never read the
# RNG counter, so only the middle case ever carried the bug -- and only the
# duplicate-seed control says anything about the other two.
_ROUTES = [
    pytest.param(-1, 1.0, id="gumbel"),
    pytest.param(VOCAB_SIZE, 0.95, id="dual_pivot"),
    pytest.param(4, 1.0, id="heap"),
]


@pytest.fixture(scope="module")
def sampler(session: InferenceSession) -> Model:
    """Compiles one ``topk_fused_sampling`` graph for every route.

    ``max_k`` and ``min_top_p`` are host-scalar graph inputs rather than
    Python constants, so the same compiled graph reaches all three routes.
    """
    device = DeviceRef.from_device(session.devices[0])
    cpu = DeviceRef.CPU()
    with Graph(
        "sampling_slot_invariance",
        input_types=[
            TensorType(DType.float32, [BATCH_SIZE, VOCAB_SIZE], device=device),
            TensorType(DType.int64, [BATCH_SIZE], device=device),
            TensorType(DType.int64, [], device=cpu),
            TensorType(DType.float32, [BATCH_SIZE], device=device),
            TensorType(DType.float32, [BATCH_SIZE], device=device),
            TensorType(DType.float32, [], device=cpu),
            TensorType(DType.uint64, [BATCH_SIZE], device=device),
        ],
    ) as graph:
        logits, top_k, max_k, temperature, top_p, min_top_p, seed = graph.inputs
        graph.output(
            topk_fused_sampling(
                logits=logits.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                temperature=temperature.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
            )
        )
    return session.load(graph)


def _context(
    request_id: str, base_seed: int = 4242, generated: int = 7
) -> TextContext:
    """Builds a context whose only free variable is its request id."""
    return TextContext(
        request_id=RequestID(request_id),
        max_length=128,
        tokens=TokenBuffer(np.arange(generated, dtype=np.int64)),
        sampling_params=SamplingParams(seed=base_seed),
    )


def _seeds(*contexts: TextContext) -> npt.NDArray[np.uint64]:
    return np.array(
        [request_row_seed(context) for context in contexts], dtype=np.uint64
    )


def _draw(
    sampler: Model,
    session: InferenceSession,
    seeds: npt.NDArray[np.uint64],
    max_k: int,
    min_top_p: float,
) -> npt.NDArray[np.int64]:
    """Samples one token per row; every row sees the identical distribution.

    Holding the logits equal across rows leaves the seed and the slot as the
    only things that can move a row's token.
    """
    device = session.devices[0]
    logits_row = np.full(VOCAB_SIZE, -100.0, dtype=np.float32)
    logits_row[:_SUPPORT] = 0.0
    logits = np.tile(logits_row, (BATCH_SIZE, 1))

    (tokens,) = sampler(
        Buffer.from_numpy(logits).to(device),
        Buffer.from_numpy(np.full(BATCH_SIZE, max_k, np.int64)).to(device),
        Buffer.from_numpy(np.array(max_k, np.int64)),
        Buffer.from_numpy(np.ones(BATCH_SIZE, np.float32)).to(device),
        Buffer.from_numpy(np.full(BATCH_SIZE, min_top_p, np.float32)).to(
            device
        ),
        Buffer.from_numpy(np.array(min_top_p, np.float32)),
        Buffer.from_numpy(seeds).to(device),
    )
    assert isinstance(tokens, Buffer)
    return tokens.to_numpy().reshape(BATCH_SIZE)


@pytest.mark.parametrize(("max_k", "min_top_p"), _ROUTES)
def test_token_follows_the_request_not_the_slot(
    session: InferenceSession,
    sampler: Model,
    max_k: int,
    min_top_p: float,
) -> None:
    """Swapping two requests' slots must not change either one's token.

    Both rows carry the same logits, so the swap moves nothing but the batch
    position. This is the bug: a preempted request re-admitted into a
    different slot used to draw a different token from the same seed.
    """
    for first_id, second_id in zip(
        _REQUEST_IDS[::2], _REQUEST_IDS[1::2], strict=True
    ):
        first, second = _context(first_id), _context(second_id)
        forward = _draw(
            sampler, session, _seeds(first, second), max_k, min_top_p
        )
        reversed_ = _draw(
            sampler, session, _seeds(second, first), max_k, min_top_p
        )
        assert forward[0] == reversed_[1], (
            f"{first_id} drew {forward[0]} at slot 0 but {reversed_[1]} at"
            " slot 1 with its seed unchanged"
        )
        assert forward[1] == reversed_[0], (
            f"{second_id} drew {forward[1]} at slot 1 but {reversed_[0]} at"
            " slot 0 with its seed unchanged"
        )


@pytest.mark.parametrize(("max_k", "min_top_p"), _ROUTES)
def test_a_different_seed_at_the_same_slot_moves_the_token(
    session: InferenceSession,
    sampler: Model,
    max_k: int,
    min_top_p: float,
) -> None:
    """Slot 0's token must vary with slot 0's seed.

    Without this, :func:`test_token_follows_the_request_not_the_slot` would
    pass against a sampler that had stopped reading the seed, or against a row
    whose distribution had collapsed to a single token.
    """
    other = _context("slot-invariance-companion")
    drawn = {
        int(
            _draw(
                sampler,
                session,
                _seeds(_context(request_id), other),
                max_k,
                min_top_p,
            )[0]
        )
        for request_id in _REQUEST_IDS
    }
    assert len(drawn) > 1, (
        f"slot 0 drew only token {drawn.pop()} across {len(_REQUEST_IDS)}"
        " distinct seeds: its draw is not keyed on its seed at all"
    )


@pytest.mark.parametrize(("max_k", "min_top_p"), _ROUTES)
def test_requests_sharing_a_seed_still_draw_independently(
    session: InferenceSession,
    sampler: Model,
    max_k: int,
    min_top_p: float,
) -> None:
    """Equal sampling params, equal token counts, different requests.

    Two co-resident requests can be handed the same ``sampling_params.seed``
    -- a client pinning a seed for reproducibility, most obviously -- and
    reach a step having generated the same number of tokens. Nothing but the
    request id then tells their keys apart, so this is where the id earns its
    place: on the dual-pivot route it replaces the slot term that used to
    separate them, and on the Gumbel and heap routes it supplies a separation
    that never existed.
    """
    disagreements = sum(
        int(row[0] != row[1])
        for row in (
            _draw(
                sampler,
                session,
                _seeds(_context(first_id), _context(second_id)),
                max_k,
                min_top_p,
            )
            for first_id, second_id in zip(
                _REQUEST_IDS[::2], _REQUEST_IDS[1::2], strict=True
            )
        )
    )
    trials = len(_REQUEST_IDS) // 2
    # The heap route's support here is 4 tokens, so ~1 trial in 4 agrees by
    # chance even when the two rows are fully independent; the wider routes
    # agree ~1 in 16. Half is comfortably below both and far above the zero a
    # lock-stepped pair would give.
    assert disagreements >= trials // 2, (
        f"{trials - disagreements} of {trials} same-seed pairs drew the same"
        " token: co-resident requests are sampling in lock step"
    )


@pytest.mark.parametrize(("max_k", "min_top_p"), _ROUTES)
def test_the_row_key_is_the_only_separator(
    session: InferenceSession,
    sampler: Model,
    max_k: int,
    min_top_p: float,
) -> None:
    """Rows handed a literally identical key now draw identically.

    Not a desirable property in itself -- it is the cost of removing the slot
    from the counter, recorded here so it cannot be forgotten. Any caller that
    hands ``topk_fused_sampling`` one key for several rows gets one token for
    all of them, which is why every production path derives its keys per
    request and why the graph-level shared seed is still spread by row index
    before it reaches the kernel.
    """
    for trial, request_id in enumerate(_REQUEST_IDS):
        duplicate = np.full(
            BATCH_SIZE, request_row_seed(_context(request_id)), dtype=np.uint64
        )
        row = _draw(sampler, session, duplicate, max_k, min_top_p)
        assert row[0] == row[1], (
            f"trial {trial}: rows carrying one key drew {row[0]} and {row[1]}"
            " -- something other than the key is still reaching the draw"
        )
