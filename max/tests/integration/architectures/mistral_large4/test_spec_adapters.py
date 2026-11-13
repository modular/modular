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
"""The Mistral Large 4 DSpark proposer's block boundary.

The block driver lays a block out request-major, ``block_size`` rows per
request, and fills the rows past each anchor with the proposer's
``mask_token_id``. Mistral Large 4's drafter has no mask token -- its rows past
the anchor are the checkpoint's trained ``backbone.mask_embedding`` parameters
-- so the value the proposer reports there is a placeholder drawn from the
model's own vocabulary. A placeholder is only safe while nothing reads it, and
what makes that true is structural: :func:`block_anchors` takes column 0 alone,
and the proposer embeds *that* through the target's token embedding rather than
the block. Both are pinned here, on eager CPU tensors, with a stub target and
drafter: the filler id never becomes an embedding row. The other half of that
boundary -- the pack and unpack between the proposer's two block phases -- is
pinned here too, on a two-rank mesh where the halves are told apart.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.mistral_large4_modulev3.spec_adapters import (
    MASK_TOKEN_ID,
    DSparkMistralLarge4Proposer,
    block_anchors,
)
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh

BATCH = 2
"""Requests in the batch: two blocks of ``DEPTH`` rows each."""

DEPTH = 3
"""The drafter's block width, the deployment's ``--num-speculative-tokens``."""

HIDDEN = 4
"""The embedding width: the test reads shapes and ids, so it stays small."""

ANCHORS = [5, 11]
"""The tokens each block commits, one per request."""

_MESH = DeviceMesh((CPU(),), (1,), ("tp",))

_TP2 = DeviceMesh((CPU(), CPU()), (2,), ("tp",))
"""Two ranks, so a packed block's two halves are told apart by their values."""


def _tensor(array: np.ndarray[Any, Any]) -> Tensor:
    return Tensor(torch.from_numpy(array), device=CPU())


def _block_ids() -> np.ndarray[Any, Any]:
    """The driver's flattened block: each anchor, then its filler rows."""
    rows = [[anchor, MASK_TOKEN_ID, MASK_TOKEN_ID] for anchor in ANCHORS]
    return np.array(rows, dtype=np.int64).reshape(-1)


class _RecordingEmbedding:
    """The target's token embedding, recording the ids it is asked for."""

    def __init__(self) -> None:
        self.looked_up: list[Tensor] = []

    def __call__(self, ids: Tensor) -> Tensor:
        self.looked_up.append(ids)
        # One row per id, so a filler row would show up in what the drafter
        # receives. Built as graph ops: the ambient realization context
        # rejects a value read while it is still open.
        rows = F.cast(ids, DType.float32).reshape([ids.shape[0], 1])
        return F.broadcast_to(rows, [ids.shape[0], HIDDEN])


def test_block_anchors_reads_only_the_blocks_first_row() -> None:
    """Column 0 of the request-major block, and nothing else."""
    with ensure_context():
        anchors = block_anchors(_tensor(_block_ids()), DEPTH)
    assert anchors.to_numpy().tolist() == ANCHORS


def test_the_mask_token_is_inert() -> None:
    """The proposer embeds the anchors, never a block's filler rows.

    ``embed_block`` runs the target's own token embedding over the anchors --
    one row per request, not one per block row -- and hands the drafter the
    block built from them. A filler id reaching that lookup would be a real
    token of the vocabulary, and would change the draft.
    """
    rows_seen: list[Tensor] = []

    def block_from_anchors(embeds: Tensor) -> Tensor:
        rows_seen.append(embeds)
        return embeds

    with ensure_context():
        embedding = _RecordingEmbedding()
        target = SimpleNamespace(
            config=SimpleNamespace(mesh=_MESH),
            language_model=SimpleNamespace(tok_embeddings=embedding),
        )
        proposer = DSparkMistralLarge4Proposer(
            cast(Any, SimpleNamespace(block_from_anchors=block_from_anchors)),
            cast(Any, target),
            block_size=DEPTH,
        )
        embeds = proposer.embed_block(cast(Any, None), _tensor(_block_ids()))

    # One lookup, of the anchors alone: no filler id is in it, and its row
    # count is the batch's rather than the block's.
    assert [rows.to_numpy().tolist() for rows in embedding.looked_up] == [
        ANCHORS
    ]
    # The block crosses to the block forward as one packed per-rank list, so
    # on this one-device mesh it is the anchors and then the rows.
    anchors, _ = embeds
    assert anchors.to_numpy().tolist() == ANCHORS
    # The drafter's block rows are the embeddings of those anchors, so the
    # filler id reached no table on the way either.
    (embedded,) = rows_seen
    np.testing.assert_array_equal(
        embedded.to_numpy()[:, 0], np.array(ANCHORS, np.float32)
    )


def test_the_mask_token_is_a_real_id() -> None:
    """The placeholder is an ordinary vocabulary entry, which is why it matters.

    The checkpoint declares no mask token of its own, so the placeholder has to
    be a value the block driver can lay out. Keeping it inside the vocabulary
    is what makes an accidental lookup an ordinary token rather than an
    out-of-range id -- which is why the inertness above is a correctness
    property and not a bounds check.
    """
    assert 0 <= MASK_TOKEN_ID < 204_800


def test_forward_block_takes_the_first_half_as_anchors_and_the_second_as_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block forward splits the packed block in ``embed_block``'s order.

    ``embed_block`` packs one per-rank list of anchors and then one of embedded
    rows, and the driver threads the result through as a single opaque value.
    A flipped or offset split would score the drafter's rows against the wrong
    half, so the two are given distinct contents on a two-rank mesh and read
    back where the block forward hands them over.
    """
    anchors = [_tensor(np.array([anchor], np.int32)) for anchor in ANCHORS]
    rows = [
        _tensor(np.full((BATCH * DEPTH, HIDDEN), fill, np.float32))
        for fill in (1.0, 2.0)
    ]
    handed_over: list[tuple[Tensor, Tensor]] = []

    def forward_block_paged(
        block_input: Tensor,
        anchor_ids: Tensor,
        anchor_positions: Tensor,
        kv: object,
        head: object,
    ) -> tuple[Tensor, Tensor]:
        handed_over.append((anchor_ids, block_input))
        return _tensor(np.zeros((BATCH, HIDDEN), np.float32)), _tensor(
            np.zeros((BATCH, DEPTH), np.int32)
        )

    with ensure_context():
        proposer = DSparkMistralLarge4Proposer(
            cast(Any, SimpleNamespace(forward_block_paged=forward_block_paged)),
            cast(
                Any,
                SimpleNamespace(
                    config=SimpleNamespace(mesh=_TP2),
                    # ``_draft_block`` reads the target's LM head and hands it
                    # to the drafter; nothing here scores through it.
                    language_model=SimpleNamespace(lm_head=None),
                ),
            ),
            block_size=DEPTH,
        )
        # The leaf is the block's cache, which the split does not touch: only
        # its lengths are read, for the row positions.
        monkeypatch.setattr(
            proposer,
            "_leaf",
            lambda caches: SimpleNamespace(
                cache_lengths=_tensor(np.array([2], np.int32))
            ),
        )
        proposer.forward_block(
            cast(Any, SimpleNamespace(n_devs=2)),
            [*anchors, *rows],
            [],
            cast(Any, []),
        )

    ((anchor_ids, block_input),) = handed_over
    assert [shard.to_numpy().tolist() for shard in anchor_ids.local_shards] == [
        [anchor] for anchor in ANCHORS
    ]
    assert [
        shard.to_numpy().tolist() for shard in block_input.local_shards
    ] == [
        [[1.0] * HIDDEN] * (BATCH * DEPTH),
        [[2.0] * HIDDEN] * (BATCH * DEPTH),
    ]


def test_forward_block_refuses_a_packed_block_of_the_wrong_length() -> None:
    """A pack that is not two per-rank lists fails at the split, not later.

    The rank count is the batch's, so the halves are that many values each. A
    pack that lost or gained one would otherwise pair the anchors of one rank
    with the rows of another, which compiles into a graph that drafts wrong.
    """
    proposer = DSparkMistralLarge4Proposer(
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace(config=SimpleNamespace(mesh=_TP2))),
        block_size=DEPTH,
    )
    with pytest.raises(ValueError, match="one value per rank"):
        proposer.forward_block(
            cast(Any, SimpleNamespace(n_devs=2)),
            [_tensor(np.array([ANCHORS[0]], np.int32))],
            [],
            cast(Any, []),
        )
