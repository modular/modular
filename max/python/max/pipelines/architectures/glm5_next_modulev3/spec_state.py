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
"""GLM-5.3-Flash's per-request state across a speculative verify, on ModuleV3.

The rollback is the graph-API model's
(:class:`~max.pipelines.architectures.glm5_next.spec_state.RecurrentStateRollback`):
the verify records the KDA recurrence into the verify ring and defers the conv
windows and the indexer tail seeds, and the rollback lands all three on the
accepted prefix. This module binds those captures to the ModuleV3 layers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from max.graph import BufferValue, Dim, TensorValue
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.pipelines.architectures.glm5_next.spec_state import (
    RecurrentStateRollback,
    deferred_seed_offsets,
)
from max.pipelines.architectures.glm5_next.state_cache import (
    VERIFY_RING_LEAF_ID,
    has_verify_ring,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from .glm5_next import Glm5Next, Glm5NextTextModel, state_leaf
from .layers.indexer import PooledIndexer, TailDeferral
from .layers.kda import KdaVerify, KimiDeltaAttention
from .layers.sparse_mla import Glm5NextSparseMLA

__all__ = ["Glm5NextSpecState"]

_StateInputs = Sequence[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]


class Glm5NextSpecState(RecurrentStateRollback):
    """The ModuleV3 target's state capture, bound around its verify.

    Used in this order: :meth:`capturing` around the target's verify,
    :meth:`roll_forward` once the accepted count is known, then
    :meth:`capturing_draft` around the draft's first step, which runs its own
    indexer over the verified window, and :meth:`replay_draft_tail` after it.
    """

    def __init__(self, target: Glm5Next) -> None:
        super().__init__()
        self.target = target

    @property
    def language_model(self) -> Glm5NextTextModel:
        """The target's decoder stack and head."""
        return self.target.language_model

    @contextmanager
    def capturing(
        self,
        state: _StateInputs,
        num_draft_tokens: Dim,
        row_offsets: Sequence[TensorValue],
        mapping: DeviceMapping,
    ) -> Iterator[None]:
        """Binds the target's state capture for the verify inside.

        Unwound on the way out, so the draft's own forward cannot append to
        the target's captures.

        Args:
            state: Per-device state leaves, the verify ring included.
            num_draft_tokens: The verify's draft width ``K``.
            row_offsets: The verified window's offsets on each device.
            mapping: The replicated mapping of the target's mesh.

        Raises:
            ValueError: If the cache declares no verify ring.
        """
        if not all(map(has_verify_ring, state)):
            raise ValueError(
                "A GLM-5.3-Flash verify records its KDA recurrence into the"
                " verify ring, and the cache declares none; build the cache"
                " with the speculative config."
            )
        self._start_verify(num_draft_tokens, len(row_offsets))
        assert self.verify_width is not None
        ring_pool, ring_rows = state_leaf(
            list(state), VERIFY_RING_LEAF_ID, mapping, head_sharded=True
        )
        kda = KdaVerify(
            ring_pool=ring_pool,
            ring_rows=ring_rows,
            verify_width=self.verify_width,
            captures=self.kda_captures,
        )
        tail = TailDeferral(
            seed_row_offsets=Tensor.from_shard_values(
                deferred_seed_offsets(num_draft_tokens, row_offsets), mapping
            ),
            captures=self.tail_captures,
        )
        attentions = [layer.self_attn for layer in self.language_model.layers]
        for attention in attentions:
            if isinstance(attention, KimiDeltaAttention):
                attention.verify = kda
            else:
                assert isinstance(attention, Glm5NextSparseMLA)
                attention.indexer.tail_deferral = tail
        try:
            yield
        finally:
            for attention in attentions:
                if isinstance(attention, KimiDeltaAttention):
                    attention.verify = None
                else:
                    assert isinstance(attention, Glm5NextSparseMLA)
                    attention.indexer.tail_deferral = None

    @contextmanager
    def capturing_draft(
        self,
        indexer: PooledIndexer,
        row_offsets: Sequence[TensorValue],
        mapping: DeviceMapping,
    ) -> Iterator[None]:
        """Binds the draft's tail capture for its first step inside.

        That step runs the draft's indexer over the same verified window, so
        its ring defers to :meth:`replay_draft_tail` like the target's defers
        to :meth:`roll_forward`.

        Args:
            indexer: The draft layer's indexer.
            row_offsets: The verified window's offsets on each device.
            mapping: The replicated mapping of the target's mesh.
        """
        self._start_draft(len(row_offsets))
        assert self.num_draft_tokens is not None
        indexer.tail_deferral = TailDeferral(
            seed_row_offsets=Tensor.from_shard_values(
                deferred_seed_offsets(self.num_draft_tokens, row_offsets),
                mapping,
            ),
            captures=self.draft_tail_captures,
        )
        try:
            yield
        finally:
            indexer.tail_deferral = None
