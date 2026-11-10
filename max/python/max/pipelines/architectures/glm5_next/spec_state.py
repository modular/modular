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

Three state kinds advance with every verified token, and none has a length
pointer to rewind:

- **KDA recurrent.** The verify reads the live pool without writing it and
  records each token's update in the verify ring, a scratch leaf of the state
  cache. The accepted records are folded into the live pool once the accepted
  count is known, as Qwen3.5 does for Gated DeltaNet.
- **KDA conv.** The verify leaves the window unwritten and the rollback
  places it at the accepted length from the copy the verify keeps of its
  input, inside the same ``kda_state_fold`` launch that folds the recurrent
  pool. Qwen3.5's Gated DeltaNet still replays its conv windows per layer
  through :func:`~max.pipelines.speculative.gated_delta_rollback.replay_conv_pools`.
- **Indexer tail ring.** It carries each sparse layer's incomplete k-pool,
  slot-indexed modulo ``index_kpool``, so a rejected draft's seed can
  overwrite a member the next step closes its pool with. The verify seeds
  nothing and the rollback seeds the accepted rows. The pooled keys a verify
  writes to the paged indexer cache need nothing: past the accepted length
  they are masked, and rewritten when that pool really closes. The MTP
  draft's first step runs its own indexer over the same window, after the
  target's rollback, so its ring is seeded once that step has run.

A window with no drafts has nothing to reject, so its verify lands all three on
the live pools and every rollback launch does nothing. The verify width that
decides this is a shape, which the ops read without a device sync.

The speculative driver hands its hooks one graph value per device; the
rollback joins them into tensors on the target's mesh, broadcasts the
batch-wide plan with :func:`~max.functional.distributed_broadcast`, and runs
each device's launches under :func:`~max.functional.call_on_mesh`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue
from max.nn._graph_ops.state_space.kimi_delta import STATE_LAYOUT
from max.nn.kernels import kda_state_fold, mla_kpool_seed_tail
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.pipelines.speculative.gated_delta_rollback import (
    accepted_lengths,
    accepted_row_plan,
)
from max.pipelines.speculative.ragged_token_merger import shape_to_scalar
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh
from max.tensor.sharding.action import PerShard

from .glm5_next import Glm5Next, Glm5NextTextModel, replicate, state_leaf
from .layers.indexer import PooledIndexer, TailDeferral
from .layers.indexer_inputs import KpoolTailReplayInputs
from .layers.kda import KdaVerify, KimiDeltaAttention, verify_width_operand
from .layers.kda_replay import KdaReplayInputs
from .layers.sparse_mla import Glm5NextSparseMLA
from .state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    VERIFY_RING_LEAF_ID,
    has_verify_ring,
)

__all__ = [
    "Glm5NextSpecState",
    "deferred_seed_offsets",
    "fold_kda_state_pools",
    "replay_tail_seeds",
]

_StateInputs = Sequence[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]


def _dim_scalar(dim: Dim, device: Device, dtype: DType) -> Tensor:
    """Returns ``dim`` as a scalar built on ``device``.

    Built there rather than copied from the host, which a device graph
    capture could not replay.
    """
    shape = F.shape_to_tensor([dim])
    return F.range(
        shape[0], shape[0] + 1, out_dim=1, dtype=dtype, device=device
    )[0]


def deferred_seed_offsets(num_draft_tokens: Dim, row_offsets: Tensor) -> Tensor:
    """Returns the row offsets a verify's tail seed reads.

    The offsets themselves without drafts, so the forward lands the ring, and
    all zeros with them, which seeds no row and leaves the ring for the
    rollback.

    Args:
        num_draft_tokens: The verify's draft width ``K``.
        row_offsets: ``[batch + 1]`` uint32 offsets, the same on every device.
    """

    def seed_offsets(offsets: Tensor) -> Tensor:
        no_drafts = F.equal(
            _dim_scalar(num_draft_tokens, offsets.device, DType.uint32), 0
        )
        return offsets.cast(DType.uint32) * no_drafts.cast(DType.uint32)

    return F.call_on_mesh(
        seed_offsets, row_offsets.mesh, out_specs=row_offsets.mapping
    )(row_offsets)


def _qkv_windows(captures: Sequence[KdaReplayInputs]) -> list[Tensor]:
    """Returns one device's per-layer copies of the verify's conv input."""
    windows = []
    for capture in captures:
        assert capture.qkv_window is not None, "a fold needs a verify"
        windows.append(capture.qkv_window)
    return windows


def fold_kda_state_pools(
    recurrent_pool: Tensor,
    recurrent_rows: Tensor,
    ring_pool: Tensor,
    ring_rows: Tensor,
    conv_pool: Tensor,
    conv_rows: Tensor,
    captures: Sequence[Sequence[KdaReplayInputs]],
    fold_accepted: Tensor,
    row_offsets: Tensor,
    verify_width: Tensor,
) -> None:
    """Applies the verify's accepted records to the live KDA pools.

    One launch per device folds the ring into the recurrent pool and places
    each request's conv window at its accepted length, and none runs at a
    verify width of zero.

    Args:
        recurrent_pool: The recurrent pool, still pre-verify.
        recurrent_rows: ``[num_layers, batch_size]`` state rows.
        ring_pool: The verify ring.
        ring_rows: ``[num_layers, batch_size]`` ring rows.
        conv_pool: The conv pool, still pre-verify.
        conv_rows: ``[num_layers, batch_size]`` conv rows.
        captures: Per-device, per-layer inputs the verify recorded; the fold
            reads each layer's ``qkv_window``.
        fold_accepted: ``[batch]`` records to fold, on one device.
        row_offsets: ``[batch + 1]`` offsets of the verified window, which
            bound each row's fold.
        verify_width: The verify's width operand.
    """
    kda_state_fold(
        recurrent_pool,
        ring_pool,
        recurrent_rows,
        ring_rows,
        replicate(fold_accepted.cast(DType.uint32), recurrent_pool.mesh),
        row_offsets,
        verify_width,
        conv_pool=conv_pool,
        conv_row_ids=conv_rows,
        qkv_windows=PerShard(map(_qkv_windows, captures)),
        state_layout=STATE_LAYOUT,
    )


def replay_tail_seeds(
    captures: Sequence[Sequence[KpoolTailReplayInputs]],
    row_indices: Tensor,
    replay_offsets: Tensor,
    num_draft_tokens: Tensor,
    mesh: DeviceMesh,
) -> None:
    """Seeds each indexer tail ring with the verify's accepted rows.

    The verify left every ring at its pre-verify members, so the seed over the
    accepted rows, from the pre-verify cache length, lands each ring where a
    forward over the accepted prefix alone would have. Without drafts the
    verify seeded the ring itself, and the offsets collapse to zero.

    Args:
        captures: Per-device, per-layer inputs the verify's indexers recorded.
        row_indices: Rows of the verify tensors the replay consumes.
        replay_offsets: ``[batch + 1]`` ragged offsets over those rows.
        num_draft_tokens: Scalar ``K`` on the device the plan lives on.
        mesh: The mesh the captures were recorded on.
    """
    has_drafts = (num_draft_tokens > 0).cast(replay_offsets.dtype)
    offsets = replicate((replay_offsets * has_drafts).cast(DType.uint32), mesh)
    rows = replicate(row_indices, mesh)

    def seed(
        layer_captures: Sequence[KpoolTailReplayInputs],
        rows: Tensor,
        offsets: Tensor,
    ) -> None:
        for capture in layer_captures:
            mla_kpool_seed_tail(
                capture.tail,
                F.gather(capture.k, rows, axis=0),
                F.gather(capture.gate, rows, axis=0),
                offsets,
                capture.cache_lengths,
                capture.slot_idx,
                capture.kpool,
            )

    F.call_on_mesh(seed, mesh)(PerShard(captures), rows, offsets)


class Glm5NextSpecState:
    """The ModuleV3 target's state capture, and the rollback it feeds.

    Used in this order: :meth:`capturing` around the target's verify,
    :meth:`roll_forward` once the accepted count is known, then
    :meth:`capturing_draft` around the draft's first step, which runs its own
    indexer over the verified window, and :meth:`replay_draft_tail` after it.
    Rolling forward before the count settles lands the state on a prefix the
    row never accepted, and seeding the draft's ring before its step has run
    lets that step seed it again, both silently.
    """

    def __init__(self, target: Glm5Next) -> None:
        self.target = target
        self.kda_captures: list[list[KdaReplayInputs]] = []
        """Per-device, per-KDA-layer conv inputs the verify recorded."""
        self.tail_captures: list[list[KpoolTailReplayInputs]] = []
        """Per-device tail-seed inputs of the target's sparse layers."""
        self.draft_tail_captures: list[list[KpoolTailReplayInputs]] = []
        """Per-device tail-seed inputs of the draft's first step."""
        self.num_draft_tokens: Dim | None = None
        """The verify's draft width ``K``, set when the verify is bound."""
        self.verify_width: Tensor | None = None
        """The verify width operand the forward and the rollback share.

        Both must read the same one: a forward that lands a window the
        rollback then replays or folds advances the state twice.
        """
        self._plan: tuple[Tensor, Tensor, Tensor] | None = None
        """The accepted rows, their offsets and ``K`` on the plan's device,
        which :meth:`roll_forward` builds and :meth:`replay_draft_tail`
        reuses."""

    @property
    def language_model(self) -> Glm5NextTextModel:
        """The target's decoder stack and head."""
        return self.target.language_model

    @property
    def mesh(self) -> DeviceMesh:
        """The target's mesh, which every capture and rollback runs on."""
        mesh = self.language_model.mesh
        assert mesh is not None
        return mesh

    @contextmanager
    def capturing(
        self,
        state: _StateInputs,
        num_draft_tokens: Dim,
        row_offsets: Tensor,
    ) -> Iterator[None]:
        """Binds the target's state capture for the verify inside.

        Unwound on the way out, so the draft's own forward cannot append to
        the target's captures.

        Args:
            state: Per-device state leaves, the verify ring included.
            num_draft_tokens: The verify's draft width ``K``.
            row_offsets: The verified window's offsets, replicated on the
                target's mesh.

        Raises:
            ValueError: If the cache declares no verify ring.
        """
        if not all(map(has_verify_ring, state)):
            raise ValueError(
                "A GLM-5.3-Flash verify records its KDA recurrence into the"
                " verify ring, and the cache declares none; build the cache"
                " with the speculative config."
            )
        self._start_verify(num_draft_tokens)
        assert self.verify_width is not None
        ring_pool, ring_rows = state_leaf(
            list(state),
            VERIFY_RING_LEAF_ID,
            row_offsets.mapping,
            head_sharded=True,
        )
        kda = KdaVerify(
            ring_pool=ring_pool,
            ring_rows=ring_rows,
            verify_width=self.verify_width,
            captures=self.kda_captures,
        )
        tail = TailDeferral(
            seed_row_offsets=deferred_seed_offsets(
                num_draft_tokens, row_offsets
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
        self, indexer: PooledIndexer, row_offsets: Tensor
    ) -> Iterator[None]:
        """Binds the draft's tail capture for its first step inside.

        That step runs the draft's indexer over the same verified window, so
        its ring defers to :meth:`replay_draft_tail` like the target's defers
        to :meth:`roll_forward`.

        Args:
            indexer: The draft layer's indexer.
            row_offsets: The verified window's offsets, replicated on the
                target's mesh.
        """
        assert self.num_draft_tokens is not None, "draft before verify"
        self.draft_tail_captures = [[] for _ in range(self.mesh.num_devices)]
        indexer.tail_deferral = TailDeferral(
            seed_row_offsets=deferred_seed_offsets(
                self.num_draft_tokens, row_offsets
            ),
            captures=self.draft_tail_captures,
        )
        try:
            yield
        finally:
            indexer.tail_deferral = None

    def _start_verify(self, num_draft_tokens: Dim) -> None:
        """Resets the captures for a verify of width ``num_draft_tokens``."""
        n_devs = self.mesh.num_devices
        self.kda_captures = [[] for _ in range(n_devs)]
        self.tail_captures = [[] for _ in range(n_devs)]
        self.num_draft_tokens = num_draft_tokens
        self.verify_width = verify_width_operand(num_draft_tokens)

    def roll_forward(
        self,
        state: _StateInputs,
        *,
        merged_offsets: TensorValue,
        merged_offsets_per_dev: Sequence[TensorValue],
        num_accepted: TensorValue,
        device: DeviceRef,
    ) -> None:
        """Lands the target's state leaves on the accepted prefix.

        Must run after the accepted count has been corrected for rows that
        carried no proposal, and before anything reads the live pools.

        Args:
            state: Per-device state leaves, the verify ring included.
            merged_offsets: Ragged offsets over the verified window.
            merged_offsets_per_dev: ``merged_offsets`` on each device.
            num_accepted: ``[batch]`` accepted draft tokens per request.
            device: The device the batch-wide tensors live on.
        """
        assert self.verify_width is not None, "roll_forward before capturing"
        assert self.num_draft_tokens is not None
        assert all(map(has_verify_ring, state)), "the cache declares no ring"
        num_draft_tokens = shape_to_scalar(self.num_draft_tokens, device)
        window = self.kda_captures[0][0].qkv_window
        assert window is not None
        # This verify never carries a prompt row, so every row is one
        # token and its drafts: the window copy is the input, row for row.
        row_indices, replay_offsets = accepted_row_plan(
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            window.shape[0],
            device,
        )
        self._plan = (
            Tensor.from_graph_value(row_indices),
            Tensor.from_graph_value(replay_offsets),
            Tensor.from_graph_value(num_draft_tokens),
        )
        replicated = DeviceMapping.replicated(self.mesh)
        conv_pool, conv_rows = state_leaf(
            list(state), CONV_LEAF_ID, replicated, head_sharded=True
        )
        recurrent_pool, recurrent_rows = state_leaf(
            list(state), RECURRENT_LEAF_ID, replicated, head_sharded=True
        )
        ring_pool, ring_rows = state_leaf(
            list(state), VERIFY_RING_LEAF_ID, replicated, head_sharded=True
        )
        fold_kda_state_pools(
            recurrent_pool,
            recurrent_rows,
            ring_pool,
            ring_rows,
            conv_pool,
            conv_rows,
            self.kda_captures,
            Tensor.from_graph_value(
                accepted_lengths(merged_offsets, num_accepted, num_draft_tokens)
            ),
            Tensor.from_shard_values(merged_offsets_per_dev, replicated),
            self.verify_width,
        )
        replay_tail_seeds(self.tail_captures, *self._plan, self.mesh)

    def replay_draft_tail(self) -> None:
        """Seeds the draft's tail ring with the verify's accepted rows.

        Must run after :meth:`roll_forward` and after the draft's first step.
        """
        assert self._plan is not None, "draft tail replay before roll_forward"
        replay_tail_seeds(self.draft_tail_captures, *self._plan, self.mesh)
