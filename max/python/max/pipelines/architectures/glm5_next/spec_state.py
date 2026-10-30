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
"""GLM-5.3-Flash's per-request state across a speculative verify.

Three state kinds advance with every verified token, and none has a length
pointer to rewind:

- **KDA recurrent.** The verify reads the live pool without writing it and
  records each token's update in the verify ring, a scratch leaf of the state
  cache. The accepted records are folded into the live pool once the accepted
  count is known, as Qwen3.5 does for Gated DeltaNet.
- **KDA conv.** The verify leaves the window unwritten and the rollback re-runs
  the conv kernel over the accepted rows of the copy the verify keeps of its
  input, through Qwen3.5's
  :func:`~max.pipelines.speculative.gated_delta_rollback.replay_conv_pools`.
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
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.legacy.kernels import mla_kpool_seed_tail
from max.nn.legacy.kv_cache import (
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.legacy.state_space import kda_state_fold, verify_width_operand
from max.nn.legacy.state_space.kimi_delta import STATE_LAYOUT
from max.pipelines.speculative.gated_delta_rollback import (
    accepted_lengths,
    accepted_row_plan,
    replay_conv_pools,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

from ..deepseekV3_2.layers.indexer import KpoolTailReplayInputs
from .glm5_next import Glm5Next
from .layers.decoder import Glm5NextDecoderLayer
from .layers.kda import KdaReplayInputs
from .layers.kimi_delta_attention import Glm5NextKdaSublayer
from .layers.sparse_mla import Glm5NextSparseMLASublayer
from .state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    VERIFY_RING_LEAF_ID,
    has_verify_ring,
    leaf_inputs,
)

__all__ = [
    "Glm5NextRecurrentState",
    "deferred_seed_offsets",
    "fold_kda_state_pools",
    "replay_tail_seeds",
]


def deferred_seed_offsets(
    num_draft_tokens: Dim, row_offsets: Sequence[TensorValue]
) -> list[TensorValue]:
    """Returns the row offsets a verify's tail seed reads, per device.

    The offsets themselves without drafts, so the forward lands the ring, and
    all zeros with them, which seeds no row and leaves the ring for the
    rollback.

    Args:
        num_draft_tokens: The verify's draft width ``K``.
        row_offsets: ``[batch + 1]`` uint32 offsets on each device.
    """
    seeds = []
    for offsets in row_offsets:
        no_drafts = (
            _shape_to_scalar(num_draft_tokens, offsets.device, DType.uint32)
            == 0
        )
        seeds.append(offsets.cast(DType.uint32) * no_drafts.cast(DType.uint32))
    return seeds


def fold_kda_state_pools(
    live_recurrent_pools: Sequence[BufferValue],
    recurrent_row_ids: Sequence[TensorValue],
    ring_pools: Sequence[BufferValue],
    ring_row_ids: Sequence[TensorValue],
    fold_accepted: TensorValue,
    row_offsets: Sequence[TensorValue],
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
) -> None:
    """Applies the verify's accepted ring records to the live KDA pool.

    One launch per device covers every layer, and none at a verify width of
    zero.

    Args:
        live_recurrent_pools: Per-device recurrent pool, still pre-verify.
        recurrent_row_ids: Per-device ``[num_layers, batch_size]`` state rows.
        ring_pools: Per-device verify ring.
        ring_row_ids: Per-device ``[num_layers, batch_size]`` ring rows.
        fold_accepted: ``[batch]`` records to fold.
        row_offsets: Per-device ``[batch + 1]`` offsets of the verified
            window, which bound each row's fold.
        signal_buffers: Used only to place ``fold_accepted`` on each device.
        verify_width: The verify's width operand.
    """
    accepted_per_dev = broadcast_per_device(
        fold_accepted.cast(DType.uint32),
        list(signal_buffers),
        len(live_recurrent_pools),
    )
    for device_idx, state_pool in enumerate(live_recurrent_pools):
        kda_state_fold(
            state_pool,
            ring_pools[device_idx],
            recurrent_row_ids[device_idx],
            ring_row_ids[device_idx],
            accepted_per_dev[device_idx],
            row_offsets[device_idx],
            verify_width,
            state_layout=STATE_LAYOUT,
        )


def replay_tail_seeds(
    captures: Sequence[Sequence[KpoolTailReplayInputs]],
    row_indices: TensorValue,
    replay_offsets: TensorValue,
    num_draft_tokens: TensorValue,
    signal_buffers: Sequence[BufferValue],
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
        signal_buffers: Used only to place the plan on each device.
    """
    has_drafts = (num_draft_tokens > 0).cast(replay_offsets.dtype)
    offsets = (replay_offsets * has_drafts).cast(DType.uint32)
    n_devs = len(captures)
    offsets_per_dev = broadcast_per_device(
        offsets, list(signal_buffers), n_devs
    )
    rows_per_dev = broadcast_per_device(
        row_indices, list(signal_buffers), n_devs
    )
    for device_idx, device_captures in enumerate(captures):
        rows = rows_per_dev[device_idx]
        for capture in device_captures:
            mla_kpool_seed_tail(
                capture.tail,
                ops.gather(capture.k, rows, axis=0),
                ops.gather(capture.gate, rows, axis=0),
                offsets_per_dev[device_idx],
                capture.cache_lengths,
                capture.slot_idx,
                capture.kpool,
            )


class Glm5NextRecurrentState:
    """The verify's state capture and the accepted-prefix rollback.

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
        """The verify's draft width ``K``. Set by :meth:`capturing`."""
        self.verify_width: TensorValue | None = None
        """The verify width operand the forward and the rollback share.

        Both must read the same one: a forward that lands a window the
        rollback then replays or folds advances the state twice.
        """
        self._plan: tuple[TensorValue, TensorValue, TensorValue] | None = None
        """The accepted rows, their offsets and ``K`` on the plan's device,
        which :meth:`roll_forward` builds and :meth:`replay_draft_tail`
        reuses."""

    def _sublayers(self) -> Iterator[object]:
        for layer in self.target.layers:
            assert isinstance(layer, Glm5NextDecoderLayer)
            yield layer.self_attn

    @contextmanager
    def capturing(
        self, num_draft_tokens: Dim, row_offsets: Sequence[TensorValue]
    ) -> Iterator[None]:
        """Binds the target's state capture for the verify inside.

        Unwound on the way out, so the draft's own forward cannot append to
        the KDA captures.

        Args:
            num_draft_tokens: The verify's draft width ``K``.
            row_offsets: The verified window's offsets on each device.
        """
        n_devs = len(row_offsets)
        self.kda_captures = [[] for _ in range(n_devs)]
        self.tail_captures = [[] for _ in range(n_devs)]
        self.num_draft_tokens = num_draft_tokens
        self.verify_width = verify_width_operand(num_draft_tokens)
        seeds = deferred_seed_offsets(num_draft_tokens, row_offsets)
        self._bind(self.kda_captures, self.verify_width, seeds)
        try:
            yield
        finally:
            self._bind(None, None, None)

    def _bind(
        self,
        kda_captures: list[list[KdaReplayInputs]] | None,
        verify_width: TensorValue | None,
        seeds: list[TensorValue] | None,
    ) -> None:
        tail_captures = None if seeds is None else self.tail_captures
        for sublayer in self._sublayers():
            if isinstance(sublayer, Glm5NextKdaSublayer):
                sublayer.replay_capture = kda_captures
                sublayer.verify_width = verify_width
            else:
                assert isinstance(sublayer, Glm5NextSparseMLASublayer)
                sublayer.tail_capture = tail_captures
                sublayer.tail_seed_offsets = seeds

    @contextmanager
    def capturing_draft(
        self,
        draft_sublayer: Glm5NextSparseMLASublayer,
        row_offsets: Sequence[TensorValue],
    ) -> Iterator[None]:
        """Binds the draft's tail capture for its first step inside.

        That step runs the draft's indexer over the same verified window, so
        its ring defers to :meth:`replay_draft_tail` like the target's defers
        to :meth:`roll_forward`.

        Args:
            draft_sublayer: The draft layer's sparse-MLA sublayer.
            row_offsets: The verified window's offsets on each device.
        """
        assert self.num_draft_tokens is not None, "draft before verify"
        self.draft_tail_captures = [[] for _ in row_offsets]
        draft_sublayer.tail_capture = self.draft_tail_captures
        draft_sublayer.tail_seed_offsets = deferred_seed_offsets(
            self.num_draft_tokens, row_offsets
        )
        try:
            yield
        finally:
            draft_sublayer.tail_capture = None
            draft_sublayer.tail_seed_offsets = None

    def roll_forward(
        self,
        state: Sequence[
            RecurrentStateInputsPerDevice[TensorValue, BufferValue]
        ],
        *,
        merged_offsets: TensorValue,
        merged_offsets_per_dev: Sequence[TensorValue],
        num_accepted: TensorValue,
        signal_buffers: Sequence[BufferValue],
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
            signal_buffers: Used only to place the plan on each device.
            device: The device the batch-wide tensors live on.
        """
        assert self.verify_width is not None, "roll_forward before capturing"
        assert self.num_draft_tokens is not None
        assert all(map(has_verify_ring, state)), "the cache declares no ring"
        num_draft_tokens = _shape_to_scalar(self.num_draft_tokens, device)
        window = self.kda_captures[0][0].qkv_window
        assert window is not None
        # The verify copied its rows whenever it had drafts, which is whenever
        # the rollback runs. Clamping to the copy keeps the gathers in bounds
        # when it does not.
        row_indices, replay_offsets = accepted_row_plan(
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            window.shape[0],
            device,
            # Each row is one token and its drafts whenever the rollback
            # runs, and the plan shrinks to a row per request when it does
            # not.
            plan_rows=Dim("batch_size") * (1 + self.num_draft_tokens),
        )
        self._plan = (row_indices, replay_offsets, num_draft_tokens)

        def leaves(
            leaf_id: str,
        ) -> list[RecurrentLeafInputs[TensorValue, BufferValue]]:
            return [
                leaf_inputs(device_state, leaf_id) for device_state in state
            ]

        conv = leaves(CONV_LEAF_ID)
        recurrent = leaves(RECURRENT_LEAF_ID)
        ring = leaves(VERIFY_RING_LEAF_ID)
        replay_conv_pools(
            self.kda_captures,
            [leaf.pool for leaf in conv],
            [leaf.live_row_ids for leaf in conv],
            row_indices,
            replay_offsets,
            signal_buffers,
            self.verify_width,
        )
        fold_kda_state_pools(
            [leaf.pool for leaf in recurrent],
            [leaf.live_row_ids for leaf in recurrent],
            [leaf.pool for leaf in ring],
            [leaf.live_row_ids for leaf in ring],
            accepted_lengths(merged_offsets, num_accepted, num_draft_tokens),
            merged_offsets_per_dev,
            signal_buffers,
            self.verify_width,
        )
        replay_tail_seeds(
            self.tail_captures,
            row_indices,
            replay_offsets,
            num_draft_tokens,
            signal_buffers,
        )

    def replay_draft_tail(self, signal_buffers: Sequence[BufferValue]) -> None:
        """Seeds the draft's tail ring with the verify's accepted rows.

        Must run after :meth:`roll_forward` and after the draft's first step.

        Args:
            signal_buffers: Used only to place the plan on each device.
        """
        assert self._plan is not None, "draft tail replay before roll_forward"
        row_indices, replay_offsets, num_draft_tokens = self._plan
        replay_tail_seeds(
            self.draft_tail_captures,
            row_indices,
            replay_offsets,
            num_draft_tokens,
            signal_buffers,
        )
