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
"""Kimi K3's KDA state across a speculative verify (ModuleV3).

Translation of ``glm5_next/spec_state.py``'s KDA half onto this arch's
Tensor-API layer, which one call runs over every device's shards; the
per-device ring tensors travel as a ``PerShard`` the same dispatch splits.

Two state kinds advance with every verified token, and neither has a length
pointer to rewind:

- **KDA recurrent.** The verify reads the live pool without writing it and
  records each token's update in the verify ring, a scratch leaf of the
  state cache. The accepted records are folded into the live pool once the
  accepted count is known, with ``kda_state_fold`` -- whose record count is
  the TOKENS to fold, anchor included, not the accepted drafts
  (:func:`~max.pipelines.speculative.gated_delta_rollback.ring_accepted_lengths`
  converts). A fold is the recurrence's own update over the same operands,
  so it lands bit for bit where the pinned-``fma`` ``kda_decode`` the base
  graph runs would have.
- **KDA conv.** The verify leaves the window unwritten and the rollback
  places it from the copy the verify keeps of its input, inside the same
  ``kda_state_fold`` launch that folds the recurrent pool -- a copy of the
  last ``kernel_size - 1`` raw inputs, no conv re-run.

K3 has no indexer tail ring to seed; that leaf is GLM-5.3-Flash's sparse-MLA
business. A window with no drafts has nothing to reject, so its verify lands
both pools directly and every rollback launch does nothing. The verify width
that decides this is a shape, which the ops read without a device sync.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn._graph_ops.state_space import verify_width_operand
from max.nn._graph_ops.state_space.kimi_delta import (
    STATE_LAYOUT,
    kda_state_fold,
)
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.pipelines.speculative.gated_delta_rollback import (
    ring_accepted_lengths,
)
from max.pipelines.speculative.ragged_token_merger import shape_to_scalar
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device
from max.tensor import Tensor
from max.tensor.sharding.action import PerShard

from .layers.kda_attention import (
    KdaConvReplayInputs,
    KdaVerifyBinding,
    KimiK3KDAAttention,
)
from .model import KimiK3Model
from .state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    LeafBuffer,
    LeafValue,
    layer_row,
    leaf_inputs,
    verify_ring_leaf,
)

__all__ = [
    "KimiK3RecurrentState",
    "fold_kda_state_pools",
]


def _pool_value(pool: BufferValue | Tensor) -> BufferValue:
    """Returns a pool's graph buffer, promoting an eager tensor's backing."""
    return pool if isinstance(pool, BufferValue) else BufferValue(pool)


def _value(value: TensorValue | Tensor) -> TensorValue:
    """Returns a value's graph value, binding an eager tensor's data."""
    return value if isinstance(value, TensorValue) else TensorValue(value)


def _ring_leaf_value(
    value: BufferValue | TensorValue | Tensor,
) -> Tensor:
    """Wraps one device's ring leaf value as the layer's Tensor API takes it.

    A graph build's leaf holds a buffer value, which becomes an unrealized
    tensor around it; an eager leaf already holds a realized tensor whose
    buffer writes must reach.
    """
    if isinstance(value, Tensor):
        return value
    return Tensor.from_graph_value(value)


def fold_kda_state_pools(
    live_recurrent_pools: Sequence[BufferValue | Tensor],
    recurrent_row_ids: Sequence[TensorValue | Tensor],
    ring_pools: Sequence[BufferValue | Tensor],
    ring_row_ids: Sequence[TensorValue | Tensor],
    fold_accepted: TensorValue,
    row_offsets: Sequence[TensorValue | Tensor],
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
    *,
    live_conv_pools: Sequence[BufferValue | Tensor] | None = None,
    conv_row_ids: Sequence[TensorValue | Tensor] | None = None,
    kda_captures: Sequence[Sequence[KdaConvReplayInputs]] | None = None,
    window_row_len: TensorValue | None = None,
) -> None:
    """Applies the verify's accepted records to the live KDA pools.

    One launch per device folds the ring into the recurrent pool and places
    each request's conv window at its accepted length from the captured
    windows, and none runs at a verify width of zero.

    Args:
        live_recurrent_pools: Per-device recurrent pool, still pre-verify.
        recurrent_row_ids: Per-device ``[num_layers, batch_size]`` state rows.
        ring_pools: Per-device verify ring.
        ring_row_ids: Per-device ``[num_layers, batch_size]`` ring rows.
        live_conv_pools: Per-device conv pool, still pre-verify.
        conv_row_ids: Per-device ``[num_layers, batch_size]`` conv rows.
        kda_captures: Per-device, per-layer verify captures; the fold reads
            each layer's ``qkv_window``.
        window_row_len: ``[1]`` int32 scalar: the capture's rows per request,
            the stride the conv kernel wrote each row's tokens at (``K + 1``).
            Zero reads each request's window at its own offset in
            ``row_offsets``; omitted, the caller keeps that layout.
        fold_accepted: ``[batch]`` records to fold -- tokens, anchor
            included; see the module docstring.
        row_offsets: Per-device ``[batch + 1]`` offsets of the verified
            window, which bound each row's fold.
        signal_buffers: Used only to place ``fold_accepted`` on each device.
        verify_width: The verify's width operand.
    """
    window_row_len_per_dev = (
        None
        if window_row_len is None
        else broadcast_per_device(
            ops.reshape(window_row_len, [1]),
            list(signal_buffers),
            len(live_recurrent_pools),
        )
    )
    accepted_per_dev = broadcast_per_device(
        fold_accepted.cast(DType.uint32),
        list(signal_buffers),
        len(live_recurrent_pools),
    )
    for device_idx, state_pool in enumerate(live_recurrent_pools):
        fold_args = (
            _pool_value(state_pool),
            _pool_value(ring_pools[device_idx]),
            _value(recurrent_row_ids[device_idx]),
            _value(ring_row_ids[device_idx]),
            accepted_per_dev[device_idx],
            _value(row_offsets[device_idx]),
            verify_width,
        )
        if (
            live_conv_pools is not None
            and conv_row_ids is not None
            and kda_captures is not None
            and window_row_len_per_dev is not None
        ):
            kda_state_fold(
                *fold_args,
                conv_pool=_pool_value(live_conv_pools[device_idx]),
                conv_row_ids=_value(conv_row_ids[device_idx]),
                qkv_windows=[
                    capture.qkv_window for capture in kda_captures[device_idx]
                ],
                window_row_len=window_row_len_per_dev[device_idx],
                state_layout=STATE_LAYOUT,
            )
        else:
            kda_state_fold(*fold_args, state_layout=STATE_LAYOUT)


class KimiK3RecurrentState:
    """The verify's state capture and the accepted-prefix rollback.

    Used in this order: :meth:`capturing` around the target's verify, then
    :meth:`roll_forward` once the accepted count is known. Rolling forward
    before the count settles lands the state on a prefix the row never
    accepted, silently; :meth:`roll_forward` also spends the capture,
    clearing it, so a later build that skips :meth:`capturing` fails the
    contract asserts instead of replaying another graph's values.

    The KDA layers read the bound ring through the
    :class:`~.layers.kda_attention.KdaVerifyBinding` this sets on each of
    them, unwound on the way out so a later forward -- the draft's, or the
    next step's -- cannot append to the capture.
    """

    def __init__(self, kda_layers: Sequence[KimiK3KDAAttention]) -> None:
        if not kda_layers:
            raise ValueError("a Kimi K3 verify needs at least one KDA layer")
        self.kda_layers = list(kda_layers)
        self.captures: list[list[KdaConvReplayInputs]] = []
        """Per-device, per-KDA-layer conv inputs the verify recorded.

        Filled by :meth:`capturing`, in layer order, and read by
        :meth:`roll_forward`."""
        self.num_draft_tokens: Dim | None = None
        """The verify's draft width ``K``. Set by :meth:`capturing`."""
        self.verify_width: TensorValue | None = None
        """The verify width operand the forward and the rollback share.

        Both must read the same one: a forward that lands a window the
        rollback then replays or folds advances the state twice."""

    @classmethod
    def from_model(cls, model: KimiK3Model) -> KimiK3RecurrentState:
        """Builds the rollback state of a target model's KDA layers."""
        return cls(
            [
                layer.attention
                for layer in model.layers
                if isinstance(layer.attention, KimiK3KDAAttention)
            ]
        )

    @contextmanager
    def capturing(
        self,
        num_draft_tokens: Dim,
        state: Sequence[RecurrentStateInputsPerDevice[LeafValue, LeafBuffer]],
    ) -> Iterator[None]:
        """Binds the KDA layers' verify ring and capture for the verify inside.

        Unwound on the way out, so the draft's own forward cannot append to
        the capture.

        Args:
            num_draft_tokens: The verify's draft width ``K``.
            state: Per-device state inputs, leaves in declaration order,
                the verify ring declared last.

        Raises:
            AssertionError: If the cache declares no ring, or its ring's
                layer count disagrees with the KDA layers bound here.
        """
        # The ring is told by its position and shape, not by a name the
        # leaves carry: binding refuses a cache that declares none and a
        # third leaf that is not shaped like one.
        ring = [verify_ring_leaf(device_state) for device_state in state]
        num_layers = int(ring[0].live_row_ids.shape[0])
        assert len(self.kda_layers) == num_layers, (
            "the verify ring holds"
            f" {num_layers} layers' rows but this state binds"
            f" {len(self.kda_layers)} KDA layers"
        )
        self.captures = [[] for _ in state]
        self.num_draft_tokens = num_draft_tokens
        self.verify_width = verify_width_operand(num_draft_tokens)
        ring_pool = PerShard(_ring_leaf_value(leaf.pool) for leaf in ring)
        try:
            for position, layer in enumerate(self.kda_layers):
                layer.verify_binding = KdaVerifyBinding(
                    verify_width=self.verify_width,
                    ring_pool=ring_pool,
                    ring_row_id=PerShard(
                        _ring_leaf_value(layer_row(leaf, position))
                        for leaf in ring
                    ),
                    captures=self.captures,
                )
            yield
        finally:
            for layer in self.kda_layers:
                layer.verify_binding = None

    def roll_forward(
        self,
        state: Sequence[RecurrentStateInputsPerDevice[LeafValue, LeafBuffer]],
        *,
        merged_offsets: TensorValue | Tensor,
        merged_offsets_per_dev: Sequence[TensorValue | Tensor],
        num_accepted: TensorValue | Tensor,
        signal_buffers: Sequence[BufferValue],
        device: DeviceRef,
    ) -> None:
        """Lands the KDA state leaves on the accepted prefix.

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
        assert self.captures and self.captures[0], (
            "roll_forward before capturing, or over a verify that captured"
            " nothing"
        )
        merged_offsets = _value(merged_offsets)
        num_accepted = _value(num_accepted)
        drafts = self.num_draft_tokens
        assert drafts is not None
        num_draft_tokens = shape_to_scalar(drafts, device)
        # A mixed verify's prompt rows were committed by the verify itself,
        # so the accept counts below zero them: no record is folded and no
        # window is re-placed for such a row.
        accepted = ring_accepted_lengths(
            ops.rebind(merged_offsets.cast(DType.int64)[1:], ["batch_size"])
            - ops.rebind(merged_offsets.cast(DType.int64)[:-1], ["batch_size"]),
            num_accepted,
            num_draft_tokens,
        )
        conv = [
            leaf_inputs(device_state, CONV_LEAF_ID) for device_state in state
        ]
        recurrent = [
            leaf_inputs(device_state, RECURRENT_LEAF_ID)
            for device_state in state
        ]
        ring = [verify_ring_leaf(device_state) for device_state in state]
        # The fold places each layer's conv window from the captured copy in
        # the same launch that folds the ring, so no per-layer conv replay
        # runs: the window is the last kernel_size - 1 raw inputs, a copy of
        # at most that many rows of the capture, read at the capture's own
        # row stride -- one verify window per request -- so a mixed verify's
        # wider prompt row lands its decode neighbours in the right slot.
        #
        fold_args = (
            [_pool_value(leaf.pool) for leaf in recurrent],
            [_value(leaf.live_row_ids) for leaf in recurrent],
            [_pool_value(leaf.pool) for leaf in ring],
            [_value(leaf.live_row_ids) for leaf in ring],
            accepted,
            [_value(offsets) for offsets in merged_offsets_per_dev],
            signal_buffers,
            self.verify_width,
        )
        # The fold places the windows in the same launch. The stride is the
        # draft width plus one, as a runtime scalar: a served graph declares
        # that width symbolically, so it is the one form that works whether
        # the extent is static or not.
        fold_kda_state_pools(
            *fold_args,
            live_conv_pools=[_pool_value(leaf.pool) for leaf in conv],
            conv_row_ids=[_value(leaf.live_row_ids) for leaf in conv],
            kda_captures=self.captures,
            window_row_len=shape_to_scalar(drafts + 1, device, DType.int32),
        )
        # The rollback spends the capture; clearing it keeps a later build
        # that skips `capturing` on the contract asserts above.
        self.captures = []
        self.verify_width = None
        self.num_draft_tokens = None
