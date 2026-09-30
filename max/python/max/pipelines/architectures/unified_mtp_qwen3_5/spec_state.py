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
"""Qwen3.5's linear-attention state across a speculative verify.

Verifying a speculative window advances every Gated DeltaNet recurrence in the
target, and no length pointer rewinds one. So the verify runs against a shadow
copy of the recurrent pool and the accepted prefix is replayed into the live one
once the accepted count is known.

The conv pool needs no copy. Its window is the last ``kernel_size - 1`` raw
inputs, so the verify leaves it unwritten and the rollback writes it at the
accepted position.

A window with no drafts has nothing to reject, so its verify writes both live
pools directly and the rollback does nothing. The verify width that decides
this is a shape, which the state ops read at launch without a device sync.

That dance is the same whichever drafter sits behind the target -- it is
parameterized in the accepted length, not in the draft width -- which is why it
lives here rather than in either arch's adapters. The MTP head and the DFlash2
block drafter differ only in what they do with the hidden states the verify
returns, and each keeps that in its own :mod:`spec_adapters`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.kv_cache import (
    KVCacheParamInterface,
    MultiKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
)
from max.nn.state_space import verify_width_operand

from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs
from ..qwen3_5.qwen3_5 import Qwen3_5, Qwen3_5LinearAttentionBlock
from ..qwen3_5.state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    RING_LEAF_ID,
    STATE_CACHE_KEY,
    shadowed_leaf_ids,
)
from .state_rollback import (
    accepted_lengths,
    accepted_row_plan,
    fold_state_pools,
    replay_state_pools,
    shadow_row_ids,
    snapshot_state_pools,
)

__all__ = [
    "LIVE_CONV_POOLS",
    "LIVE_CONV_ROW_IDS",
    "LIVE_RECURRENT_POOLS",
    "LIVE_RECURRENT_ROW_IDS",
    "POSITION_IDS",
    "RING_POOLS",
    "RING_ROW_IDS",
    "SHADOW_RECURRENT_POOLS",
    "Qwen3_5RecurrentState",
    "graph_kv_params",
    "state_tail",
]

LIVE_CONV_POOLS = "live_conv_pools"
LIVE_RECURRENT_POOLS = "live_recurrent_pools"
LIVE_CONV_ROW_IDS = "live_conv_row_ids"
LIVE_RECURRENT_ROW_IDS = "live_recurrent_row_ids"
SHADOW_RECURRENT_POOLS = "shadow_recurrent_pools"
RING_POOLS = "ring_pools"
RING_ROW_IDS = "ring_row_ids"
POSITION_IDS = "position_ids"
"""Keys this architecture's own graph inputs ride under in a driver batch's
``extra``.

The tail is declared by the graph signature and read back by the model, so
these name the one handoff between them that the driver does not understand.
``POSITION_IDS`` is ``None`` unless the target runs M-RoPE.

``RING_POOLS`` and ``RING_ROW_IDS`` hold the ring's pool and rows per device,
or ``None`` on the snapshot rollback. The ring is a scratch leaf of the state
cache, so its rows come from the engine like the live leaves'.
``SHADOW_RECURRENT_POOLS`` is ``None`` on the ring rollback. There is no conv
shadow on either rollback.
"""


def graph_kv_params(kv_params: KVCacheParamInterface) -> MultiKVCacheParams:
    """Returns the cache the fused graph's signature is built from.

    This is the allocated cache without its recurrent state child. The KV
    tree sits in the middle of the spec-decode signature and the state pools
    are declared in the trailing tail, so keeping the child would shift
    every later input.

    Args:
        kv_params: The pipeline's cache, state child included.

    Returns:
        The same tree with the state child dropped.
    """
    assert isinstance(kv_params, MultiKVCacheParams), (
        f"expected MultiKVCacheParams, got {type(kv_params).__name__}"
    )
    return MultiKVCacheParams.from_params(
        {
            key: child
            for key, child in kv_params.children.items()
            if key != STATE_CACHE_KEY
        }
    )


def state_tail(
    trailing: Iterator[Any],
    regions: Sequence[RecurrentStateRegion],
    ring_len: int,
    num_devices: int,
) -> dict[str, Any]:
    """Reads the state tail :meth:`.UnifiedMTPQwen3_5.input_types` declares.

    Consumes exactly the tail and leaves ``trailing`` at the next input.

    Args:
        trailing: The graph's trailing inputs, positioned at the first pool.
        regions: The state regions, in declaration order.
        ring_len: Records one verify-ring row holds, or zero for no ring.
        num_devices: Devices the tail is declared across.

    Returns:
        The ``extra`` entries :class:`Qwen3_5RecurrentState` reads.
    """
    pools = {
        region.leaf_id: [next(trailing).buffer for _ in range(num_devices)]
        for region in regions
    }
    rows = {
        region.leaf_id: [next(trailing).tensor for _ in range(num_devices)]
        for region in regions
    }
    shadows = {
        region.leaf_id: [next(trailing).buffer for _ in range(num_devices)]
        for region in regions
        if region.leaf_id in shadowed_leaf_ids(ring_len)
    }

    return {
        LIVE_CONV_POOLS: pools[CONV_LEAF_ID],
        LIVE_RECURRENT_POOLS: pools[RECURRENT_LEAF_ID],
        LIVE_CONV_ROW_IDS: rows[CONV_LEAF_ID],
        LIVE_RECURRENT_ROW_IDS: rows[RECURRENT_LEAF_ID],
        SHADOW_RECURRENT_POOLS: shadows.get(RECURRENT_LEAF_ID),
        RING_POOLS: pools.get(RING_LEAF_ID),
        RING_ROW_IDS: rows.get(RING_LEAF_ID),
    }


_ShadowState = list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]


class Qwen3_5RecurrentState:
    """The shadow-verify and accepted-prefix replay, for one target.

    Used in three steps, in this order: :meth:`snapshot` before the verify,
    :meth:`capturing` around it, and :meth:`roll_forward` once the accepted
    count is known. Doing them out of order is a silent correctness bug --
    replaying before the count settles rolls the state onto a prefix the row
    never accepted -- so each method says what it depends on.
    """

    def __init__(
        self,
        target: Qwen3_5,
        state_regions: Sequence[RecurrentStateRegion],
        ring_len: int = 0,
    ) -> None:
        self.target = target
        self.regions = {region.leaf_id: region for region in state_regions}
        self.num_layers = len(target.linear_layer_indices)
        self.ring_len = ring_len
        """Records one ring row holds, or zero for snapshot-and-replay."""
        self.captures: list[list[GatedDeltaReplayInputs]] = []
        """Per-device, per-layer state-kernel inputs the verify recorded.

        The replay re-runs the two state kernels over exactly these rather
        than recomputing the projections, so the arithmetic it repeats is the
        arithmetic the verify did. Filled by :meth:`capturing` and read by
        :meth:`roll_forward`.
        """
        self.num_draft_tokens: Dim | None = None
        """The verify's draft width ``K``. Set by :meth:`capturing`."""
        self.verify_width: TensorValue | None = None
        """The verify width operand the forward and the rollback share.

        Both must read the same one: a forward that lands a window the
        rollback then replays or folds advances the state twice. Set by
        :meth:`capturing`.
        """

    def snapshot(
        self, extra: Mapping[str, Any], devices: Sequence[DeviceRef]
    ) -> _ShadowState:
        """Returns the verify's state, copying the recurrent leaf to a shadow.

        Must run before the verify. Only the snapshot rollback copies, and
        only the recurrent leaf. The live leaves come first either way, since
        a verify with no drafts writes them. The shadow or the ring follows.
        The shadow's rows are this graph's own, and the ring's rows are the
        ones the engine supplied.
        """
        num_layers = self.num_layers
        live_recurrent_rows = extra[LIVE_RECURRENT_ROW_IDS]

        def live_leaves(
            i: int,
        ) -> tuple[RecurrentLeafInputs[TensorValue, BufferValue], ...]:
            return (
                RecurrentLeafInputs(
                    pool=extra[LIVE_CONV_POOLS][i],
                    live_row_ids=extra[LIVE_CONV_ROW_IDS][i],
                ),
                RecurrentLeafInputs(
                    pool=extra[LIVE_RECURRENT_POOLS][i],
                    live_row_ids=live_recurrent_rows[i],
                ),
            )

        def verify_leaves(
            i: int,
        ) -> tuple[RecurrentLeafInputs[TensorValue, BufferValue], ...]:
            if self.ring_len:
                return (
                    RecurrentLeafInputs(
                        pool=extra[RING_POOLS][i],
                        live_row_ids=extra[RING_ROW_IDS][i],
                    ),
                )
            return (
                RecurrentLeafInputs(
                    pool=extra[SHADOW_RECURRENT_POOLS][i],
                    live_row_ids=shadow_row_ids(num_layers, devices[i]),
                ),
            )

        if not self.ring_len:
            snapshot_state_pools(
                extra[LIVE_RECURRENT_POOLS],
                extra[SHADOW_RECURRENT_POOLS],
                live_recurrent_rows,
                ops.shape_to_tensor([live_recurrent_rows[0].shape[1]])[0]
                * num_layers,
            )
        return [
            RecurrentStateInputsPerDevice(
                leaves=(*live_leaves(i), *verify_leaves(i)),
            )
            for i in range(len(devices))
        ]

    @contextmanager
    def capturing(self, n_devs: int, num_draft_tokens: Dim) -> Iterator[None]:
        """Binds the linear blocks' replay capture for the verify inside.

        The binding is unwound on the way out so the capture cannot leak into
        a later phase -- the draft's own forward would otherwise append to it
        and the replay would repeat arithmetic the verify never did.

        Args:
            n_devs: Devices the target is sharded across.
            num_draft_tokens: The verify's draft width ``K``.
        """
        self.captures = [[] for _ in range(n_devs)]
        self.num_draft_tokens = num_draft_tokens
        self.verify_width = verify_width_operand(num_draft_tokens)
        self._bind(self.captures, self.verify_width)
        try:
            yield
        finally:
            self._bind(None, None)

    def _bind(
        self,
        capture: list[list[GatedDeltaReplayInputs]] | None,
        verify_width: TensorValue | None,
    ) -> None:
        for layer_idx in self.target.linear_layer_indices:
            block = self.target.layers[layer_idx]
            assert isinstance(block, Qwen3_5LinearAttentionBlock)
            block.replay_capture = capture
            block.verify_width = verify_width
            block.verify_ring = bool(self.ring_len)

    def roll_forward(
        self,
        extra: Mapping[str, Any],
        *,
        merged_offsets: TensorValue,
        num_accepted: TensorValue,
        num_draft_tokens: TensorValue,
        total_rows: Dim,
        signal_buffers: Sequence[BufferValue],
        device: DeviceRef,
    ) -> None:
        """Replays the accepted prefix into the live pools.

        Must run after the accepted count has been corrected for rows that
        carried no proposal, and before anything downstream reads the live
        pools.

        This also writes the conv window, which the verify left unwritten.
        A verify width of zero launches nothing, because the verify's forward
        already landed both leaves.

        Args:
            extra: The batch's model-owned inputs, holding the state tail.
            merged_offsets: Ragged offsets over the verified window.
            num_accepted: ``[batch]`` accepted draft tokens per request.
            num_draft_tokens: This step's draft width, as a scalar on
                ``device``, zero on a prefill.
            total_rows: Row count of the verified window.
            signal_buffers: Used only to place the plan on each device.
            device: The device the batch-wide tensors live on.
        """
        assert self.verify_width is not None, "roll_forward before capturing"
        assert self.num_draft_tokens is not None
        row_indices, replay_offsets = accepted_row_plan(
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            total_rows,
            device,
            # Each row is one token and its drafts whenever the rollback
            # runs, and the plan shrinks to a row per request when it does
            # not.
            plan_rows=Dim("batch_size") * (1 + self.num_draft_tokens),
        )
        replay_state_pools(
            self.captures,
            extra[LIVE_CONV_POOLS],
            None if self.ring_len else extra[LIVE_RECURRENT_POOLS],
            extra[LIVE_CONV_ROW_IDS],
            None if self.ring_len else extra[LIVE_RECURRENT_ROW_IDS],
            row_indices,
            replay_offsets,
            signal_buffers,
            self.verify_width,
        )
        if not self.ring_len:
            return

        fold_state_pools(
            extra[LIVE_RECURRENT_POOLS],
            extra[LIVE_RECURRENT_ROW_IDS],
            extra[RING_POOLS],
            extra[RING_ROW_IDS],
            accepted_lengths(merged_offsets, num_accepted, num_draft_tokens),
            signal_buffers,
            self.verify_width,
        )
