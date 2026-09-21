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
copy of the state pools and the accepted prefix is replayed into the live ones
once the accepted count is known.

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
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
)

from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs
from ..qwen3_5.qwen3_5 import Qwen3_5, Qwen3_5LinearAttentionBlock
from .state_rollback import (
    accepted_row_plan,
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
    "SHADOW_CONV_POOLS",
    "SHADOW_RECURRENT_POOLS",
    "Qwen3_5RecurrentState",
]

LIVE_CONV_POOLS = "live_conv_pools"
LIVE_RECURRENT_POOLS = "live_recurrent_pools"
LIVE_CONV_ROW_IDS = "live_conv_row_ids"
LIVE_RECURRENT_ROW_IDS = "live_recurrent_row_ids"
SHADOW_CONV_POOLS = "shadow_conv_pools"
SHADOW_RECURRENT_POOLS = "shadow_recurrent_pools"
POSITION_IDS = "position_ids"
"""Keys this architecture's own graph inputs ride under in a driver batch's
``extra``.

The tail is declared by the graph signature and read back by the model, so
these name the one handoff between them that the driver does not understand.
``POSITION_IDS`` is ``None`` unless the target runs M-RoPE.
"""

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
    ) -> None:
        self.target = target
        self.regions = {region.leaf_id: region for region in state_regions}
        self.num_layers = len(target.linear_layer_indices)
        self.captures: list[list[GatedDeltaReplayInputs]] = []
        """Per-device, per-layer state-kernel inputs the verify recorded.

        The replay re-runs the two state kernels over exactly these rather
        than recomputing the projections, so the arithmetic it repeats is the
        arithmetic the verify did. Filled by :meth:`capturing` and read by
        :meth:`roll_forward`.
        """

    def snapshot(
        self, extra: Mapping[str, Any], devices: Sequence[DeviceRef]
    ) -> _ShadowState:
        """Copies the live pools into the shadow and returns the shadow state.

        Must run before the verify: afterwards the live pools no longer hold
        the pre-verify state the replay starts from.
        """
        conv_row_ids = extra[LIVE_CONV_ROW_IDS]
        shadow_conv = extra[SHADOW_CONV_POOLS]
        shadow_recurrent = extra[SHADOW_RECURRENT_POOLS]

        num_layers = self.num_layers
        batch_dim = conv_row_ids[0].shape[1]
        shadow_span = ops.shape_to_tensor([batch_dim])[0] * num_layers
        snapshot_state_pools(
            extra[LIVE_CONV_POOLS], shadow_conv, conv_row_ids, shadow_span
        )
        snapshot_state_pools(
            extra[LIVE_RECURRENT_POOLS],
            shadow_recurrent,
            extra[LIVE_RECURRENT_ROW_IDS],
            shadow_span,
        )

        # The shadow's layout is this graph's own, so its rows are built here.
        def shadow_leaf(
            pool: BufferValue, device: DeviceRef
        ) -> RecurrentLeafInputs[TensorValue, BufferValue]:
            return RecurrentLeafInputs(
                pool=pool,
                # The snapshot already put the pre-verify state here; the
                # verify reads and writes it in place.
                live_row_ids=shadow_row_ids(num_layers, device),
            )

        return [
            RecurrentStateInputsPerDevice(
                leaves=(
                    shadow_leaf(shadow_conv[i], devices[i]),
                    shadow_leaf(shadow_recurrent[i], devices[i]),
                ),
            )
            for i in range(len(devices))
        ]

    @contextmanager
    def capturing(self, n_devs: int) -> Iterator[None]:
        """Binds the linear blocks' replay capture for the verify inside.

        The binding is unwound on the way out so the capture cannot leak into
        a later phase -- the draft's own forward would otherwise append to it
        and the replay would repeat arithmetic the verify never did.
        """
        self.captures = [[] for _ in range(n_devs)]
        self._bind(self.captures)
        try:
            yield
        finally:
            self._bind(None)

    def _bind(self, capture: list[list[GatedDeltaReplayInputs]] | None) -> None:
        for layer_idx in self.target.linear_layer_indices:
            block = self.target.layers[layer_idx]
            assert isinstance(block, Qwen3_5LinearAttentionBlock)
            block.replay_capture = capture

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

        Args:
            extra: The batch's model-owned inputs, holding the state tail.
            merged_offsets: Ragged offsets over the verified window.
            num_accepted: ``[batch]`` accepted draft tokens per request.
            num_draft_tokens: This step's draft width, as a scalar on
                ``device``; zero on a prefill, where the plan then covers the
                whole prompt with no phase branch.
            total_rows: Row count of the verified window.
            signal_buffers: Used only to place the plan on each device.
            device: The device the batch-wide tensors live on.
        """
        row_indices, replay_offsets = accepted_row_plan(
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            total_rows,
            device,
        )
        replay_state_pools(
            self.captures,
            extra[LIVE_CONV_POOLS],
            extra[LIVE_RECURRENT_POOLS],
            extra[LIVE_CONV_ROW_IDS],
            extra[LIVE_RECURRENT_ROW_IDS],
            row_indices,
            replay_offsets,
            signal_buffers,
        )
