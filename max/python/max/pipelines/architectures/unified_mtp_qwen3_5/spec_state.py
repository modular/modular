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
target, and no length pointer rewinds one. So the verify reads the live
recurrent pool without writing it, records each token's update in a ring, and
the accepted records are folded into the live pool once the accepted count is
known.

The conv pool needs no ring. Its window is the last ``kernel_size - 1`` raw
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

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    TensorType,
    TensorValue,
)
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
)
from .state_rollback import (
    accepted_lengths,
    accepted_row_plan,
    fold_state_pools,
    replay_conv_pools,
)

__all__ = [
    "LIVE_CONV_POOLS",
    "LIVE_CONV_ROW_IDS",
    "LIVE_RECURRENT_POOLS",
    "LIVE_RECURRENT_ROW_IDS",
    "POSITION_IDS",
    "RING_POOLS",
    "RING_ROW_IDS",
    "Qwen3_5RecurrentState",
    "graph_kv_params",
    "state_tail",
    "state_tail_types",
]

LIVE_CONV_POOLS = "live_conv_pools"
LIVE_RECURRENT_POOLS = "live_recurrent_pools"
LIVE_CONV_ROW_IDS = "live_conv_row_ids"
LIVE_RECURRENT_ROW_IDS = "live_recurrent_row_ids"
RING_POOLS = "ring_pools"
RING_ROW_IDS = "ring_row_ids"
POSITION_IDS = "position_ids"
"""Keys this architecture's own graph inputs ride under in a driver batch's
``extra``.

The tail is declared by the graph signature and read back by the model, so
these name the one handoff between them that the driver does not understand.
``POSITION_IDS`` is ``None`` unless the target runs M-RoPE.

The ring is a scratch leaf of the state cache, so its rows come from the
engine like the live leaves'.
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


def state_tail_types(
    regions: Sequence[RecurrentStateRegion], devices: Sequence[DeviceRef]
) -> list[TensorType | BufferType]:
    """Returns the state tail a fused Qwen3.5 speculative graph declares.

    A pool per region, then the ``[num_layers, batch_size]`` rows each layer
    of each request occupies in it, every block region-major and
    device-minor. One buffer per leaf rather than one per layer, so the
    caller picks the row layout.

    Args:
        regions: The state regions, the ring included, in declaration order.
        devices: Devices the target is sharded across.
    """
    tail: list[TensorType | BufferType] = []
    for region in regions:
        tail.extend(
            BufferType(
                region.dtype,
                shape=[region.rows_dim, *region.row_shape],
                device=device,
            )
            for device in devices
        )
    for region in regions:
        tail.extend(
            TensorType(
                DType.uint32,
                shape=[region.num_layers, "batch_size"],
                device=device,
            )
            for device in devices
        )
    return tail


def state_tail(
    trailing: Iterator[Any],
    regions: Sequence[RecurrentStateRegion],
    num_devices: int,
) -> dict[str, Any]:
    """Reads the state tail :func:`state_tail_types` declares.

    Consumes exactly the tail and leaves ``trailing`` at the next input.

    Args:
        trailing: The graph's trailing inputs, positioned at the first pool.
        regions: The state regions, in declaration order.
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
    return {
        LIVE_CONV_POOLS: pools[CONV_LEAF_ID],
        LIVE_RECURRENT_POOLS: pools[RECURRENT_LEAF_ID],
        LIVE_CONV_ROW_IDS: rows[CONV_LEAF_ID],
        LIVE_RECURRENT_ROW_IDS: rows[RECURRENT_LEAF_ID],
        RING_POOLS: pools[RING_LEAF_ID],
        RING_ROW_IDS: rows[RING_LEAF_ID],
    }


class Qwen3_5RecurrentState:
    """The ring verify and the accepted-prefix rollback, for one target.

    Used in three steps, in this order: :meth:`verify_state` for the verify's
    state inputs, :meth:`capturing` around the verify, and
    :meth:`roll_forward` once the accepted count is known. Rolling forward
    before the count settles folds the state onto a prefix the row never
    accepted, a silent correctness bug, so each method says what it depends
    on.
    """

    def __init__(
        self,
        target: Qwen3_5,
        state_regions: Sequence[RecurrentStateRegion],
    ) -> None:
        if RING_LEAF_ID not in {region.leaf_id for region in state_regions}:
            raise ValueError(
                "a speculative Qwen3.5 verify needs the ring leaf among its"
                " state regions"
            )
        self.target = target
        self.captures: list[list[GatedDeltaReplayInputs]] = []
        """Per-device, per-layer state-kernel inputs the verify recorded.

        The conv replay re-runs the conv kernel over exactly these rather
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

    @staticmethod
    def verify_state(
        extra: Mapping[str, Any], n_devs: int
    ) -> list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]:
        """Returns the verify's state inputs: the live leaves, then the ring.

        The live leaves come first, since a verify with no drafts writes
        them.
        """

        def leaf(
            pools: str, rows: str, i: int
        ) -> RecurrentLeafInputs[TensorValue, BufferValue]:
            return RecurrentLeafInputs(
                pool=extra[pools][i], live_row_ids=extra[rows][i]
            )

        return [
            RecurrentStateInputsPerDevice(
                leaves=(
                    leaf(LIVE_CONV_POOLS, LIVE_CONV_ROW_IDS, i),
                    leaf(LIVE_RECURRENT_POOLS, LIVE_RECURRENT_ROW_IDS, i),
                    leaf(RING_POOLS, RING_ROW_IDS, i),
                ),
            )
            for i in range(n_devs)
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
        """Lands the live pools on the accepted prefix.

        Must run after the accepted count has been corrected for rows that
        carried no proposal, and before anything downstream reads the live
        pools.

        Folds the ring's accepted records into the recurrent pool and writes
        the conv window, which the verify left unwritten. A verify width of
        zero launches nothing, because the verify's forward already landed
        both leaves.

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
        replay_conv_pools(
            self.captures,
            extra[LIVE_CONV_POOLS],
            extra[LIVE_CONV_ROW_IDS],
            row_indices,
            replay_offsets,
            signal_buffers,
            self.verify_width,
        )
        fold_state_pools(
            extra[LIVE_RECURRENT_POOLS],
            extra[LIVE_RECURRENT_ROW_IDS],
            extra[RING_POOLS],
            extra[RING_ROW_IDS],
            accepted_lengths(merged_offsets, num_accepted, num_draft_tokens),
            signal_buffers,
            self.verify_width,
        )
