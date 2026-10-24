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
"""Qwen3.5's call, and its linear-attention state across a speculative verify.

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

That dance is the same whichever draft sits behind the target: it is
parameterized in the accepted length, not in the draft width. The graph
declares the state past the canonical tail, where the engines that bind the
Qwen3.5 speculative graphs slot by slot find it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from max.driver import Buffer, Device
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
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
    recurrent_leaf,
)
from max.nn.state_space import GatedDeltaReplayInputs, verify_width_operand
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.context import TextContext
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.gated_delta_rollback import (
    accepted_lengths,
    accepted_row_plan,
    fold_state_pools,
    replay_conv_pools,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
    verify_lengths,
)
from max.pipelines.speculative.spec_target import DistributedVerify, Verified

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = [
    "LIVE_CONV_POOLS",
    "LIVE_CONV_ROW_IDS",
    "LIVE_RECURRENT_POOLS",
    "LIVE_RECURRENT_ROW_IDS",
    "POSITION_IDS",
    "RING_POOLS",
    "RING_ROW_IDS",
    "Qwen3_5RecurrentState",
    "state_tail",
    "state_tail_types",
]

PRIMARY_KV = "attn"

TAIL_KV = "state"
"""The target's recurrent state, which the graph declares in its tail."""


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
``POSITION_IDS`` is set only for an MTP head over a target that runs M-RoPE.

The ring is a scratch leaf of the state cache, so its rows come from the
engine like the live leaves'.
"""


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
    # ``linear_state_regions`` declares the conv leaf, the recurrent leaf,
    # then the ring.
    pools = [
        [next(trailing).buffer for _ in range(num_devices)] for _ in regions
    ]
    rows = [
        [next(trailing).tensor for _ in range(num_devices)] for _ in regions
    ]
    return {
        LIVE_CONV_POOLS: pools[0],
        LIVE_RECURRENT_POOLS: pools[1],
        LIVE_CONV_ROW_IDS: rows[0],
        LIVE_RECURRENT_ROW_IDS: rows[1],
        RING_POOLS: pools[2],
        RING_ROW_IDS: rows[2],
    }


@runtime_checkable
class _ReplayCapturing(Protocol):
    """A linear-attention block that records its state kernels' inputs."""

    replay_capture: list[list[GatedDeltaReplayInputs]] | None
    verify_width: TensorValue | None


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
        target: Any,
        state_regions: Sequence[RecurrentStateRegion],
    ) -> None:
        if not any(region.scratch for region in state_regions):
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
            assert isinstance(block, _ReplayCapturing)
            block.replay_capture = capture
            block.verify_width = verify_width

    def roll_forward(
        self,
        extra: Mapping[str, Any],
        *,
        merged_offsets: TensorValue,
        num_accepted: TensorValue,
        num_draft_tokens: TensorValue,
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
            signal_buffers: Used only to place the plan on each device.
            device: The device the batch-wide tensors live on.
        """
        assert self.verify_width is not None, "roll_forward before capturing"
        assert self.num_draft_tokens is not None
        window = self.captures[0][0].qkv_window
        assert window is not None
        # The verify copied its rows whenever it had drafts, which is
        # whenever the rollback runs. Clamping to the copy keeps the gather in
        # bounds when it does not.
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


def _regions(target_config: Any) -> tuple[RecurrentStateRegion, ...]:
    state = recurrent_leaf(target_config.kv_params)
    assert state is not None, "a Qwen3.5 target keeps a recurrent state"
    return tuple(state.regions)


def trailing_input_types(
    target_config: Any,
) -> list[TensorType | BufferType]:
    """Returns the state tail, the verify ring included."""
    return state_tail_types(_regions(target_config), target_config.devices)


def extra(target_config: Any, trailing: Sequence[Any]) -> dict[str, Any]:
    """Names the state tail's pools and rows for the verify's state."""
    return state_tail(
        iter(trailing), _regions(target_config), len(target_config.devices)
    )


def tail_buffers(
    state: Sequence[RecurrentStateInputsPerDevice[Buffer, Buffer]],
) -> list[Buffer]:
    """Returns the cache's state inputs in the state tail's order.

    That is each leaf's pools, then each leaf's rows, device-minor.
    """
    leaves = range(len(state[0].leaves))
    return [
        *(device.leaves[i].pool for i in leaves for device in state),
        *(device.leaves[i].live_row_ids for i in leaves for device in state),
    ]


def _mtp_trailing_input_types(
    target_config: Any,
) -> list[TensorType | BufferType]:
    """Returns the state tail.

    Then the merged window's 3-axis positions for a target that runs M-RoPE,
    shared across devices.
    """
    types = trailing_input_types(target_config)
    if target_config.mrope_enabled:
        types.append(
            TensorType(
                DType.int64,
                shape=[3, "merged_total_seq_len"],
                device=target_config.devices[0],
            )
        )
    return types


def _mtp_extra(target_config: Any, trailing: Sequence[Any]) -> dict[str, Any]:
    """Names the state tail, and the positions that follow it."""
    rest = iter(trailing)
    state = state_tail(
        rest, _regions(target_config), len(target_config.devices)
    )
    if target_config.mrope_enabled:
        state[POSITION_IDS] = next(rest).tensor
    return state


def _mtp_trailing_buffers(
    target_config: Any,
    contexts: Sequence[TextContext],
    devices: Sequence[Device],
) -> list[Buffer]:
    """Returns the merged window's positions, for a target that runs M-RoPE.

    A text-only pairing holds no image, so each request's positions count on
    from its processed length on all three axes, through its drafts.
    """
    if not target_config.mrope_enabled:
        return []
    positions = np.concatenate(
        [
            ctx.tokens.processed_length + np.arange(length, dtype=np.int64)
            for ctx, length in zip(
                contexts, verify_lengths(contexts), strict=True
            )
        ]
    )
    return [Buffer.from_numpy(np.tile(positions, (3, 1))).to(devices[0])]


def _mtp_takes_vision_outputs(target_config: Any) -> bool:
    """M-RoPE marks a checkpoint whose prompts can hold images."""
    return target_config.mrope_enabled


def check(speculative: SpeculativeConfig, structured_output: bool) -> None:
    """Raises if a Qwen3.5 target can't verify under the speculative config.

    Raises:
        ValueError: If acceptance is greedy or synthetic, or the graph takes
            no constrained-decoding bitmask.
    """
    if speculative.use_greedy_acceptance:
        raise ValueError(
            "Speculative Qwen3.5 requires stochastic acceptance: the greedy"
            " path ignores token_bitmasks, and this checkpoint's lm_head"
            " padding rows are only excluded through the bitmask."
        )
    if speculative.synthetic_acceptance_rate is not None:
        raise ValueError(
            "synthetic acceptance would bypass the state rollback's"
            " accepted-length plan"
        )
    if not structured_output:
        raise ValueError(
            "Speculative Qwen3.5 needs the constrained-decoding bitmask input:"
            " this checkpoint's lm_head has live padding rows past"
            " sampleable_vocab_size and the in-graph acceptance sampler"
            " excludes them only through that mask. Exporting a MEF with"
            " mach/tools/gen-mef turns it on by default and"
            " --no-sampler-grammar turns it off; elsewhere it follows"
            " --enable-structured-output (or a tool parser that implies"
            " it)."
        )


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a Qwen3.5 model.

    It records its state's updates in the verify ring and rolls the state onto
    the accepted prefix.

    Raises:
        ValueError: If the target wasn't built to return the selected
            layers a block draft reads, or the last normalized layer the
            MTP head reads.
    """
    del draft_config
    if hidden_states not in (
        ReturnHiddenStates.SELECTED_LAYERS,
        ReturnHiddenStates.ALL_NORMALIZED,
    ):
        raise ValueError(
            "A Qwen3.5 target verifies only for drafts that read its"
            " selected layers or its last normalized one. Got"
            f" {hidden_states.value}."
        )
    # The rollback reads each linear layer's state-kernel inputs out of the
    # verify pass, and those captures cannot cross a subgraph boundary.
    target_config.use_subgraphs = False
    # Mach's slot maps for both Qwen3.5 graphs carry the distributed
    # inputs, which this verify doesn't read.
    return DistributedVerify(
        _verify, Qwen3_5RecurrentState(target, _regions(target_config))
    )


def _verify(
    state: Qwen3_5RecurrentState, batch: SequentialBatch | BlockBatch
) -> Verified[list[TensorValue]]:
    verify_state = state.verify_state(batch.extra, batch.n_devs)
    with state.capturing(batch.n_devs, batch.num_draft_tokens):
        # The positions cover the merged window. ``None`` keeps the target on
        # its static rope table. Images arrive only in a K = 0 step, which
        # merges no drafts, so the scatter indices into the step's tokens
        # need no remapping.
        outputs = state.target(
            batch.merged_tokens,
            batch.kv_collections,
            batch.return_n_logits,
            batch.merged_offsets,
            batch.signal_buffers,
            verify_state,
            position_ids=batch.extra.get(POSITION_IDS),
            image_embeddings=batch.vision_embeddings or None,
            image_token_indices=batch.vision_scatter_indices or None,
        )
    roll_forward = partial(_roll_forward, state)
    if isinstance(batch, SequentialBatch):
        return Verified(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
            roll_forward=roll_forward,
        )
    return Verified(
        logits=outputs.logits["logits"],
        hidden=fuse_captured_hidden_states(
            captures_by_device(
                list(outputs.hidden_states.tensors()), batch.n_devs
            )
        ),
        roll_forward=roll_forward,
    )


def _roll_forward(
    state: Qwen3_5RecurrentState, batch: SequentialBatch | BlockBatch
) -> None:
    assert batch.num_accepted is not None
    state.roll_forward(
        batch.extra,
        merged_offsets=batch.merged_offsets,
        num_accepted=batch.num_accepted,
        num_draft_tokens=_shape_to_scalar(
            batch.num_draft_tokens, batch.device0
        ),
        signal_buffers=batch.signal_buffers,
        device=batch.device0,
    )


_BASE = TargetAdapter(
    primary_kv=PRIMARY_KV,
    tail_kv=TAIL_KV,
    verify=verify,
    tail_buffers=tail_buffers,
    check=check,
    trailing_input_types=trailing_input_types,
    extra=extra,
)
# The MTP head reads 3-axis positions past the state tail.
ADAPTER = replace(
    _BASE,
    by_method={
        "mtp": replace(
            _BASE,
            trailing_input_types=_mtp_trailing_input_types,
            extra=_mtp_extra,
            trailing_buffers=_mtp_trailing_buffers,
            takes_vision_outputs=_mtp_takes_vision_outputs,
        )
    },
)
