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
known. A row longer than the verify window, as a prompt mixed into the batch,
is committed by the verify itself at its last real token, both leaves, and the
rollback replays and folds nothing of it.

The conv pool needs no ring. Its window is the last ``kernel_size - 1`` raw
inputs, so the verify leaves it unwritten and the rollback writes it at the
accepted position.

A window with no drafts has nothing to reject, so its verify writes both live
pools directly and the rollback does nothing. The verify width that decides
this is a shape, which the state ops read at launch without a device sync.

That dance is the same whichever draft sits behind the target: it is
parameterized in the accepted length, not in the draft width. The state is the
cache's ``state`` child, which the graph declares among its KV inputs and the
verify reads from the batch's ``passthrough_kv``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from max import tree
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
from max.nn._graph_ops.state_space import (
    GatedDeltaReplayInputs,
    verify_width_operand,
)
from max.nn.kv_cache import (
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
    recurrent_leaf,
)
from max.nn.legacy.graph_groups import MRoPEPositions
from max.nn.legacy.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.context import TextContext
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.gated_delta_rollback import (
    fold_state_pools,
    replay_conv_pools,
    window_row_plan,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
    verify_lengths,
)
from max.pipelines.speculative.spec_target import DistributedVerify, Verified

from .batch_processor import context_position_rows
from .state_cache import ATTN_CACHE_KEY, STATE_CACHE_KEY

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = [
    "POSITION_IDS",
    "Qwen3_5RecurrentState",
]

PRIMARY_KV = ATTN_CACHE_KEY

POSITION_IDS = "position_ids"
"""Where a driver batch's ``extra`` holds the merged window's M-RoPE
positions, for a target that runs M-RoPE."""


@runtime_checkable
class _ReplayCapturing(Protocol):
    """A linear-attention block that records its state kernels' inputs."""

    replay_capture: list[list[GatedDeltaReplayInputs]] | None
    verify_width: TensorValue | None


class Qwen3_5RecurrentState:
    """The ring verify and the accepted-prefix rollback, for one target.

    Used in three steps, in this order: :meth:`verify_state` for the verify's
    state inputs, :meth:`capturing` around the verify, and
    :meth:`roll_forward` on those inputs once the accepted count is known. Rolling forward
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
        self.verify_width: TensorValue | None = None
        """The verify width operand the forward and the rollback share.

        Both must read the same one: a forward that lands a window the
        rollback then replays or folds advances the state twice. Set by
        :meth:`capturing`.
        """

    @staticmethod
    def verify_state(
        passthrough_kv: Mapping[str, Sequence[Any]],
    ) -> list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]:
        """Returns the cache's ``state`` child, per device.

        Each device's leaves are the live conv and recurrent leaves, then the
        ring, as ``linear_state_regions`` declares them. The live leaves come
        first, since a verify with no drafts writes them.

        Args:
            passthrough_kv: The batch's KV leaves past its primary one.

        Raises:
            ValueError: If a device's state isn't those three leaves.
        """
        state = list(passthrough_kv[STATE_CACHE_KEY])
        for device in state:
            assert isinstance(device, RecurrentStateInputsPerDevice)
            if len(device.leaves) != 3:
                raise ValueError(
                    "a speculative Qwen3.5 verify needs the live conv and"
                    " recurrent leaves and the ring; the KV cache declares"
                    f" {len(device.leaves)} state leaves"
                )
        return state

    @contextmanager
    def capturing(
        self,
        n_devs: int,
        num_draft_tokens: Dim,
    ) -> Iterator[None]:
        """Binds the linear blocks' replay capture for the verify inside.

        The binding is unwound on the way out so the capture cannot leak into
        a later phase -- the draft's own forward would otherwise append to it
        and the replay would repeat arithmetic the verify never did.

        Args:
            n_devs: Devices the target is sharded across.
            num_draft_tokens: The verify's draft width ``K``.
        """
        self.captures = [[] for _ in range(n_devs)]
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
        state: Sequence[
            RecurrentStateInputsPerDevice[TensorValue, BufferValue]
        ],
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
            state: The :meth:`verify_state` the verify ran on.
            merged_offsets: Ragged offsets over the verified window.
            num_accepted: ``[batch]`` accepted draft tokens per request.
            num_draft_tokens: This step's draft width, as a scalar on
                ``device``, zero on a prefill.
            signal_buffers: Used only to place the plan on each device.
            device: The device the batch-wide tensors live on.
        """
        assert self.verify_width is not None, "roll_forward before capturing"
        window = self.captures[0][0].qkv_window
        assert window is not None
        row_indices, replay_offsets = window_row_plan(
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            window.shape[0],
            device,
        )
        conv, recurrent, ring = (
            [device.leaves[i] for device in state] for i in range(3)
        )
        replay_conv_pools(
            self.captures,
            [leaf.pool for leaf in conv],
            [leaf.live_row_ids for leaf in conv],
            row_indices,
            replay_offsets,
            signal_buffers,
            self.verify_width,
        )
        fold_state_pools(
            [leaf.pool for leaf in recurrent],
            [leaf.live_row_ids for leaf in recurrent],
            [leaf.pool for leaf in ring],
            [leaf.live_row_ids for leaf in ring],
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            signal_buffers,
            self.verify_width,
        )


def _regions(target_config: Any) -> tuple[RecurrentStateRegion, ...]:
    state = recurrent_leaf(target_config.kv_params)
    assert state is not None, "a Qwen3.5 target keeps a recurrent state"
    return tuple(state.regions)


def _mrope_input_types(
    target_config: Any,
) -> list[TensorType | BufferType]:
    """Returns the leaves of :func:`_mrope_input_tree`."""
    named = _mrope_input_tree(target_config)
    return [] if named is None else tree.leaves(named)


def _mrope_input_tree(
    target_config: Any,
) -> OrderedDict[str, object] | None:
    """Names the M-RoPE positions, for a target that runs M-RoPE.

    They are the merged window's 3-axis positions, shared across devices.
    """
    if not target_config.mrope_enabled:
        return None
    return OrderedDict(
        mrope=MRoPEPositions(
            position_ids=TensorType(
                DType.int64,
                shape=[3, "merged_total_seq_len"],
                device=target_config.devices[0],
            )
        )
    )


def _mrope_extra(target_config: Any, trailing: Sequence[Any]) -> dict[str, Any]:
    """Names the M-RoPE positions, when the graph takes them."""
    if not target_config.mrope_enabled:
        return {}
    return {POSITION_IDS: trailing[0].tensor}


def _mrope_buffers(
    target_config: Any,
    contexts: Sequence[TextContext],
    devices: Sequence[Device],
) -> list[Buffer]:
    """Returns the merged window's positions, for a target that runs M-RoPE.

    Each request's run covers its tokens and the drafts it verifies.
    """
    if not target_config.mrope_enabled:
        return []
    positions = np.concatenate(
        [
            context_position_rows(ctx, length)
            for ctx, length in zip(
                contexts, verify_lengths(contexts), strict=True
            )
        ],
        axis=1,
    ).astype(np.int64)
    return [Buffer.from_numpy(positions).to(devices[0])]


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
            " excludes them only through that mask. Turn it on with"
            " --enable-structured-output (or a tool parser that implies"
            " it), or leave it on when exporting the graph."
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
    verify_state = state.verify_state(batch.passthrough_kv)
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
        state.verify_state(batch.passthrough_kv),
        merged_offsets=batch.merged_offsets,
        num_accepted=batch.num_accepted,
        num_draft_tokens=_shape_to_scalar(
            batch.num_draft_tokens, batch.device0
        ),
        signal_buffers=batch.signal_buffers,
        device=batch.device0,
    )


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    check=check,
    # The conv verify and the gated delta ring both commit a long row.
    verify_commits_long_rows=True,
    # The verify reads 3-axis positions past the canonical tail. They are
    # the target's, so every method takes them.
    trailing_input_types=_mrope_input_types,
    trailing_input_tree=_mrope_input_tree,
    extra=_mrope_extra,
    trailing_buffers=_mrope_buffers,
)
