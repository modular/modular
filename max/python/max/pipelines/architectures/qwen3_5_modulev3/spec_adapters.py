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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/spec_adapters.py
# at main ff595979200.
"""Qwen3.5's call on ModuleV3, and its linear-attention state across a verify.

Verifying a speculative window advances every Gated DeltaNet recurrence in the
target, and no length pointer rewinds one. So the verify reads the live
recurrent pool without writing it, records each token's update in a ring, and
the accepted records are folded into the live pool once the accepted count is
known. A row longer than the verify window, as a prompt mixed into the batch,
is committed by the verify itself, and the rollback replays none of it.

The conv pool needs no ring. Its window is the last ``kernel_size - 1`` raw
inputs, so the verify leaves it unwritten and the rollback writes it at the
accepted position. A window with no drafts has nothing to reject, so its
verify writes both live pools and the rollback does nothing.

The verify runs on graph values, as the driver hands them; the target and
the MTP head run on ``Tensor`` values.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    DimLike,
    TensorType,
    TensorValue,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    RecurrentStateInputsPerDevice,
    RecurrentStateRegion,
    recurrent_leaf,
)
from max.nn.legacy.graph_groups import MRoPEPositions
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.context import TextContext
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    TensorAdapter,
)
from max.pipelines.speculative.gated_delta_rollback import (
    fold_state_pools,
    replay_conv_pools,
    window_row_plan,
)
from max.pipelines.speculative.ragged_token_merger import shape_to_scalar
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
    always_takes_prefill_rows,
    verify_lengths,
)
from max.pipelines.speculative.spec_target import DistributedVerify, Verified
from max.pipelines.speculative.unified_graph_ops import (
    gather_accepted_hidden_states,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from ..dflash2_qwen3_5_modulev3.proposer import dflash2_block_proposer
from .batch_processor import context_position_rows
from .layers.gated_deltanet import GatedDeltaReplay, GatedDeltaVerify
from .qwen3_5 import (
    Qwen3_5Decoder,
    Qwen3_5TextModel,
    distributed_state,
    replicate,
)
from .state_cache import ATTN_CACHE_KEY, STATE_CACHE_KEY

if TYPE_CHECKING:
    from max.pipelines.speculative.driver import SequentialProposer
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "POSITION_IDS", "Qwen3_5MTPProposer"]

PRIMARY_KV = ATTN_CACHE_KEY

POSITION_IDS = "position_ids"
"""Where a driver batch's ``extra`` holds the merged window's M-RoPE
positions, for a target that runs M-RoPE."""


def verify_width_operand(num_draft_tokens: DimLike) -> Tensor:
    """Returns the ``[K]`` host operand the state kernels read ``K`` from.

    The kernels read only its length, host metadata, so choosing between
    landing the state on the forward and deferring it to the rollback costs
    no device sync. At a width of zero there is no draft to reject.
    """
    return F.range(
        0,
        Dim(num_draft_tokens),
        1,
        out_dim=num_draft_tokens,
        dtype=DType.int64,
        device=CPU(),
    )


@dataclass(frozen=True)
class _ConvReplay:
    """One layer's conv replay inputs on one device, as the rollback reads."""

    qkv_window: TensorValue | None
    conv_weight: TensorValue


def _conv_replay(capture: GatedDeltaReplay) -> _ConvReplay:
    return _ConvReplay(
        qkv_window=(
            None
            if capture.qkv_window is None
            else TensorValue(capture.qkv_window)
        ),
        conv_weight=TensorValue(capture.conv_weight),
    )


class Qwen3_5RecurrentState:
    """The ring verify and the accepted-prefix rollback, for one target.

    Used in three steps, in this order: :meth:`verify_state` for the verify's
    state inputs, :meth:`capturing` around the verify, and
    :meth:`roll_forward` once the accepted count is known. Rolling forward
    before the count settles folds the state onto a prefix the row never
    accepted.
    """

    def __init__(
        self,
        model: Qwen3_5TextModel,
        state_regions: Sequence[RecurrentStateRegion],
    ) -> None:
        if not any(region.scratch for region in state_regions):
            raise ValueError(
                "a speculative Qwen3.5 verify needs the ring leaf among its"
                " state regions"
            )
        self.model = model
        self.verify: GatedDeltaVerify | None = None
        """The verify width and the replay inputs the verify recorded.

        Set by :meth:`capturing` and read by :meth:`roll_forward`: a forward
        that lands a window the rollback then replays or folds would advance
        the state twice, so both read one width.
        """

    @staticmethod
    def verify_state(
        passthrough_kv: Mapping[str, Sequence[Any]],
    ) -> list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]:
        """Returns the cache's ``state`` child, per device.

        Each device's leaves are the live conv and recurrent leaves, then the
        ring. The live leaves come first, since a verify with no drafts
        writes them.

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
    def capturing(self, n_devs: int, num_draft_tokens: Dim) -> Iterator[None]:
        """Binds the linear layers' verify for the forward inside.

        Unbound on the way out, so the draft's own forward can't append to
        the capture.
        """
        self.verify = GatedDeltaVerify(
            verify_width=verify_width_operand(num_draft_tokens),
            captures=[[] for _ in range(n_devs)],
        )
        self._bind(self.verify)
        try:
            yield
        finally:
            self._bind(None)

    def _bind(self, verify: GatedDeltaVerify | None) -> None:
        for index in self.model.linear_layer_indices:
            linear_attn = self.model.layers[index].linear_attn
            assert linear_attn is not None
            linear_attn.verify = verify

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

        Folds the ring's accepted records into the recurrent pool and writes
        the conv window, which the verify left unwritten. A verify width of
        zero launches nothing, because the verify already landed both.
        """
        verify = self.verify
        assert verify is not None, "roll_forward before capturing"
        captures = [
            [_conv_replay(capture) for capture in device]
            for device in verify.captures
        ]
        window = captures[0][0].qkv_window
        assert window is not None
        verify_width = TensorValue(verify.verify_width)
        row_indices, replay_offsets = window_row_plan(
            merged_offsets,
            num_accepted,
            num_draft_tokens,
            window.shape[0],
            device,
        )
        conv, recurrent, ring = (
            [device_state.leaves[i] for device_state in state] for i in range(3)
        )
        replay_conv_pools(
            captures,
            [leaf.pool for leaf in conv],
            [leaf.live_row_ids for leaf in conv],
            row_indices,
            replay_offsets,
            signal_buffers,
            verify_width,
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
            verify_width,
        )


def _language_model(target: Any) -> Qwen3_5TextModel:
    """Returns the model a built target runs, under the decoder's names."""
    if isinstance(target, Qwen3_5Decoder):
        target = target.model
    assert isinstance(target, Qwen3_5TextModel)
    return target


def _regions(target_config: Any) -> tuple[RecurrentStateRegion, ...]:
    state = recurrent_leaf(target_config.kv_params)
    assert state is not None, "a Qwen3.5 target keeps a recurrent state"
    return tuple(state.regions)


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

    It records its state's updates in the verify ring and rolls the state
    onto the accepted prefix.

    Raises:
        ValueError: If the target wasn't built to return the selected layers
            a block draft reads, or the last normalized layer the MTP head
            reads.
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
    model = _language_model(target)
    return DistributedVerify(
        _verify,
        Qwen3_5RecurrentState(model, _regions(target_config)),
        hidden_states,
    )


def _verify(
    state: Qwen3_5RecurrentState,
    hidden_states: ReturnHiddenStates,
    batch: SequentialBatch | BlockBatch,
) -> Verified[list[TensorValue]]:
    model = state.model
    mesh = model.mesh
    replicated = DeviceMapping.replicated(mesh)
    verify_state = state.verify_state(batch.passthrough_kv)
    position_ids = batch.extra.get(POSITION_IDS)
    # Images arrive only in a step that merges no drafts, so the scatter
    # indices into the step's tokens need no remapping.
    vision = (
        VisionInputs(
            embeddings=Tensor.from_shard_values(
                batch.vision_embeddings, replicated
            ),
            scatter_indices=Tensor.from_shard_values(
                batch.vision_scatter_indices, replicated
            ),
        )
        if batch.vision_embeddings
        else None
    )
    offsets = replicate(Tensor.from_graph_value(batch.merged_offsets), mesh)
    capture_layers = (
        model.config.target_layer_ids or []
        if hidden_states == ReturnHiddenStates.SELECTED_LAYERS
        else []
    )
    with state.capturing(batch.n_devs, batch.num_draft_tokens):
        h, captures = model.hidden_states(
            replicate(Tensor.from_graph_value(batch.merged_tokens), mesh),
            offsets,
            PagedCacheValues.from_upstream(batch.kv_collections, replicated),
            distributed_state(verify_state, replicated),
            vision=vision,
            position_ids=(
                None
                if position_ids is None
                else replicate(Tensor.from_graph_value(position_ids), mesh)
            ),
            capture_layers=capture_layers,
            subgraphs=False,
        )
    rows = model.window_rows(
        offsets, Tensor.from_graph_value(batch.return_n_logits)[0]
    )
    logits = model.logits(F.gather(h, rows, axis=0))
    hidden = (
        model.norm(h)
        if hidden_states == ReturnHiddenStates.ALL_NORMALIZED
        else captures[0]
        if len(captures) == 1
        else F.concat(captures, axis=-1)
    )
    return Verified(
        logits=TensorValue(logits),
        hidden=[TensorValue(shard) for shard in hidden.local_shards],
        roll_forward=partial(_roll_forward, state),
    )


def _roll_forward(
    state: Qwen3_5RecurrentState, batch: SequentialBatch | BlockBatch
) -> None:
    assert batch.num_accepted is not None
    state.roll_forward(
        state.verify_state(batch.passthrough_kv),
        merged_offsets=batch.merged_offsets,
        num_accepted=batch.num_accepted,
        num_draft_tokens=shape_to_scalar(batch.num_draft_tokens, batch.device0),
        signal_buffers=batch.signal_buffers,
        device=batch.device0,
    )


class Qwen3_5MTPProposer(TensorAdapter):
    """Drafts one token a step through the MTP head and the target's head.

    The head returns its normalized hidden states; the target's head
    projects them at one row per request rather than over the whole
    verified window.
    """

    reuse: ReuseSpec | None = None
    # A one-token step must not inherit the merged window's prompt length,
    # or attention picks the prefill kernel for a one-row query.
    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache: DraftCache = DraftCache.OWN
    split_prefix: str = "mtp"
    # One replica holds the whole batch, so the carry keeps the graph's own
    # ``batch_size``, which the concat inside the head needs.
    carry_dim_names: CarryDimNames = CarryDimNames()
    step_hidden_mode: ReturnHiddenStates = ReturnHiddenStates.ALL
    uses_thinking_phase: bool = True
    supports_zero_draft_rows: bool = False
    takes_distributed_inputs: bool = True

    def __init__(self, draft: Any, model: Qwen3_5TextModel) -> None:
        self.draft = draft
        self.model = model
        self.mesh = model.mesh
        self.hidden_dim: DimLike = model.config.hidden_size

    @property
    def _replicated(self) -> DeviceMapping:
        return DeviceMapping.replicated(self.mesh)

    def _hidden(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        tokens: Tensor,
        hidden: Sequence[Tensor],
    ) -> list[Tensor]:
        replicated = self._replicated
        h = self.draft(
            self.model.embed_tokens(replicate(tokens, self.mesh)),
            Tensor.from_shard_values(list(hidden), replicated),
            PagedCacheValues.from_upstream(
                batch.draft_kv_collections, replicated
            ),
            Tensor.from_shard_values(batch.query_offsets_per_dev, replicated),
        )
        return list(h.local_shards)

    def _logits(self, hidden: Sequence[Tensor]) -> Tensor:
        return self.model.head(
            Tensor.from_shard_values(list(hidden), self._replicated)
        )

    def prefill(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        tokens: Tensor,
        target_hidden: list[Tensor],
    ) -> Proposed[Tensor]:
        """Runs the first draft step over the target's hidden states."""
        assert batch.num_accepted is not None
        hidden = self._hidden(batch, tokens, target_hidden)
        carried: list[Tensor] = F.functional(gather_accepted_hidden_states)(
            hidden,
            merged_offsets=batch.merged_offsets,
            merged_offsets_per_dev=list(batch.merged_offsets_per_dev),
            num_accepted=batch.num_accepted,
            num_draft_tokens=batch.num_draft_tokens,
            data_parallel_degree=batch.data_parallel_degree,
            data_parallel_splits=batch.dist.data_parallel_splits,
            signal_buffers=list(batch.signal_buffers),
            device=batch.device0,
            split_prefix=self.split_prefix,
        )
        logits = self._logits(carried)
        return Proposed(
            logits=None,
            hidden=hidden,
            carry=carried,
            token=F.argmax(logits, axis=-1).reshape([-1]),
        )

    def step(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        draft_input: DraftStepInput[Tensor],
        index: int,
    ) -> Proposed[Tensor]:
        """Runs one later draft step on the previous step's output."""
        del index
        hidden = self._hidden(batch, draft_input.tokens, draft_input.hidden)
        return Proposed(logits=self._logits(hidden), hidden=hidden)


def sequential_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[Any, Any, Any, Any]:
    """Returns the MTP head's proposer, beside the verify's target."""
    del draft_config, speculative
    assert isinstance(verify, DistributedVerify)
    state = verify.args[0]
    assert isinstance(state, Qwen3_5RecurrentState)
    return Qwen3_5MTPProposer(draft, state.model)


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


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    check=check,
    # The conv verify and the gated delta ring both commit a long row.
    verify_commits_long_rows=True,
    verify_takes_prefill_rows=always_takes_prefill_rows,
    sequential_proposer=sequential_proposer,
    block_proposer=dflash2_block_proposer,
    # Right after return_n_logits, where the graph-API pairings take them.
    signal_buffer_index=3,
    # The verify reads 3-axis positions past the canonical tail. They are
    # the target's, so every method takes them.
    trailing_input_types=_mrope_input_types,
    trailing_input_tree=_mrope_input_tree,
    extra=_mrope_extra,
    trailing_buffers=_mrope_buffers,
)
