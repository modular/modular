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
"""GLM-5.3-Flash's call, and the MTP layer its checkpoint ships.

Verifying K draft tokens advances 34 KDA recurrences and every sparse layer's
indexer tail ring, none of which a length pointer can rewind. The verify
defers all of them and rolls them forward over the accepted prefix once the
accepted count is known -- see :mod:`.spec_state`. The state rides beside the
latent cache in the driver batch's ``passthrough_kv``, under its name in the
cache tree.

The verify merges a batch's image rows into its embeddings as the target does
serving alone. The MTP layer embeds token ids only, an image's placeholders
included, as vLLM's does: it sees an image through the target's hidden states.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, DimLike, TensorValue, ops
from max.nn.legacy.kv_cache import RecurrentStateInputsPerDevice
from max.nn.legacy.rotary_embedding import RotaryEmbedding
from max.nn.legacy.transformer import ReturnHiddenStates
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
)
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import DistributedVerify, Verified
from max.pipelines.speculative.unified_graph_ops import (
    gather_accepted_hidden_states,
)

from ..deepseekV3.spec_adapters import ep_input_types
from .glm5_next import Glm5Next
from .layers.decoder import Glm5NextDecoderLayer
from .layers.mlp import Glm5NextMlpSublayerInputs
from .layers.sparse_mla import (
    Glm5NextSparseMLASublayer,
    SparseMLASublayerInputs,
)
from .mtp import Glm5NextMTP
from .spec_state import Glm5NextRecurrentState
from .state_cache import (
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    STATE_CACHE_KEY,
    TAIL_RING_LEAF_ID,
    leaf_inputs,
)

if TYPE_CHECKING:
    from max.pipelines.speculative.driver import SequentialProposer
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "Glm5NextMTPProposer"]

_TargetHidden = list[TensorValue]
"""Per-device normalized hidden states, one entry per rank."""

_PREFILL_TOKEN_SHARD = "mtp_moe_token_shard"
_STEP_TOKEN_SHARD = "mtp_step_moe_token_shard"
"""Token-shard dim names of the draft's MoE, one per row count it splits."""


def check(speculative: SpeculativeConfig, structured_output: bool) -> None:
    """Raises if a GLM-5.3-Flash target can't verify under the config.

    Raises:
        ValueError: If acceptance is synthetic.
    """
    del structured_output
    if speculative.synthetic_acceptance_rate is not None:
        raise ValueError(
            "synthetic acceptance would bypass the KDA state rollback's"
            " accepted-length plan"
        )


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a GLM-5.3-Flash model.

    It records its KDA state's updates in the verify ring, defers its
    indexers' tail seeds, and rolls both onto the accepted prefix.

    Raises:
        ValueError: If the target wasn't built to return its last normalized
            layer, which the MTP layer reads.
    """
    del draft_config
    if hidden_states != ReturnHiddenStates.ALL_NORMALIZED:
        raise ValueError(
            "A GLM-5.3-Flash target verifies only for drafts that read its"
            f" last normalized layer. Got {hidden_states.value}."
        )
    assert isinstance(target, Glm5Next)
    # The rollback reads every layer's verify inputs, which cannot cross a
    # subgraph boundary.
    target_config.use_subgraphs = False
    # The verify reads only the logits over its window, so it skips the
    # last-token projection, a vocab-wide matmul per step.
    target.emit_last_token_logits = False
    return DistributedVerify(_verify, Glm5NextRecurrentState(target))


def _state(
    batch: SequentialBatch,
) -> list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]:
    """Returns the per-device state leaves the batch carries."""
    state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]] = []
    for device in batch.passthrough_kv[STATE_CACHE_KEY]:
        assert isinstance(device, RecurrentStateInputsPerDevice)
        state.append(device)
    return state


def _verify(
    state: Glm5NextRecurrentState, batch: SequentialBatch | BlockBatch
) -> Verified[_TargetHidden]:
    assert isinstance(batch, SequentialBatch)
    with state.capturing(batch.num_draft_tokens, batch.merged_offsets_per_dev):
        outputs = state.target(
            batch.merged_tokens,
            # A step with images to encode merges no drafts, mixed or not:
            # `_should_verify_drafts` (overlap_text_generation.py) verifies
            # none while any context `needs_vision_encoding`. So the scatter
            # indices hold in the merged sequence.
            batch.vision_embeddings,
            batch.vision_scatter_indices,
            batch.signal_buffers,
            batch.kv_collections,
            batch.passthrough_kv[INDEXER_CACHE_KEY],
            batch.return_n_logits,
            # The target broadcasts the offsets itself.
            batch.merged_offsets,
            batch.dist.host_merged_offsets,
            _state(batch),
            batch.ep_inputs,
        )
    return Verified(
        logits=outputs.logits["logits"],
        hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
        roll_forward=partial(_roll_forward, state),
    )


def _roll_forward(
    state: Glm5NextRecurrentState, batch: SequentialBatch
) -> None:
    assert batch.num_accepted is not None
    state.roll_forward(
        _state(batch),
        merged_offsets=batch.merged_offsets,
        merged_offsets_per_dev=batch.merged_offsets_per_dev,
        num_accepted=batch.num_accepted,
        signal_buffers=batch.signal_buffers,
        device=batch.device0,
    )


def sequential_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[_TargetHidden]:
    """Returns the proposer of the MTP layer, beside the verify's state."""
    del speculative
    assert isinstance(verify, DistributedVerify)
    (state,) = verify.args
    assert isinstance(state, Glm5NextRecurrentState)
    assert isinstance(draft, Glm5NextMTP)
    return Glm5NextMTPProposer(draft, state, draft_config.hidden_size)


def _padding_rope(target: Glm5Next) -> RotaryEmbedding:
    """Returns the padding rotary table the target's sparse layers share."""
    layer = target.layers[target.config.sparse_attention_layers[0]]
    assert isinstance(layer, Glm5NextDecoderLayer)
    sublayer = layer.self_attn
    assert isinstance(sublayer, Glm5NextSparseMLASublayer)
    return sublayer.rope


class Glm5NextMTPProposer:
    """The MTP draft, projecting through the target's ``lm_head``.

    Step 0 runs the draft's own indexer over the verified window; every later
    step reuses that selection, gathered at the accepted position, with its
    lag. So only step 0 touches the draft's tail ring, and it is seeded with
    the accepted rows once that step has run.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
        DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache = DraftCache.OWN
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames()
    step_hidden_mode = ReturnHiddenStates.ALL
    uses_thinking_phase = True
    supports_zero_draft_rows = False
    takes_distributed_inputs = True

    def __init__(
        self,
        draft: Glm5NextMTP,
        state: Glm5NextRecurrentState,
        hidden_size: DimLike,
    ) -> None:
        self.draft = draft
        self.state = state
        self.hidden_dim = hidden_size
        self.reuse: ReuseSpec | None = ReuseSpec(
            dim=draft.selection_width, split_prefix="mtp_topk"
        )
        draft.share_rope(_padding_rope(state.target))

    def _attn_inputs(
        self,
        batch: SequentialBatch,
        selection: list[TensorValue],
        prev_selection: list[TensorValue],
    ) -> SparseMLASublayerInputs:
        layer = self.draft.cache_layer_idx
        rings = [
            leaf_inputs(device, TAIL_RING_LEAF_ID) for device in _state(batch)
        ]
        return SparseMLASublayerInputs(
            signal_buffers=batch.signal_buffers,
            input_row_offsets=batch.query_offsets_per_dev,
            mla_kv_collections=batch.draft_kv_collections,
            indexer_kv_collections=batch.passthrough_kv[INDEXER_CACHE_KEY],
            layer_idx=ops.constant(layer, DType.uint32, device=DeviceRef.CPU()),
            tail_pools=[ring.pool for ring in rings],
            tail_row_ids=[ring.live_row_id(layer) for ring in rings],
            prev_selection=prev_selection,
            selection=selection,
        )

    def _mlp_inputs(self, batch: SequentialBatch) -> Glm5NextMlpSublayerInputs:
        return Glm5NextMlpSublayerInputs(
            ep_buffers=list(batch.ep_inputs or []),
            signal_buffers=batch.signal_buffers,
            host_input_row_offsets=batch.dist.host_query_offsets,
        )

    def _logits(
        self, hidden: _TargetHidden, signal_buffers: list[BufferValue]
    ) -> TensorValue:
        return self.draft.lm_head(hidden, signal_buffers)[0]

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: _TargetHidden,
    ) -> Proposed:
        assert batch.num_accepted is not None
        selection: list[TensorValue] = []
        self.draft.mlp.token_shard_name = _PREFILL_TOKEN_SHARD
        with self.state.capturing_draft(
            self.draft.self_attn, batch.merged_offsets_per_dev
        ):
            hidden = self.draft(
                tokens,
                target_hidden,
                batch.signal_buffers,
                self._attn_inputs(batch, selection, prev_selection=[]),
                self._mlp_inputs(batch),
            )
        self.state.replay_draft_tail(batch.signal_buffers)

        carry = gather_accepted_hidden_states(
            hidden,
            merged_offsets=batch.merged_offsets,
            merged_offsets_per_dev=batch.merged_offsets_per_dev,
            num_accepted=batch.num_accepted,
            num_draft_tokens=batch.num_draft_tokens,
            data_parallel_degree=batch.data_parallel_degree,
            data_parallel_splits=batch.dist.data_parallel_splits,
            signal_buffers=batch.signal_buffers,
            device=batch.device0,
            split_prefix=self.split_prefix,
        )
        token = ops.argmax(self._logits(carry, batch.signal_buffers), axis=-1)
        return Proposed(
            logits=None,
            hidden=hidden,
            reuse=selection,
            carry=carry,
            token=token.reshape([-1]),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        # Step 0's selection was made for the token one position before step
        # 1's, and every later step reuses it.
        token_shard_name = self.draft.mlp.token_shard_name
        self.draft.self_attn.selection_lag = index
        self.draft.mlp.token_shard_name = _STEP_TOKEN_SHARD
        try:
            hidden = self.draft(
                draft_input.tokens,
                draft_input.hidden,
                batch.signal_buffers,
                self._attn_inputs(
                    batch, selection=[], prev_selection=draft_input.reuse
                ),
                self._mlp_inputs(batch),
            )
        finally:
            self.draft.self_attn.selection_lag = 0
            self.draft.mlp.token_shard_name = token_shard_name
        return Proposed(
            logits=self._logits(hidden, batch.signal_buffers), hidden=hidden
        )


ADAPTER = TargetAdapter(
    primary_kv=MLA_CACHE_KEY,
    verify=verify,
    ep_input_types=ep_input_types,
    check=check,
    sequential_proposer=sequential_proposer,
)
