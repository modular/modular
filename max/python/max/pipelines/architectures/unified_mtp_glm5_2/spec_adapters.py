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
"""Speculative-decoding adapters for a GLM-5.2 (DeepSeek-V3.2) target."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, DimLike, TensorType, TensorValue
from max.nn.graph_groups import SparseIndices
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import forward_sharded_layers
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)
from max.pipelines.speculative.spec_target import Verified

from ..deepseekV3_2.deepseekV3_2 import DeepseekV3_2
from ..deepseekV3_2.layers.sparse_mla import sparse_selection_width
from ..deepseekV3_2_nextn.deepseekV3_2_nextn import DeepseekV3_2NextN

__all__ = [
    "INDEXER_DRAFT_KV",
    "INDEXER_TARGET_KV",
    "Glm5_2MTPProposer",
    "Glm5_2Target",
]

INDEXER_TARGET_KV = "target_indexer"
"""Names the target's lightning-indexer cache leaf in ``passthrough_kv``."""

INDEXER_DRAFT_KV = "draft_indexer"
"""Names the draft's lightning-indexer cache leaf in ``passthrough_kv``."""


class Glm5_2Target:
    """The DeepSeek-V3.2 target entry point and its output layout."""

    def __init__(self, target: DeepseekV3_2) -> None:
        self.target = target

    def verify(self, batch: SequentialBatch) -> Verified[list[TensorValue]]:
        outputs = self.target(
            batch.merged_tokens,
            batch.signal_buffers,
            batch.kv_collections,
            batch.passthrough_kv[INDEXER_TARGET_KV],
            batch.return_n_logits,
            # ``DeepseekV3_2.__call__`` broadcasts internally, so this target
            # takes the merged tensor rather than the per-device list.
            batch.merged_offsets,
            batch.dist.host_merged_offsets,
            batch.dist.data_parallel_splits,
            batch.batch_context_lengths,
            batch.ep_inputs,
        )
        return Verified(
            logits=outputs.logits["logits"],
            hidden=list(
                outputs.hidden_states[self.target.return_hidden_states.value]
            ),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        if self.target.ep_manager is None:
            return ()
        return self.target.ep_manager.input_types()


class Glm5_2MTPProposer:
    """The single-layer sparse NextN draft."""

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
        DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache = DraftCache.OWN
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step", per_device=True)
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True
    supports_zero_draft_rows = False

    def __init__(self, draft: DeepseekV3_2NextN) -> None:
        self.draft = draft
        self.hidden_dim: DimLike = draft.config.hidden_size
        assert isinstance(draft.config.kv_params, MultiKVCacheParams)
        mla_kv_params = draft.config.kv_params.children["mla"]
        assert isinstance(mla_kv_params, KVCacheParams)
        self.reuse: ReuseSpec | None = ReuseSpec(
            dim=sparse_selection_width(mla_kv_params, draft.config.index_topk),
            split_prefix="mtp_topk",
        )

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        # Step 0 needs ALL hidden states and VARIABLE logits; ``last_logits``
        # goes unread, so suppress its vocab-sized projection.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        self.draft.emit_last_token_logits = False
        outputs = self.draft(
            tokens,
            target_hidden,
            batch.signal_buffers,
            batch.draft_kv_collections,
            batch.passthrough_kv[INDEXER_DRAFT_KV],
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            batch.dist.host_query_offsets,
            batch.dist.data_parallel_splits,
            batch.batch_context_lengths,
            batch.ep_inputs,
            prev_topk_indices=None,
            reuse_prev_topk=False,
        )
        # Steps 1..K-1 read last_logits at index 0 and take the LAST_PER_DEVICE
        # path, whose allgather fences successive invocations.
        self.draft.return_hidden_states = self.step_hidden_mode
        self.draft.return_logits = ReturnLogits.LAST_TOKEN
        self.draft.emit_last_token_logits = True

        return Proposed(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states[ReturnHiddenStates.ALL.value]),
            reuse=list(outputs.group("topk_indices", SparseIndices)["rows"]),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        # ``hnorm`` expects the post-final-norm hidden state, which the target
        # hands over at step 0. ``shared_head_norm`` is row-wise, so
        # normalizing the one carried row here is equivalent and far cheaper.
        hidden = forward_sharded_layers(
            self.draft.shared_head_norm_shards, draft_input.hidden
        )
        outputs = self.draft(
            draft_input.tokens,
            hidden,
            batch.signal_buffers,
            batch.draft_kv_collections,
            batch.passthrough_kv[INDEXER_DRAFT_KV],
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            batch.dist.host_query_offsets,
            batch.dist.data_parallel_splits,
            batch.batch_context_lengths,
            batch.ep_inputs,
            prev_topk_indices=draft_input.reuse,
            reuse_prev_topk=True,
            # Every step reuses step 0's selection, made for the token one
            # position before step 1's.
            selection_lag=index,
            split_prefix=f"{self.split_prefix}_draft_step{index}",
        )
        return Proposed(
            logits=outputs.logits["next_token_logits"],
            hidden=list(outputs.hidden_states[self.step_hidden_mode.value]),
        )
