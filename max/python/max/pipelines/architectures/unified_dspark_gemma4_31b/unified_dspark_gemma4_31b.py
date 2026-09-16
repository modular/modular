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
"""DSpark + Gemma4-31B: the Speculators block draft on the block driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..dspark_draft.dspark_speculators_draft import DSparkSpeculatorsDraft
from ..gemma4.block_spec_adapters import SLIDING_KV, Gemma4BlockTarget
from ..gemma4.gemma4 import Gemma4TextModel
from .model_config import UnifiedDSparkGemma4_31BConfig
from .spec_adapters import DSparkGemma4_31BProposer

__all__ = ["SLIDING_KV", "UnifiedDSparkGemma4_31B"]


class UnifiedDSparkGemma4_31B(BlockDriver[TensorValue]):
    """Spark + Gemma4-31B: the Speculators block draft on the block driver

    The draft ships its own embedding, head and d2t map, so it reuses
    nothing of the target past the context KV. The phases live in
    :class:`BlockDriver`.
    """

    target: Gemma4TextModel
    draft: DSparkSpeculatorsDraft

    def __init__(
        self,
        config: UnifiedDSparkGemma4_31BConfig,
        enable_structured_output: bool = False,
    ) -> None:
        target = Gemma4TextModel(config.target)
        draft = DSparkSpeculatorsDraft(
            hidden_size=config.draft.hidden_size,
            num_hidden_layers=config.draft.num_hidden_layers,
            num_attention_heads=config.draft.num_attention_heads,
            num_key_value_heads=config.draft.num_key_value_heads,
            head_dim=config.draft.head_dim,
            intermediate_size=config.draft.intermediate_size,
            rms_norm_eps=config.draft.rms_norm_eps,
            rope_theta=config.draft.rope_theta,
            sliding_window=config.draft.sliding_window,
            layer_causal=[config.draft.causal] * config.draft.num_hidden_layers,
            vocab_size=config.draft.vocab_size,
            draft_vocab_size=config.draft.draft_vocab_size,
            markov_rank=config.draft.markov_rank,
            block_size=config.draft.block_size,
            sample_from_anchor=config.draft.sample_from_anchor,
            mask_token_id=config.draft.mask_token_id,
            num_context_features=config.draft.num_context_features,
            max_seq_len=config.draft.max_seq_len,
            kv_params=config.draft_kv_params,
            devices=list(config.target.devices),
            dtype=config.target.unquantized_dtype,
        )
        # The block stream embeds RAW rows (runtime-semantics contract: no
        # gemma sqrt(hidden) scale on the speculators draft path), so the
        # draft owns a plain Embedding loaded verbatim from the checkpoint's
        # frozen full-vocab copy rather than routing through the target's
        # ScaledWordEmbedding.
        draft.embed_tokens = Embedding(
            vocab_size=config.draft.vocab_size,
            hidden_dim=config.draft.hidden_size,
            dtype=config.target.unquantized_dtype,
            device=config.target.devices[0],
        )
        block_size = config.effective_block_size
        super().__init__(
            Gemma4BlockTarget(
                target,
                hidden_size=config.target.text_config.hidden_size,
                dtype=config.target.unquantized_dtype,
            ),
            DSparkGemma4_31BProposer(
                draft,
                block_size=block_size,
                mask_token_id=int(config.mask_token_id),
                hidden_size=config.draft.hidden_size,
            ),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices,
                distributed=False,
                # Gemma4's embedding and lm_head are collective even at one
                # device.
                include_signal_buffers=True,
                include_in_thinking_phase=True,
                enable_structured_output=enable_structured_output,
            ),
            speculative_config=config.speculative_config,
            enable_structured_output=enable_structured_output,
            relaxed_acceptance=True,
        )
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)

    @override
    @property
    def signature_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.config.target.kv_params,
                "draft": self.config.draft_kv_params,
            }
        )
