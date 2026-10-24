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
"""The standalone DSpark draft architecture's config and module builder.

Speculators-format DSpark drafts (registered HuggingFace architecture name
``DSparkDraftModel``) are never invoked as a standalone pipeline: they run
only beside a target, inside a unified spec-decode graph. The config builds
the draft's module for that graph, and supplies the draft's KV leaf.
"""

from __future__ import annotations

from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.pipelines.speculative import validate_draft_state_dict

from ..speculators_common.draft_config import (
    DSparkSpeculatorsDraftArchConfig,
    DSparkSpeculatorsDraftConfig,
)
from ..speculators_common.weight_adapters import (
    SKIPPED_DRAFT_KEYS,
    SKIPPED_DRAFT_PREFIXES,
    validate_draft_checkpoint_weights,
)
from .dspark_speculators_draft import DSparkSpeculatorsDraft


class DSparkDraftArchConfig(DSparkSpeculatorsDraftArchConfig):
    """The speculators-format DSpark draft's config, which builds its module."""

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DSparkSpeculatorsDraft, DSparkSpeculatorsDraftConfig]:
        """Builds the DSpark draft from the draft checkpoint ``context`` names.

        The draft computes in its checkpoint's own precision. It brings its
        own embedding, pruned-vocabulary head and ``d2t`` map, so it takes
        nothing from its target but the captured layers.

        Raises:
            ValueError: If the method isn't ``dspark``, or its checkpoint
                lacks a weight the draft needs.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_dspark():
            raise ValueError(
                "The DSparkDraftModel draft is a DSpark draft and requires"
                " --speculative-method dspark, got"
                f" {speculative.speculative_method!r}."
            )
        config = DSparkSpeculatorsDraftConfig.from_huggingface_config(
            draft_model.huggingface_config
        )
        # Speculators keys match the draft module 1:1.
        state_dict = {
            name: weight.data()
            for name, weight in context.weights.items()
            if name not in SKIPPED_DRAFT_KEYS
        }
        validate_draft_checkpoint_weights(state_dict, config)
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        devices = list(kv_params.devices)
        dtype = supported_encoding_dtype(
            draft_model.quantization_encoding or "bfloat16"
        )
        draft = DSparkSpeculatorsDraft(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            sliding_window=config.sliding_window,
            layer_causal=[config.causal] * config.num_hidden_layers,
            vocab_size=config.vocab_size,
            draft_vocab_size=config.draft_vocab_size,
            markov_rank=config.markov_rank,
            sample_from_anchor=config.sample_from_anchor,
            mask_token_id=config.mask_token_id,
            num_context_features=config.num_context_features,
            max_seq_len=config.max_seq_len,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
        )
        # The block embeds raw rows (no sqrt(hidden) scale on this path), so
        # the draft owns a plain embedding loaded from its own frozen copy.
        draft.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            dtype=dtype,
            device=devices[0],
        )
        loaded = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith(SKIPPED_DRAFT_PREFIXES)
        }
        draft.load_state_dict(
            loaded,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        # An unpruned drafter ships no ``d2t``, and its zero offsets map each
        # draft id to the same target id.
        validate_draft_state_dict(
            (
                name
                for name in draft.raw_state_dict()
                if config.has_d2t or name != "d2t"
            ),
            loaded.keys(),
        )
        return draft, config
