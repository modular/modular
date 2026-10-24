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
"""Gemma 4's assistant: its config, which builds its module.

The assistant runs only inside a unified spec-decode graph beside its target.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.pipeline_parallel import stage_layers
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict

from .gemma4_assistant import Gemma4Assistant
from .model_config import Gemma4AssistantConfig

__all__ = ["Gemma4AssistantDraftArchConfig", "Gemma4AssistantDraftConfig"]


@dataclass(frozen=True)
class Gemma4AssistantDraftConfig:
    """The assistant's module config, and what the pairing reads beside it."""

    module: Gemma4AssistantConfig

    @property
    def hidden_size(self) -> int:
        """Returns the target's hidden size, which the assistant drafts in."""
        return self.module.backbone_hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary the assistant's head scores."""
        return self.module.vocab_size


class Gemma4AssistantDraftArchConfig(Gemma4AssistantConfig):
    """The assistant's config, which builds its module."""

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Gemma4Assistant, Gemma4AssistantDraftConfig]:
        """Builds the assistant over the target's KV params, which
        ``context`` names.

        It embeds through the target's embedding, so spec decoding shares
        that onto it.

        Raises:
            ValueError: If the method isn't ``mtp``.
        """
        speculative = context.pipeline_config.speculative
        assert speculative is not None
        if not speculative.is_mtp():
            raise ValueError(
                "Gemma 4's assistant requires --speculative-method mtp, got"
                f" {speculative.speculative_method!r}."
            )
        target_kv = context.kv_params
        assert isinstance(target_kv, MultiKVCacheParams)
        sliding = target_kv.children["sliding_attention"]
        full = target_kv.children["full_attention"]
        assert isinstance(sliding, KVCacheParams)
        assert isinstance(full, KVCacheParams)
        draft_model = context.pipeline_config.draft_model
        assert draft_model is not None
        target_text = context.huggingface_config.text_config
        config = Gemma4AssistantConfig.from_huggingface_config(
            draft_model.huggingface_config,
            backbone_hidden_size=target_text.hidden_size,
            devices=list(sliding.devices),
        )
        # A pipeline stage's caches hold only its own layers, so the
        # assistant finds the last sliding and full layers among those.
        layer_types = list(target_text.layer_types)
        layers = stage_layers(context.pipeline_config, len(layer_types))
        draft = Gemma4Assistant(
            config,
            target_layer_types=layer_types[layers.start : layers.stop],
            target_sliding_kv_params=sliding,
            target_global_kv_params=full,
        )
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        draft.load_state_dict(state_dict, weight_alignment=1, strict=False)
        validate_draft_state_dict(
            draft.raw_state_dict().keys(),
            state_dict.keys(),
            DraftAliases(always=("embed_tokens.",)),
        )
        return draft, Gemma4AssistantDraftConfig(config)
