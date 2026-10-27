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
"""Gemma 4's ModuleV3 assistant: its config, which builds its module."""

from __future__ import annotations

from typing import Any

from max import functional as F
from max.nn.legacy.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh

from ..gemma4_assistant.model import Gemma4AssistantDraftConfig
from ..gemma4_assistant.model_config import Gemma4AssistantConfig
from .gemma4_assistant_modulev3 import Gemma4AssistantModuleV3

__all__ = ["Gemma4AssistantModuleV3DraftArchConfig"]


class Gemma4AssistantModuleV3DraftArchConfig(Gemma4AssistantConfig):
    """The ModuleV3 assistant's config, which builds its module."""

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Gemma4AssistantModuleV3, Gemma4AssistantDraftConfig]:
        """Builds the assistant over the target's KV params, which
        ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. It embeds through the target's
        embedding, so spec decoding shares that onto it.

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
        devices = [ref.to_device() for ref in sliding.devices]
        mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
        with F.lazy(), default_dtype(config.dtype), default_device(mesh):
            draft = Gemma4AssistantModuleV3(
                config,
                list(target_text.layer_types),
                sliding,
                full,
                mesh,
            )
        return draft, Gemma4AssistantDraftConfig(config)

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: Gemma4AssistantDraftConfig
    ) -> dict[str, Any]:
        """Returns the weights the built assistant compiles with."""
        del config
        return parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
