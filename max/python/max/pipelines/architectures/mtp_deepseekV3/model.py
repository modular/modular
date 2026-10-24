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
"""The NextN head of a DeepseekV3 checkpoint: its config and module builder.

The head runs only beside its target, inside a unified spec-decode graph.
The config builds the head's module for that graph from the target's own
checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict
from transformers import AutoConfig

from ..deepseekV3.model import DeepseekV3Model
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3_nextn.deepseekV3_nextn import DeepseekV3NextN
from ..deepseekV3_nextn.model_config import DeepseekV3NextNConfig

__all__ = ["MTPDeepseekV3DraftArchConfig"]

logger = logging.getLogger("max.pipelines")

_NEXTN_NORM = "decoder_layer.self_attn.kv_a_layernorm.weight"
"""The NextN layer's norm, which the config probes for the norm dtype."""


@dataclass
class MTPDeepseekV3DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> MTPDeepseekV3DraftArchConfig:
        """Records the sequence length the pipeline resolved."""
        del pipeline_config, model_config
        return cls(max_seq_len=max_seq_len)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns a one-layer view of the target's cache, for the one
        decoder layer the head has."""
        target = DeepseekV3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        assert isinstance(target, KVCacheParams)
        return replace(target, num_layers=1)

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DeepseekV3NextN, DeepseekV3NextNConfig]:
        """Builds the NextN head from the target's checkpoint, which
        ``context`` names.

        It embeds and projects through the target's embedding and head, so
        spec decoding shares those onto it.

        Raises:
            KeyError: If the checkpoint has no NextN layer.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config = _draft_config(context, state_dict)
        context.start_ep(config.ep_config)
        draft = DeepseekV3NextN(config)
        draft.load_state_dict(state_dict, weight_alignment=1, strict=False)
        validate_draft_state_dict(
            draft.raw_state_dict().keys(),
            state_dict.keys(),
            DraftAliases(always=("embed_tokens.", "lm_head.")),
        )
        return draft, draft.config


def _draft_config(
    context: ModelLoadContext, state_dict: dict[str, WeightData]
) -> DeepseekV3NextNConfig:
    """The target's config, finalized against the NextN layer's weights."""
    if _NEXTN_NORM not in state_dict:
        raise KeyError(
            f"Expected NextN norm key '{_NEXTN_NORM}' not found in the draft"
            f" state_dict. Available keys: {list(state_dict)[:10]}..."
        )
    base = DeepseekV3Model.finalized_config(
        context.pipeline_config,
        # The target's config probes its first layer's norm.
        {
            **state_dict,
            "layers.0.self_attn.kv_a_layernorm.weight": state_dict[_NEXTN_NORM],
        },
        max_seq_len=context.max_seq_len,
        max_batch_total_tokens=context.max_batch_total_tokens,
        return_logits=context.return_logits,
        return_hidden_states=ReturnHiddenStates.LAST,
    )
    if (
        base.quant_config is not None
        and base.quant_config.is_nvfp4
        and not any("weight_scale_2" in key for key in state_dict)
    ):
        logger.info(
            "NextN weights are BF16 (no weight_scale_2 found); "
            "disabling NVFP4 config for draft."
        )
        base.quant_config = None
        base.dtype = DType.bfloat16
        if base.ep_config is not None:
            base.ep_config.dispatch_dtype = DType.bfloat16
            base.ep_config.dispatch_quant_config = None
    return DeepseekV3NextNConfig(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
