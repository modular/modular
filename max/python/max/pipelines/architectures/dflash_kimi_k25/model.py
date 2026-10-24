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
"""The DFlash draft over a Kimi K2.5 target: its config and module builder.

The draft runs only beside its target, inside a unified spec-decode graph.
Its config builds the draft's module for that graph from the draft's own
checkpoint, into a bfloat16 MHA cache of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config
from transformers import AutoConfig

from ..llama3.weight_adapters import _convert_safetensor_with_model_config
from .dflash_kimi_k25 import DFlashKimiK25, DFlashKimiK25DraftConfig

__all__ = ["DFlashKimiK25DraftArchConfig"]

logger = logging.getLogger("max.pipelines")


@dataclass
class DFlashKimiK25DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> DFlashKimiK25DraftArchConfig:
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
        """Returns the draft's cache, in bfloat16 whatever the target caches
        in, and as wide as the block it writes."""
        del cache_dtype
        speculative = pipeline_config.speculative
        assert speculative is not None
        return kv_cache_config.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=int(huggingface_config.num_key_value_heads),
            head_dim=int(huggingface_config.head_dim),
            num_layers=int(huggingface_config.num_hidden_layers),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            is_mla=False,
            num_q_heads=int(huggingface_config.num_attention_heads),
            speculative_method=speculative.speculative_method,
            num_draft_tokens=_block_size(huggingface_config, pipeline_config),
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DFlashKimiK25, DFlashKimiK25DraftConfig]:
        """Builds the draft from the draft checkpoint ``context`` names.

        It embeds and projects through its target's embedding and head, so
        spec decoding shares those onto it.

        Raises:
            ValueError: If the method isn't ``dflash``.
            NotImplementedError: If the checkpoint names the target's first
                layer, whose input is the raw embeddings.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if speculative.speculative_method != "dflash":
            raise ValueError(
                "The Kimi K2.5 DFlash draft requires --speculative-method"
                f" dflash, got {speculative.speculative_method!r}."
            )
        draft_hf = draft_model.huggingface_config
        dflash = parse_dflash_draft_hf_config(draft_hf)
        # The checkpoint names the layers whose inputs the draft read, which
        # the target captures as the outputs of the layers below.
        layer_ids = [i - 1 for i in dflash.target_layer_ids]
        if any(i < 0 for i in layer_ids):
            raise NotImplementedError(
                "DFlash dflash_config.target_layer_ids must not contain 0 "
                "for the current MAX impl — capturing layer-0's input "
                "(= raw token embeddings) is not yet wired. Got "
                f"target_layer_ids={dflash.target_layer_ids}."
            )
        state_dict = _convert_safetensor_with_model_config(
            dict(context.weights.items()), draft_hf, draft_model
        )
        assert isinstance(context.kv_params, KVCacheParams)
        config = _draft_config(
            draft_hf,
            state_dict,
            kv_params=context.kv_params,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            target_layer_ids=layer_ids,
            sliding_window_override=draft_model.sliding_window,
        )
        config.mask_token_id = dflash.mask_token_id
        config.block_size = _block_size(draft_hf, pipeline_config)
        draft = DFlashKimiK25(config)
        draft.load_state_dict(state_dict, weight_alignment=1, strict=False)
        validate_draft_state_dict(
            draft.raw_state_dict().keys(),
            state_dict.keys(),
            DraftAliases(always=("embed_tokens.", "lm_head.")),
        )
        return draft, config


def _block_size(draft_hf: Any, pipeline_config: PipelineConfig) -> int:
    """The block the draft was trained on, or one past the configured width
    for a checkpoint that declares none."""
    speculative = pipeline_config.speculative
    assert speculative is not None
    return (
        parse_dflash_draft_hf_config(draft_hf).block_size
        or speculative.draft_width + 1
    )


def _draft_config(
    draft_hf: Any,
    state_dict: dict[str, WeightData],
    *,
    kv_params: KVCacheParams,
    data_parallel_degree: int,
    target_layer_ids: list[int],
    sliding_window_override: int | None = None,
) -> DFlashKimiK25DraftConfig:
    """Builds :class:`DFlashKimiK25DraftConfig` from the draft HF config.

    Raises:
        ValueError: If the HF config declares no Deepseek-yarn RoPE.
    """
    rope_scaling = getattr(draft_hf, "rope_scaling", None)
    if (
        rope_scaling is None
        or rope_scaling.get("rope_type", rope_scaling.get("type")) != "yarn"
    ):
        raise ValueError(
            "Draft HF config must declare a Deepseek-yarn rope_scaling block."
        )

    use_override = sliding_window_override is not None
    raw = (
        sliding_window_override
        if use_override
        else getattr(draft_hf, "sliding_window", None)
    )
    sliding_window = int(raw) if raw is not None and int(raw) > 0 else None
    logger.info(
        "DFlash Kimi draft: sliding_window=%s (from %s)",
        sliding_window,
        "MAXModelConfig override" if use_override else "draft HF config",
    )

    return DFlashKimiK25DraftConfig(
        hidden_size=int(draft_hf.hidden_size),
        num_attention_heads=int(draft_hf.num_attention_heads),
        num_key_value_heads=int(draft_hf.num_key_value_heads),
        head_dim=int(draft_hf.head_dim),
        intermediate_size=int(draft_hf.intermediate_size),
        num_hidden_layers=int(draft_hf.num_hidden_layers),
        vocab_size=int(draft_hf.vocab_size),
        rms_norm_eps=float(draft_hf.rms_norm_eps),
        rope_theta=float(get_rope_theta(draft_hf)),
        max_position_embeddings=int(draft_hf.max_position_embeddings),
        devices=list(kv_params.devices),
        data_parallel_degree=data_parallel_degree,
        dtype=DType.bfloat16,
        norm_dtype=state_dict["norm.weight"].dtype,
        kv_params=kv_params,
        rope_scaling=dict(rope_scaling),
        target_layer_ids=list(target_layer_ids),
        sliding_window=sliding_window,
    )
