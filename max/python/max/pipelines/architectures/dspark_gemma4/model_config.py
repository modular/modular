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
"""Configs for the native DSpark draft for Gemma4 targets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib.config import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.pipelines.speculative import validate_draft_state_dict
from transformers import AutoConfig

from ..speculators_common.draft_config import construct_draft_kv_params
from .dspark_gemma4 import DSparkGemma4, DSparkGemma4DraftConfig, _get

_SKIPPED_PREFIXES = ("confidence_head.", "embed_tokens.", "lm_head.")
"""The checkpoint ships a confidence head the draft doesn't run, and frozen
copies of the target's embedding and head, which the draft reads from the
target instead."""

__all__ = [
    "Gemma4DSparkDraftArchConfig",
    "dspark_draft_width",
    "gemma4_dspark_12b_width",
]

logger = logging.getLogger("max.pipelines")


def dspark_draft_width(
    speculative: SpeculativeConfig,
    draft_huggingface_config: Any,
    *,
    warn: bool = True,
) -> int:
    """Returns the resolved DSpark draft width from the draft HF config."""
    raw_block_size = _get(draft_huggingface_config, "block_size", None)
    block_size = int(raw_block_size) if raw_block_size is not None else 0
    if block_size <= 0:
        if speculative.num_speculative_tokens is None:
            raise ValueError(
                "The DSpark draft config declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return speculative.num_speculative_tokens
    actual_spec = speculative.num_speculative_tokens
    if warn and actual_spec is not None and actual_spec != block_size:
        logger.warning(
            "DSpark draft was trained at block_size=%d and drafts at"
            " every block position, so num_speculative_tokens is"
            " being overridden from %d to %d.",
            block_size,
            actual_spec,
            block_size,
        )
    return block_size


@dataclass
class Gemma4DSparkDraftArchConfig:
    """Thin ArchConfig for the standalone ``Gemma4DSparkModel`` registration.

    A DSpark checkpoint is only ever served as ``--draft-model`` next to a
    Gemma4 target, inside the unified spec-decode graph, so the registry
    needs just the draft-side max-sequence-length clamp from this config.
    """

    max_position_embeddings: int = 262144

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Gemma4DSparkDraftArchConfig:
        del pipeline_config
        assert model_config is not None
        huggingface_config = model_config.huggingface_config
        assert huggingface_config is not None
        return cls(
            max_position_embeddings=cls.calculate_max_seq_len(
                huggingface_config, model_config
            )
        )

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del model_config
        max_pos = getattr(huggingface_config, "max_position_embeddings", None)
        if isinstance(huggingface_config, dict):
            max_pos = huggingface_config.get("max_position_embeddings")
        assert max_pos is not None, (
            "DSpark draft config is missing max_position_embeddings."
        )
        return int(max_pos)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the draft's KV leaf, from its own geometry.

        The draft's cache is bfloat16 whatever the target's is: it is
        trained against bfloat16, and its attention takes no fp8 query.
        """
        del kv_cache_config, cache_dtype
        draft = DSparkGemma4DraftConfig.from_huggingface_config(
            huggingface_config
        )
        return construct_draft_kv_params(
            pipeline_config,
            devices,
            n_kv_heads=draft.num_key_value_heads,
            head_dim=draft.head_dim,
            num_layers=draft.num_hidden_layers,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DSparkGemma4, DSparkGemma4DraftConfig]:
        """Builds the DSpark draft from the draft checkpoint ``context`` names.

        The draft computes in its checkpoint's own precision. It embeds and
        projects through its target's embedding and head, so spec decoding
        shares those onto it.

        Raises:
            ValueError: If the method isn't ``dspark``.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_dspark():
            raise ValueError(
                "The Gemma4DSparkModel draft is a DSpark draft and requires"
                " --speculative-method dspark, got"
                f" {speculative.speculative_method!r}."
            )
        config = DSparkGemma4DraftConfig.from_huggingface_config(
            draft_model.huggingface_config
        )
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        draft = DSparkGemma4(
            config,
            kv_params=kv_params,
            devices=list(kv_params.devices),
            dtype=supported_encoding_dtype(
                draft_model.quantization_encoding or "bfloat16"
            ),
        )
        # Keys match the draft module 1:1.
        loaded = {
            name: weight.data()
            for name, weight in context.weights.items()
            if not name.startswith(_SKIPPED_PREFIXES)
        }
        draft.load_state_dict(
            loaded,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        validate_draft_state_dict(draft.raw_state_dict().keys(), loaded.keys())
        return draft, config


def gemma4_dspark_12b_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the block size: on the 12B draft every slot drafts a token."""
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DSpark requires a draft model.")
    return dspark_draft_width(speculative, draft_huggingface_config)
