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
"""Config for DFlash Kimi K2.5 unified pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from typing_extensions import Self

from ..deepseekV3.model_config import DeepseekV3Config
from ..dflash_kimi_k25 import DFlashKimiK25DraftConfig
from ..kimik2_5.model_config import KimiK2_5TextConfig
from ..unified_dflash_llama3.model_config import (  # re-exported helpers
    DflashDraftHFConfig,
    parse_dflash_draft_hf_config,
)

__all__ = [
    "DflashDraftHFConfig",
    "UnifiedDflashKimiK25Config",
    "parse_dflash_draft_hf_config",
]

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class UnifiedDflashKimiK25Config(ArchConfigWithKVCache):
    """Unified config for the DFlash Kimi K2.5 pipeline.

    Holds the Kimi target (``DeepseekV3Config`` populated from a
    ``KimiK25ForConditionalGeneration`` HF config) and the DFlash draft
    (``DFlashKimiK25DraftConfig`` built from the draft HF config).
    """

    target: DeepseekV3Config
    draft: DFlashKimiK25DraftConfig
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    mask_token_id: int = 0
    block_size: int = 0

    def __post_init__(self) -> None:
        if len(self.target.devices) != len(self.draft.devices):
            raise ValueError(
                "Target and draft must have the same number of devices."
                f" Got target={len(self.target.devices)}"
                f" draft={len(self.draft.devices)}."
            )

    def validate_dflash_fields(self) -> None:
        """Strict validation run from
        :meth:`UnifiedDflashKimiK25Model.load_model` once the DFlash-specific
        fields have been populated. ``__post_init__`` accepts the empty
        placeholder config produced by :meth:`initialize` so we can't enforce
        these there.
        """
        if not self.target_layer_ids:
            raise ValueError(
                "DFlash requires non-empty target_layer_ids (one per draft"
                " hidden layer)."
            )
        if len(self.target_layer_ids) != self.draft.num_hidden_layers:
            raise ValueError(
                "DFlash invariant: len(target_layer_ids) must equal the"
                " draft's num_hidden_layers."
                f" Got len(target_layer_ids)={len(self.target_layer_ids)}"
                f" draft.num_hidden_layers={self.draft.num_hidden_layers}."
            )
        if not 0 <= self.mask_token_id < self.target.vocab_size:
            raise ValueError(
                "DFlash mask_token_id must be in [0, target.vocab_size)."
                f" Got mask_token_id={self.mask_token_id}"
                f" target.vocab_size={self.target.vocab_size}."
            )
        if self.block_size > 0:
            expected_spec = self.block_size - 1
            actual_spec = self.speculative_config.num_speculative_tokens
            if actual_spec != expected_spec:
                logger.warning(
                    "DFlash draft was trained at block_size=%d, so"
                    " num_speculative_tokens is being overridden from %d to"
                    " %d. The DFlash draft's behavior is only defined at"
                    " its trained block_size.",
                    self.block_size,
                    actual_spec,
                    expected_spec,
                )
                self.speculative_config.num_speculative_tokens = expected_spec

    def resolve_block_size(self, *, default: int | None = None) -> int:
        if self.block_size > 0:
            return self.block_size
        if default is not None:
            return default
        return self.speculative_config.num_speculative_tokens + 1

    def get_kv_params(self) -> KVCacheParamInterface:
        target_kv = self.target.get_kv_params()
        assert isinstance(target_kv, KVCacheParams)
        return MultiKVCacheParams.from_params(
            {"target": target_kv, "draft": self.draft.kv_params}
        )

    @property
    def devices(self) -> list[DeviceRef]:
        """Exposes the target's devices so this unified config satisfies the
        ``ModelConfigWithKVCache`` protocol ``KimiK25MemoryPlanner`` requires
        (target and draft share placement; ``__post_init__`` checks the device
        count, and both are built from the target's devices)."""
        return self.target.devices

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        """Build an early placeholder config for KV memory estimation.

        The DFlash-specific fields are populated in
        :meth:`UnifiedDflashKimiK25Model.load_model` once the draft HF config
        has been parsed; we then re-instantiate the config with the real
        values.
        """
        model_config = model_config or pipeline_config.model
        assert model_config.huggingface_config is not None
        assert pipeline_config.draft_model is not None
        assert pipeline_config.draft_model.huggingface_config is not None
        assert pipeline_config.speculative is not None

        target_config = KimiK2_5TextConfig.initialize(
            pipeline_config, model_config
        )
        target_kv = target_config.kv_params
        assert isinstance(target_kv, KVCacheParams)
        placeholder_draft = DFlashKimiK25DraftConfig(
            hidden_size=target_config.hidden_size,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=target_config.hidden_size,
            intermediate_size=target_config.hidden_size,
            num_hidden_layers=1,
            vocab_size=target_config.vocab_size,
            rms_norm_eps=target_config.rms_norm_eps,
            rope_theta=target_config.rope_theta,
            max_position_embeddings=target_config.max_position_embeddings,
            devices=list(target_config.devices),
            data_parallel_degree=target_config.data_parallel_degree,
            dtype=target_config.dtype,
            norm_dtype=target_config.norm_dtype,
            kv_params=target_kv,
            rope_scaling={
                "factor": 1.0,
                "original_max_position_embeddings": (
                    target_config.max_position_embeddings
                ),
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
            target_layer_ids=[0],
        )
        return cls(
            target=target_config,
            draft=placeholder_draft,
            speculative_config=pipeline_config.speculative,
            target_layer_ids=[],
            mask_token_id=0,
            block_size=0,
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()
