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
"""Config for the unified DSpark Nemotron-H pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self

from ..nemotron_h_modulev3.model_config import NemotronHConfig, is_sm100
from .dspark_qwen3 import (
    DSparkQwen3DraftConfig,
    parse_dspark_draft_config,
)

logger = logging.getLogger("max.pipelines")

__all__ = [
    "Qwen3DSparkDraftArchConfig",
    "UnifiedDSparkNemotronHConfig",
    "check_draft_runs_on",
    "dspark_nemotron_draft_width",
]


def check_draft_runs_on(
    device_specs: Sequence[DeviceSpec],
    draft: DSparkQwen3DraftConfig,
    *,
    devices_loader: Callable[[Sequence[DeviceSpec]], Sequence[Any]] = (
        load_devices
    ),
) -> None:
    """Checks that the devices run the draft's quantized matmuls.

    The draft's NVFP4 weights run on the block-scaled NVFP4 matmul, which is
    SM100-only. They are never dequantized, so another device has no way to
    run them.

    Args:
        device_specs: The devices the pipeline runs on.
        draft: The parsed draft config.
        devices_loader: Loads the devices from their specs.

    Raises:
        ValueError: If the draft is NVFP4 and a device is not SM100.
    """
    if not draft.quantized:
        return
    for device in devices_loader(device_specs):
        if not is_sm100(device):
            raise ValueError(
                "The NVFP4 DSpark draft runs its W4A16_NVFP4 linears on the"
                " block-scaled NVFP4 matmul, which needs SM100 GPUs, and"
                f" {device.architecture_name} is not one. Use a BF16 drafter"
                " for this model instead."
            )


def dspark_nemotron_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
    *,
    warn: bool = True,
) -> int:
    """Returns the draft width the checkpoint was trained for.

    The drafter only works at its trained block size. With
    ``sample_from_anchor: false`` the block's first slot holds the committed
    token and predicts nothing, so a block of 8 drafts 7 tokens.

    Args:
        speculative: The speculative config, whose width the checkpoint
            overrides.
        target_huggingface_config: Unused.
        draft_huggingface_config: The draft's Hugging Face config.
        warn: Whether to warn on an overridden width.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DSpark requires a draft model.")
    width = parse_dspark_draft_config(draft_huggingface_config).num_drafts
    requested = speculative.num_speculative_tokens
    if warn and requested is not None and requested != width:
        logger.warning(
            "The DSpark draft was trained at its block size, so"
            " num_speculative_tokens is being overridden from %d to %d.",
            requested,
            width,
        )
    return width


@dataclass
class Qwen3DSparkDraftArchConfig:
    """The thin config of the standalone ``Qwen3DSparkModel`` registration.

    A DSpark checkpoint is only ever served as ``--draft-model`` next to a
    Nemotron-H target, which the registry rewrites to the unified
    architecture, so the registry needs only the draft side's maximum
    sequence length from this.
    """

    max_position_embeddings: int = 1 << 20

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Qwen3DSparkDraftArchConfig:
        del pipeline_config, max_seq_len
        assert model_config is not None
        hf = model_config.huggingface_config
        assert hf is not None
        return cls(max_position_embeddings=int(hf.max_position_embeddings))

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del model_config
        return int(huggingface_config.max_position_embeddings)


def construct_draft_kv_params(
    pipeline_config: PipelineConfig,
    draft_config: DSparkQwen3DraftConfig,
    devices: list[DeviceRef],
    *,
    page_size: int | None = None,
) -> KVCacheParams:
    """Builds the draft's KV leaf from its own geometry.

    Pinned to BF16, since the draft fills the cache with its own unquantized
    projections. The cache is full-length: the draft's sliding window is the
    attention mask's, not the cache's.
    """
    assert pipeline_config.speculative is not None
    return pipeline_config.model.kv_cache.to_params(
        dtype=DType.bfloat16,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=pipeline_config.speculative.speculative_method,
        num_draft_tokens=pipeline_config.speculative.draft_width,
        page_size=page_size,
    )


def construct_target_kv_params(
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
) -> MultiKVCacheParams:
    """The target's attention and Mamba state."""
    assert pipeline_config.speculative is not None
    return NemotronHConfig.construct_kv_params(
        huggingface_config,
        pipeline_config,
        devices,
        kv_cache_config,
        cache_dtype,
        speculative_method=pipeline_config.speculative.speculative_method,
        num_draft_tokens=pipeline_config.speculative.draft_width,
    )


@dataclass(kw_only=True)
class UnifiedDSparkNemotronHConfig(ArchConfigWithKVCache):
    """Nemotron-H and its DSpark draft, side by side."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = (
        NemotronHConfig.DEFAULT_ENCODING
    )
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = (
        NemotronHConfig.SUPPORTED_ENCODINGS
    )

    target: NemotronHConfig
    draft: DSparkQwen3DraftConfig
    draft_kv_params: KVCacheParams
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    """The target layers whose *inputs* are tapped for the draft."""
    mask_token_id: int = 0
    block_size: int = 0

    def __post_init__(self) -> None:
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.target.target_layer_ids = tuple(self.target_layer_ids)
        self.target.capture_mamba_replay = True

    def validate(self) -> None:
        """Checks the draft against the target it is paired with.

        Raises:
            ValueError: If the pair cannot run together.
        """
        if self.target.vocab_size != self.draft.vocab_size:
            raise ValueError(
                "The DSpark draft shares the target's embedding and lm_head,"
                " so their vocabularies must match. Got"
                f" draft={self.draft.vocab_size}"
                f" target={self.target.vocab_size}."
            )
        if self.target.hidden_size != self.draft.hidden_size:
            raise ValueError(
                "The DSpark draft reads the target's hidden states directly,"
                " so their widths must match. Got"
                f" draft={self.draft.hidden_size}"
                f" target={self.target.hidden_size}."
            )
        layers = len(self.target.layer_kinds)
        if not self.target_layer_ids or any(
            not 1 <= i <= layers for i in self.target_layer_ids
        ):
            raise ValueError(
                "The DSpark draft's target_layer_ids must name layers of the"
                f" {layers}-layer target, got"
                f" {[i - 1 for i in self.target_layer_ids]}."
            )
        if not 0 <= self.mask_token_id < self.draft.vocab_size:
            raise ValueError(
                f"mask_token_id {self.mask_token_id} is outside the"
                f" vocabulary of {self.draft.vocab_size}."
            )
        width = self.speculative_config.num_speculative_tokens
        if width is not None and width != self.draft.num_drafts:
            raise ValueError(
                "The DSpark draft proposes"
                f" {self.draft.num_drafts} tokens per block, but"
                f" num_speculative_tokens is {width}."
            )

    @property
    def devices(self) -> list[DeviceRef]:
        """The devices the unified model runs on."""
        return list(self.target.devices)

    def get_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.target.kv_params,
                "draft": self.draft_kv_params,
            }
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        assert model_config.huggingface_config is not None
        assert pipeline_config.draft_model is not None
        draft_hf = pipeline_config.draft_model.huggingface_config
        assert draft_hf is not None
        assert pipeline_config.speculative is not None

        draft = parse_dspark_draft_config(draft_hf)
        # Rejects an NVFP4 draft off SM100 before anything is loaded.
        check_draft_runs_on(model_config.device_specs, draft)

        target = NemotronHConfig.initialize(
            pipeline_config, model_config, max_seq_len=max_seq_len
        )
        encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        target.kv_params = construct_target_kv_params(
            model_config.huggingface_config,
            pipeline_config,
            target.devices,
            model_config.kv_cache,
            cache_dtype_for_encoding(
                encoding, model_config.kv_cache.kv_cache_format
            ),
        )
        draft_kv_params = construct_draft_kv_params(
            pipeline_config,
            draft,
            list(target.devices),
            page_size=target.kv_params.page_size,
        )
        config = cls(
            target=target,
            draft=draft,
            draft_kv_params=draft_kv_params,
            speculative_config=pipeline_config.speculative,
            target_layer_ids=list(draft.tap_layer_inputs),
            mask_token_id=draft.mask_token_id,
            block_size=draft.block_size,
        )
        config.validate()
        return config

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return NemotronHConfig.calculate_max_seq_len(
            huggingface_config, model_config
        )
