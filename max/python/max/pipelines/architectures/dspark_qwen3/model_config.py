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
"""Configuration for the registered Nemotron-H DSpark draft."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from transformers import AutoConfig

from ..nemotron_h_modulev3.model_config import is_sm100
from .dspark_qwen3 import (
    DSparkQwen3DraftConfig,
    parse_dspark_draft_config,
)

logger = logging.getLogger("max.pipelines")

__all__ = [
    "Qwen3DSparkDraftArchConfig",
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
    """The planning config of the ``Qwen3DSparkModel`` draft.

    A DSpark checkpoint is only ever served as ``--draft-model`` next to a
    Nemotron-H target, in the unified spec-decode graph. It records the
    draft checkpoint's maximum sequence length for planning.
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
        """Builds the config from the checkpoint's position bound."""
        del pipeline_config, max_seq_len
        assert model_config is not None
        hf = model_config.huggingface_config
        assert hf is not None
        return cls(max_position_embeddings=int(hf.max_position_embeddings))

    def get_max_seq_len(self) -> int:
        """The maximum sequence length the config allows."""
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Returns the checkpoint's ``max_position_embeddings``."""
        del model_config
        return int(huggingface_config.max_position_embeddings)

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

        Pinned to BF16, since the draft fills the cache with its own
        unquantized projections. The cache is full-length: the draft's
        sliding window is the attention mask's, not the cache's.
        """
        del cache_dtype
        draft = parse_dspark_draft_config(huggingface_config)
        speculative = pipeline_config.speculative
        assert speculative is not None
        return kv_cache_config.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=draft.num_key_value_heads,
            head_dim=draft.head_dim,
            num_layers=draft.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=speculative.speculative_method,
            num_draft_tokens=speculative.draft_width,
        )


@dataclass(frozen=True, kw_only=True)
class DSparkQwen3Config(DSparkQwen3DraftConfig):
    """Adds the planned cache and sequence bound to the checkpoint config."""

    kv_params: KVCacheParams
    max_seq_len: int


def configure_dspark_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> DSparkQwen3Config:
    """Finalizes the draft config without constructing or loading its module."""
    del state_dict
    pipeline_config = context.pipeline_config
    draft_model = pipeline_config.draft_model
    assert draft_model is not None
    config = parse_dspark_draft_config(draft_model.huggingface_config)
    check_draft_runs_on(pipeline_config.model.device_specs, config)
    assert isinstance(context.kv_params, KVCacheParams)
    return DSparkQwen3Config(
        **{field.name: getattr(config, field.name) for field in fields(config)},
        kv_params=context.kv_params,
        max_seq_len=context.max_seq_len,
    )
