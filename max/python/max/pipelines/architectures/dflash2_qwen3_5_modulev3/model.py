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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/dflash2_qwen3_5/model.py
# at main 4df8111afba.
"""The DFlash2 draft's arch config and its draft role.

The draft runs only beside a Qwen3.5 target, inside the fused spec-decode
graph, so it has no pipeline model: its arch config sizes its KV leaf, and
its registration finalizes the config the framework builds the module from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_draft import DraftRegistration
from transformers import AutoConfig
from typing_extensions import Self

from .dflash2_qwen3_5 import DFlash2Qwen3_5
from .model_config import (
    DRAFT_SLIDING_WINDOW,
    DFlash2Qwen3_5Config,
    draft_head_dim,
)

__all__ = [
    "DFlash2Qwen3_5DraftArchConfig",
    "configure_dflash2_draft",
    "dflash2_draft_registration",
]


@dataclass
class DFlash2Qwen3_5DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
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
        """Returns the draft's windowed KV leaf, from its own geometry.

        Its dtype and window differ from the target's leaf. The drafter's
        mask never reaches further back than :data:`DRAFT_SLIDING_WINDOW`, so
        the leaf costs a per-slot constant.
        """
        del cache_dtype
        return kv_cache_config.to_params(
            # Pinned rather than following --kv-cache-dtype: the drafter
            # fills this cache with its own unquantized projections. Mach's
            # registry pins the matching group and the MEF sidecar's
            # draft_kv_cache_dtype makes a disagreement a startup error.
            dtype=DType.bfloat16,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=draft_head_dim(huggingface_config),
            num_layers=huggingface_config.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            window_size=DRAFT_SLIDING_WINDOW,
        )


def configure_dflash2_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> DFlash2Qwen3_5Config:
    """Returns the drafter's config, read off its own checkpoint.

    Raises:
        ValueError: If the drafter would span more than one device, or its
            config names a convention it lacks.
    """
    del state_dict
    draft_model = context.pipeline_config.draft_model
    assert draft_model is not None and draft_model.max_length is not None
    kv_params = context.kv_params
    assert isinstance(kv_params, KVCacheParams)
    if len(kv_params.devices) != 1:
        raise ValueError(
            "The DFlash2 drafter runs on one device; its target spans"
            f" {len(kv_params.devices)}."
        )
    return DFlash2Qwen3_5Config.from_huggingface_config(
        draft_model.huggingface_config,
        kv_params=kv_params,
        max_seq_len=draft_model.max_length,
    )


dflash2_draft_registration = DraftRegistration[
    DFlash2Qwen3_5, DFlash2Qwen3_5Config
](method="dflash2", configure=configure_dflash2_draft)
