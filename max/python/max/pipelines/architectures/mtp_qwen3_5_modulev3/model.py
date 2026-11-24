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
# max/python/max/pipelines/architectures/mtp_qwen3_5/model.py
# at main 4df8111afba.
"""The config of the MTP head a Qwen3.5 checkpoint ships, and its draft role.

The head runs only beside its target, inside the fused spec-decode graph. Its
config is the target's, over a KV leaf of its own one layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface, KVCacheParams
from max.pipelines.architectures.qwen3_5_modulev3.model_config import (
    Qwen3_5Config,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_draft import DraftRegistration
from transformers import AutoConfig

from .mtp import Qwen3_5MTP

__all__ = [
    "MTPQwen3_5DraftArchConfig",
    "configure_mtp_draft",
    "mtp_draft_registration",
]

_EMBED_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
"""The target's embedding table, in a multimodal and a text-only checkpoint."""


class MTPQwen3_5DraftArchConfig(Qwen3_5Config):
    """The head's config: its target's, with a KV leaf for its one layer."""

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams:
        """Returns a one-layer view of the target's attention cache."""
        del allow_kv_head_replication
        attn = cls._attn_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        return replace(attn, num_layers=1)


def configure_mtp_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> Qwen3_5Config:
    """Returns the target's config, finalized against the head's weights.

    The head's tensors are BF16 in every published checkpoint, the NVFP4 one
    included; the target's embedding table rides along, since the config
    reads the checkpoint's compute dtype off it.

    Raises:
        ValueError: If the checkpoint ships no MTP head.
    """
    if "fc.weight" not in state_dict:
        raise ValueError(
            "The checkpoint ships no MTP head: expected its mtp.fc.weight."
            " Serve it without --speculative-method mtp."
        )
    weights = dict(state_dict)
    for name in _EMBED_NAMES:
        if context.weights[name].exists():
            weights["embed_tokens.weight"] = context.weights[name].data()
    config = Qwen3_5Config.initialize_from_config(
        context.pipeline_config,
        context.huggingface_config,
        max_seq_len=context.max_seq_len,
    )
    assert isinstance(context.kv_params, KVCacheParamInterface)
    config.kv_params = context.kv_params
    return config.finalize(
        context.huggingface_config, weights, ModelOutputConfig()
    )


mtp_draft_registration = DraftRegistration[Qwen3_5MTP, Qwen3_5Config](
    method="mtp", configure=configure_mtp_draft
)
