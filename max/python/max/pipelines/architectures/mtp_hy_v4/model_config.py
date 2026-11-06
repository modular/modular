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
"""Configuration for HYV4's MTP layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from ..hy_v4.model_config import HYV4Config
from ..hy_v4.mtp import draft_config

__all__ = ["MTPHYV4DraftArchConfig"]


class MTPHYV4DraftArchConfig(HYV4Config):
    """The MTP layer's config: its target's, with caches for its one layer."""

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Returns the target's MLA and indexer leaves, each holding the one layer."""
        target = HYV4Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        draft = {}
        for name, params in target.children.items():
            assert isinstance(params, KVCacheParams)
            draft[name] = replace(params, num_layers=1)
        return MultiKVCacheParams.from_params(draft)


def configure_mtp_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> HYV4Config:
    """Finalizes the single MTP layer against the adapted checkpoint."""
    assert isinstance(context.kv_params, MultiKVCacheParams)
    config = HYV4Config.initialize(
        context.pipeline_config, max_seq_len=context.max_seq_len
    )
    if config.num_nextn_predict_layers < 1:
        raise ValueError(
            "Hy4 MTP requires a checkpoint with num_nextn_predict_layers > 0"
        )
    config.max_batch_context_length = (
        context.max_batch_total_tokens or config.max_batch_context_length
    )
    config.quant_config = parse_quant_config(
        context.pipeline_config.model.huggingface_config,
        dict(state_dict),
        config.dtype,
    )
    return draft_config(config, context.kv_params)
