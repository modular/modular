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
"""The fused Hy4 MTP graph's config."""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers import AutoConfig

from ..hy_v4.model_config import HYV4Config

__all__ = ["UnifiedMTPHYV4Config"]


class UnifiedMTPHYV4Config(HYV4Config):
    """Hy4's config with the MTP draft's caches beside the target's, so
    memory planning budgets the cache tree the pipeline allocates."""

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """``{target: {mla, indexer}, draft: {mla, indexer}}``; the draft's
        caches hold its one layer."""
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
        return MultiKVCacheParams.from_params(
            {"target": target, "draft": MultiKVCacheParams.from_params(draft)}
        )
