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
"""The ModuleV3 GLM archs build the same NVFP4 MLA cache as the graph arch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_1.model_config import (
    Glm5_1Config as Glm5_1GraphConfig,
)
from max.pipelines.architectures.glm5_1_modulev3.arch import (
    glm5_1_modulev3_arch,
)
from max.pipelines.architectures.glm5_1_modulev3.model_config import (
    Glm5_1Config,
)
from max.pipelines.architectures.unified_mtp_glm5_2_modulev3.arch import (
    unified_mtp_glm5_2_modulev3_arch,
)
from max.pipelines.lib import KVCacheConfig, SupportedArchitecture

_INDEX_TOPK = 2048
_QK_ROPE_HEAD_DIM = 64


def _huggingface_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        num_hidden_layers=4,
        num_attention_heads=64,
        index_topk=_INDEX_TOPK,
        index_head_dim=128,
        indexer_types=["full", "shared", "full", "shared"],
    )


def _pipeline_config() -> Mock:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = None
    return pipeline_config


def _mla_kv_params(
    config: type[Glm5_1Config | Glm5_1GraphConfig], cache_dtype: DType
) -> MLAKVCacheParams:
    kv_params = config.construct_kv_params(
        huggingface_config=_huggingface_config(),
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.GPU(0)],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=cache_dtype,
    )
    assert isinstance(kv_params, MultiKVCacheParams)
    mla = kv_params.children["mla"]
    assert isinstance(mla, MLAKVCacheParams)
    return mla


@pytest.mark.parametrize(
    "arch", [glm5_1_modulev3_arch, unified_mtp_glm5_2_modulev3_arch]
)
def test_modulev3_glm_archs_accept_an_nvfp4_kv_cache(
    arch: SupportedArchitecture,
) -> None:
    assert arch.supports_nvfp4_kv_cache is True


def test_nvfp4_mla_cache_keeps_the_rope_dims_fp8() -> None:
    mla = _mla_kv_params(Glm5_1Config, DType.uint8)

    assert mla.is_nvfp4_kv_cache
    assert mla.kvcache_quant_config is not None
    assert mla.kvcache_quant_config.fp8_tail_dim == _QK_ROPE_HEAD_DIM
    # The indexer's top-k carries each selection's scale row after its value
    # row.
    assert mla.sparse_indices_stride == 2 * _INDEX_TOPK


def test_nvfp4_mla_cache_matches_the_graph_arch() -> None:
    """Mach serves the ModuleV3 export against MAX's tested NVFP4 layout."""
    assert _mla_kv_params(Glm5_1Config, DType.uint8) == _mla_kv_params(
        Glm5_1GraphConfig, DType.uint8
    )


def test_fp8_mla_cache_stays_scale_free() -> None:
    mla = _mla_kv_params(Glm5_1Config, DType.float8_e4m3fn)

    assert mla.kvcache_quant_config is None
    assert mla.sparse_indices_stride == _INDEX_TOPK
