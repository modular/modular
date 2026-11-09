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

from __future__ import annotations

from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config as DeepseekV3_2ModuleV3Config,
)
from max.pipelines.architectures.deepseekV3_2_nextn.model_config import (
    DeepseekV3_2NextNConfig,
)
from max.pipelines.lib import KVCacheConfig
from transformers import DeepseekV32Config


@pytest.mark.parametrize(
    "config_cls",
    [DeepseekV3_2Config, DeepseekV3_2ModuleV3Config, DeepseekV3_2NextNConfig],
    ids=["legacy", "modulev3", "nextn"],
)
@pytest.mark.parametrize("cache_dtype", [DType.float8_e4m3fn, DType.uint8])
def test_indexer_capture_identity_matches_physical_mla(
    config_cls: type[DeepseekV3_2Config | DeepseekV3_2ModuleV3Config],
    cache_dtype: DType,
) -> None:
    """Preserves the MLA dispatch form in the narrower indexer cache leaf."""
    huggingface_config = DeepseekV32Config(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_topk=2048,
        num_attention_heads=128,
        num_hidden_layers=2,
    )
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = None
    params = config_cls.construct_kv_params(
        huggingface_config=huggingface_config,
        pipeline_config=pipeline_config,
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=cache_dtype,
    )
    assert isinstance(params, MultiKVCacheParams)
    mla = params.children["mla"]
    indexer = params.children["indexer"]
    assert isinstance(mla, MLAKVCacheParams)
    assert isinstance(indexer, MLAKVCacheParams)

    # The indexer stores narrower rows but joins the MLA launch identity.
    assert mla.head_dim == 576
    assert indexer.head_dim == 128
    assert mla.resolved_sparse_dispatch_head_dim == 576
    assert (
        indexer.resolved_sparse_dispatch_head_dim
        == mla.resolved_sparse_dispatch_head_dim
    )
    assert mla.resolved_sparse_physical_indices
    assert (
        indexer.resolved_sparse_physical_indices
        == mla.resolved_sparse_physical_indices
    )
    assert indexer.resolved_sparse_kv_form == mla.resolved_sparse_kv_form
    assert mla.sparse_fixed_topk
    assert indexer.sparse_fixed_topk == mla.sparse_fixed_topk
    assert mla.sparse_indices_stride == (
        4096 if cache_dtype == DType.uint8 else 2048
    )
    assert indexer.sparse_indices_stride == mla.sparse_indices_stride
