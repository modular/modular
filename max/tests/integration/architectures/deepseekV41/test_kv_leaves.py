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
"""Tests building the KV cache parameters from the real config."""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41 import kv_leaves
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"

_LEAF_KEYS = ("swa", "c2a", "c2a_state", "idx_c2a", "c1a", "idx_c1a")


def _real_text_config() -> AutoConfig:
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    return DeepseekV41Config.get_text_config(config)


def _mock_pipeline_config(
    *, speculative: object = None, data_parallel_degree: int = 1
) -> Mock:
    model = Mock()
    model.data_parallel_degree = data_parallel_degree
    pipeline_config = Mock()
    pipeline_config.model = model
    pipeline_config.speculative = speculative
    return pipeline_config


def _build_params(
    cache_dtype: DType = DType.bfloat16,
    *,
    devices: Sequence[DeviceRef] = (DeviceRef.CPU(),),
) -> MultiKVCacheParams:
    params = kv_leaves.construct_kv_params(
        huggingface_config=_real_text_config(),
        pipeline_config=_mock_pipeline_config(),
        devices=list(devices),
        kv_cache_config=KVCacheConfig(),
        cache_dtype=cache_dtype,
    )
    assert isinstance(params, MultiKVCacheParams)
    return params


def _leaf(params: MultiKVCacheParams, key: str) -> KVCacheParams:
    leaf = params.children[key]
    assert isinstance(leaf, KVCacheParams)
    return leaf


def test_construct_kv_params_succeeds_on_the_real_config() -> None:
    params = _build_params()
    assert set(params.children) == set(_LEAF_KEYS)
    for key in _LEAF_KEYS:
        _leaf(params, key)


def test_construct_kv_params_succeeds_with_four_gpu_devices() -> None:
    """Every leaf carries all four devices."""
    devices = [DeviceRef("gpu", i) for i in range(4)]
    params = _build_params(devices=devices)
    assert set(params.children) == set(_LEAF_KEYS)
    for key in _LEAF_KEYS:
        assert list(_leaf(params, key).devices) == devices
