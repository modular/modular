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
"""ModuleV3 Gemma 4 over more devices than global KV heads."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.quant_tensor import NVFP4Tensor
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.gemma4_modulev3.layers.attention import (
    Gemma4Attention,
)
from max.pipelines.architectures.gemma4_modulev3.model_config import (
    Gemma4Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh

LOCAL_KV_HEADS = 8
GLOBAL_KV_HEADS = 2
N_HEADS = 8
HEAD_DIM = 8
GLOBAL_HEAD_DIM = 16
HIDDEN = 32


def _text_config() -> Any:
    return SimpleNamespace(
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=1024,
        num_key_value_heads=LOCAL_KV_HEADS,
        num_global_key_value_heads=GLOBAL_KV_HEADS,
        head_dim=HEAD_DIM,
        global_head_dim=GLOBAL_HEAD_DIM,
    )


def _pipeline_config() -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(pipeline_stage=None, data_parallel_degree=1),
        speculative=None,
    )


def _leaves(params: MultiKVCacheParams) -> dict[str, MHAKVCacheParams]:
    leaves: dict[str, MHAKVCacheParams] = {}
    for name, leaf in params.children.items():
        assert isinstance(leaf, MHAKVCacheParams)
        leaves[name] = leaf
    return leaves


def _modulev3(n_devices: int) -> dict[str, MHAKVCacheParams]:
    return _leaves(
        Gemma4Config.construct_kv_params(
            SimpleNamespace(text_config=_text_config()),
            _pipeline_config(),
            [DeviceRef.GPU(i) for i in range(n_devices)],
            KVCacheConfig(),
            DType.bfloat16,
        )
    )


def test_a_degree_above_the_global_heads_replicates_them() -> None:
    leaves = _modulev3(4)
    full = leaves["full_attention"]
    assert full.n_kv_heads == GLOBAL_KV_HEADS
    assert full.n_kv_heads_per_device == 1
    assert leaves["sliding_attention"].n_kv_heads_per_device == 2


def test_the_graph_api_config_still_rejects_a_degree_above_the_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Gemma4ForConditionalGenerationConfig.construct_kv_params(
            SimpleNamespace(text_config=_text_config()),
            _pipeline_config(),
            [DeviceRef.GPU(i) for i in range(4)],
            KVCacheConfig(),
            DType.bfloat16,
        )
    assert _modulev3(4)["full_attention"].allow_kv_head_replication


def test_a_dividing_degree_keeps_the_split_params() -> None:
    split = _leaves(
        Gemma4TextConfig.construct_kv_params(
            _text_config(),
            _pipeline_config(),
            [DeviceRef.GPU(i) for i in range(2)],
            KVCacheConfig(),
            DType.bfloat16,
        )
    )
    leaves = _modulev3(2)
    assert leaves.keys() == split.keys()
    for name, leaf in leaves.items():
        assert replace(leaf, allow_kv_head_replication=False) == split[name]


@contextmanager
def _global_attention(
    n_devices: int, quant_config: QuantConfig | None = None
) -> Iterator[Gemma4Attention]:
    """Builds a global attention layer on a CPU mesh, staying lazy so the
    weights the graph prepares can be read without running it."""
    mesh = DeviceMesh((CPU(),) * n_devices, (n_devices,), ("tp",))
    with (
        F.lazy(),
        default_dtype(DType.bfloat16),
        default_device(mesh),
        ensure_context(),
    ):
        yield Gemma4Attention(
            rope_global=NonCallableMock(),
            rope_local=NonCallableMock(),
            num_attention_heads=N_HEADS,
            num_key_value_heads=LOCAL_KV_HEADS,
            num_global_key_value_heads=GLOBAL_KV_HEADS,
            attention_k_eq_v=False,
            hidden_size=HIDDEN,
            kv_params=_modulev3(n_devices)["full_attention"],
            layer_idx_in_cache=0,
            is_sliding=False,
            quant_config=quant_config,
        )


def _shapes(values: Iterable[Tensor]) -> list[tuple[int, ...]]:
    return [tuple(int(d) for d in v.shape) for v in values]


def test_each_device_holds_one_whole_global_kv_head() -> None:
    """The k and v projections stay on the host as stored, and the fused
    weight gives each device its group's head."""
    with _global_attention(4) as attention:
        for proj in (attention.k_proj, attention.v_proj):
            assert isinstance(proj.weight, Tensor)
            assert _shapes([proj.weight]) == [
                (GLOBAL_KV_HEADS * GLOBAL_HEAD_DIM, HIDDEN)
            ]
            assert proj.weight.mesh.num_devices == 1
        q_rows = N_HEADS // 4 * GLOBAL_HEAD_DIM
        assert (
            _shapes(attention.wqkv.local_shards)
            == [(q_rows + 2 * GLOBAL_HEAD_DIM, HIDDEN)] * 4
        )


def test_a_dividing_degree_splits_the_global_kv_heads() -> None:
    with _global_attention(2) as attention:
        assert attention.kv_repeats == 1
        for proj in (attention.k_proj, attention.v_proj):
            assert isinstance(proj.weight, Tensor)
            assert (
                _shapes(proj.weight.local_shards)
                == [(GLOBAL_HEAD_DIM, HIDDEN)] * 2
            )


def test_nvfp4_kv_weights_repeat_with_their_scales() -> None:
    nvfp4 = QuantConfig.nvfp4(mlp_quantized_layers=(), attn_quantized_layers=())
    with _global_attention(4, nvfp4) as attention:
        for proj in (attention.k_proj, attention.v_proj):
            weight = attention._kv_weight(proj)
            assert isinstance(weight, NVFP4Tensor)
            assert (
                _shapes(weight.weight.local_shards)
                == [(GLOBAL_HEAD_DIM, HIDDEN // 2)] * 4
            )
            assert (
                _shapes(weight.weight_scale.local_shards)
                == [(GLOBAL_HEAD_DIM, HIDDEN // 16)] * 4
            )
