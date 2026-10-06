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
"""Tests the MiMo-V2 config against the published checkpoint config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from max.driver import DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures import hf_config_shims
from max.pipelines.architectures.mimo_v2.memory_planner import (
    adapted_weights_size,
)
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
    attention_head_dim,
    layer_types,
    validate_devices,
)
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

LAYERS = 48
FULL_LAYERS = [0, 5, 11, 17, 23, 29, 35, 41, 47]


def _config(**overrides: Any) -> PretrainedConfig:
    """The text fields of the published ``config.json``."""
    values: dict[str, Any] = dict(
        vocab_size=152576,
        hidden_size=4096,
        num_hidden_layers=LAYERS,
        layernorm_epsilon=1e-6,
        hybrid_layer_pattern=[
            0 if i in FULL_LAYERS else 1 for i in range(LAYERS)
        ],
        moe_layer_freq=[0] + [1] * (LAYERS - 1),
        num_attention_heads=64,
        swa_num_attention_heads=64,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        partial_rotary_factor=0.334,
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        sliding_window=128,
        attention_chunk_size=128,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        intermediate_size=16384,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=None,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        attention_projection_layout="fused_qkv",
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                f"model.layers.{layer}.mlp.experts.{expert}.{proj}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for layer in range(1, LAYERS)
                for expert in range(256)
                for proj in ("gate_proj", "up_proj", "down_proj")
            },
        },
    )
    values.update(overrides)
    return PretrainedConfig(**values)


def _build(hf: PretrainedConfig, num_devices: int) -> MiMoV2Config:
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    # A stand-in; the config only stores it.
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=attention_head_dim(hf),
        num_layers=len(FULL_LAYERS),
        devices=devices[:1],
    )
    return MiMoV2Config.from_huggingface_config(
        hf, devices=devices, kv_params=kv_params, max_seq_len=4096
    )


@pytest.mark.parametrize("num_devices", [1, 2, 4])
def test_published_config(num_devices: int) -> None:
    hf = _config()
    config = _build(hf, num_devices)
    groups = layer_types(hf)
    assert [i for i, g in enumerate(groups) if g == FULL] == FULL_LAYERS
    assert groups.count(SLIDING) == 39
    assert config.moe_layers == set(range(1, LAYERS))
    # Q, K and V are padded to 256, but the scale is from Q/K's 192.
    assert attention_head_dim(hf) == 256
    assert config.attention_scale == 192**-0.5
    # int(192 * 0.334) = 64: RoPE covers the first 64 dims, not 64.128.
    assert config.rotary_dim == 64
    assert config.rope_thetas == {SLIDING: 10000.0, FULL: 10000000.0}
    assert config.sinks == {SLIDING: True, FULL: False}
    assert config.sliding_window == 128
    assert config.attention_value_scale == 0.707
    # One chunk per full-attention KV head, padded to whole FP8 blocks.
    full, sliding = config.qkv_layouts[FULL], config.qkv_layouts[SLIDING]
    assert (full.chunks, full.q_rows, full.k_rows, full.v_rows) == (
        4,
        3072,
        192,
        128,
    )
    assert full.chunks * full.padded_rows == 13824
    assert (sliding.q_rows, sliding.k_rows, sliding.v_rows) == (3072, 384, 256)
    assert sliding.chunks * sliding.padded_rows == 14848


def test_tp8_would_split_a_qkv_chunk() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        _build(_config(), 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_shared_experts": 1},
        {"scoring_func": "softmax"},
        {"n_group": 8, "topk_group": 4},
        {"routed_scaling_factor": 2.5},
        {"swa_head_dim": 128},
        {"attention_projection_layout": "split"},
        {"tie_word_embeddings": True},
    ],
)
def test_unimplemented_variants_fail(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="MiMo-V2"):
        _build(_config(**overrides), 1)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            {
                "max_position_embeddings": 1048576,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 262144,
                },
            },
            "rope_type 'yarn'",
        ),
        # The legacy key and spelling, which the reference also reads.
        (
            {"rope_scaling": {"type": "linear", "factor": 2.0}},
            "rope_type 'linear'",
        ),
    ],
)
def test_scaled_rope_fails(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build(_config(**overrides), 1)


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_type": "default", "type": "default", "rope_theta": 1e7},
        {"rope_theta": 1e7},
    ],
)
def test_default_rope_builds(rope: dict[str, Any]) -> None:
    _build(_config(rope_parameters=rope, routed_scaling_factor=1.0), 1)


def test_cpu_devices_fail() -> None:
    with pytest.raises(ValueError, match="only on NVIDIA SM100"):
        validate_devices([DeviceSpec.cpu()])


def test_adapted_weights_size_of_the_published_config() -> None:
    # 160.86 GB of MXFP4 experts, 3.22 GB of BF16 o_proj, 2.88 GB of FP8
    # qkv_proj, 2.50 GB of embedding and head, 0.20 GB of router and 0.20 GB
    # of dense MLP: 158.2 GiB, against 180.6 GiB of root shards on disk.
    one = adapted_weights_size(_config(), 1)
    assert one == 169_862_523_776
    # A second device holds its own copy of every router and norm.
    router = 47 * (256 * 4096 + 256) * 4
    norms = (2 * LAYERS + 1) * 4096 * 2
    assert adapted_weights_size(_config(), 2) == one + router + norms


def test_config_loads_without_remote_code(tmp_path: Path) -> None:
    published = _config().to_dict() | {
        "model_type": "mimo_v2",
        "auto_map": {"AutoConfig": "configuration_mimo_v2.MiMoV2Config"},
        "max_position_embeddings": 1048576,
        "rope_parameters": {
            "partial_rotary_factor": 0.334,
            "rope_theta": 10000000.0,
            "rope_type": "default",
            "type": "default",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(published))

    loaded = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, hf_config_shims._MiMoV2HFConfig)
    assert _build(loaded, 2) == _build(_config(), 2)
