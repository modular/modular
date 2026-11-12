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
"""The deployments ``Glm5NextConfig.initialize`` refuses before it builds.

Both refusals name a failure that would otherwise show up late and far from
its cause:

- MTP needs the vision tower. Every MTP pairing encodes the batch's images
  with the target's tower, so a checkpoint that ships none would fail only
  after the fused graph compiled and the cache was allocated.
- With the vision encoder cache off, chunked prefill and prefix caching put
  image rows on the wrong placeholders: a step's encoder rows cover only the
  images in its window, while the scatter indices cover every image of the
  prompt.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import cast

import pytest
from max.driver import DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next import model_config
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers import PretrainedConfig

NUM_LAYERS = 2
HIDDEN = 256

TEXT_CONFIG: dict[str, object] = {
    "dtype": "bfloat16",
    "vocab_size": 512,
    "hidden_size": HIDDEN,
    "num_hidden_layers": NUM_LAYERS,
    "layer_types": ["linear_attention", "deepseek_sparse_attention"],
    "first_k_dense_replace": 1,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "hidden_act": "silu",
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "kv_lora_rank": 512,
    "q_lora_rank": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 0,
    "v_head_dim": 192,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "index_head_dim": 128,
    "index_n_heads": 4,
    "index_topk": 64,
    "linear_attn_config": {"num_heads": 8, "head_dim": 128},
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
}

VISION_CONFIG: dict[str, object] = {
    "depth": 1,
    "hidden_size": 128,
    "num_heads": 2,
    "intermediate_size": 256,
    "projection_intermediate_size": 256,
    "out_hidden_size": HIDDEN,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
}


def _kv_params(**_: object) -> MultiKVCacheParams:
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
        num_q_heads=8,
    )
    return MultiKVCacheParams.from_params({"mla": leaf, "indexer": leaf})


@pytest.fixture(autouse=True)
def _no_cache_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skips the cache tree, which these guards never read."""
    monkeypatch.setattr(
        Glm5NextConfig, "construct_kv_params", staticmethod(_kv_params)
    )
    monkeypatch.setattr(
        model_config,
        "_select_quantization_encoding",
        lambda config, default: "bfloat16",
    )


def _initialize(
    *,
    vision: bool = True,
    speculative: bool = False,
    vision_cache_utilization: float = 0.1,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
) -> Glm5NextConfig:
    hf_config = PretrainedConfig(
        text_config=PretrainedConfig(**TEXT_CONFIG),
        vision_config=PretrainedConfig(**VISION_CONFIG) if vision else None,
    )
    model = SimpleNamespace(
        huggingface_config=hf_config,
        model_path="zai-org/GLM-5.3-Flash",
        kv_cache=KVCacheConfig(enable_prefix_caching=enable_prefix_caching),
        device_specs=[DeviceSpec.cpu()],
        data_parallel_degree=1,
    )
    pipeline_config = SimpleNamespace(
        model=model,
        speculative=object() if speculative else None,
        runtime=SimpleNamespace(
            vision_cache_utilization=vision_cache_utilization,
            enable_chunked_prefill=enable_chunked_prefill,
        ),
    )
    return Glm5NextConfig.initialize(
        cast(PipelineConfig, pipeline_config), max_seq_len=1024
    )


def test_mtp_without_the_vision_tower_is_refused() -> None:
    with pytest.raises(
        ValueError, match=re.escape("needs GLM-5.3-Flash's vision tower")
    ):
        _initialize(vision=False, speculative=True)


def test_mtp_with_the_vision_tower_builds() -> None:
    config = _initialize(speculative=True)
    assert config.vision_config is not None


def test_a_text_only_checkpoint_builds_without_mtp() -> None:
    assert _initialize(vision=False).vision_config is None


@pytest.mark.parametrize(
    ("enable_chunked_prefill", "enable_prefix_caching"),
    [(True, False), (False, True), (True, True)],
)
def test_no_vision_cache_with_split_prompts_is_refused(
    enable_chunked_prefill: bool, enable_prefix_caching: bool
) -> None:
    with pytest.raises(ValueError, match="vision encoder cache"):
        _initialize(
            vision_cache_utilization=0,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
        )


def test_no_vision_cache_builds_when_prompts_stay_whole() -> None:
    """Without chunked prefill or prefix caching an image's prompt runs in
    one step, which encodes every image it scatters."""
    config = _initialize(vision_cache_utilization=0)
    assert config.vision_config is not None


def test_a_text_only_checkpoint_needs_no_vision_cache() -> None:
    config = _initialize(
        vision=False,
        vision_cache_utilization=0,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )
    assert config.vision_config is None
