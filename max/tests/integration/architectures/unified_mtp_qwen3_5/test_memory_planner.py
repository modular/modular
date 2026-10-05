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
"""Tests the fused Qwen3.5 MTP memory planner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.unified_mtp_qwen3_5.memory_planner import (
    UnifiedMTPQwen3_5MemoryPlanner,
)
from max.pipelines.lib import PipelineConfig

NUM_DRAFTS = 3
_LAYER_TYPES = ["linear_attention", "linear_attention", "full_attention"]


def _config() -> Qwen3_5Config:
    devices = [DeviceRef.GPU(0)]
    return Qwen3_5Config(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=len(_LAYER_TYPES),
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=64,
        interleaved_rope_weights=True,
        vocab_size=64,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            devices=devices,
            n_kv_heads=2,
            head_dim=16,
            num_layers=1,
            page_size=16,
        ),
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=0.25,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        layer_types=list(_LAYER_TYPES),
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
    )


def _pipeline_config() -> PipelineConfig:
    """Returns the fields of a pipeline config the planner reads."""
    return cast(
        "PipelineConfig",
        SimpleNamespace(speculative=SimpleNamespace(draft_width=NUM_DRAFTS)),
    )


def test_a_request_is_priced_its_ring() -> None:
    """Checks a request's extra bytes are its verify ring alone."""
    config = _config()
    planner = UnifiedMTPQwen3_5MemoryPlanner(config)

    ring_bytes = config._per_request_ring_bytes(NUM_DRAFTS + 1)
    assert ring_bytes > 0
    assert planner.spec_state_bytes(_pipeline_config()) == ring_bytes
