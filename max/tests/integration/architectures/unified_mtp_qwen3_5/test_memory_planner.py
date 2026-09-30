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

import pytest
from max.driver import Device
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


def _pipeline_config(
    rollback: str, max_batch_size: int | None = 16
) -> PipelineConfig:
    """Returns the fields of a pipeline config the planner reads."""
    return cast(
        "PipelineConfig",
        SimpleNamespace(
            runtime=SimpleNamespace(max_batch_size=max_batch_size),
            model=SimpleNamespace(
                kv_cache=SimpleNamespace(device_memory_utilization=0.9)
            ),
            speculative=SimpleNamespace(
                recurrent_state_rollback=rollback, draft_width=NUM_DRAFTS
            ),
        ),
    )


def _devices(free_memory: int) -> list[Device]:
    return cast(
        "list[Device]", [SimpleNamespace(stats={"free_memory": free_memory})]
    )


def test_the_ring_arm_shadows_far_less_than_the_snapshot() -> None:
    """Checks the ring arm reserves less shadow than the snapshot arm."""
    planner = UnifiedMTPQwen3_5MemoryPlanner(_config())

    ring = planner.shadow_bytes_per_request(_pipeline_config("ring"))
    snapshot = planner.shadow_bytes_per_request(_pipeline_config("snapshot"))

    assert ring < snapshot / 4


@pytest.mark.parametrize("rollback", ["snapshot", "ring"])
def test_the_shadows_are_reserved_outside_the_pool(rollback: str) -> None:
    """Checks activation memory reserves only the shadow pools."""
    planner = UnifiedMTPQwen3_5MemoryPlanner(_config())
    pipeline_config = _pipeline_config(rollback, max_batch_size=16)

    reserved = planner.estimate_activation_memory(pipeline_config, None)

    assert reserved == 16 * planner.shadow_bytes_per_request(pipeline_config)


def test_an_unset_batch_size_reserves_what_inference_chose() -> None:
    """Checks the shadows are reserved at the inferred batch size."""
    planner = UnifiedMTPQwen3_5MemoryPlanner(_config())
    pipeline_config = _pipeline_config("ring", max_batch_size=None)

    inferred = planner.infer_max_batch_size(
        pipeline_config, _devices(free_memory=40 * 1024**3), 1024**3
    )
    reserved = planner.estimate_activation_memory(pipeline_config, None)

    assert inferred is not None and inferred >= 1
    assert reserved == inferred * planner.shadow_bytes_per_request(
        pipeline_config
    )


@pytest.mark.parametrize(
    ("rollback", "ring_len"), [("snapshot", 0), ("ring", NUM_DRAFTS + 1)]
)
def test_a_request_is_priced_its_shadow_and_ring(
    rollback: str, ring_len: int
) -> None:
    """Checks a request's extra bytes are its shadows and ring alone."""
    config = _config()
    planner = UnifiedMTPQwen3_5MemoryPlanner(config)
    pipeline_config = _pipeline_config(rollback)

    assert planner.shadow_state_bytes(pipeline_config) == (
        planner.shadow_bytes_per_request(pipeline_config)
        + config._per_request_ring_bytes(ring_len)
    )
