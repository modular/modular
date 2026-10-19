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
"""Tests the planner's TP2 plan on two B200s and its DP4 sizing.

The planner's weight and activation estimates feed the estimator's KV cache
budget. Measured constants come from device memory traces on B200 GPUs.
"""

from pathlib import Path
from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.driver import Device
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41.memory_planner import (
    CheckpointTensor,
    DeepseekV41MemoryPlanner,
    DeviceWeightLayout,
    device_weight_bytes,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
    data_parallel_ep_config,
)
from max.pipelines.kv_cache.memory_planner import ModelConfigWithKVCache
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.memory_estimation import MemoryEstimator
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"
GIB = 1 << 30


def _pipeline_config(
    *,
    max_length: int = 4096,
    max_num_input_tokens: int = 8192,
    num_devices: int = 1,
) -> NonCallableMock:
    config = NonCallableMock(spec=PipelineConfig)
    config.model = MagicMock()
    # These plans serve text only.
    config.model.vision_config_overrides = {"max_images_per_request": 0}
    config.runtime = MagicMock()
    config.model.max_length = max_length
    config.model.kv_cache.kv_cache_page_size = 128
    config.model.device_specs = [MagicMock() for _ in range(num_devices)]
    config.runtime.max_batch_input_tokens = 8192
    config.runtime.max_num_input_tokens = max_num_input_tokens
    return config


def _real_huggingface_config() -> AutoConfig:
    """Returns the real checkpoint's top-level config, text config nested."""
    return AutoConfig.from_pretrained(str(TESTDATA / "config.json"))


def _planner() -> DeepseekV41MemoryPlanner:
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock()
    return DeepseekV41MemoryPlanner(config)


# The real NVFP4 checkpoint's 491.06 GiB of tensor bytes, grouped by device
# placement and summed from its safetensors headers.
_REAL_CHECKPOINT = [
    # Routed experts, split across devices: NVFP4 weights, their e4m3 block
    # scales, and the per-expert float32 scalars.
    CheckpointTensor("layers.0.ffn.experts.0.w1.weight", "U8", 271_790_899_200),
    CheckpointTensor(
        "layers.0.ffn.experts.0.w1.weight_scale", "F8_E4M3", 33_973_862_400
    ),
    CheckpointTensor("layers.0.ffn.experts.0.w1.input_scale", "F32", 368_640),
    # Head-sharded attention: fp8 projections (wq_b, indexer.wq_b) and
    # their e8m0 scales, the indexer's weights_proj, attn_sink.
    CheckpointTensor("layers.0.attn.wq_b.weight", "F8_E4M3", 1_677_721_600),
    CheckpointTensor("layers.0.attn.wq_b.scale", "F8_E8M0", 1_638_400),
    CheckpointTensor(
        "layers.0.attn.indexer.wq_b.weight", "F8_E4M3", 41_943_040
    ),
    CheckpointTensor("layers.0.attn.indexer.wq_b.scale", "F8_E8M0", 40_960),
    CheckpointTensor(
        "layers.0.attn.indexer.weights_proj.weight", "BF16", 2_621_440
    ),
    CheckpointTensor("layers.0.attn.attn_sink", "F32", 10_240),
    # wo_a: MXFP8 by default, host-dequantized to bf16 with its switch off.
    CheckpointTensor("layers.0.attn.wo_a.weight", "F8_E4M3", 1_342_177_280),
    CheckpointTensor("layers.0.attn.wo_a.scale", "F8_E8M0", 1_310_720),
    # Replicated on every device: wo_b, embedding, head, the shared
    # experts' fp8 projections and their scales.
    CheckpointTensor("layers.0.attn.wo_b.weight", "F8_E4M3", 1_677_721_600),
    CheckpointTensor("layers.0.attn.wo_b.scale", "F8_E8M0", 1_638_400),
    CheckpointTensor("embed.weight", "BF16", 1_323_827_200),
    CheckpointTensor("head.weight", "BF16", 1_323_827_200),
    CheckpointTensor(
        "layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", 1_415_577_600
    ),
    CheckpointTensor(
        "layers.0.ffn.shared_experts.w1.scale", "F8_E8M0", 1_382_400
    ),
    # Engram's wkv, split by columns across devices: weight and scale.
    CheckpointTensor("layers.1.engram.wkv.weight", "F8_E4M3", 314_572_800),
    CheckpointTensor("layers.1.engram.wkv.scale", "F8_E8M0", 307_200),
    # wq_a and wkv: dequantized into one fused bf16 projection, scales folded.
    CheckpointTensor("layers.0.attn.wq_a.weight", "F8_E4M3", 262_144_000),
    CheckpointTensor("layers.0.attn.wq_a.scale", "F8_E8M0", 256_000),
    CheckpointTensor("layers.0.attn.wkv.weight", "F8_E4M3", 104_857_600),
    CheckpointTensor("layers.0.attn.wkv.scale", "F8_E8M0", 102_400),
    # The router gate (float32 on device), other bf16, mHC, norms.
    CheckpointTensor("layers.0.ffn.gate.weight", "BF16", 157_286_400),
    CheckpointTensor("layers.0.attn.q_norm.weight", "BF16", 524_288),
    CheckpointTensor("layers.0.hc_attn_fn", "F32", 157_417_920),
    CheckpointTensor("layers.0.attn_norm.weight", "BF16", 37_841_920),
    # Pinned host memory: Engram's hash tables and their e8m0 scales.
    CheckpointTensor(
        "layers.1.engram.embed.weight", "F8_E4M3", 196_613_849_600
    ),
    CheckpointTensor("layers.1.engram.embed.scale", "F8_E8M0", 6_144_182_800),
    # Not loaded by the trunk: DSpark stages and the vision tower.
    CheckpointTensor("mtp.0.ffn.experts.0.w1.weight", "I8", 7_572_112_128),
    CheckpointTensor("vision.blocks.0.mlp.w1.weight", "BF16", 1_331_299_464),
]

_DEFAULT_LAYOUT = DeviceWeightLayout(
    head_bf16=True,
    host_dequantized=lambda base: False,
    expert_scales_preinterleaved=True,
)


# Per-device free memory on a B200 with
# MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=97 (173 GiB of
# 178.35 GiB), and the signal buffers planned at TP2.
_TP2_MM97_FREE = 185_757_335_552
_TP2_SIGNAL_BYTES = 588_775_424
# Measured TP2 per-device KV pool for max_length = 40000.
_TP2_KV_40K_PER_DEVICE = 2.3 * GIB


def _tp2_devices() -> list[Device]:
    devices: list[Device] = []
    for _ in range(2):
        device = MagicMock()
        device.stats = {
            "free_memory": _TP2_MM97_FREE,
            "total_memory": _TP2_MM97_FREE,
        }
        devices.append(device)
    return devices


def _tp2_plan_inputs(
    *, chunk: int, max_length: int, layout: DeviceWeightLayout
) -> tuple[NonCallableMock, int, int]:
    config = _pipeline_config(
        max_length=max_length, max_num_input_tokens=chunk, num_devices=2
    )
    config.runtime.max_batch_input_tokens = chunk
    weights, _ = device_weight_bytes(_REAL_CHECKPOINT, 2, layout=layout)
    activation = _planner().estimate_activation_memory(
        config, _real_huggingface_config()
    )
    return config, 2 * weights, activation


def _kv_budget_per_device(
    *,
    chunk: int,
    max_length: int,
    utilization: float,
    layout: DeviceWeightLayout,
) -> float:
    config, weights, activation = _tp2_plan_inputs(
        chunk=chunk, max_length=max_length, layout=layout
    )
    config.model.kv_cache.device_memory_utilization = utilization
    return (
        MemoryEstimator._available_kv_cache_memory(
            weights,
            activation,
            config.model,
            _tp2_devices(),
            _TP2_SIGNAL_BYTES,
        )
        / 2
    )


@pytest.mark.parametrize("utilization", [0.97, 1.0])
def test_tp2_4096_row_chunk_at_40k_leaves_the_kv_pool_room(
    utilization: float,
) -> None:
    """Weights, forward reserve and a 40000-token KV pool fit the budget."""
    kv = _kv_budget_per_device(
        chunk=4096,
        max_length=40000,
        utilization=utilization,
        layout=_DEFAULT_LAYOUT,
    )
    assert kv >= _TP2_KV_40K_PER_DEVICE
    _, weights, activation = _tp2_plan_inputs(
        chunk=4096, max_length=40000, layout=_DEFAULT_LAYOUT
    )
    assert (weights + activation + _TP2_SIGNAL_BYTES) / 2 + kv <= (
        _TP2_MM97_FREE
    )


@pytest.mark.parametrize("max_length", [40000, 131072, 1048576])
def test_reserve_grows_only_by_the_candidate_keep_step(
    max_length: int,
) -> None:
    """Past the arena, the reserve is the indexer's score window plus the
    ``[T, blocks]`` keep step, never a ``[T, max_length]`` score."""
    rows = 8192
    _, _, activation = _tp2_plan_inputs(
        chunk=rows, max_length=max_length, layout=_DEFAULT_LAYOUT
    )
    keep = rows * -(-max_length // 8) * 28
    assert activation / 2 - keep <= (1.25 + 16 + 0.5 + 0.25) * GIB


def test_tp2_8192_row_chunk_at_40k_fits_the_full_budget() -> None:
    """Without ``[T, S]`` scores an 8192-row chunk at 40000 fits TP2 at
    utilization 1.0."""
    config, weights, activation = _tp2_plan_inputs(
        chunk=8192, max_length=40000, layout=_DEFAULT_LAYOUT
    )
    config.model.kv_cache.device_memory_utilization = 1.0
    kv = MemoryEstimator._available_kv_cache_memory(
        weights, activation, config.model, _tp2_devices(), _TP2_SIGNAL_BYTES
    )
    assert kv / 2 >= _TP2_KV_40K_PER_DEVICE


def test_tp2_in_graph_interleave_layout_is_refused_at_plan_time() -> None:
    """In-graph scale interleave at TP2 fails planning for 40000/4096 rows."""
    in_graph = DeviceWeightLayout(
        head_bf16=True,
        host_dequantized=lambda base: False,
        expert_scales_preinterleaved=False,
    )
    config, weights, activation = _tp2_plan_inputs(
        chunk=4096, max_length=40000, layout=in_graph
    )
    config.model.kv_cache.device_memory_utilization = 0.97
    with pytest.raises(
        RuntimeError, match=r"exceeds available memory|room for KV cache"
    ):
        MemoryEstimator.plan_from_sizes(
            config,
            config.model,
            MagicMock(),
            _tp2_devices(),
            weights,
            activation,
            _TP2_SIGNAL_BYTES,
        )


def test_dp4_splits_only_the_routed_experts() -> None:
    """Under DP4 each device holds attention and Engram whole and a quarter
    of the routed experts."""
    experts = [
        t
        for t in _REAL_CHECKPOINT
        if t.name.startswith("layers.") and ".ffn.experts." in t.name
    ]
    whole, _ = device_weight_bytes(_REAL_CHECKPOINT, 1, layout=_DEFAULT_LAYOUT)
    expert_bytes, _ = device_weight_bytes(experts, 1, layout=_DEFAULT_LAYOUT)
    dp4, _ = device_weight_bytes(
        _REAL_CHECKPOINT, 4, layout=_DEFAULT_LAYOUT, data_parallel=True
    )
    assert abs(dp4 - (whole - expert_bytes * 3 / 4)) <= 1


def _dp4_config(data_parallel_degree: int) -> NonCallableMock:
    config = _pipeline_config(
        max_length=40000, max_num_input_tokens=8192, num_devices=4
    )
    config.model.data_parallel_degree = data_parallel_degree
    for i, spec in enumerate(config.model.device_specs):
        spec.id = i
    config.speculative = None
    return config


def test_dp4_reserves_the_expert_parallel_buffers() -> None:
    """Each DP4 device also reserves the expert-parallel buffers and 12 KiB
    for each row its experts can receive: every replica's 8192 rows, six
    routed copies each."""
    hf_config = _real_huggingface_config()
    extra = (
        _planner().estimate_activation_memory(_dp4_config(4), hf_config)
        - _planner().estimate_activation_memory(_dp4_config(1), hf_config)
    ) // 4
    ep = data_parallel_ep_config(
        _dp4_config(4), DeepseekV41Config.get_text_config(hf_config)
    )
    assert ep is not None
    assert ep.get_max_recv_tokens() == 4 * 8192 * 6
    assert extra == ep.estimate_memory_usage() + 4 * 8192 * 6 * (12 << 10)
