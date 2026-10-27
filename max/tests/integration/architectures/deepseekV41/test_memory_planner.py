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

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.driver import Device
from max.dtype import DType
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41.layers.indexer import (
    candidate_window_rows,
)
from max.pipelines.architectures.deepseekV41.memory_planner import (
    CheckpointTensor,
    DeepseekV41MemoryPlanner,
    DeviceWeightLayout,
    csa2_activation_bytes,
    device_weight_bytes,
    ep_receive_pool_bytes,
    graph_arena_bytes,
    graph_capture_pool_bytes,
    kv_leaf_bytes_per_token,
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
    config.runtime.device_graph_capture = False
    return config


def _real_huggingface_config() -> AutoConfig:
    """Returns the real checkpoint's top-level config, text config nested."""
    return AutoConfig.from_pretrained(str(TESTDATA / "config.json"))


def _planner() -> DeepseekV41MemoryPlanner:
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock(num_draft_tokens=0)
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
    # The router gate (bf16 on the fused route), other bf16, mHC, norms.
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
    router_gate_float32=False,
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
    """Past the arena and the attention's fixed workspaces, the reserve is
    the indexer's score window plus the ``[T, kept]`` block ids and a fixed
    window scratch, never a ``[T, blocks]`` or ``[T, max_length]`` tensor."""
    rows = 8192
    _, _, activation = _tp2_plan_inputs(
        chunk=rows, max_length=max_length, layout=_DEFAULT_LAYOUT
    )
    kept_ids = rows * 2048 * 4
    blocks = -(-max_length // 8)
    window = candidate_window_rows(rows, blocks, 8) * (blocks * 4 + 4)
    keep = kept_ids + -(-window // (256 << 20)) * (256 << 20)
    attention = csa2_activation_bytes(live_rows=rows, num_heads=64)
    assert attention == GIB // 2
    # Arena, score window, window scratch page and one page of headroom.
    assert (
        activation / 2 - keep - attention
        <= (1.25 + 16 + 0.5 + 0.25 + 0.25) * GIB
    )
    # Only the window scratch grows with max_length, by at most one page.
    _, _, short = _tp2_plan_inputs(
        chunk=rows, max_length=40000, layout=_DEFAULT_LAYOUT
    )
    assert activation / 2 - short / 2 <= 256 << 20


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


def test_router_gate_doubles_only_on_the_graph_route() -> None:
    """The fused route reads the bf16 gate as is; the graph route holds a
    float32 copy."""
    gates = [
        CheckpointTensor("layers.0.ffn.gate.weight", "BF16", 3_932_160),
        CheckpointTensor("layers.1.ffn.gate.weight", "BF16", 3_932_160),
    ]
    fused, _ = device_weight_bytes(gates, 2, layout=_DEFAULT_LAYOUT)
    graph, _ = device_weight_bytes(
        gates,
        2,
        layout=dataclasses.replace(_DEFAULT_LAYOUT, router_gate_float32=True),
    )
    assert fused == 2 * 3_932_160
    assert graph == 2 * fused


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
    """Each DP4 device also reserves the expert-parallel buffers and the
    receive tensors for every replica's 8192 rows, six routed copies each,
    once live and once as fragmentation headroom; its arena drops to 1.25 MiB
    per row."""
    hf_config = _real_huggingface_config()
    extra = (
        _planner().estimate_activation_memory(_dp4_config(4), hf_config)
        - _planner().estimate_activation_memory(_dp4_config(1), hf_config)
    ) // 4
    ep = data_parallel_ep_config(
        _dp4_config(4), DeepseekV41Config.get_text_config(hf_config)
    )
    assert ep is not None
    rows = 4 * 8192 * 6
    assert ep.get_max_recv_tokens() == rows
    pool = ep_receive_pool_bytes(ep)
    # NVFP4 rows, bf16 down projection, source info, block scales.
    assert pool >= rows * (5120 // 2 + 5120 * 2 + 8)
    arena_drop = 8192 * ((2 << 20) - (5 << 18))
    assert extra == (
        ep.estimate_memory_usage() + 2 * pool - (256 << 20) - arena_drop
    )


# Measured on four B200s, DP4EP4, DSpark k=5, max_batch_size 128,
# max_batch_input_tokens 8192, max_length 1048576 (MemoryManager log): the
# largest per-forward planned pool; device 0's captured graphs' pool with one
# graph's working memory live at a time; device 0's extra head bytes over
# 3072 x 129280 entries (bfloat16 logits, their float32 cast, and sampling
# and verification working memory); and device 0's eager transient peak over
# an 8192-row prefill chunk.
_DP4_MEASURED_FORWARD_POOL = 2_790_020_608
_DP4_MEASURED_GRAPH_POOL = 9_620_885_504
_DP4_MEASURED_HEAD_BYTES = 794_296_320 + 1_588_592_640 + 1_721_597_952
_DP4_MEASURED_EAGER_PEAK = 13.05 * GIB


def _dp4_served_config(*, graph_capture: bool) -> NonCallableMock:
    config = _dp4_config(4)
    config.model.max_length = 1048576
    config.runtime.max_batch_size = 128
    config.runtime.device_graph_capture = graph_capture
    config.speculative = MagicMock()
    config.speculative.is_dspark.return_value = True
    config.speculative.num_speculative_tokens = 5
    return config


def test_dp4_reserves_the_measured_graph_pool_and_eager_peak() -> None:
    """The served DP4 plan reserves at least the measured captured-graph
    pool, and its eager terms cover the measured eager peak."""
    hf_config = _real_huggingface_config()
    captured = _planner().estimate_activation_memory(
        _dp4_served_config(graph_capture=True), hf_config
    )
    eager_only = _planner().estimate_activation_memory(
        _dp4_served_config(graph_capture=False), hf_config
    )
    ep = data_parallel_ep_config(
        _dp4_served_config(graph_capture=True),
        DeepseekV41Config.get_text_config(hf_config),
    )
    assert ep is not None
    assert ep.get_max_recv_tokens() == (8192 + 128 * 5) * 24
    pool = ep_receive_pool_bytes(ep)
    assert pool >= _DP4_MEASURED_FORWARD_POOL
    graph_pool = (captured - eager_only) // 4
    assert graph_pool >= _DP4_MEASURED_GRAPH_POOL
    # Within 1.5 GiB of the measurement: tight enough not to starve KV.
    assert graph_pool <= _DP4_MEASURED_GRAPH_POOL + 1.5 * GIB
    head = graph_capture_pool_bytes(
        capture_rows=1,
        head_rows=4 * 128 * 6,
        vocab_size=129280,
        row_bytes=0,
        indexer_bytes=0,
        attention_bytes=0,
        ep_pool_bytes=0,
    ) - graph_arena_bytes(live_rows=1, row_bytes=0)
    assert _DP4_MEASURED_HEAD_BYTES <= head
    assert head <= _DP4_MEASURED_HEAD_BYTES + GIB // 2
    # Arena and receive tensors alone, without the indexer or headroom.
    arena = -(-((5 << 28) + 8192 * (5 << 18)) // (256 << 20)) * (256 << 20)
    assert arena + pool >= _DP4_MEASURED_EAGER_PEAK


def test_graph_capture_pool_holds_the_csa2_workspaces() -> None:
    """Captured decode graphs run the CSA2 attention, so the pool holds its
    workspaces on top of the other terms."""
    attention = csa2_activation_bytes(live_rows=128 * 6, num_heads=64)
    assert attention > 0

    def pool(attention_bytes: int) -> int:
        return graph_capture_pool_bytes(
            capture_rows=128 * 6,
            head_rows=4 * 128 * 6,
            vocab_size=129280,
            row_bytes=0,
            indexer_bytes=0,
            attention_bytes=attention_bytes,
            ep_pool_bytes=0,
        )

    assert pool(attention) == pool(0) + attention


def _bytes_per_token(csa_records: bool) -> dict[str, float]:
    return kv_leaf_bytes_per_token(
        DeepseekV41Config.get_text_config(_real_huggingface_config()),
        page_size=128,
        cache_dtype=DType.bfloat16,
        csa_records=csa_records,
    )


def test_kv_bytes_per_token_per_leaf() -> None:
    """bf16 rows take 1 KiB per layer per token in the window and every
    compressed leaf; records take a 528-byte slot per token in the window
    and a 288-byte slot per compression group."""
    unchanged = {"c2a_state": 192.0, "idx_c2a": 102.0, "idx_c1a": 68.0}
    assert _bytes_per_token(False) == {
        "swa": 43 * 1024.0,
        "c2a": 3 * 1024.0,
        "c1a": 1024.0,
        **unchanged,
    }
    assert _bytes_per_token(True) == {
        "swa": 43 * 528.0,
        "c2a": 3 * 288 / 2,
        "c1a": 288.0,
        **unchanged,
    }


def test_dp4_kv_capacity_per_device() -> None:
    """Context tokens a DP4 device's KV budget holds, counting only the
    leaves that grow with context: about 4.8x more with records, less the
    record attention's workspaces."""
    hf_config = _real_huggingface_config()
    config = _dp4_config(4)
    config.model.kv_cache.device_memory_utilization = 0.97
    weights, _ = device_weight_bytes(
        _REAL_CHECKPOINT, 4, layout=_DEFAULT_LAYOUT, data_parallel=True
    )
    activation = _planner().estimate_activation_memory(config, hf_config)
    devices: list[Device] = []
    for _ in range(4):
        device = MagicMock()
        device.stats = {
            "free_memory": _TP2_MM97_FREE,
            "total_memory": _TP2_MM97_FREE,
        }
        devices.append(device)
    budget = (
        MemoryEstimator._available_kv_cache_memory(
            4 * weights, activation, config.model, devices, _TP2_SIGNAL_BYTES
        )
        / 4
    )
    windowed = ("swa", "c2a_state")
    capacity = {
        records: budget
        / sum(
            v for k, v in _bytes_per_token(records).items() if k not in windowed
        )
        for records in (False, True)
    }
    assert 60 * GIB < budget < 61 * GIB
    assert 15.2e6 < capacity[False] < 15.3e6
    assert 73.0e6 < capacity[True] < 73.2e6
