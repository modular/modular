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
"""Tests the serving MoE's tensor-parallel split of its interleaved scales.

MAX Python holds each expert stack whole and slices it per device in the
graph, while Mach's loader interleaves each device's row-major slice on its
own. The two agree only if slicing the stored layout on the module's own
sharding strategies gives each device the interleave of its slice.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy
from max.pipelines.architectures.mimo_v2.layers.moe import MiMoV2MoE
from max.pipelines.architectures.mimo_v2.quant import parse_quant_scheme
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    interleave_e8m0,
)
from transformers.configuration_utils import PretrainedConfig

EXPERTS, HIDDEN, WIDTH = 2, 256, 512
SCALES = ("experts_gate_up_proj_scale", "experts_down_proj_scale")


def _moe(scale_dtype: DType = DType.float8_e8m0fnu) -> MiMoV2MoE:
    config = PretrainedConfig(
        num_hidden_layers=2,
        moe_layer_freq=[0, 1],
        n_routed_experts=EXPERTS,
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                f"model.layers.1.mlp.experts.{e}.{p}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for e in range(EXPERTS)
                for p in ("gate_proj", "up_proj", "down_proj")
            },
        },
    )
    experts = parse_quant_scheme(config).experts
    weight_scale = dataclasses.replace(experts.weight_scale, dtype=scale_dtype)
    return MiMoV2MoE(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_dim=WIDTH,
        norm_topk_prob=True,
        quant_config=dataclasses.replace(experts, weight_scale=weight_scale),
        devices=[DeviceRef.CPU()],
    )


def _rank_slice(name: str, scales: np.ndarray, rank: int, n: int) -> np.ndarray:
    """Rank ``rank``'s row-major scales: its gate rows then its up rows, or
    its down columns."""
    if name == "experts_gate_up_proj_scale":
        rows = WIDTH // n
        gate = scales[:, rank * rows : (rank + 1) * rows]
        up = scales[:, WIDTH + rank * rows : WIDTH + (rank + 1) * rows]
        return np.concatenate([gate, up], axis=1)
    cols = scales.shape[2] // n
    return scales[:, :, rank * cols : (rank + 1) * cols]


def _interleaved(scales: np.ndarray) -> np.ndarray:
    return np.stack([interleave_e8m0(np.ascontiguousarray(e)) for e in scales])


@pytest.mark.parametrize("num_devices", [2, 4])
def test_each_shard_is_the_interleave_of_its_slice(num_devices: int) -> None:
    rng = np.random.default_rng(0)
    row_major = {
        "experts_gate_up_proj_scale": rng.integers(
            0, 255, (EXPERTS, 2 * WIDTH, HIDDEN // 32), dtype=np.uint8
        ),
        "experts_down_proj_scale": rng.integers(
            0, 255, (EXPERTS, HIDDEN, WIDTH // 32), dtype=np.uint8
        ),
    }
    # The CPU has no E8M0 type, and slicing moves bytes whatever their type.
    moe = _moe(scale_dtype=DType.uint8)
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
    state = {
        name: Buffer.from_numpy(_interleaved(scales))
        for name, scales in row_major.items()
    }
    moe.load_state_dict(state, weight_alignment=1, strict=False)
    shards = moe.shard([DeviceRef.CPU()] * num_devices)

    with Graph("scale_shards") as graph:
        graph.output(
            *(getattr(shard, name) for name in SCALES for shard in shards)
        )
    outputs = (
        InferenceSession(devices=[CPU()])
        .load(graph, weights_registry=state)
        .execute()
    )

    for i, name in enumerate(SCALES):
        for rank in range(num_devices):
            got = outputs[i * num_devices + rank]
            assert isinstance(got, Buffer)
            want = _interleaved(
                _rank_slice(name, row_major[name], rank, num_devices)
            )
            np.testing.assert_array_equal(
                got.to_numpy(), want, err_msg=f"{name} rank {rank}"
            )


def test_serving_declares_only_the_expert_stacks() -> None:
    weights = _moe().raw_state_dict()

    assert sorted(weights) == [
        "experts_down_proj",
        "experts_down_proj_scale",
        "experts_gate_up_proj",
        "experts_gate_up_proj_scale",
        "gate.e_score_correction_bias",
        "gate.gate_score.weight",
    ]
    # The scales in the grouped matmul's layout, [E, N/128, K/128, 32, 4, 4].
    shapes = {name: list(weight.shape) for name, weight in weights.items()}
    assert shapes["experts_gate_up_proj"] == [EXPERTS, 2 * WIDTH, HIDDEN // 2]
    assert shapes["experts_gate_up_proj_scale"] == [
        EXPERTS,
        2 * WIDTH // 128,
        HIDDEN // 128,
        32,
        4,
        4,
    ]
    assert shapes["experts_down_proj"] == [EXPERTS, HIDDEN, WIDTH // 2]
    assert shapes["experts_down_proj_scale"] == [
        EXPERTS,
        HIDDEN // 128,
        WIDTH // 128,
        32,
        4,
        4,
    ]
