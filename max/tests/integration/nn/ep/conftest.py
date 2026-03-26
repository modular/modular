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

import os
from dataclasses import dataclass

import pytest
import torch
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.moe import MoE, MoEGate
from max.nn.transformer.distributed_transformer import (
    forward_sharded_layers,
)

"""
Fixtures for EP tests, including dummy weights.
"""

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64
WEIGHTS_STDDEV = 0.01
N_DEVICES = 4
TOP_K = 8


@dataclass
class CompiledEPModels:
    """Pre-compiled MoE and MoEGate models for EP integration tests."""

    moe_model: Model
    gate_model: Model
    ep_comm_init: EPCommInitializer
    devices: list[Accelerator]


@pytest.fixture
def moe_weights_fp8() -> dict[str, torch.Tensor]:
    """Generate FP8 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fp8_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float32
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = torch.finfo(fp8_dtype).min

    moe_weights = {}

    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * WEIGHTS_STDDEV
    )

    FP8_WEIGHTS_MULTIPLIER = 100
    SCALE_MIN = 0.5 * (WEIGHTS_STDDEV / FP8_WEIGHTS_MULTIPLIER)
    SCALE_MAX = WEIGHTS_STDDEV / FP8_WEIGHTS_MULTIPLIER
    for expert_idx in range(NUM_EXPERTS):
        moe_weights[f"experts.{expert_idx}.gate_proj.weight"] = (
            (
                torch.randn(
                    MOE_DIM,
                    HIDDEN_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.gate_proj.weight_scale"] = (
            torch.rand(
                MOE_DIM // 128,
                HIDDEN_DIM // 128,
                dtype=scale_dtype,
                device=device,
            )
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

        moe_weights[f"experts.{expert_idx}.up_proj.weight"] = (
            (
                torch.randn(
                    MOE_DIM,
                    HIDDEN_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.up_proj.weight_scale"] = (
            torch.rand(
                MOE_DIM // 128,
                HIDDEN_DIM // 128,
                dtype=scale_dtype,
                device=device,
            )
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

        moe_weights[f"experts.{expert_idx}.down_proj.weight"] = (
            (
                torch.randn(
                    HIDDEN_DIM,
                    MOE_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.down_proj.weight_scale"] = (
            torch.rand(
                HIDDEN_DIM // 128,
                MOE_DIM // 128,
                dtype=scale_dtype,
                device=device,
            )
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

    moe_weights["shared_experts.down_proj.weight"] = (
        (
            torch.randn(
                HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16, device=device
            )
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.down_proj.weight_scale"] = (
        torch.rand(
            HIDDEN_DIM // 128, MOE_DIM // 128, dtype=scale_dtype, device=device
        )
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )
    moe_weights["shared_experts.gate_proj.weight"] = (
        (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.gate_proj.weight_scale"] = (
        torch.rand(
            MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype, device=device
        )
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )
    moe_weights["shared_experts.up_proj.weight"] = (
        (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.up_proj.weight_scale"] = (
        torch.rand(
            MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype, device=device
        )
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )

    return moe_weights


@pytest.fixture
def moe_weights_fp4() -> dict[str, torch.Tensor]:
    """Generate FP4 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fp4_scale_min = 50.0
    fp4_scale_max = 150.0

    def _add_fp4_proj(
        moe_weights: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
        weight_scale_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = torch.randint(
            0,
            256,
            (out_dim, in_dim // 2),
            dtype=torch.uint8,
            device=device,
        )
        weight_scale = (
            torch.rand(
                out_dim,
                weight.shape[1] // 8,
                dtype=torch.float32,
                device=device,
            )
            * (fp4_scale_max - fp4_scale_min)
            + fp4_scale_min
        ).to(torch.float8_e4m3fn)

        if weight_scale_2 is None:
            weight_scale_2 = (
                torch.rand((), dtype=torch.float32, device=device) * 1e-4
            )

        moe_weights[f"{prefix}.weight"] = weight
        moe_weights[f"{prefix}.weight_scale"] = weight_scale
        moe_weights[f"{prefix}.weight_scale_2"] = weight_scale_2
        moe_weights[f"{prefix}.input_scale"] = torch.ones(
            (), dtype=torch.float32, device=device
        )
        return weight_scale_2

    moe_weights = {}

    # Gate weights for router
    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    )

    # Individual expert weights -- gate_proj and up_proj share weight_scale_2.
    for expert_idx in range(NUM_EXPERTS):
        gate_up_scale_2 = _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.gate_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.up_proj",
            MOE_DIM,
            HIDDEN_DIM,
            weight_scale_2=gate_up_scale_2,
        )
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
        )

    # Shared experts weights -- gate_proj and up_proj share weight_scale_2.
    shared_gate_up_scale_2 = _add_fp4_proj(
        moe_weights,
        "shared_experts.gate_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_fp4_proj(
        moe_weights,
        "shared_experts.up_proj",
        MOE_DIM,
        HIDDEN_DIM,
        weight_scale_2=shared_gate_up_scale_2,
    )
    _add_fp4_proj(
        moe_weights,
        "shared_experts.down_proj",
        HIDDEN_DIM,
        MOE_DIM,
    )

    return moe_weights


@pytest.fixture(scope="module")
def moe_weights() -> dict[str, torch.Tensor]:
    """Generate random BF16 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    moe_weights: dict[str, torch.Tensor] = {}

    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * WEIGHTS_STDDEV
    )

    for expert_idx in range(NUM_EXPERTS):
        moe_weights[f"experts.{expert_idx}.gate_proj.weight"] = (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * WEIGHTS_STDDEV
        )
        moe_weights[f"experts.{expert_idx}.up_proj.weight"] = (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * WEIGHTS_STDDEV
        )
        moe_weights[f"experts.{expert_idx}.down_proj.weight"] = (
            torch.randn(
                HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16, device=device
            )
            * WEIGHTS_STDDEV
        )

    moe_weights["shared_experts.gate_proj.weight"] = (
        torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
        * WEIGHTS_STDDEV
    )
    moe_weights["shared_experts.up_proj.weight"] = (
        torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
        * WEIGHTS_STDDEV
    )
    moe_weights["shared_experts.down_proj.weight"] = (
        torch.randn(HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16, device=device)
        * WEIGHTS_STDDEV
    )

    return moe_weights


@pytest.fixture(scope="module")
def compiled_ep_models(
    moe_weights: dict[str, torch.Tensor],
) -> CompiledEPModels | None:
    """Compile MoE and MoEGate graphs once, shared across parametrized runs.

    Returns None when hardware requirements are not met.
    """
    if accelerator_count() < N_DEVICES:
        return None

    n_devices = N_DEVICES
    max_tokens_per_rank = 128
    dtype = DType.bfloat16

    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=dtype,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
    )

    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    moe = MoE(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=dtype,
        apply_router_weight_first=False,
        ep_batch_manager=ep_batch_manager,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)
    moe_weights_cpu = {k: v.cpu() for k, v in moe_weights.items()}
    moe.load_state_dict(moe_weights_cpu)

    ep_comm_init.ep_init(session)

    per_device_input_types: list[TensorType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", HIDDEN_DIM),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]

    with Graph(
        "EPMoE",
        input_types=[
            *per_device_input_types,
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        inputs_tensors = [x.tensor for x in graph.inputs[:n_devices]]
        ep_batch_manager.fetch_buffers(graph.inputs[n_devices:])
        outputs = forward_sharded_layers(moe_shards, inputs_tensors)
        graph.output(*outputs)

    moe_model = session.load(graph, weights_registry=moe.state_dict())

    moe_gate = MoEGate(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        dtype=dtype,
    )
    moe_gate.sharding_strategy = ShardingStrategy.replicate(n_devices)
    moe_gate_shards = moe_gate.shard(devices_ref)

    gate_weight_dict = {
        "gate_score.weight": moe_weights_cpu["gate.gate_score.weight"]
    }
    moe_gate.load_state_dict(gate_weight_dict)

    with Graph(
        "MoEGate",
        input_types=per_device_input_types,
    ) as gate_graph:
        gate_inputs = [x.tensor for x in gate_graph.inputs[:n_devices]]
        gate_outputs: list[TensorValue] = []
        for moe_gate_shard, inp in zip(
            moe_gate_shards, gate_inputs, strict=False
        ):
            gate_outputs.extend(moe_gate_shard(inp))
        gate_graph.output(*gate_outputs)

    gate_model = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )

    return CompiledEPModels(
        moe_model=moe_model,
        gate_model=gate_model,
        ep_comm_init=ep_comm_init,
        devices=devices,
    )
