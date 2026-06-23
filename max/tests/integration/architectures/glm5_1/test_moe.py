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
"""MoE layer test for GLM-5.1 (sigmoid + noaux_tc routing)."""

from __future__ import annotations

import functools

import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Device, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.linear import Linear
from max.nn.moe import MoE
from max.pipelines.architectures.deepseekV3_2.layers import (
    DeepseekV3_2MLP,
    DeepseekV3_2TopKRouter,
)
from torch_reference.configuration_glm import GlmMoeDsaConfig
from torch_reference.modeling_glm import GlmMoeDsaMoE

pytestmark = [
    pytest.mark.skipif(
        not __import__(
            "torch_reference.modeling_glm", fromlist=["TORCH_REFERENCE_READY"]
        ).TORCH_REFERENCE_READY,
        reason="GLM torch reference not installed",
    ),
    pytest.mark.skipif(
        accelerator_api() == "hip",
        reason=(
            "AMD ROCm (HIP): fused MoE kernel requires WARP_SIZE divisible by "
            "num_threads; incompatible with n_routed_experts=32 in testdata "
            "(max/kernels/src/nn/moe.mojo)"
        ),
    ),
]


def generate_torch_outputs(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    shared_expert_weights: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    layer = GlmMoeDsaMoE(config).to(torch.bfloat16).to(device)
    layer.eval()

    input_tensor = input_tensor.to(device)
    layer.gate.weight.data = gate_weight.to(device)
    layer.experts.gate_up_proj.data = gate_up_proj.to(device)
    layer.experts.down_proj.data = down_proj.to(device)

    for name, param in layer.shared_experts.named_parameters():
        param.data = shared_expert_weights[name].to(torch.bfloat16).to(device)

    return layer(input_tensor)


def generate_max_outputs(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    shared_expert_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()

    moe_dim = config.moe_intermediate_size
    hidden_dim = config.hidden_size
    n_experts = config.n_routed_experts

    state_dict: dict[str, torch.Tensor] = {
        "gate.gate_score.weight": gate_weight.cpu(),
        "gate.e_score_correction_bias": torch.zeros(
            n_experts, dtype=torch.float32
        ),
    }
    for i in range(n_experts):
        gate_up = gate_up_proj[i].cpu()
        gate_w, up_w = gate_up.chunk(2, dim=0)
        state_dict[f"experts.{i}.gate_proj.weight"] = gate_w
        state_dict[f"experts.{i}.up_proj.weight"] = up_w
        state_dict[f"experts.{i}.down_proj.weight"] = down_proj[i].cpu()

    state_dict["shared_experts.gate_proj.weight"] = shared_expert_weights[
        "gate_proj.weight"
    ].cpu()
    state_dict["shared_experts.down_proj.weight"] = shared_expert_weights[
        "down_proj.weight"
    ].cpu()
    state_dict["shared_experts.up_proj.weight"] = shared_expert_weights[
        "up_proj.weight"
    ].cpu()

    assert config.n_shared_experts is not None
    shared_dim = config.moe_intermediate_size * config.n_shared_experts
    # BF16 MoE layers use ``MoE`` in prod; ``DeepseekV3_2MoE`` is for quantized paths only.
    moe = MoE(
        dtype=dtype,
        devices=[DeviceRef.GPU()] if is_gpu else [DeviceRef.CPU()],
        hidden_dim=hidden_dim,
        num_experts=n_experts,
        num_experts_per_token=config.num_experts_per_tok,
        moe_dim=moe_dim,
        gate_cls=functools.partial(
            DeepseekV3_2TopKRouter,
            routed_scaling_factor=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            gate_dtype=DType.bfloat16,
            correction_bias_dtype=DType.float32,
            linear_cls=Linear,
        ),
        mlp_cls=DeepseekV3_2MLP,
        has_shared_experts=True,
        shared_experts_dim=shared_dim,
        apply_router_weight_first=False,
    )
    moe.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator(0)])
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "GlmMoeDsaMoE",
        moe,
        input_types=(
            TensorType(
                dtype,
                (input_tensor.shape[0], hidden_dim),
                device=DeviceRef.GPU() if is_gpu else DeviceRef.CPU(),
            ),
        ),
    )
    compiled = session.load(graph, weights_registry=moe.state_dict())
    return compiled.execute(input_tensor)


def test_moe(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    shared_expert_weights: dict[str, torch.Tensor],
) -> None:
    torch_dtype = torch.bfloat16
    max_dtype = DType.bfloat16
    torch_output = generate_torch_outputs(
        config,
        input_tensor,
        gate_weight,
        gate_up_proj,
        down_proj,
        shared_expert_weights,
        torch.device("cuda"),
    )
    max_output = generate_max_outputs(
        config,
        input_tensor.squeeze(),
        gate_weight,
        gate_up_proj,
        down_proj,
        shared_expert_weights,
        max_dtype,
        Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(),
        torch.from_dlpack(max_output[0]).to(torch_dtype).squeeze(),
        rtol=1e-4,
        atol=2 * torch.finfo(torch.bfloat16).eps,
    )
