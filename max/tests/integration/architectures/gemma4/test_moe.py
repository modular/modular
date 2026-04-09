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
"""Tests for Gemma4 MoE."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    MOE_TEXT_HIDDEN_ACTIVATION,
    MOE_TEXT_HIDDEN_SIZE,
    MOE_TEXT_MOE_INTERMEDIATE_SIZE,
    MOE_TEXT_NUM_EXPERTS,
    MOE_TEXT_RMS_NORM_EPS,
    MOE_TEXT_TOP_K_EXPERTS,
    TorchGemma4TextExperts,
    TorchGemma4TextRouter,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.layers.moe import (
    Gemma4TextExperts,
    Gemma4TextRouter,
)
from transformers import PreTrainedConfig

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

torch.manual_seed(42)


def generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


@pytest.fixture()
def text_moe_block_state_dict() -> dict[str, torch.Tensor]:
    return {
        "gate_up_proj": generate_tensor(
            (
                MOE_TEXT_NUM_EXPERTS,
                2 * MOE_TEXT_MOE_INTERMEDIATE_SIZE,
                MOE_TEXT_HIDDEN_SIZE,
            )
        ),
        "down_proj": generate_tensor(
            (
                MOE_TEXT_NUM_EXPERTS,
                MOE_TEXT_HIDDEN_SIZE,
                MOE_TEXT_MOE_INTERMEDIATE_SIZE,
            )
        ),
    }


@pytest.fixture()
def max_text_moe_block() -> Gemma4TextExperts:
    model = Gemma4TextExperts(
        MAX_DTYPE,
        DeviceRef.GPU(),
        MOE_TEXT_NUM_EXPERTS,
        MOE_TEXT_TOP_K_EXPERTS,
        MOE_TEXT_HIDDEN_SIZE,
        MOE_TEXT_MOE_INTERMEDIATE_SIZE,
    )
    return model


@pytest.fixture()
def torch_text_moe_block() -> TorchGemma4TextExperts:
    config = PreTrainedConfig()
    config.num_experts = MOE_TEXT_NUM_EXPERTS
    config.hidden_size = MOE_TEXT_HIDDEN_SIZE
    config.moe_intermediate_size = MOE_TEXT_MOE_INTERMEDIATE_SIZE
    config.hidden_activation = MOE_TEXT_HIDDEN_ACTIVATION
    model = TorchGemma4TextExperts(config).to("cuda").to(TORCH_DTYPE)
    return model


def test_text_moe_block_matches_torch(
    torch_text_moe_block: TorchGemma4TextExperts,
    max_text_moe_block: Gemma4TextExperts,
    text_moe_block_state_dict: dict[str, torch.Tensor],
) -> None:
    n = 23
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    hidden_states = generate_tensor((n, MOE_TEXT_HIDDEN_SIZE)).to("cuda")
    top_k_index = torch.randint(
        0, MOE_TEXT_NUM_EXPERTS, (n, MOE_TEXT_TOP_K_EXPERTS)
    ).to("cuda")
    top_k_weights = generate_tensor((n, MOE_TEXT_TOP_K_EXPERTS)).to("cuda")

    torch_text_moe_block.load_state_dict(text_moe_block_state_dict)
    ref_output = torch_text_moe_block(hidden_states, top_k_index, top_k_weights)
    ref_output = ref_output.cpu()

    session = InferenceSession(devices=[device])
    with Graph(
        "test_text_moe_block",
        input_types=[
            TensorType(
                MAX_DTYPE, tuple(hidden_states.shape), device=device_ref
            ),
            TensorType(
                DType.int64, tuple(top_k_index.shape), device=device_ref
            ),
            TensorType(
                MAX_DTYPE, tuple(top_k_weights.shape), device=device_ref
            ),
        ],
    ) as graph:
        hidden_states_input, top_k_index_input, top_k_weights_input = (
            graph.inputs
        )
        assert isinstance(hidden_states_input, TensorValue)
        assert isinstance(top_k_index_input, TensorValue)
        assert isinstance(top_k_weights_input, TensorValue)

        output = max_text_moe_block(
            hidden_states_input, top_k_index_input, top_k_weights_input
        )
        graph.output(output)

    compiled = session.load(graph, weights_registry=text_moe_block_state_dict)

    hidden_states_gpu = Buffer.from_dlpack(hidden_states)
    top_k_index_gpu = Buffer.from_dlpack(top_k_index)
    top_k_weights_gpu = Buffer.from_dlpack(top_k_weights)
    (result_buf,) = compiled.execute(
        hidden_states_gpu,
        top_k_index_gpu,
        top_k_weights_gpu,
    )
    assert isinstance(result_buf, Buffer)
    max_output = torch.from_dlpack(result_buf).cpu()

    assert torch.allclose(ref_output, max_output, rtol=1e-5, atol=1e-5)


@pytest.fixture()
def text_moe_router_state_dict() -> dict[str, torch.Tensor]:
    return {
        "scale": generate_tensor((MOE_TEXT_HIDDEN_SIZE,)),
        "proj.weight": generate_tensor(
            (MOE_TEXT_NUM_EXPERTS, MOE_TEXT_HIDDEN_SIZE)
        ),
        "per_expert_scale": generate_tensor((MOE_TEXT_NUM_EXPERTS,)),
    }


@pytest.fixture()
def max_text_moe_router() -> Gemma4TextRouter:
    model = Gemma4TextRouter(
        MAX_DTYPE,
        DeviceRef.GPU(),
        MOE_TEXT_HIDDEN_SIZE,
        MOE_TEXT_NUM_EXPERTS,
        MOE_TEXT_TOP_K_EXPERTS,
        MOE_TEXT_RMS_NORM_EPS,
    )
    return model


@pytest.fixture()
def torch_text_moe_router() -> TorchGemma4TextRouter:
    config = PreTrainedConfig()
    config.hidden_size = MOE_TEXT_HIDDEN_SIZE
    config.num_experts = MOE_TEXT_NUM_EXPERTS
    config.rms_norm_eps = MOE_TEXT_RMS_NORM_EPS
    config.top_k_experts = MOE_TEXT_TOP_K_EXPERTS
    model = TorchGemma4TextRouter(config).to(device="cuda", dtype=TORCH_DTYPE)
    return model


def test_text_moe_router_matches_torch(
    torch_text_moe_router: TorchGemma4TextRouter,
    max_text_moe_router: Gemma4TextRouter,
    text_moe_router_state_dict: dict[str, torch.Tensor],
) -> None:
    n = 23
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    hidden_states = generate_tensor((n, MOE_TEXT_HIDDEN_SIZE)).to("cuda")

    torch_text_moe_router.load_state_dict(text_moe_router_state_dict)
    ref_top_k_weights, ref_top_k_index = torch_text_moe_router(hidden_states)
    ref_top_k_weights = ref_top_k_weights.cpu()
    ref_top_k_index = ref_top_k_index.cpu()

    session = InferenceSession(devices=[device])
    with Graph(
        "test_text_moe_router",
        input_types=[
            TensorType(
                MAX_DTYPE, tuple(hidden_states.shape), device=device_ref
            ),
        ],
    ) as graph:
        hidden_states_input = graph.inputs[0]
        assert isinstance(hidden_states_input, TensorValue)

        top_k_weights, top_k_index = max_text_moe_router(hidden_states_input)
        graph.output(top_k_weights, top_k_index)

    compiled = session.load(
        graph,
        weights_registry={
            "scale": text_moe_router_state_dict["scale"],
            "weight": text_moe_router_state_dict["proj.weight"],
            "per_expert_scale": text_moe_router_state_dict["per_expert_scale"],
        },
    )

    (max_top_k_weights, max_top_k_index) = compiled.execute(
        Buffer.from_dlpack(hidden_states)
    )
    max_top_k_weights = torch.from_dlpack(max_top_k_weights).cpu()
    max_top_k_index = torch.from_dlpack(max_top_k_index).cpu()

    # Compare sorted indices per row: top-k may return tied experts in
    # different order due to bfloat16 precision differences.
    assert torch.equal(
        ref_top_k_index.sort(dim=-1).values, max_top_k_index.sort(dim=-1).values
    )
