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

"""Tests for Qwen3VL-MoE MoE layer."""

import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen3vl_moe.nn.moe import (
    Qwen3VLMoE,
    Qwen3VLMoEGate,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import load_weights_to_hf_moe_block
from utils.weight_generator import get_weight_generator

# Looser tolerances for BF16 comparisons.
MOE_RTOL = 2e-2
MOE_ATOL = 3e-2


@torch.no_grad()
def generate_torch_moe_outputs(
    hidden_states: torch.Tensor,
    moe_weights: dict[str, torch.Tensor],
    text_config: dict,
    device: torch.device,
) -> torch.Tensor:
    """Generate reference outputs using HF Qwen3VL-MoE MoE implementation."""
    hf_config = Qwen3VLMoeTextConfig(**text_config)

    # Create the sparse MoE block which contains the gate and experts
    sparse_moe_block = (
        Qwen3VLMoeTextSparseMoeBlock(hf_config).to(device).to(torch.bfloat16)
    )
    sparse_moe_block.eval()

    # Load weights into the HF model
    load_weights_to_hf_moe_block(sparse_moe_block, moe_weights, device)

    # HF expects [batch_size, seq_len, hidden_size]
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)

    hidden_states = hidden_states.to(device=device, dtype=torch.bfloat16)

    hf_out = sparse_moe_block(hidden_states)

    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Return [seq_len, hidden_size] format
    return hf_out.squeeze(0)


def generate_max_moe_outputs(
    hidden_states: torch.Tensor,
    moe_weights: dict[str, torch.Tensor],
    text_config: dict,
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX Qwen3VL-MoE MoE implementation."""
    assert isinstance(device, Accelerator), (
        "Qwen3VL-MoE MoE tests expect an Accelerator (GPU) device"
    )

    torch_device = torch.device("cuda")
    hidden_states = hidden_states.to(torch_device)

    seq_len, hidden_size = hidden_states.shape
    num_experts = text_config["num_experts"]
    num_experts_per_token = text_config["num_experts_per_tok"]
    moe_intermediate_size = text_config["moe_intermediate_size"]

    # Convert weights to MAX format
    state_dict = {
        weight_name: value.to(dtype.to_torch()).cpu()
        for weight_name, value in moe_weights.items()
    }

    device_ref = DeviceRef.GPU()

    moe = Qwen3VLMoE(
        devices=[device_ref],
        hidden_dim=hidden_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        moe_dim=moe_intermediate_size,
        gate_cls=Qwen3VLMoEGate,
        dtype=dtype,
    )
    moe.load_state_dict(state_dict, strict=True)

    session = InferenceSession(devices=[device])

    input_type = TensorType(dtype, [seq_len, hidden_size], device=device_ref)

    with Graph("Qwen3VLMoE", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = moe(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=moe.state_dict())

    result = compiled.execute(
        Buffer.from_dlpack(hidden_states.to(torch_device)).to(device)
    )
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize("seq_len", [16, 32])
def test_qwen3vl_moe(seq_len: int) -> None:
    """HF vs MAX Qwen3VL-MoE MoE layer on a single sequence."""
    torch.manual_seed(42)

    config_loader = get_config_loader()
    qwen3_config = config_loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)
    text_config = qwen3_config["text_config"]
    hidden_size = text_config["hidden_size"]

    weight_generator = get_weight_generator(ConfigNames.QWEN3VL_30B)
    moe_weights = weight_generator.generate_moe_weights()

    torch_device = torch.device("cuda")

    # Single sequence [L, H]
    hidden_states = torch.randn(
        seq_len, hidden_size, dtype=torch.bfloat16, device=torch_device
    )

    torch_output = generate_torch_moe_outputs(
        hidden_states=hidden_states,
        moe_weights=moe_weights,
        text_config=text_config,
        device=torch_device,
    )

    max_output = generate_max_moe_outputs(
        hidden_states=hidden_states,
        moe_weights=moe_weights,
        text_config=text_config,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    expected_shape = (seq_len, hidden_size)
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Basic sanity: no NaNs/Infs anywhere in MAX output
    assert torch.all(torch.isfinite(max_output)), (
        "MAX output contains NaNs or infs"
    )

    assert_tensors_close(
        torch_output,
        max_output,
        rtol=MOE_RTOL,
        atol=MOE_ATOL,
        message="Qwen3VL-MoE MoE layer outputs do not match",
    )

    del hidden_states, torch_output, max_output, moe_weights
    torch.cuda.empty_cache()
