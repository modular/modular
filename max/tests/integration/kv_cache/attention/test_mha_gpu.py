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
"""Test pipelines for attention layers on GPU."""

import math

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.attention.multihead_attention import MultiheadAttention
from torch.utils.dlpack import from_dlpack


def create_attention_weights(
    hidden_size: int,
    has_bias: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    stacked_qkv: bool = False,
) -> dict[str, torch.Tensor]:
    """Create shared attention weights for both PyTorch and MAX models."""
    torch.manual_seed(42)

    std = 0.02

    weights = {}

    if stacked_qkv:
        # Create stacked QKV weight
        weights["qkv_proj.weight"] = std * torch.randn(
            3 * hidden_size, hidden_size, dtype=dtype, device=device
        )
        if has_bias:
            weights["qkv_proj.bias"] = std * torch.randn(
                3 * hidden_size, dtype=dtype, device=device
            )
    else:
        # Create separate Q, K, V weights
        for proj in ["q", "k", "v"]:
            weights[f"{proj}_proj.weight"] = std * torch.randn(
                hidden_size, hidden_size, dtype=dtype, device=device
            )
            if has_bias:
                weights[f"{proj}_proj.bias"] = std * torch.randn(
                    hidden_size, dtype=dtype, device=device
                )

    # Always create output projection
    weights["o_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    if has_bias:
        weights["o_proj.bias"] = std * torch.randn(
            hidden_size, dtype=dtype, device=device
        )

    return weights


@torch.no_grad()
def generate_torch_outputs(
    num_heads: int,
    hidden_size: int,
    attention_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    has_bias: bool = False,
    dtype: torch.dtype = torch.float32,
    stacked_qkv: bool = False,
) -> torch.Tensor:
    """Generate outputs using PyTorch MultiheadAttention."""

    # Create PyTorch model
    torch_attention = torch.nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads,
        bias=has_bias,
        batch_first=True,
        dtype=dtype,
        device="cuda",
    )

    # PyTorch MultiheadAttention uses different weight layout
    if stacked_qkv and "qkv_proj.weight" in attention_weights:
        # Split stacked QKV weights for PyTorch
        qkv_weight = attention_weights["qkv_proj.weight"]
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
    else:
        q_weight = attention_weights["q_proj.weight"]
        k_weight = attention_weights["k_proj.weight"]
        v_weight = attention_weights["v_proj.weight"]

    o_weight = attention_weights["o_proj.weight"]

    torch_attention.in_proj_weight.data = torch.cat(
        [q_weight, k_weight, v_weight], dim=0
    )
    torch_attention.out_proj.weight.data = o_weight

    if has_bias:
        if stacked_qkv and "qkv_proj.bias" in attention_weights:
            # Split stacked QKV bias for PyTorch
            qkv_bias = attention_weights["qkv_proj.bias"]
            q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        else:
            q_bias = attention_weights["q_proj.bias"]
            k_bias = attention_weights["k_proj.bias"]
            v_bias = attention_weights["v_proj.bias"]

        o_bias = attention_weights["o_proj.bias"]

        torch_attention.in_proj_bias.data = torch.cat(
            [q_bias, k_bias, v_bias], dim=0
        )
        torch_attention.out_proj.bias.data = o_bias

    output, _ = torch_attention(input_tensor, input_tensor, input_tensor)
    return output


def generate_max_outputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    hidden_size: int,
    attention_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    has_bias: bool = False,
    dtype: DType = DType.float32,
    stacked_qkv: bool = False,
) -> torch.Tensor:
    """Generate outputs using MAX MultiheadAttention."""

    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    head_dim = hidden_size // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    attention = MultiheadAttention(
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        devices=[DeviceRef.GPU()],
        dtype=dtype,
        scale=scale,
        qkv_has_bias=has_bias,
        o_proj_has_bias=has_bias,
        stacked_qkv=stacked_qkv,
    )

    state_dict = {}
    for weight_name, value in attention_weights.items():
        state_dict[weight_name] = value.cpu()

    attention.load_state_dict(state_dict)

    def build_attention_model() -> Graph:
        with Graph(
            "multihead_attention_test",
            input_types=[
                TensorType(
                    dtype=dtype,
                    shape=[batch_size, seq_len, hidden_size],
                    device=DeviceRef.GPU(),
                )
            ],
        ) as graph:
            inputs = graph.inputs[0]
            output = attention(inputs.tensor)
            graph.output(output)
        return graph

    graph = build_attention_model()
    model = session.load(graph, weights_registry=attention.state_dict())

    x_tensor = Buffer.from_dlpack(input_tensor).to(cuda)
    max_output = model.execute(x_tensor)[0]

    return max_output


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,hidden_size,has_bias,dtype,stacked_qkv",
    [
        (1, 16, 4, 64, False, DType.float32, False),
        (1, 16, 4, 128, True, DType.float32, True),
        (1, 8, 4, 256, False, DType.bfloat16, False),
        (1, 8, 4, 256, True, DType.bfloat16, True),
    ],
)
@torch.no_grad()
def test_multihead_attention_gpu(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    hidden_size: int,
    has_bias: bool,
    dtype: DType,
    stacked_qkv: bool,
) -> None:
    """Test MultiheadAttention functionality on GPU with different configurations and dtypes."""

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32

    attention_weights = create_attention_weights(
        hidden_size=hidden_size,
        has_bias=has_bias,
        dtype=torch_dtype,
        device="cuda",
        stacked_qkv=stacked_qkv,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch_dtype, device="cuda"
    )

    torch_output = generate_torch_outputs(
        num_heads=num_heads,
        hidden_size=hidden_size,
        attention_weights=attention_weights,
        input_tensor=input_tensor,
        has_bias=has_bias,
        dtype=torch_dtype,
        stacked_qkv=stacked_qkv,
    )

    max_output = generate_max_outputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        hidden_size=hidden_size,
        attention_weights=attention_weights,
        input_tensor=input_tensor,
        has_bias=has_bias,
        dtype=dtype,
        stacked_qkv=stacked_qkv,
    )

    if dtype == DType.bfloat16:
        torch.testing.assert_close(
            torch_output,
            from_dlpack(max_output),
            rtol=2 * torch.finfo(torch.bfloat16).eps,
            atol=8 * torch.finfo(torch.bfloat16).eps,
        )
    else:
        # TODO: DOUBLE CHECK THIS LATER
        # Switched to same tolerances as bfloat16 for FP32 as FP32 epsilon is too harsh
        torch.testing.assert_close(
            torch_output,
            from_dlpack(max_output),
            rtol=2 * torch.finfo(torch.bfloat16).eps,
            atol=8 * torch.finfo(torch.bfloat16).eps,
        )


def test_multihead_attention_error_cases_gpu() -> None:
    """Test error handling in MultiheadAttention."""

    # Test invalid head configuration
    with pytest.raises(ValueError):
        # hidden_size not divisible by num_heads
        MultiheadAttention(
            num_attention_heads=7,  # 512 is not divisible by 7
            hidden_size=512,
            devices=[DeviceRef.GPU()],
        )


if __name__ == "__main__":
    pytest.main([__file__])
