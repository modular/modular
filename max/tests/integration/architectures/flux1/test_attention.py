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


from typing import Any

import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.architectures.flux1.layers.flux_attention import (
    FluxAttention as MaxFluxAttention,
)
from torch.utils.dlpack import from_dlpack


class FluxAttentionWrapper(MaxFluxAttention):
    """Wrapper for the MAX FluxAttention layer.

    Max compiler does not support model input as a tuple or list.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb_0: torch.Tensor,
        image_rotary_emb_1: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb=(image_rotary_emb_0, image_rotary_emb_1),
        )


def weight_adapter(
    attention_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Adapts the attention weights to the MAX FluxAttention layer."""
    adapted_weights = {}
    keys = list(attention_weights.keys())
    for key in keys:
        if key == "to_out.0.weight":
            adapted_weights["to_out.weight"] = attention_weights[key]
        elif key == "to_out.0.bias":
            adapted_weights["to_out.bias"] = attention_weights[key]
        else:
            adapted_weights[key] = attention_weights[key]
    return adapted_weights


@torch.no_grad()
def generate_torch_outputs(
    flux_config: dict[str, Any],
    input_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Generates the outputs of the PyTorch FluxAttention layer."""
    query_dim = (
        flux_config["num_attention_heads"] * flux_config["attention_head_dim"]
    )

    # For dual-stream attention, need added_kv_proj_dim
    # Flux1 dev uses bias=True for all projections
    layer = (
        FluxAttention(
            query_dim=query_dim,
            heads=flux_config["num_attention_heads"],
            dim_head=flux_config["attention_head_dim"],
            bias=True,  # Main stream projections have bias
            added_kv_proj_dim=query_dim,  # For dual-stream attention
            added_proj_bias=True,  # Encoder projections have bias
            out_bias=True,  # Output projections have bias
        )
        .to(torch.bfloat16)
        .to("cuda")
    )
    layer.load_state_dict(attention_weights)

    output = layer(
        hidden_states=input_tensor,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=image_rotary_emb,
    )

    # For dual-stream, output is a tuple: (hidden_states, encoder_hidden_states)
    # Return only the image hidden states
    return output[0] if isinstance(output, tuple) else output


def generate_max_outputs(
    flux_config: dict[str, Any],
    input_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Generates the outputs of the MAX FluxAttention layer."""
    device_ref = Accelerator()
    query_dim = (
        flux_config["num_attention_heads"] * flux_config["attention_head_dim"]
    )
    attention_weights = weight_adapter(attention_weights)

    with F.lazy():
        # Create MAX FluxAttention layer with dual-stream support
        # Flux1 dev uses bias=True for all projections
        attention = FluxAttentionWrapper(
            query_dim=query_dim,
            heads=flux_config["num_attention_heads"],
            dim_head=flux_config["attention_head_dim"],
            bias=True,  # Main stream projections have bias
            added_kv_proj_dim=query_dim,  # For dual-stream attention
            added_proj_bias=True,  # Encoder projections have bias
            out_bias=True,  # Output projections have bias
        )
        attention.to(device_ref)

    # Define input types for the graph
    batch_size, img_seq_len, hidden_dim = input_tensor.shape
    text_seq_len = encoder_hidden_states.shape[1]

    hidden_states_type = TensorType(
        DType.bfloat16,
        [batch_size, img_seq_len, hidden_dim],
        device_ref,
    )
    encoder_hidden_states_type = TensorType(
        DType.bfloat16,
        [batch_size, text_seq_len, hidden_dim],
        device_ref,
    )
    cos_type = TensorType(
        DType.float32,  # Rotary embeddings are in float32
        [image_rotary_emb[0].shape[0], image_rotary_emb[0].shape[1]],
        device_ref,
    )
    sin_type = TensorType(
        DType.float32,  # Rotary embeddings are in float32
        [image_rotary_emb[1].shape[0], image_rotary_emb[1].shape[1]],
        device_ref,
    )

    # Compile with correct input types
    compiled_attention = attention.compile(
        hidden_states_type,
        encoder_hidden_states_type,
        cos_type,
        sin_type,
        weights=attention_weights,
    )

    # Execute with dual-stream inputs
    output = compiled_attention(
        Tensor.from_dlpack(input_tensor.cuda()),
        Tensor.from_dlpack(encoder_hidden_states.cuda()),
        Tensor.from_dlpack(image_rotary_emb[0].cuda()),
        Tensor.from_dlpack(image_rotary_emb[1].cuda()),
    )

    # For dual-stream, output is a tuple: (hidden_states, encoder_hidden_states)
    # Return only the image hidden states
    return output[0] if isinstance(output, (tuple, list)) else output


def test_flux_attention(
    flux_config: dict[str, Any],
    input_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that MAX FluxAttention matches PyTorch FluxAttention output."""
    torch_output = generate_torch_outputs(
        flux_config,
        input_tensor,
        encoder_hidden_states,
        attention_weights,
        image_rotary_emb,
    )

    max_output = generate_max_outputs(
        flux_config,
        input_tensor,
        encoder_hidden_states,
        attention_weights,
        image_rotary_emb,
    )

    # Convert MAX output back to torch for comparison
    max_output_torch = from_dlpack(max_output).to(torch.bfloat16)
    # Compare outputs
    torch.testing.assert_close(
        torch_output.to(torch.bfloat16),
        max_output_torch.to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=16 * torch.finfo(torch.bfloat16).eps,
    )
