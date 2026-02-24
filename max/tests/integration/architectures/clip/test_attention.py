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


from types import SimpleNamespace
from typing import Any

import torch
from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.architectures.clip.clip import (
    CLIPAttention as MaxCLIPAttention,
)
from max.pipelines.architectures.clip.model_config import ClipConfigBase
from torch.utils.dlpack import from_dlpack
from transformers.models.clip.modeling_clip import CLIPAttention


@torch.no_grad()
def generate_torch_outputs(
    clip_config: dict[str, Any],
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generates the outputs of the PyTorch CLIPAttention layer."""

    # Create a config object that matches what CLIPAttention expects
    config = SimpleNamespace(
        hidden_size=clip_config["hidden_size"],
        num_attention_heads=clip_config["num_attention_heads"],
        attention_dropout=clip_config["attention_dropout"],
        _attn_implementation="eager",  # Use eager implementation for consistency
    )

    # Create CLIPAttention layer
    layer = CLIPAttention(config).to(torch.bfloat16).to("cuda")

    # Load weights into the layer
    layer.load_state_dict(attention_weights)

    # Run forward pass
    output, _ = layer(
        hidden_states=input_tensor,
        output_attentions=False,
    )

    return output


def generate_max_outputs(
    clip_config: dict[str, Any],
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generates the outputs of the MAX CLIPAttention layer."""

    device_ref = Accelerator()

    # Create MAX config
    max_config = ClipConfigBase(
        hidden_size=clip_config["hidden_size"],
        num_attention_heads=clip_config["num_attention_heads"],
        attention_dropout=clip_config["attention_dropout"],
        dtype=DType.bfloat16,
    )

    with F.lazy():
        # Create MAX CLIPAttention layer
        attention = MaxCLIPAttention(max_config)
        attention.load_state_dict(attention_weights)
        attention.to(device_ref)

    # Define input types for the graph
    batch_size, seq_length, hidden_size = input_tensor.shape

    hidden_states_type = TensorType(
        DType.bfloat16,
        [batch_size, seq_length, hidden_size],
        device_ref,
    )

    compiled_attention = attention.compile(
        hidden_states_type,
        weights=attention_weights,
    )

    output = compiled_attention(
        Tensor.from_dlpack(input_tensor.cuda()),
    )
    return output


def test_clip_attention(
    clip_config: dict[str, Any],
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    """Test that MAX CLIPAttention matches PyTorch CLIPAttention output."""

    torch_output = generate_torch_outputs(
        clip_config,
        input_tensor,
        attention_weights,
    )

    max_output = generate_max_outputs(
        clip_config,
        input_tensor,
        attention_weights,
    )

    # Convert MAX output back to torch for comparison
    max_output_torch = from_dlpack(max_output).to(torch.bfloat16)

    # Compare outputs
    torch.testing.assert_close(
        torch_output.to(torch.bfloat16),
        max_output_torch.to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=4 * torch.finfo(torch.bfloat16).eps,
    )
