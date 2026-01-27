# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import typing

import pytest
import torch
from max.driver import accelerator_api
from torch_reference.configuration_deepseek import DeepseekV2Config
from torch_reference.modeling_deepseek import DeepseekV2Attention


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = DeepseekV2Attention(config=config, layer_idx=0).to(torch.bfloat16)
    layer.load_state_dict(attention_weights)
    torch_output = layer(input_tensor, attention_mask=attention_mask)
    return torch_output[0]


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_latent_attention_decode(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    generate_latent_attention_max_outputs: typing.Callable[..., torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    max_output = generate_latent_attention_max_outputs(
        config, input_tensor, attention_weights, use_prefill=False
    )
    torch.testing.assert_close(torch_output, max_output, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_latent_attention_prefill(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    generate_latent_attention_max_outputs: typing.Callable[..., torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    max_output = generate_latent_attention_max_outputs(
        config, input_tensor, attention_weights, use_prefill=True
    )
    torch.testing.assert_close(torch_output, max_output, rtol=5e-4, atol=5e-4)
