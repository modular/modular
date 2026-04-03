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

"""Test that MAX QwenImageTransformerBlock matches diffusers output."""

from typing import Any

import torch
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformerBlock as DiffusersBlock,
)
from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.architectures.qwen_image.layers.qwen_image_attention import (
    QwenImageTransformerBlock as MaxBlock,
)
from torch.utils.dlpack import from_dlpack


class QwenImageBlockWrapper(MaxBlock):
    """Wrapper to flatten tuple inputs for MAX compiler."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        freqs_cis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return super().forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=freqs_cis,
        )


@torch.no_grad()
def generate_torch_outputs(
    qwen_config: dict[str, Any],
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    block_weights: dict[str, torch.Tensor],
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run diffusers QwenImageTransformerBlock and return outputs."""
    inner_dim = (
        qwen_config["num_attention_heads"] * qwen_config["attention_head_dim"]
    )

    layer = (
        DiffusersBlock(
            dim=inner_dim,
            num_attention_heads=qwen_config["num_attention_heads"],
            attention_head_dim=qwen_config["attention_head_dim"],
            qk_norm="rms_norm",
            eps=qwen_config["eps"],
        )
        .to(torch.bfloat16)
        .to("cuda")
    )
    layer.load_state_dict(block_weights)

    txt_out, img_out = layer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=None,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
    )
    return txt_out, img_out


def generate_max_outputs(
    qwen_config: dict[str, Any],
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    block_weights: dict[str, torch.Tensor],
    image_rotary_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MAX QwenImageTransformerBlock and return outputs."""
    device_ref = Accelerator()
    inner_dim = (
        qwen_config["num_attention_heads"] * qwen_config["attention_head_dim"]
    )

    with F.lazy():
        block = QwenImageBlockWrapper(
            dim=inner_dim,
            num_attention_heads=qwen_config["num_attention_heads"],
            attention_head_dim=qwen_config["attention_head_dim"],
            mlp_ratio=4.0,
            eps=qwen_config["eps"],
            bias=True,
        )
        block.to(device_ref)

    batch_size, img_seq_len, _ = hidden_states.shape
    txt_seq_len = encoder_hidden_states.shape[1]

    compiled = block.compile(
        TensorType(
            DType.bfloat16, [batch_size, img_seq_len, inner_dim], device_ref
        ),
        TensorType(
            DType.bfloat16, [batch_size, txt_seq_len, inner_dim], device_ref
        ),
        TensorType(DType.bfloat16, [batch_size, inner_dim], device_ref),
        TensorType(DType.float32, list(image_rotary_emb.shape), device_ref),
        weights=block_weights,
    )

    result = compiled(
        Tensor.from_dlpack(hidden_states),
        Tensor.from_dlpack(encoder_hidden_states),
        Tensor.from_dlpack(temb),
        Tensor.from_dlpack(image_rotary_emb),
    )
    return result[0], result[1]


def test_qwen_image_block(
    qwen_config: dict[str, Any],
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    block_weights: dict[str, torch.Tensor],
    image_rotary_emb_diffusers: tuple[torch.Tensor, torch.Tensor],
    image_rotary_emb_max: torch.Tensor,
) -> None:
    """Test that MAX QwenImageTransformerBlock matches diffusers output."""
    torch_txt_out, torch_img_out = generate_torch_outputs(
        qwen_config,
        hidden_states,
        encoder_hidden_states,
        temb,
        block_weights,
        image_rotary_emb_diffusers,
    )

    max_txt_out, max_img_out = generate_max_outputs(
        qwen_config,
        hidden_states,
        encoder_hidden_states,
        temb,
        block_weights,
        image_rotary_emb_max,
    )

    max_txt_torch = from_dlpack(max_txt_out).to(torch.bfloat16)
    max_img_torch = from_dlpack(max_img_out).to(torch.bfloat16)

    # Image stream output
    torch.testing.assert_close(
        torch_img_out.to(torch.bfloat16),
        max_img_torch,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=16 * torch.finfo(torch.bfloat16).eps,
    )

    # Text stream output
    torch.testing.assert_close(
        torch_txt_out.to(torch.bfloat16),
        max_txt_torch,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=16 * torch.finfo(torch.bfloat16).eps,
    )
