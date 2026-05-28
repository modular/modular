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
"""MLA layer tests for GLM-5.1.

Torch reference skips the DSA indexer; MAX uses ``NaiveGlmMlaAttention`` (dense
matmul + softmax) instead of flare ``mla_*_graph`` kernels.
"""

from __future__ import annotations

import typing

import pytest
import torch
from max.dtype import DType
from torch_reference.configuration_glm import GlmMoeDsaConfig
from torch_reference.dense_mla_torch import glm_mla_forward_dense_no_indexer
from torch_reference.modeling_glm import (
    GlmMoeDsaAttention,
    GlmMoeDsaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

pytestmark = pytest.mark.skipif(
    not __import__(
        "torch_reference.modeling_glm", fromlist=["TORCH_REFERENCE_READY"]
    ).TORCH_REFERENCE_READY,
    reason="GLM torch reference not installed",
)

TOLERANCES = {
    DType.bfloat16: {"rtol": 5e-3, "atol": 5e-3},
    DType.float8_e4m3fn: {"rtol": 5e-2, "atol": 5e-2},
}


def _position_embeddings(
    config: GlmMoeDsaConfig,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_emb = GlmMoeDsaRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1)
    probe = torch.zeros(
        batch_size,
        seq_len,
        config.num_attention_heads,
        config.qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )
    return rotary_emb(probe, position_ids)


def generate_torch_outputs(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    *,
    skip_indexer: bool = True,
) -> torch.Tensor:
    device = input_tensor.device
    batch_size, seq_len, _ = input_tensor.shape
    layer = GlmMoeDsaAttention(config=config, layer_idx=0).to(torch.bfloat16)
    layer.eval()
    layer.load_state_dict(attention_weights, strict=False)

    position_embeddings = _position_embeddings(
        config, batch_size, seq_len, device, torch.bfloat16
    )
    mask_4d = attention_mask
    if mask_4d.dim() == 4:
        mask_4d = mask_4d[:, :, :seq_len, :seq_len]

    if skip_indexer:
        return glm_mla_forward_dense_no_indexer(
            layer,
            input_tensor,
            position_embeddings,
            mask_4d,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            eager_attention_forward=eager_attention_forward,
        )

    attn_output, _, _ = layer(
        input_tensor,
        position_embeddings,
        mask_4d,
        past_key_values=None,
    )
    return attn_output


def test_mla_decode(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    generate_mla_max_outputs: typing.Callable[..., torch.Tensor],
    kv_dtype: DType,
) -> None:
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    max_output = generate_mla_max_outputs(
        config,
        input_tensor,
        attention_mask,
        attention_weights,
        use_prefill=False,
    )
    torch.testing.assert_close(torch_output, max_output, **TOLERANCES[kv_dtype])


def test_mla_prefill(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    generate_mla_max_outputs: typing.Callable[..., torch.Tensor],
    kv_dtype: DType,
) -> None:
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    max_output = generate_mla_max_outputs(
        config,
        input_tensor,
        attention_mask,
        attention_weights,
        use_prefill=True,
    )
    torch.testing.assert_close(torch_output, max_output, **TOLERANCES[kv_dtype])
