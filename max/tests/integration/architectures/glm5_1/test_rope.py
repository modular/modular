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
"""Standard RoPE layer test for GLM-5.1 (HF rotate_half / NeoX-style)."""

from __future__ import annotations

import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.rotary_embedding import RotaryEmbedding
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_glm import GlmMoeDsaConfig
from torch_reference.modeling_glm import (
    GlmMoeDsaRotaryEmbedding,
    apply_rotary_pos_emb,
)

pytestmark = pytest.mark.skipif(
    not __import__(
        "torch_reference.modeling_glm", fromlist=["TORCH_REFERENCE_READY"]
    ).TORCH_REFERENCE_READY,
    reason="GLM torch reference not installed",
)


def generate_torch_outputs(
    config: GlmMoeDsaConfig,
    input_tensor_rope: torch.Tensor,
) -> torch.Tensor:
    """BHSD tensor [batch, heads, seq, qk_rope_head_dim]."""
    batch_size, num_heads, seq_len, _ = input_tensor_rope.shape
    position_ids = torch.arange(
        seq_len, device=input_tensor_rope.device
    ).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1)

    rotary_emb = GlmMoeDsaRotaryEmbedding(config)
    probe = torch.zeros(
        batch_size,
        seq_len,
        num_heads,
        config.qk_rope_head_dim,
        dtype=input_tensor_rope.dtype,
        device=input_tensor_rope.device,
    )
    cos, sin = rotary_emb(probe, position_ids)
    return apply_rotary_pos_emb(input_tensor_rope, cos, sin, unsqueeze_dim=1)


def generate_max_outputs(
    config: GlmMoeDsaConfig,
    input_tensor_rope: torch.Tensor,
) -> torch.Tensor:
    session = InferenceSession(devices=[CPU()])
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "GlmRotaryEmbedding",
        RotaryEmbedding(
            dim=config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.qk_rope_head_dim,
            interleaved=False,  # config.rope_interleave,
        ),
        input_types=(
            TensorType(
                DType.bfloat16,
                tuple(input_tensor_rope.transpose(1, 2).contiguous().shape),
                DeviceRef.CPU(),
            ),
        ),
    )
    compiled = session.load(graph)
    max_output = compiled.execute(
        input_tensor_rope.transpose(1, 2).contiguous()
    )
    return from_dlpack(max_output[0]).to(torch.bfloat16)


def test_rope(
    config: GlmMoeDsaConfig,
    input_tensor_rope: torch.Tensor,
) -> None:
    assert config.rope_interleave, "GLM-5.1 tests expect interleaved RoPE"
    torch_output = generate_torch_outputs(config, input_tensor_rope)
    max_output = generate_max_outputs(config, input_tensor_rope)
    torch.testing.assert_close(
        torch_output,
        max_output.transpose(1, 2),
        rtol=1e-4,
        atol=2 * torch.finfo(torch.bfloat16).eps,
    )
