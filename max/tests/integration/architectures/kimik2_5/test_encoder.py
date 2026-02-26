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
"""Tests for Kimi K2.5 encoder."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn as nn
from conftest import TorchMLP2, TorchRope2DPosEmbRepeated
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.kimik2_5.layers.encoder import EncoderLayer

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

NUM_HEADS = 16
HIDDEN_DIM = 1152
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
MLP_DIM = 4304

ROPE_DIM = HEAD_DIM
ROPE_MAX_HEIGHT = 512
ROPE_MAX_WIDTH = 512
ROPE_THETA = 10000.0

torch.manual_seed(42)


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def torch_apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def torch_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_row_offsets: torch.Tensor,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(input_row_offsets)):
        attention_mask[
            ...,
            input_row_offsets[i - 1] : input_row_offsets[i],
            input_row_offsets[i - 1] : input_row_offsets[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        q.dtype
    )

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


class TorchMoonViTEncoderLayer(nn.Module):
    """Near-verbatim PyTorch reference from Kimi-K2.5 modeling_kimi_k25.py."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = NUM_HEADS
        self.hidden_dim = HIDDEN_DIM
        self.hidden_size_per_attention_head = HIDDEN_DIM // NUM_HEADS

        self.norm0 = nn.LayerNorm(HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)

        self.wqkv = nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 3)
        self.wo = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.mlp = TorchMLP2(dim=(HIDDEN_DIM, MLP_DIM, HIDDEN_DIM))

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        input_row_offsets: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = torch_apply_rope(xq, xk, rope_freqs_cis)

        attn_out = torch_eager_attention(
            xq, xk, xv, input_row_offsets=input_row_offsets
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_row_offsets: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(
            hidden_states, input_row_offsets, rope_freqs_cis
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _create_weights() -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {
        "attn.wqkv.weight": _generate_tensor((HIDDEN_DIM * 3, HIDDEN_DIM)),
        "attn.wqkv.bias": _generate_tensor((HIDDEN_DIM * 3,)),
        "attn.wo.weight": _generate_tensor((HIDDEN_DIM, HIDDEN_DIM)),
        "attn.wo.bias": _generate_tensor((HIDDEN_DIM,)),
        "norm0.weight": _generate_tensor((HIDDEN_DIM,)),
        "norm0.bias": _generate_tensor((HIDDEN_DIM,)),
        "norm1.weight": _generate_tensor((HIDDEN_DIM,)),
        "norm1.bias": _generate_tensor((HIDDEN_DIM,)),
        "mlp.up_proj.weight": _generate_tensor((MLP_DIM, HIDDEN_DIM)),
        "mlp.up_proj.bias": _generate_tensor((MLP_DIM,)),
        "mlp.down_proj.weight": _generate_tensor((HIDDEN_DIM, MLP_DIM)),
        "mlp.down_proj.bias": _generate_tensor((HIDDEN_DIM,)),
    }
    return weights


def _assert_close(expected: torch.Tensor, actual: Buffer) -> None:
    rtol = 2e-2
    atol = 4 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected,
        torch.from_dlpack(actual).cpu(),
        rtol=rtol,
        atol=atol,
    )


def _build_and_run(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    input_row_offsets: torch.Tensor,
    max_seq_len: torch.Tensor,
    rope_freqs_cis: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with the encoder layer, execute it, and return outputs."""
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    encoder = EncoderLayer(
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=MLP_DIM,
        dtype=MAX_DTYPE,
        device=device_ref,
        has_bias=True,
    )
    encoder.load_state_dict(state_dict)

    session = InferenceSession(devices=[device])

    with Graph(
        "kimik2_5_encoder_test",
        input_types=[
            TensorType(
                MAX_DTYPE,
                ["n_patches", HIDDEN_DIM],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.uint32,
                ["num_seqs"],
                device=DeviceRef.GPU(),
            ),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            TensorType(
                DType.float32,
                ["n_patches", HEAD_DIM // 2, 2],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        x_in, input_row_offsets_in, max_seq_len_in, freqs_cis_in = graph.inputs
        assert isinstance(x_in, TensorValue)
        assert isinstance(input_row_offsets_in, TensorValue)
        assert isinstance(max_seq_len_in, TensorValue)
        assert isinstance(freqs_cis_in, TensorValue)
        graph.output(
            encoder(x_in, input_row_offsets_in, max_seq_len_in, freqs_cis_in)
        )

    compiled = session.load(graph, weights_registry=encoder.state_dict())
    x_gpu = Buffer.from_dlpack(x).to(device)
    input_row_offsets_gpu = Buffer.from_dlpack(input_row_offsets).to(device)
    rope_gpu = Buffer.from_dlpack(rope_freqs_cis).to(device)
    (result,) = compiled.execute(
        x_gpu, input_row_offsets_gpu, max_seq_len, rope_gpu
    )
    assert isinstance(result, Buffer)
    return result


@pytest.mark.parametrize(
    "grid_thws",
    [
        [(2, 4, 4)],
        [(4, 3, 4), (2, 4, 4), (1, 2, 6)],
    ],
    ids=["single_sequence", "multiple_sequences"],
)
def test_vision_attention(grid_thws: list[tuple[int, int, int]]) -> None:
    """Test EncoderLayer E2E on single GPU."""
    seq_lens = [t * h * w for t, h, w in grid_thws]
    n_patches = sum(seq_lens)
    input_row_offsets = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.uint32
    )

    state_dict = _create_weights()

    x = _generate_tensor((n_patches, HIDDEN_DIM))

    rope_ref = TorchRope2DPosEmbRepeated(
        ROPE_DIM, ROPE_MAX_HEIGHT, ROPE_MAX_WIDTH, ROPE_THETA
    )
    rope_freqs_cis_complex = rope_ref.get_freqs_cis(
        torch.tensor(grid_thws), device=x.device
    )
    rope_freqs_cis_real = torch.view_as_real(rope_freqs_cis_complex)

    max_seq_len = torch.tensor([max(seq_lens)], dtype=torch.uint32)

    max_output = _build_and_run(
        state_dict, x, input_row_offsets, max_seq_len, rope_freqs_cis_real
    )

    ref = TorchMoonViTEncoderLayer()
    # Strip "attn." prefix so keys match the torch reference module.
    torch_state_dict = {
        k.removeprefix("attn."): v for k, v in state_dict.items()
    }
    ref.load_state_dict(torch_state_dict)
    ref = ref.to(dtype=TORCH_DTYPE)
    torch_input_row_offsets = input_row_offsets.to(torch.int32)
    torch_output = ref(
        x, torch_input_row_offsets, rope_freqs_cis_complex
    ).detach()

    _assert_close(torch_output, max_output)
