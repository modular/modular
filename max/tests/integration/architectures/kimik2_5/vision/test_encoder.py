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
from conftest import TorchEncoder, TorchEncoderBlock, TorchRope2DPosEmbRepeated
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.kimik2_5.layers.vision.data_processing import (
    compute_position_ids,
)
from max.pipelines.architectures.kimik2_5.layers.vision.encoder import (
    Encoder,
    EncoderBlock,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

NUM_HEADS = 16
HIDDEN_DIM = 1152
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
MLP_DIM = 4304

ROPE_MAX_HEIGHT = 512
ROPE_MAX_WIDTH = 512
ROPE_THETA = 10000.0


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def _create_encoder_layer_weights() -> dict[str, torch.Tensor]:
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


def _build_and_run_encoder_layer(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    input_row_offsets: torch.Tensor,
    max_seq_len: torch.Tensor,
    rope_freqs_cis: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with the encoder layer, execute it, and return outputs."""
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    encoder = EncoderBlock(
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
        "kimik2_5_encoder_layer_test",
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
def test_encoder_layer(grid_thws: list[tuple[int, int, int]]) -> None:
    """Test EncoderBlock E2E on single GPU."""
    torch.manual_seed(42)
    seq_lens = [t * h * w for t, h, w in grid_thws]
    n_patches = sum(seq_lens)
    input_row_offsets = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.uint32
    )

    state_dict = _create_encoder_layer_weights()

    x = _generate_tensor((n_patches, HIDDEN_DIM))

    rope_ref = TorchRope2DPosEmbRepeated(
        HEAD_DIM, ROPE_MAX_HEIGHT, ROPE_MAX_WIDTH, ROPE_THETA
    )
    rope_freqs_cis_complex = rope_ref.get_freqs_cis(
        torch.tensor(grid_thws), device=x.device
    )
    rope_freqs_cis_real = torch.view_as_real(rope_freqs_cis_complex)

    max_seq_len = torch.tensor([max(seq_lens)], dtype=torch.uint32)

    max_output = _build_and_run_encoder_layer(
        state_dict, x, input_row_offsets, max_seq_len, rope_freqs_cis_real
    )

    ref = TorchEncoderBlock(NUM_HEADS, HIDDEN_DIM, MLP_DIM)
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


def _create_encoder_weights(num_layers: int) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        for k, v in _create_encoder_layer_weights().items():
            weights[f"blocks.{i}.{k}"] = v
    weights["norm.weight"] = _generate_tensor((HIDDEN_DIM,))
    weights["norm.bias"] = _generate_tensor((HIDDEN_DIM,))
    return weights


def _build_and_run_encoder(
    state_dict: dict[str, torch.Tensor],
    num_layers: int,
    x: torch.Tensor,
    input_row_offsets: torch.Tensor,
    max_seq_len: torch.Tensor,
    position_ids: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with the encoder, execute it, and return outputs."""
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    encoder = Encoder(
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=MLP_DIM,
        num_layers=num_layers,
        rope_max_height=ROPE_MAX_HEIGHT,
        rope_max_width=ROPE_MAX_WIDTH,
        rope_theta=ROPE_THETA,
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
                DType.int64,
                ["n_patches"],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        x_in, input_row_offsets_in, max_seq_len_in, position_ids_in = (
            graph.inputs
        )
        assert isinstance(x_in, TensorValue)
        assert isinstance(input_row_offsets_in, TensorValue)
        assert isinstance(max_seq_len_in, TensorValue)
        assert isinstance(position_ids_in, TensorValue)
        graph.output(
            encoder(x_in, input_row_offsets_in, max_seq_len_in, position_ids_in)
        )

    compiled = session.load(graph, weights_registry=encoder.state_dict())
    x_gpu = Buffer.from_dlpack(x).to(device)
    input_row_offsets_gpu = Buffer.from_dlpack(input_row_offsets).to(device)
    position_ids_gpu = Buffer.from_dlpack(position_ids).to(device)
    (result,) = compiled.execute(
        x_gpu, input_row_offsets_gpu, max_seq_len, position_ids_gpu
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
def test_encoder(grid_thws: list[tuple[int, int, int]]) -> None:
    """Test Encoder E2E on single GPU."""
    torch.manual_seed(42)
    num_layers = 3
    seq_lens = [t * h * w for t, h, w in grid_thws]
    n_patches = sum(seq_lens)
    input_row_offsets = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.uint32
    )

    state_dict = _create_encoder_weights(num_layers)

    x = _generate_tensor((n_patches, HIDDEN_DIM))

    # Position IDs for MAX (computed outside the graph).
    position_ids = torch.from_numpy(
        compute_position_ids(grid_thws, ROPE_MAX_WIDTH)
    )

    max_seq_len = torch.tensor([max(seq_lens)], dtype=torch.uint32)

    max_output = _build_and_run_encoder(
        state_dict, num_layers, x, input_row_offsets, max_seq_len, position_ids
    )

    ref = TorchEncoder(
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=MLP_DIM,
        num_layers=num_layers,
        rope_max_height=ROPE_MAX_HEIGHT,
        rope_max_width=ROPE_MAX_WIDTH,
        rope_theta=ROPE_THETA,
    )
    # Strip "attn." within block keys so they match the torch reference.
    torch_state_dict = {
        k.replace(".attn.", "."): v for k, v in state_dict.items()
    }
    ref.load_state_dict(torch_state_dict)
    ref = ref.to(dtype=TORCH_DTYPE)
    torch_input_row_offsets = input_row_offsets.to(torch.int32)
    torch_output = ref(
        x, torch_input_row_offsets, torch.tensor(grid_thws)
    ).detach()

    _assert_close(torch_output, max_output)
