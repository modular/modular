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
"""Tests for Kimi K2.5 vision transformer."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn as nn
from conftest import TorchEncoder, TorchPatchEmbed, TorchPatchMergerMLP
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.kimik2_5.layers.vision.data_processing import (
    compute_position_ids,
)
from max.pipelines.architectures.kimik2_5.layers.vision.transformer import (
    Transformer,
)
from max.pipelines.architectures.kimik2_5.model_config import VisionConfig

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

NUM_HEADS = 16
HIDDEN_DIM = 1152
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
MLP_DIM = 4304

ROPE_MAX_HEIGHT = 512
ROPE_MAX_WIDTH = 512
ROPE_THETA = 10000.0

PATCH_SIZE = 14
IN_CHANNELS = 3
INIT_POS_EMB_HEIGHT = 64
INIT_POS_EMB_WIDTH = 64
INIT_POS_EMB_TIME = 4
MERGE_KERNEL_SIZE = (2, 2)
DECODER_HIDDEN_SIZE = 7168
VT_NUM_LAYERS = 2


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def _assert_close(expected: torch.Tensor, actual: Buffer) -> None:
    rtol = 2e-2
    atol = 4 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected,
        torch.from_dlpack(actual).cpu(),
        rtol=rtol,
        atol=atol,
    )


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


def _torch_tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int],
) -> list[torch.Tensor]:
    """Torch reference for tpool_patch_merger; math matches HuggingFace reference.

    Reference: https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/modeling_kimi_k25.py
    (tpool_patch_merger). Temporal pooling then spatial reshape. Returns one tensor per
    video of shape (new_h * new_w, kH * kW, D) for PatchMergerMLP.
    """
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kH, kW = merge_kernel_size
        new_h, new_w = h // kH, w // kW
        reshaped = seq.view(t, new_h, kH, new_w, kW, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        # Match reference: (new_h, new_w, kH, kW, D) -> (new_h*new_w, kH*kW, D)
        outputs.append(reshaped.view(new_h * new_w, kH * kW, -1))
        pre_sum += t * h * w
    return outputs


class TorchTransformer(nn.Module):
    """Torch reference for transformer."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.patch_embed = TorchPatchEmbed(
            out_dim=HIDDEN_DIM,
            in_dim=IN_CHANNELS,
            patch_size=PATCH_SIZE,
            pos_emb_height=INIT_POS_EMB_HEIGHT,
            pos_emb_width=INIT_POS_EMB_WIDTH,
            pos_emb_time=INIT_POS_EMB_TIME,
        )
        self.encoder = TorchEncoder(
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            mlp_dim=MLP_DIM,
            num_layers=num_layers,
            rope_max_height=ROPE_MAX_HEIGHT,
            rope_max_width=ROPE_MAX_WIDTH,
            rope_theta=ROPE_THETA,
        )
        self.merge_kernel_size = MERGE_KERNEL_SIZE
        self.patch_merger = TorchPatchMergerMLP(
            mm_hidden_size=HIDDEN_DIM,
            decoder_hidden_size=DECODER_HIDDEN_SIZE,
            merge_kernel_size=MERGE_KERNEL_SIZE,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        input_row_offsets: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.patch_embed(pixel_values, grid_thws)
        hidden = self.encoder(hidden, input_row_offsets, grid_thws)
        # List of (n_spatial_i, kH*kW, D) per video; cat to (total_n_spatial, kH*kW, D)
        merged = _torch_tpool_patch_merger(
            hidden, grid_thws, self.merge_kernel_size
        )
        return self.patch_merger(torch.cat(merged, dim=0))


def _create_transformer_weights(
    num_layers: int,
) -> dict[str, torch.Tensor]:
    """Creates weights for Transformer matching MAX state_dict keys.

    Weight names follow the Transformer's registered sublayers:
    ``patch_embed``, ``encoder``, and ``patch_merger``.
    """
    weights: dict[str, torch.Tensor] = {}
    weights["patch_embed.proj.weight"] = _generate_tensor(
        (HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    )
    weights["patch_embed.proj.bias"] = _generate_tensor((HIDDEN_DIM,))
    weights["patch_embed.pos_emb.weight"] = _generate_tensor(
        (INIT_POS_EMB_HEIGHT, INIT_POS_EMB_WIDTH, HIDDEN_DIM)
    )
    for i in range(num_layers):
        for k, v in _create_encoder_layer_weights().items():
            weights[f"encoder.blocks.{i}.{k}"] = v
    weights["encoder.norm.weight"] = _generate_tensor((HIDDEN_DIM,))
    weights["encoder.norm.bias"] = _generate_tensor((HIDDEN_DIM,))
    # PatchMergerMLP: input_dim = HIDDEN_DIM * (kH * kW), hidden_size = DECODER_HIDDEN_SIZE
    merger_input_dim = HIDDEN_DIM * (
        MERGE_KERNEL_SIZE[0] * MERGE_KERNEL_SIZE[1]
    )
    weights["patch_merger.pre_norm.weight"] = _generate_tensor((HIDDEN_DIM,))
    weights["patch_merger.pre_norm.bias"] = _generate_tensor((HIDDEN_DIM,))
    weights["patch_merger.linear1.weight"] = _generate_tensor(
        (merger_input_dim, merger_input_dim)
    )
    weights["patch_merger.linear1.bias"] = _generate_tensor((merger_input_dim,))
    weights["patch_merger.linear2.weight"] = _generate_tensor(
        (DECODER_HIDDEN_SIZE, merger_input_dim)
    )
    weights["patch_merger.linear2.bias"] = _generate_tensor(
        (DECODER_HIDDEN_SIZE,)
    )
    return weights


def _remap_transformer_keys_for_torch(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remaps MAX Transformer weight keys to torch reference naming.

    Strips the attention namespace so encoder.blocks.i.attn.* -> encoder.blocks.i.*
    for the TorchEncoder / TorchPatchEmbed reference.
    """
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        remapped[k.replace(".attn.", ".")] = v
    return remapped


def _build_and_run_transformer(
    state_dict: dict[str, torch.Tensor],
    num_layers: int,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    input_row_offsets: torch.Tensor,
    max_seq_len: torch.Tensor,
    position_ids: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with Transformer, execute, return output."""
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    vision_config = VisionConfig(
        dtype=MAX_DTYPE,
        devices=[device_ref],
        init_pos_emb_height=INIT_POS_EMB_HEIGHT,
        init_pos_emb_time=INIT_POS_EMB_TIME,
        init_pos_emb_width=INIT_POS_EMB_WIDTH,
        merge_kernel_size=list(MERGE_KERNEL_SIZE),
        mm_hidden_size=HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        projector_ln_eps=1e-5,
        text_hidden_size=DECODER_HIDDEN_SIZE,
        vt_hidden_size=HIDDEN_DIM,
        vt_intermediate_size=MLP_DIM,
        vt_num_attention_heads=NUM_HEADS,
        vt_num_hidden_layers=num_layers,
        in_channels=IN_CHANNELS,
        rope_max_height=ROPE_MAX_HEIGHT,
        rope_max_width=ROPE_MAX_WIDTH,
        rope_theta=ROPE_THETA,
    )
    vision_tower = Transformer(vision_config)
    vision_tower.load_state_dict(state_dict)

    session = InferenceSession(devices=[device])

    with Graph(
        "kimik2_5_transformer_test",
        input_types=[
            TensorType(
                MAX_DTYPE,
                ["n_patches", IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.int64,
                ["n_videos", 3],
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
        (
            pixel_values_in,
            grid_thws_in,
            input_row_offsets_in,
            max_seq_len_in,
            position_ids_in,
        ) = graph.inputs
        assert isinstance(pixel_values_in, TensorValue)
        assert isinstance(grid_thws_in, TensorValue)
        assert isinstance(input_row_offsets_in, TensorValue)
        assert isinstance(max_seq_len_in, TensorValue)
        assert isinstance(position_ids_in, TensorValue)
        outs = vision_tower(
            [pixel_values_in],
            [grid_thws_in],
            [input_row_offsets_in],
            [max_seq_len_in],
            [position_ids_in],
            [],
        )
        graph.output(outs[0])

    compiled = session.load(graph, weights_registry=vision_tower.state_dict())
    (result,) = compiled.execute(
        Buffer.from_dlpack(pixel_values).to(device),
        Buffer.from_dlpack(grid_thws).to(device),
        Buffer.from_dlpack(input_row_offsets).to(device),
        max_seq_len,
        Buffer.from_dlpack(position_ids).to(device),
    )
    assert isinstance(result, Buffer)
    return result


@pytest.mark.parametrize(
    "grid_thws",
    [
        [(1, 4, 4)],
        [(2, 4, 4)],
        [(1, 4, 4), (2, 4, 6)],
    ],
    ids=["single_image", "single_video", "mixed_batch"],
)
def test_transformer(
    grid_thws: list[tuple[int, int, int]],
) -> None:
    """Test Transformer E2E on single GPU."""
    torch.manual_seed(42)

    seq_lens = [t * h * w for t, h, w in grid_thws]
    n_patches = sum(seq_lens)
    input_row_offsets = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.uint32
    )

    state_dict = _create_transformer_weights(VT_NUM_LAYERS)

    pixel_values = _generate_tensor(
        (n_patches, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    )

    position_ids = torch.from_numpy(
        compute_position_ids(grid_thws, ROPE_MAX_WIDTH)
    )

    max_seq_len = torch.tensor([max(seq_lens)], dtype=torch.uint32)

    max_output = _build_and_run_transformer(
        state_dict,
        VT_NUM_LAYERS,
        pixel_values,
        torch.tensor(grid_thws, dtype=torch.int64),
        input_row_offsets,
        max_seq_len,
        position_ids,
    )

    # Build and run torch reference
    ref = TorchTransformer(VT_NUM_LAYERS)
    torch_state_dict = _remap_transformer_keys_for_torch(state_dict)
    ref.load_state_dict(torch_state_dict)
    ref = ref.to(dtype=TORCH_DTYPE)
    torch_grid = torch.tensor(grid_thws)
    torch_output = ref(pixel_values, torch_grid, input_row_offsets).detach()

    _assert_close(torch_output, max_output)
