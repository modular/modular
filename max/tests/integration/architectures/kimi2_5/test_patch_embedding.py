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

"""Tests for Kimi2.5 PatchEmbeddingLayer (MoonVision3dPatchEmbed).

Reference: nvidia/Kimi-K2.5-NVFP4 modeling_kimi_k25.py
- MoonVision3dPatchEmbed: proj (Conv2d) + pos_emb (Learnable2DInterpPosEmbDivided_fixed).
- Forward: x = proj(x).view(x.size(0), -1); x = pos_emb(x, grid_thws).

MAX PatchEmbeddingLayer implements the same proj; pos_emb is currently a no-op
in the graph, so we compare against torch proj(x).view only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.kimi2_5.layers.patch_embedding import (
    PatchEmbeddingLayer,
)
from torch.utils.dlpack import from_dlpack

# Kimi K2.5 vision config values (from checkpoint / reference)
PATCH_SIZE = 14
IN_CHANNELS = 3
HIDDEN_SIZE = 1152
INIT_POS_EMB_HEIGHT = 64
INIT_POS_EMB_WIDTH = 64
INIT_POS_EMB_TIME = 4

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

# 1 image, 2x2 patches -> 4 patches total; grid_thws (1, 3) = (t=1, h=2, w=2)
N_IMAGES = 1
GRID_T, GRID_H, GRID_W = 1, 2, 2
N_PATCHES = GRID_T * GRID_H * GRID_W

SEED_PROJ_WEIGHT = 42
SEED_PROJ_BIAS = 43
SEED_POS_EMB = 44
SEED_INPUT = 45


def _rand_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(dtype)


def _create_state_dict(
    dtype: torch.dtype, has_bias: bool
) -> dict[str, torch.Tensor]:
    """State dict keys match Module.raw_state_dict (proj has name='proj' -> proj.proj.*)."""
    state: dict[str, torch.Tensor] = {
        "proj.proj.weight": _rand_tensor(
            (HIDDEN_SIZE, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE),
            dtype,
            SEED_PROJ_WEIGHT,
        ),
        "pos_emb.weight": _rand_tensor(
            (INIT_POS_EMB_HEIGHT, INIT_POS_EMB_WIDTH, HIDDEN_SIZE),
            dtype,
            SEED_POS_EMB,
        ),
    }
    if has_bias:
        state["proj.proj.bias"] = _rand_tensor(
            (HIDDEN_SIZE,), dtype, SEED_PROJ_BIAS
        )
    return state


def _create_patch_embedding(
    dtype: DType, device: DeviceRef, has_bias: bool = True
) -> PatchEmbeddingLayer:
    """Build PatchEmbeddingLayer with config values (no JSON)."""
    return PatchEmbeddingLayer(
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        init_pos_emb_height=INIT_POS_EMB_HEIGHT,
        init_pos_emb_width=INIT_POS_EMB_WIDTH,
        init_pos_emb_time=INIT_POS_EMB_TIME,
        dtype=dtype,
        device=device,
        has_bias=has_bias,
    )


def _torch_proj_only(
    pixel_values: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    has_bias: bool,
    device: torch.device,
) -> torch.Tensor:
    """Reference: Conv2d(3, 1152, 14, stride=14) then view to (L, 1152)."""
    conv = nn.Conv2d(
        IN_CHANNELS,
        HIDDEN_SIZE,
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE,
        bias=has_bias,
    ).to(device=device, dtype=pixel_values.dtype)
    conv.weight = nn.Parameter(state_dict["proj.proj.weight"].to(device))
    if has_bias:
        conv.bias = nn.Parameter(state_dict["proj.proj.bias"].to(device))
    conv.eval()
    with torch.no_grad():
        out = conv(pixel_values)
    return out.view(out.size(0), -1)


def _build_and_run_max(
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    dtype: DType,
    has_bias: bool,
    n_gpus: int = 0,
) -> torch.Tensor:
    """Build MAX graph, run PatchEmbeddingLayer, return output as torch tensor."""
    devices: list[Device] = (
        [Accelerator(i) for i in range(n_gpus)] if n_gpus > 0 else [CPU(0)]
    )
    device_ref = DeviceRef.GPU() if n_gpus > 0 else DeviceRef.CPU()

    layer = _create_patch_embedding(dtype, device_ref, has_bias)
    layer.load_state_dict(state_dict)

    session = InferenceSession(devices=devices)

    pixel_type = TensorType(dtype, pixel_values.shape, device_ref)
    grid_type = TensorType(DType.int64, grid_thws.shape, device_ref)

    with Graph(
        "kimi2_5_patch_embed_test",
        input_types=(pixel_type, grid_type),
    ) as graph:
        pv, grid = graph.inputs
        out = layer(pv.tensor, grid.tensor)
        graph.output(out)

    compiled = session.load(graph, weights_registry=layer.state_dict())
    run_device = pixel_values.cuda() if n_gpus > 0 else pixel_values.cpu()
    grid_device = grid_thws.cuda() if n_gpus > 0 else grid_thws.cpu()
    result = compiled.execute(
        Buffer.from_dlpack(run_device),
        Buffer.from_dlpack(grid_device),
    )
    return from_dlpack(result[0])


def _assert_close(expected: torch.Tensor, actual: torch.Tensor) -> None:
    rtol = 2e-2
    atol = 4 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected.cpu().float(),
        actual.cpu().float(),
        rtol=rtol,
        atol=atol,
    )


# -----------------------------------------------------------------------------
# Full torch reference (MoonVision3dPatchEmbed + Learnable2DInterpPosEmbDivided_fixed)
# for test_patch_embedding_full_matches_torch once MAX pos_emb is implemented.
# Reference: nvidia/Kimi-K2.5-NVFP4 modeling_kimi_k25.py
# -----------------------------------------------------------------------------


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """1D sincos positional embedding. Returns (t_size, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    grid_t = np.arange(t_size, dtype=np.float32)
    out = np.einsum("m,d->md", grid_t, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def _get_rope_shape(
    org: torch.Tensor, interpolation_mode: str, shape: tuple[int, int]
) -> torch.Tensor:
    """Interpolate 2D pos grid (H, W, C) to (h, w), return (h*w, C)."""
    # org: (H, W, C) -> interpolate to (h, w) -> (h*w, C)
    x = org.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=shape, mode=interpolation_mode)
    x = x.squeeze(0).permute(1, 2, 0).flatten(0, 1)
    return x


class _Learnable2DInterpPosEmbDividedFixedTorch(nn.Module):
    """Torch reference for Learnable2DInterpPosEmbDivided_fixed."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        time_embed = _get_1d_sincos_pos_embed(dim, num_frames)
        self.register_buffer(
            "time_weight",
            torch.from_numpy(time_embed).float().unsqueeze(1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames
            if (h, w) == (self.weight.shape[0], self.weight.shape[1]):
                pos_emb_2d = self.weight.flatten(0, 1)
            else:
                pos_emb_2d = _get_rope_shape(
                    self.weight, self.interpolation_mode, (h, w)
                )
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1)
                    + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, self.dim))
        return x + torch.cat(pos_embs, dim=0)


class _MoonVision3dPatchEmbedTorch(nn.Module):
    """Full torch reference: proj + pos_emb (MoonVision3dPatchEmbed)."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=has_bias,
        )
        self.pos_emb = _Learnable2DInterpPosEmbDividedFixedTorch(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


def _torch_full_patch_embed(
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    has_bias: bool,
    device: torch.device,
) -> torch.Tensor:
    """Full reference: MoonVision3dPatchEmbed (proj + pos_emb)."""
    model = _MoonVision3dPatchEmbedTorch(
        out_dim=HIDDEN_SIZE,
        in_dim=IN_CHANNELS,
        patch_size=PATCH_SIZE,
        pos_emb_height=INIT_POS_EMB_HEIGHT,
        pos_emb_width=INIT_POS_EMB_WIDTH,
        pos_emb_time=INIT_POS_EMB_TIME,
        has_bias=has_bias,
    ).to(device=device, dtype=pixel_values.dtype)
    model.proj.weight = nn.Parameter(state_dict["proj.proj.weight"].to(device))
    if has_bias:
        model.proj.bias = nn.Parameter(state_dict["proj.proj.bias"].to(device))
    model.pos_emb.weight = nn.Parameter(state_dict["pos_emb.weight"].to(device))
    model.eval()
    with torch.no_grad():
        return model(pixel_values, grid_thws)


@pytest.mark.skip(
    reason="MAX pos_emb is not yet implemented; enable when PatchEmbeddingLayer applies position embeddings."
)
def test_patch_embedding_full_matches_torch() -> None:
    """Compare full PatchEmbeddingLayer (proj + pos_emb) to torch MoonVision3dPatchEmbed.

    Enable this test once Learnable2DInterpPosEmbDividedFixed applies position
    embeddings in the graph (interpolation + time_weight + per-image slicing).
    """
    has_bias = True
    torch.manual_seed(SEED_INPUT)
    pixel_values = _rand_tensor(
        (N_PATCHES, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE),
        TORCH_DTYPE,
        SEED_INPUT,
    )
    grid_thws = torch.tensor([[GRID_T, GRID_H, GRID_W]], dtype=torch.int64)
    state_dict = _create_state_dict(TORCH_DTYPE, has_bias)
    device = torch.device("cuda")
    pixel_values = pixel_values.to(device)
    grid_thws = grid_thws.to(device)

    torch_out = _torch_full_patch_embed(
        pixel_values, grid_thws, state_dict, has_bias, device
    )
    max_out = _build_and_run_max(
        pixel_values, grid_thws, state_dict, MAX_DTYPE, has_bias, n_gpus=1
    )

    _assert_close(torch_out, max_out)
    assert max_out.shape == (N_PATCHES, HIDDEN_SIZE)


def test_patch_embedding_forward_matches_torch_proj() -> None:
    """PatchEmbeddingLayer output matches torch Conv2d projection (pos_emb is no-op)."""
    has_bias = True  # Model uses bias in patch_embed.proj
    torch.manual_seed(SEED_INPUT)
    pixel_values = _rand_tensor(
        (N_PATCHES, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE),
        TORCH_DTYPE,
        SEED_INPUT,
    )
    grid_thws = torch.tensor([[GRID_T, GRID_H, GRID_W]], dtype=torch.int64)

    state_dict = _create_state_dict(TORCH_DTYPE, has_bias)
    device = torch.device("cuda")
    pixel_values = pixel_values.to(device)
    grid_thws = grid_thws.to(device)

    torch_out = _torch_proj_only(pixel_values, state_dict, has_bias, device)
    max_out = _build_and_run_max(
        pixel_values, grid_thws, state_dict, MAX_DTYPE, has_bias, n_gpus=1
    )

    _assert_close(torch_out, max_out)
    assert max_out.shape == (N_PATCHES, HIDDEN_SIZE)
