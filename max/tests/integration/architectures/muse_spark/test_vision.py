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
"""The Muse Spark vision encoder's host-side RoPE tables, window order and
position-table interpolation against a torch transcription of the reference
(transformers 5.20.0 ``modeling_muse_spark.py`` and the ``vision_utils.py``
helpers it calls). A placebo reading the position table on the untransposed
grid must not match.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as TF
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.muse_glimmer.vision import data_processing
from max.pipelines.architectures.muse_spark.batch_vision_inputs import (
    pack_uncached_images,
)
from max.pipelines.architectures.muse_spark.vision import (
    parse_vision_config,
    vision_inputs,
)
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"
NUM_LAYERS = 4
# Torch's float32 pow is not correctly rounded, so the tables need a
# tolerance; a float64 table misses by ~2e-6.
ROPE_ATOL = 4 * float(np.finfo(np.float32).eps)
# 448x448 and 224x896 after smart_resize: patch grids 32x32 and 16x64.
IMAGE_GRIDS = [[1, 32, 32], [1, 16, 64]]
ROPE_GRIDS = [[[1, 32, 32]], [[1, 16, 64]], [[1, 64, 16]], [[1, 46, 74]]]
ROPE_GRIDS += [IMAGE_GRIDS, [[1, 128, 128]]]

GEN = torch.Generator().manual_seed(0)


def _published() -> dict[str, Any]:
    return json.loads((TESTDATA / "config_nvfp4.json").read_text())


def _vision_ref_config() -> SimpleNamespace:
    """Read from the published config, independent of MAX's parsing."""
    c = _published()
    v = c["vision_config"]
    return SimpleNamespace(
        hidden_size=v["hidden_size"],
        num_heads=v["num_attention_heads"],
        head_dim=v["hidden_size"] // v["num_attention_heads"],
        layer_norm_eps=v["layer_norm_eps"],
        layer_types=v["layer_types"][:NUM_LAYERS],
        patch_size=v["patch_size"],
        pos_emb_height=v["pos_emb_height"],
        merge_size=v["merge_size"],
        rope_theta=float(v["rope_parameters"]["rope_theta"]),
        text_rms_norm_eps=c["text_config"]["rms_norm_eps"],
    )


REF = _vision_ref_config()


# --------------------------------------------------------------------------- #
# Reference: vision_utils.py helpers
# --------------------------------------------------------------------------- #


def ref_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )
    return TF.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def ref_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    position_ids = []
    for t, h, w in grid_thw.tolist():
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij"
        )
        m = spatial_merge_size
        block_shape = (h // m, m, w // m, m)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(
            torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        )
    return torch.cat(position_ids, dim=0)


def ref_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    window_index: list[torch.Tensor] = []
    cu_window_seqlens: list[int] = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2
    for grid_t, grid_h, grid_w in grid_thw.tolist():
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = TF.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = (
            seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        )
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w
    cu = torch.tensor(cu_window_seqlens, dtype=torch.int32)
    return torch.cat(window_index, dim=0), torch.unique_consecutive(cu)


def _ref_axis_taps(
    index: torch.Tensor, size: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # mode="bilinear", align_corners=False, padding="zeros".
    index = index.to(torch.float32)
    src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    offsets = torch.arange(0, 2)
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    weights = (1 - distance).clamp(min=0)
    weights = weights * ((raw_taps >= 0) & (raw_taps <= side - 1))
    return taps, weights


def ref_interpolation(
    grid_thw: torch.Tensor, num_grid_per_side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # spatial_merge_size=1: raster patch order.
    side = num_grid_per_side
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(
        TF.pad(counts.cumsum(0)[:-1], (1, 0)), counts
    )
    within = (torch.arange(int(counts.sum())) - starts) % (heights * widths)
    row, col = within // widths, within % widths
    h_taps, h_weights = _ref_axis_taps(row, heights, side)
    w_taps, w_weights = _ref_axis_taps(col, widths, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


# --------------------------------------------------------------------------- #
# Reference: modeling_muse_spark.py
# --------------------------------------------------------------------------- #


def ref_rotary(
    position_ids: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``MuseSparkVisionRotaryEmbedding`` (rope_type axial)."""
    spatial_dim = REF.head_dim // 2
    inv_freq = 1.0 / (
        REF.rope_theta
        ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
    )
    freqs = position_ids[..., None].float() * inv_freq.to(
        device=position_ids.device
    )

    def recompose(freq: torch.Tensor) -> torch.Tensor:
        freq_hw = torch.cat([freq[:, 0], freq[:, 1]], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

    # cos/sin of the float32 angles, rounded once from float64: torch's
    # float32 cos once returned values 1.5e-4 off on a CI host.
    freqs = freqs.double()
    return recompose(freqs.cos()).to(dtype), recompose(freqs.sin()).to(dtype)


def ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """``apply_rotary_pos_emb_vision`` for one of q, k."""
    orig = x.dtype
    x = x.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((x * cos) + (ref_rotate_half(x) * sin)).to(orig)


def ref_rope_inputs(
    grid_thw: torch.Tensor, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``MuseSparkVisionModel.forward``'s window order and RoPE tables."""
    window_index, cu_window_seqlens = ref_window_index(
        grid_thw,
        spatial_merge_size=1,
        window_size=REF.pos_emb_height * REF.patch_size,
        patch_size=REF.patch_size,
    )
    position_ids = ref_position_ids(grid_thw, spatial_merge_size=1).flip(-1) + 1
    cos, sin = ref_rotary(position_ids[window_index, :], dtype)
    return window_index, cu_window_seqlens, cos, sin


def _max_config() -> Any:
    hf = AutoConfig.from_pretrained(str(TESTDATA / "config_nvfp4.json"))
    config = parse_vision_config(hf)
    return dataclasses.replace(
        config,
        num_hidden_layers=NUM_LAYERS,
        layer_types=config.layer_types[:NUM_LAYERS],
    )


@pytest.mark.parametrize("grids", ROPE_GRIDS, ids=str)
def test_rope_matches_reference(grids: list[list[int]]) -> None:
    grid_thw = torch.tensor(grids)
    window_index, cu_window, cos, sin = ref_rope_inputs(grid_thw)
    ins = vision_inputs(np.array(grids), _max_config())

    np.testing.assert_array_equal(ins.window_index, window_index.numpy())
    np.testing.assert_array_equal(ins.cu_window_seqlens, cu_window.numpy())
    np.testing.assert_allclose(ins.rot_cos, cos.numpy(), rtol=0, atol=ROPE_ATOL)
    np.testing.assert_allclose(ins.rot_sin, sin.numpy(), rtol=0, atol=ROPE_ATOL)

    # The tower rotates q and k with GLIMMER/vision's _apply_rope:
    # x * cos + rotate_half(x) * sin in float32 on these tables.
    x = torch.randn(
        len(window_index), REF.num_heads, REF.head_dim, generator=GEN
    )
    x1, x2 = x.split(REF.head_dim // 2, dim=-1)
    rotated = (
        x * torch.from_numpy(ins.rot_cos)[:, None]
        + torch.cat([-x2, x1], dim=-1) * torch.from_numpy(ins.rot_sin)[:, None]
    )
    expected = ref_apply_rotary(x[None], cos, sin)[0]
    bound = 2 * ROPE_ATOL * float(x.abs().max())
    assert float((rotated - expected).abs().max()) <= bound


def test_rope_position_table_reads_transposed_grid() -> None:
    """Patch (h, w) reads the position table at (w, h) on a non-square grid."""
    grids = [[1, 16, 64]]
    idx, wts = ref_interpolation(
        torch.tensor(grids)[:, [0, 2, 1]], REF.pos_emb_height
    )
    ins = vision_inputs(np.array(grids), _max_config())
    np.testing.assert_array_equal(ins.interp_idx, idx.numpy())
    np.testing.assert_allclose(ins.interp_w, wts.numpy(), rtol=0, atol=1e-6)
    # The placebo: reading the table on the untransposed grid misses.
    placebo, _ = ref_interpolation(torch.tensor(grids), REF.pos_emb_height)
    assert not np.array_equal(ins.interp_idx, placebo.numpy())


def test_glimmer_float64_rope_tables_miss_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Glimmer's float64 tables fall outside the bar Spark's float32 meet."""
    grids = [[1, 32, 32]]
    _, _, cos, _ = ref_rope_inputs(torch.tensor(grids))
    monkeypatch.setattr(
        "max.pipelines.architectures.muse_spark.vision.data_processing"
        ".rope_tables",
        data_processing.rope_tables,
    )
    ins = vision_inputs(np.array(grids), _max_config())
    assert float(np.abs(ins.rot_cos - cos.numpy()).max()) > ROPE_ATOL


def test_pack_uncached_images_emits_the_tower_inputs_in_order() -> None:
    """Selected images pack as patches, then the vision inputs, in order."""
    config = _max_config()
    width = 3 * config.patch_temporal * config.patch_size**2
    images: list[Any] = []
    start = 0
    for t, h, w in IMAGE_GRIDS:
        tokens = t * h * w // config.merge_size**2
        pixels = np.full((t * h * w, width), len(images) + 1, np.float32)
        images.append(
            SimpleNamespace(
                pixel_values=pixels, start_idx=start, end_idx=start + tokens
            )
        )
        start += tokens
    ctx: Any = SimpleNamespace(
        request_id="r",
        images=images,
        extra_model_args={"image_grid_thw": np.array(IMAGE_GRIDS)},
    )
    assert (
        pack_uncached_images([(ctx, [])], CPU(), config, DType.float32) is None
    )
    # Selection order is ctx.images' order, not the selection's.
    packed = pack_uncached_images(
        [(ctx, images[::-1])], CPU(), config, DType.float32
    )
    assert packed is not None
    expected = vision_inputs(np.array(IMAGE_GRIDS), config)
    assert len(packed) == 1 + len(expected)
    np.testing.assert_array_equal(
        packed[0].to_numpy(),
        np.concatenate([img.pixel_values for img in images]),
    )
    for buffer, array in zip(packed[1:], expected, strict=True):
        np.testing.assert_array_equal(buffer.to_numpy(), array)
