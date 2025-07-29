# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Union

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.pipelines.architectures.qwen2_5vl.nn.input_processing_fns import (
    get_cu_seqlens_numpy,
    get_cu_window_seqlens_numpy,
    get_window_index_numpy,
    rot_pos_emb_numpy,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)


def cu_seqlens_torch(
    cu_window_seqlens: list[int], grid_thw: torch.Tensor
) -> torch.Tensor:
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    return cu_seqlens, cu_window_seqlens


def rot_pos_emb_torch(
    grid_thw: Union[np.ndarray, torch.Tensor], spatial_merge_size: int
) -> torch.Tensor:
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max()
    return pos_ids, max_grid_size


def test_get_window_index_and_cumulative_seqlens():
    grid_thw = np.array([[1, 98, 146], [1, 76, 114]], dtype=np.int64)
    window_size = 112
    spatial_merge_size = 2
    patch_size = 14

    vision_transformer = Qwen2_5_VisionTransformerPretrainedModel._from_config(
        Qwen2_5_VLVisionConfig()
    )
    window_index_torch, cu_window_seqlens_torch = (
        vision_transformer.get_window_index(grid_thw)
    )

    window_index_numpy, cu_window_seqlens_numpy = get_window_index_numpy(
        grid_thw, window_size, spatial_merge_size, patch_size
    )

    # Convert numpy window_index to torch tensor for comparison
    window_index_numpy_torch = torch.from_numpy(window_index_numpy)

    # Compare window indices (convert to same dtype first)
    assert torch.allclose(
        window_index_torch.float(),
        window_index_numpy_torch.float(),
        rtol=1e-6,
        atol=1e-8,
    )

    # Compare cu_window_seqlens (convert to numpy arrays for comparison)
    assert np.array_equal(cu_window_seqlens_torch, cu_window_seqlens_numpy)

    cu_seqlens_torch_result, cu_window_seqlens_torch_result = cu_seqlens_torch(
        cu_window_seqlens_torch, torch.from_numpy(grid_thw)
    )
    cu_seqlens_numpy = get_cu_seqlens_numpy(grid_thw)
    cu_window_seqlens_numpy = get_cu_window_seqlens_numpy(
        cu_window_seqlens_numpy
    )

    assert np.array_equal(
        cu_window_seqlens_torch_result, cu_window_seqlens_numpy
    )

    assert np.array_equal(cu_seqlens_torch_result, cu_seqlens_numpy)


def test_rot_pos_emb():
    grid_thw = np.array([[1, 98, 146], [1, 76, 114]], dtype=np.int64)
    spatial_merge_size = 2

    torch_result, torch_max_grid_size = rot_pos_emb_torch(
        grid_thw, spatial_merge_size
    )
    numpy_result, numpy_max_grid_size = rot_pos_emb_numpy(
        grid_thw, spatial_merge_size
    )

    # Convert numpy result to torch tensor for comparison
    numpy_result_torch = torch.from_numpy(numpy_result)

    assert torch.allclose(
        torch_result, numpy_result_torch, rtol=1e-6, atol=1e-8
    )

    assert torch_max_grid_size == numpy_max_grid_size


if __name__ == "__main__":
    pytest.main([__file__])
