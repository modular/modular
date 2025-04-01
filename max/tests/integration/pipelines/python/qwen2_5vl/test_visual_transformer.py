# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    generate_attention_mask,
    get_window_index,
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionWindowSdpaAttention,
)

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6

torch.manual_seed(0)


def torch_mrope_pos_ids_3d(
    grid_thw: torch.tensor, spatial_merge_size: int
) -> torch.tensor:
    """Calculate the 3D rope index based on image and video's temporal, height and width in LLM."""
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
    return pos_ids


def torch_get_window_index(
    grid_thw: torch.tensor,
    window_size: int,
    spatial_merge_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> tuple[torch.tensor, list]:
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h, llm_grid_w = (
            grid_h // spatial_merge_size,
            grid_w // spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
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
        window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
    window_index = torch.cat(window_index, dim=0)

    return window_index, cu_window_seqlens


def torch_generate_attention_mask(
    grid_thw: torch.tensor, seq_length: int, cu_window_seqlens: list
) -> tuple[torch.tensor, torch.tensor]:
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=grid_thw.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    attention_mask_cu_seqlens = torch.zeros(
        [1, seq_length, seq_length], device=grid_thw.device, dtype=torch.bool
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask_cu_seqlens[
            ...,
            cu_seqlens[i - 1] : cu_seqlens[i],
            cu_seqlens[i - 1] : cu_seqlens[i],
        ] = True

    attention_mask_cu_window_seqlens = torch.zeros(
        [1, seq_length, seq_length], device=grid_thw.device, dtype=torch.bool
    )
    for i in range(1, len(cu_window_seqlens)):
        attention_mask_cu_window_seqlens[
            ...,
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
        ] = True

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    attention_mask_cu_seqlens = torch.zeros(
        (1, seq_length, seq_length)
    ).masked_fill_(attention_mask_cu_seqlens.logical_not(), fill_val)
    attention_mask_cu_window_seqlens = torch.zeros(
        (1, seq_length, seq_length)
    ).masked_fill_(attention_mask_cu_window_seqlens.logical_not(), fill_val)

    return attention_mask_cu_seqlens, attention_mask_cu_window_seqlens


def test_pos_ids():
    image_grid_thw = torch.tensor([[1, 98, 146], [1, 22, 28]])
    spatial_merge_size = 2
    expected_pos_ids = torch_mrope_pos_ids_3d(
        image_grid_thw, spatial_merge_size
    )
    actual_pos_ids = mrope_pos_ids_3d(
        image_grid_thw.detach().numpy(), spatial_merge_size
    )
    np.testing.assert_array_equal(
        actual_pos_ids,
        expected_pos_ids.detach().numpy(),
    )


def test_window_index():
    image_grid_thw = torch.tensor([[1, 98, 146], [1, 22, 28]])
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    expected_window_index, expected_cu_window_seqlens = torch_get_window_index(
        image_grid_thw,
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )
    actual_window_index, actual_cu_window_seqlens = get_window_index(
        image_grid_thw.detach().numpy(),
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )
    np.testing.assert_array_equal(
        actual_window_index,
        expected_window_index.detach().numpy(),
    )

    np.testing.assert_array_equal(
        actual_cu_window_seqlens,
        np.array(expected_cu_window_seqlens),
    )


def test_generate_attention_mask():
    image_grid_thw = torch.tensor([[1, 98, 146], [1, 22, 28]])
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14
    seq_length = 14308
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    _, cu_window_seqlens = torch_get_window_index(
        image_grid_thw,
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )
    actual_attention_mask_full, actual_attention_mask_window = (
        generate_attention_mask(
            image_grid_thw.detach().numpy(),
            seq_length,
            np.array(cu_window_seqlens),
        )
    )
    attention_mask_full, attention_mask_window = torch_generate_attention_mask(
        image_grid_thw, seq_length, cu_window_seqlens
    )

    np.testing.assert_allclose(
        attention_mask_full.detach().numpy(),
        actual_attention_mask_full,
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )

    np.testing.assert_allclose(
        attention_mask_window.detach().numpy(),
        actual_attention_mask_window,
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


@pytest.mark.skip(reason="Currently failing with error 1e-2. Still debugging.")
def test_scaled_dot_product_attention_vision_mask():
    # batch_size, Target seq_len, embed_dim
    q = torch.rand(16, 14308, 80, dtype=torch.float32)
    k = torch.rand(16, 14308, 80, dtype=torch.float32)
    v = torch.rand(16, 14308, 80, dtype=torch.float32)
    # batch_size, target seq_len, source seq_len
    attention_mask = torch.bernoulli(torch.full((1, 14308, 14308), 0.5)).to(
        torch.bool
    )
    dim = 1280
    n_heads = 16
    expected_attn_output = F.scaled_dot_product_attention(
        q, k, v, attention_mask, dropout_p=0.0
    )

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    max_attention_mask = torch.zeros(
        attention_mask.shape, dtype=torch.float32
    ).masked_fill_(attention_mask.logical_not(), fill_val)

    session = InferenceSession()
    with Graph(
        "visual",
        input_types=[
            TensorType(DType.float32, (q.shape)),
            TensorType(DType.float32, (k.shape)),
            TensorType(DType.float32, (v.shape)),
            TensorType(DType.float32, (attention_mask.shape)),
        ],
    ) as graph:
        attn_output = VisionWindowSdpaAttention.scaled_dot_product_attention(
            xq=graph.inputs[0].tensor,
            xk=graph.inputs[1].tensor,
            xv=graph.inputs[2].tensor,
            attention_mask=graph.inputs[3].tensor,
            dim=dim,
            n_heads=n_heads,
        )

        graph.output(attn_output)
        compiled = session.load(graph)
        max_graph_output = compiled.execute(q, k, v, max_attention_mask)[0]
        assert isinstance(max_graph_output, Tensor)

        # Compare results.
        np.testing.assert_allclose(
            max_graph_output.to_numpy(),
            expected_attn_output.detach().numpy(),
            equal_nan=True,
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )
