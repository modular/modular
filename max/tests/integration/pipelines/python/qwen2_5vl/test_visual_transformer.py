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
from max.graph import Graph, TensorType, Weight
from max.nn import Conv3D
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    get_window_index,
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionPatchEmbed,
    VisionRotaryEmbedding,
    VisionTransformer,
)
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-5


@pytest.fixture
def torch_model_and_inputs():
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="float", device_map="auto"
    ).visual
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image mode=RGB size=2044x1372
    inputs = process_vision_info(messages)
    processed_inputs = processor(
        text=[text],
        images=inputs[0],
        videos=inputs[1],
        padding=True,
        return_tensors="pt",
    )
    return model, processed_inputs


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


@pytest.mark.skip(
    reason="Loads the model and real images. Used for debugging but not needed in CI."
)
def test_vision_patch_embed(torch_model_and_inputs):
    model, inputs = torch_model_and_inputs

    # Inputs to the visual transformer for one image and no videos:
    # hidden_states of shape resized_width x resized_height = [14308, 1176] and grid_thw of shape [1, 3]
    pixel_values = inputs["pixel_values"]

    temporal_patch_size = model.patch_embed.temporal_patch_size  # 2
    patch_size = model.patch_embed.patch_size  # 14
    in_channels = model.patch_embed.in_channels  # 3
    embed_dim = model.patch_embed.embed_dim  # 1280
    kernel_size = (temporal_patch_size, patch_size, patch_size)

    # Get torch model outputs and permute it to match graph API output shape.
    with torch.no_grad():
        torch_patch_embeds = model.patch_embed(pixel_values)

    graph_api_pixel_values = pixel_values

    # Register Layer weights in weights_registry.
    weights_registry = {}

    def weight(name: str, array) -> Weight:
        """Creates a Weight from the input array of and add it to weights_registry."""
        if array.dtype != torch.float32:
            array = array.float()
        weights_registry[name] = array
        return Weight(
            name=name,
            dtype=DType.float32,
            shape=array.shape,
        )

    # Permute the weights of Conv layer.
    graph_api_conv_weights = torch.permute(
        model.patch_embed.proj.weight.data, (2, 3, 4, 1, 0)
    ).contiguous()

    # Get graph api outputs.
    graph_api_proj = Conv3D(
        weight("patch_embed", graph_api_conv_weights),
        bias=None,
        stride=kernel_size,
    )

    graph_api_patch_embed = VisionPatchEmbed(
        proj=graph_api_proj,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    session = InferenceSession()
    with Graph(
        "visual",
        input_types=[
            TensorType(DType.float32, (pixel_values.shape)),
        ],
    ) as graph:
        graph_api_patch_embeds = graph_api_patch_embed(graph.inputs[0].tensor)
        graph.output(graph_api_patch_embeds)
    compiled = session.load(graph, weights_registry=weights_registry)
    graph_api_output = compiled.execute(graph_api_pixel_values)[0]
    assert isinstance(graph_api_output, Tensor)

    # Compare results.
    np.testing.assert_allclose(
        graph_api_output.to_numpy(),
        torch_patch_embeds.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


@pytest.mark.skip(
    reason="Loads the model and real images. Used for debugging but not needed in CI."
)
def test_rot_embed(torch_model_and_inputs):
    model, inputs = torch_model_and_inputs

    # Inputs to the visual transformer for one image and no videos:
    # hidden_states of shape resized_width x resized_height = [14308, 1176] and grid_thw of shape [1, 3]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    # Permute (batch_size, in_channels, depth, height, width) inputs to (batch_size, depth, height, width, in_channels) for our Graph API.
    # graph_api_pixel_values = torch.permute(
    #     pixel_values, (0, 2, 3, 4, 1)
    # ).contiguous()
    graph_api_pixel_values = pixel_values

    temporal_patch_size = model.patch_embed.temporal_patch_size  # 2
    patch_size = model.patch_embed.patch_size  # 14
    in_channels = model.patch_embed.in_channels  # 3
    embed_dim = model.patch_embed.embed_dim  # 1280
    kernel_size = (temporal_patch_size, patch_size, patch_size)
    spatial_merge_size = model.spatial_merge_size
    hidden_size = 1280
    n_heads = 16
    theta = 10000.0
    window_size = model.window_size
    spatial_merge_unit = model.spatial_merge_unit

    # Get torch model outputs and permute it to match graph API output shape.
    with torch.no_grad():
        hidden_states = model.patch_embed(pixel_values)
        rotary_pos_emb = model.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = model.get_window_index(image_grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // spatial_merge_unit, spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // spatial_merge_unit, spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        torch_model_output = (emb.cos(), emb.sin())[0]

    # Register Layer weights in weights_registry.
    weights_registry = {}

    def weight(name: str, array) -> Weight:
        """Creates a Weight from the input array of and add it to weights_registry."""
        weights_registry[name] = array
        if array.dtype != torch.float32:
            array = array.float()
        return Weight(
            name=name,
            dtype=DType.float32,
            shape=array.shape,
        )

    # Permute the weights of Conv layer.
    graph_api_conv_weights = torch.permute(
        model.patch_embed.proj.weight.data, (2, 3, 4, 1, 0)
    ).contiguous()

    # Get graph api outputs.
    graph_api_proj = Conv3D(
        weight("patch_embed", graph_api_conv_weights),
        bias=None,
        stride=kernel_size,
    )
    graph_api_patch_embed = VisionPatchEmbed(
        proj=graph_api_proj,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    rotary_pos_emb = VisionRotaryEmbedding(
        dim=hidden_size, n_heads=n_heads, theta=theta
    )
    visual_transformer = VisionTransformer(
        patch_embed=graph_api_patch_embed,
        rotary_pos_emb=rotary_pos_emb,
        spatial_merge_unit=spatial_merge_unit,
    )

    rot_pos_ids = mrope_pos_ids_3d(image_grid_thw, spatial_merge_size)
    window_index, cu_window_seqlens = get_window_index(
        image_grid_thw,
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit,
    )

    max_grid_size = image_grid_thw[:, 1:].max().item()

    session = InferenceSession()
    with Graph(
        "visual",
        input_types=[
            TensorType(DType.float32, (pixel_values.shape)),
            TensorType(DType.int64, (image_grid_thw.shape)),
            TensorType(DType.int64, (rot_pos_ids.shape)),
            TensorType(DType.int64, (window_index.shape)),
            TensorType(DType.int64, (cu_window_seqlens.shape)),
        ],
    ) as graph:
        visual_transformer_output = visual_transformer(
            x=graph.inputs[0].tensor,
            grid_thw=graph.inputs[1].tensor,
            rot_pos_ids=graph.inputs[2].tensor,
            max_grid_size=max_grid_size,
            window_index=graph.inputs[3].tensor,
            cu_window_seqlens=graph.inputs[4].tensor,
        )
        graph.output(visual_transformer_output)

    compiled = session.load(graph, weights_registry=weights_registry)
    graph_api_output = compiled.execute(
        graph_api_pixel_values,
        image_grid_thw,
        rot_pos_ids,
        window_index,
        cu_window_seqlens,
    )[0]
    assert isinstance(graph_api_output, Tensor)

    # Compare results.
    np.testing.assert_allclose(
        graph_api_output.to_numpy(),
        torch_model_output.detach().float().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
