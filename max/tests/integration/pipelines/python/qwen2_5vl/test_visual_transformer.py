# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight
from max.nn import Conv3D
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionPatchEmbed,
)
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


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
