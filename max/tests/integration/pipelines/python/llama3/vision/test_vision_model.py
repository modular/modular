# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision model tests by comparing it against the transformers
package reference implementation.
"""

import pytest
from pathlib import Path
from llama_vision.vision_model import instantiate_vision_model
from max.pipelines import PipelineConfig, SupportedEncoding
from max.dtype import DType
from max.graph import Graph, ops
from max.graph.weights import SafetensorWeights


def generate_test_vision_model() -> Graph:
    """
    This helper function generates a test vision model instance for testing purposes.
    """

    pipeline_config = PipelineConfig(
        architecture="MllamaForConditionalGeneration",
        huggingface_repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization_encoding=SupportedEncoding.bfloat16,
        weight_path=[
            Path("model-00001-of-00005.safetensors"),
            Path("model-00002-of-00005.safetensors"),
            Path("model-00003-of-00005.safetensors"),
            Path("model-00004-of-00005.safetensors"),
            Path("model-00005-of-00005.safetensors"),
        ],
    )
    vision_config = pipeline_config.huggingface_config.vision_config

    weights = pipeline_config.load_weights()
    assert isinstance(
        weights, SafetensorWeights
    ), "only safetensor weights supported currently"

    with Graph("test_llama_vision") as graph:
        print("building vision model...")
        vision_model = instantiate_vision_model(
            dtype=pipeline_config.dtype,
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            supported_aspect_ratios=vision_config.supported_aspect_ratios,
            hidden_size=vision_config.hidden_size,
            max_num_tiles=vision_config.max_num_tiles,
            num_channels=vision_config.num_channels,
            norm_eps=vision_config.norm_eps,
            attention_heads=vision_config.attention_heads,
            num_hidden_layers=vision_config.num_hidden_layers,
            intermediate_size=vision_config.intermediate_size,
            num_global_layers=vision_config.num_global_layers,
            intermediate_layers_indices=[3, 7, 15, 23, 30],
            weights=weights,
        )

        graph.output(
            ops.constant(
                len(vision_model.intermediate_layers_indices), dtype=DType.int32
            ),
        )

        return graph


@pytest.mark.skip("requires internet and very large 20GB+")
def test_build_vision_model():
    """
    This test is not meant to be run in CI.
    It will require the internet and download over 20gb of weights.
    It is primarily meant to be run as a double check function that the vision model continues to build.
    """
    vision_model = generate_test_vision_model()
    assert isinstance(vision_model, Graph)
