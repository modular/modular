# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import Tuple

import numpy as np
import pytest
import torch
from max.driver import DLPackArray
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight
from nn.linear import Linear

# from pixtral.llava import LlavaConditionalGeneration
from pixtral.llava_projector import LlavaMultiModalConnector
from pixtral.vision_encoder.vision_encoder import VisionEncoder
from transformers import LlavaConfig, MistralConfig, PixtralVisionConfig
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

ACCURACY_RTOL = 1e-1
ACCURACY_ATOL = 1e-1


@pytest.fixture
def pytorch_connector():
    text_config = MistralConfig(
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        head_dim=128,
        hidden_act="silu",
        hidden_size=5120,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=1024000,
        model_type="mistral",
        num_attention_heads=32,
        num_hidden_layers=40,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
        vocab_size=131072,
    )
    vision_config = PixtralVisionConfig()
    return LlavaMultiModalProjector(
        LlavaConfig(
            vision_config,
            text_config,
            vision_feature_layer=-1,
            image_token_index=10,
            vision_feature_select_strategy="full",
            image_seq_length=1,
        )
    )


@pytest.fixture
def img_dtype():
    return torch.float32


@pytest.fixture
def size():
    return (1, 160, 1024)


@pytest.fixture
def img_features(size: Tuple, img_dtype: torch.dtype):
    # Gnerate imgs of shape (batch_size, seq_len=num_patches, hidden_size).
    return torch.randint(low=0, high=1, size=size).to(img_dtype)


@pytest.fixture
def graph_api_connector(pytorch_connector: LlavaMultiModalProjector):
    weights_registry: dict[str, DLPackArray] = {}
    weights_registry["linear_1"] = pytorch_connector.linear_1.weight.data
    weights_registry["linear_2"] = pytorch_connector.linear_2.weight.data

    linear_1 = Linear(
        Weight(
            name="linear_1",
            dtype=DType.from_numpy(weights_registry["linear_1"].numpy().dtype),  # type: ignore
            shape=weights_registry["linear_1"].shape,  # type: ignore
        )
    )
    linear_2 = Linear(
        Weight(
            name="linear_2",
            dtype=DType.from_numpy(weights_registry["linear_2"].numpy().dtype),  # type: ignore
            shape=weights_registry["linear_2"].shape,  # type: ignore
        )
    )
    connector = LlavaMultiModalConnector(linear_1, linear_2)
    return connector, weights_registry


@pytest.fixture
def graph_api_pixtral(
    graph_api_connector: Tuple[LlavaMultiModalConnector, dict]
):
    # vision_encoder =
    connector, weights_registry = graph_api_connector
    # LlavaConditionalGeneration(vision_encoder: VisionEncoder
    # multi_modal_projector=graph_api_connector,
    # vocab_size: int
    # language_model: Transformer
    # pad_token_id= -1
    # image_token_index= 10)
    pass


def test_connector(
    img_features: torch.Tensor,
    pytorch_connector: LlavaMultiModalProjector,
    graph_api_connector: Tuple[LlavaMultiModalConnector, dict],
    size: Tuple,
):
    weights_registry: dict[str, DLPackArray] = {}
    connector = graph_api_connector[0]
    weights_registry.update(graph_api_connector[1])

    session = InferenceSession()
    graph = Graph(
        "Llava_MLP",
        connector,
        input_types=(TensorType(DType.float32, size),),
    )

    compiled = session.load(graph, weights_registry=weights_registry)

    output = compiled.execute(img_features)[0].to_numpy()  # type: ignore
    pytorch_output = pytorch_connector(img_features).detach().numpy()

    np.testing.assert_allclose(
        output,
        pytorch_output,
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
