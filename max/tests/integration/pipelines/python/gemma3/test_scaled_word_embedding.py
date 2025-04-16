# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
)


def generate_torch_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    embedding_weights: torch.Tensor,
) -> torch.Tensor:
    layer = (
        Gemma3TextScaledWordEmbedding(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
            padding_idx=text_config.pad_token_id,
            embed_scale=1.0,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    layer.weight.data = embedding_weights.to(torch.bfloat16).to("cuda")
    return layer(input_tensor.to("cuda")).to(torch.bfloat16)


def generate_max_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    embedding_weights: torch.Tensor,
) -> torch.Tensor:
    layer = ScaledWordEmbedding(
        vocab_size=text_config.vocab_size,
        hidden_dim=text_config.hidden_size,
        dtype=DType.bfloat16,
        device=DeviceRef.GPU(),
        quantization_encoding=None,
        name="embeddings",
        embed_scale=1.0,
    )

    # Weights need to be passed as CPU tensors
    state_dict = {"embeddings": embedding_weights.cpu()}
    layer.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator()])
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "ScaledWordEmbedding",
        layer,
        input_types=(
            TensorType(
                DType.int64,
                Shape(input_tensor.shape),
                device=DeviceRef.GPU(),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=state_dict)
    max_output = compiled.execute(input_tensor.to("cuda"))
    return torch.from_dlpack(max_output[0]).to(torch.bfloat16)


def test_scaled_word_embedding(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    embedding_weights: torch.Tensor,
) -> None:
    torch_output = generate_torch_outputs(
        text_config, input_tensor, embedding_weights
    )
    max_output = generate_max_outputs(
        text_config, input_tensor, embedding_weights
    )

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=1e-3,
        atol=1e-3,
    )
