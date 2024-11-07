# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision encoder layer.
"""

import numpy as np
import pytest

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from modular_graph_test import modular_graph_test
from nn import Linear, LPLayerNorm
from llama3.vision.encoder import (
    VisionEncoder,
    VisionEncoderLayer,
)
from llama3.vision.mlp import MLP
from torch_vision_encoder import MllamaVisionEncoder, MllamaVisionEncoderLayer


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"]),
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["x", "y", "z", "dim"]),
    ],
)
def test_vision_encoder_layer(
    session: InferenceSession, input_type: TensorType
) -> None:
    dim = input_type.shape[-1]
    mlp_fc1_type = TensorType(input_type.dtype, ["hidden_dim", dim])
    mlp_fc2_type = TensorType(input_type.dtype, [dim, "hidden_dim"])
    encoder_layernorm_w1_type = TensorType(input_type.dtype, [dim])
    encoder_layernorm_w2_type = encoder_layernorm_w1_type

    # TODO: Set these as is for now - can look to parameterizing them later on.
    eps = 1e-5
    is_gated = False
    gate_attn = None
    gate_ffn = None

    with Graph(
        "vision_encoder_layer",
        input_types=[
            input_type,
            mlp_fc1_type,
            mlp_fc2_type,
            encoder_layernorm_w1_type,
            encoder_layernorm_w2_type,
        ],
    ) as graph:
        x, mlp_fc1, mlp_fc2, encoder_layernorm_w1, encoder_layernorm_w2 = (
            graph.inputs
        )

        vision_encoder_layer = VisionEncoderLayer(
            mlp=MLP(Linear(mlp_fc1), Linear(mlp_fc2)),
            input_layernorm=LPLayerNorm(encoder_layernorm_w1, eps),
            post_attention_layernorm=LPLayerNorm(encoder_layernorm_w2, eps),
            is_gated=is_gated,
            gate_attn=gate_attn,
            gate_ffn=gate_ffn,
        )
        graph.output(vision_encoder_layer(x)[0])

        @modular_graph_test(session, graph, max_magnitude=1 / 128)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            x, mlp_fc1, mlp_fc2, encoder_layernorm_w1, encoder_layernorm_w2 = (
                torch_inputs
            )

            # Transpose weights to match our Linear semantics.
            expected = (
                MllamaVisionEncoderLayer(
                    eps=eps,
                    mlp_fc1=mlp_fc1,
                    mlp_fc2=mlp_fc2,
                    encoder_layernorm_w1=encoder_layernorm_w1,
                    encoder_layernorm_w2=encoder_layernorm_w2,
                )(x)[0]
                .detach()
                .numpy()
            )
            # Relative L2 norm threshold
            threshold = 1e-4
            assert (
                np.linalg.norm(result - expected)
                / (np.linalg.norm(expected) + np.finfo(np.float32).eps)
                < threshold
            )


@pytest.mark.skip(reason="AIPIPE-176")
@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"]),
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["x", "y", "z", "dim"]),
    ],
)
def test_vision_encoder(
    session: InferenceSession, input_type: TensorType
) -> None:
    dim = input_type.shape[-1]
    mlp_fc1_type = TensorType(input_type.dtype, ["hidden_dim", dim])
    mlp_fc2_type = TensorType(input_type.dtype, [dim, "hidden_dim"])
    encoder_layernorm_w1_type = TensorType(input_type.dtype, [dim])
    encoder_layernorm_w2_type = encoder_layernorm_w1_type

    # TODO: Set these as is for now - can look to parameterizing them later on.
    eps = 1e-5
    num_layers = 32
    is_gated = False
    gate_attn = None
    gate_ffn = None

    with Graph(
        "vision_encoder",
        input_types=[
            input_type,
            mlp_fc1_type,
            mlp_fc2_type,
            encoder_layernorm_w1_type,
            encoder_layernorm_w2_type,
        ],
    ) as graph:
        x, mlp_fc1, mlp_fc2, encoder_layernorm_w1, encoder_layernorm_w2 = (
            graph.inputs
        )

        layers = [
            VisionEncoderLayer(
                mlp=MLP(Linear(mlp_fc1), Linear(mlp_fc2)),
                input_layernorm=LPLayerNorm(encoder_layernorm_w1, eps),
                post_attention_layernorm=LPLayerNorm(encoder_layernorm_w2, eps),
                is_gated=is_gated,
                gate_attn=gate_attn,
                gate_ffn=gate_ffn,
            )
            for _ in range(num_layers)
        ]
        vision_encoder = VisionEncoder(layers)
        graph.output(vision_encoder(x))

        @modular_graph_test(session, graph, max_magnitude=1 / 256)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            x, mlp_fc1, mlp_fc2, encoder_layernorm_w1, encoder_layernorm_w2 = (
                torch_inputs
            )

            # Transpose weights to match our Linear semantics.
            expected = (
                MllamaVisionEncoder(
                    eps=eps,
                    mlp_fc1=mlp_fc1,
                    mlp_fc2=mlp_fc2,
                    encoder_layernorm_w1=encoder_layernorm_w1,
                    encoder_layernorm_w2=encoder_layernorm_w2,
                )(x)
                .detach()
                .numpy()
            )

            # Relative L2 norm threshold
            threshold = 20
            assert (
                np.linalg.norm(result - expected)
                / (np.linalg.norm(expected) + np.finfo(np.float32).eps)
                < threshold
            )
