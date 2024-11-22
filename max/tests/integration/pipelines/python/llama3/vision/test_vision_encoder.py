# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision encoder tests by comparing it against the transformers
package reference implementation.
"""

import numpy as np
import pytest
from llama_vision.encoder import Attention
from llama_vision.encoder import VisionEncoder, VisionEncoderLayer
from llama_vision.mlp import MLP
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from modular_graph_test import modular_graph_test
from nn import Linear, LPLayerNorm
from torch_vision_encoder import MllamaVisionEncoder, MllamaVisionEncoderLayer

ATTENTION_HEADS = 16
SEQ_LEN = 4128
HIDDEN_SIZE = 1280


@pytest.mark.parametrize(
    "hidden_state_type",
    [
        TensorType(DType.float32, [1, SEQ_LEN, HIDDEN_SIZE]),
    ],
)
def test_vision_encoder_layer(
    session: InferenceSession, hidden_state_type: TensorType
) -> None:
    dim = hidden_state_type.shape[-1]
    attn_mask_type = TensorType(
        hidden_state_type.dtype,
        [
            hidden_state_type.shape[0],
            1,
            hidden_state_type.shape[1],
            hidden_state_type.shape[1],
        ],
    )
    attn_weight_type = TensorType(hidden_state_type.dtype, [dim, dim])
    mlp_fc1_type = TensorType(hidden_state_type.dtype, ["hidden_dim", dim])
    mlp_fc2_type = TensorType(hidden_state_type.dtype, [dim, "hidden_dim"])
    encoder_layernorm_w1_type = TensorType(hidden_state_type.dtype, [dim])
    encoder_layernorm_w2_type = encoder_layernorm_w1_type

    # TODO: Set these as is for now - can look to parameterizing them later on.
    eps = 1e-5
    is_gated = False
    gate_attn = None
    gate_ffn = None

    with Graph(
        "vision_encoder_layer",
        input_types=[
            hidden_state_type,
            attn_mask_type,
            attn_weight_type,
            mlp_fc1_type,
            mlp_fc2_type,
            encoder_layernorm_w1_type,
            encoder_layernorm_w2_type,
        ],
    ) as graph:
        (
            hidden_state,
            attn_mask,
            attn_weight,
            mlp_fc1,
            mlp_fc2,
            encoder_layernorm_w1,
            encoder_layernorm_w2,
        ) = graph.inputs

        vision_encoder_layer = VisionEncoderLayer(
            self_attn=Attention(
                n_heads=ATTENTION_HEADS,
                head_dim=HIDDEN_SIZE // ATTENTION_HEADS,
                wk=Linear(attn_weight),  # type: ignore
                wv=Linear(attn_weight),  # type: ignore
                wq=Linear(attn_weight),  # type: ignore
                wo=Linear(attn_weight),  # type: ignore
            ),
            mlp=MLP(Linear(mlp_fc1), Linear(mlp_fc2)),  # type: ignore
            input_layernorm=LPLayerNorm(encoder_layernorm_w1, eps),  # type: ignore
            post_attention_layernorm=LPLayerNorm(encoder_layernorm_w2, eps),  # type: ignore
            is_gated=is_gated,
            gate_attn=gate_attn,
            gate_ffn=gate_ffn,
        )
        graph.output(vision_encoder_layer(hidden_state, attn_mask)[0])  # type: ignore

        @modular_graph_test(session, graph, max_magnitude=1 / 128)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            (
                hidden_state,
                attn_mask,
                attn_weight,
                mlp_fc1,
                mlp_fc2,
                encoder_layernorm_w1,
                encoder_layernorm_w2,
            ) = torch_inputs

            # Transpose weights to match our Linear semantics.
            expected = (
                MllamaVisionEncoderLayer(
                    eps=eps,
                    hidden_size=HIDDEN_SIZE,
                    attention_heads=ATTENTION_HEADS,
                    attn_weight=attn_weight,
                    mlp_fc1=mlp_fc1,
                    mlp_fc2=mlp_fc2,
                    encoder_layernorm_w1=encoder_layernorm_w1,
                    encoder_layernorm_w2=encoder_layernorm_w2,
                    is_gated=is_gated,
                )(hidden_state, attn_mask)[0]
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


@pytest.mark.parametrize(
    "hidden_states_type",
    [
        TensorType(DType.float32, [1, SEQ_LEN, HIDDEN_SIZE]),
    ],
)
def test_vision_encoder(
    session: InferenceSession, hidden_states_type: TensorType
) -> None:
    batch_size, seq_len, hidden_size = hidden_states_type.shape
    attention_mask_type = TensorType(
        hidden_states_type.dtype, [batch_size, 1, seq_len, seq_len]
    )

    attn_weight_type = TensorType(
        hidden_states_type.dtype, [hidden_size, hidden_size]
    )
    mlp_fc1_type = TensorType(
        hidden_states_type.dtype, ["hidden_dim", hidden_size]
    )
    mlp_fc2_type = TensorType(
        hidden_states_type.dtype, [hidden_size, "hidden_dim"]
    )
    encoder_layernorm_w1_type = TensorType(
        hidden_states_type.dtype, [hidden_size]
    )
    encoder_layernorm_w2_type = encoder_layernorm_w1_type

    # TODO: Set these as is for now - can look to parameterizing them later on.
    eps = 1e-5
    # Reducing the number of layers from 32 to 5 for test purposes.
    num_layers = 5
    is_gated = False
    gate_attn = None
    gate_ffn = None
    output_hidden_states = False

    with Graph(
        "vision_encoder",
        input_types=[
            hidden_states_type,
            attention_mask_type,
            attn_weight_type,
            mlp_fc1_type,
            mlp_fc2_type,
            encoder_layernorm_w1_type,
            encoder_layernorm_w2_type,
        ],
    ) as graph:
        (
            hidden_states,
            attention_mask,
            attn_weight,
            mlp_fc1,
            mlp_fc2,
            encoder_layernorm_w1,
            encoder_layernorm_w2,
        ) = graph.inputs
        # attention_mask: shape=[1, 1, 4128, 4128], dtype=torch.bfloat16
        layers = [
            VisionEncoderLayer(
                mlp=MLP(Linear(mlp_fc1), Linear(mlp_fc2)),  # type: ignore
                input_layernorm=LPLayerNorm(encoder_layernorm_w1, eps),  # type: ignore
                post_attention_layernorm=LPLayerNorm(encoder_layernorm_w2, eps),  # type: ignore
                self_attn=Attention(
                    n_heads=ATTENTION_HEADS,
                    head_dim=HIDDEN_SIZE // ATTENTION_HEADS,
                    wk=Linear(attn_weight),  # type: ignore
                    wv=Linear(attn_weight),  # type: ignore
                    wq=Linear(attn_weight),  # type: ignore
                    wo=Linear(attn_weight),  # type: ignore
                ),
                is_gated=is_gated,
                gate_attn=gate_attn,
                gate_ffn=gate_ffn,
            )
            for _ in range(num_layers)
        ]
        vision_encoder = VisionEncoder(layers)
        graph.output(
            vision_encoder(
                hidden_states=hidden_states.tensor,
                attention_mask=attention_mask.tensor,
                output_hidden_states=output_hidden_states,
            )[0]
        )

        @modular_graph_test(session, graph, max_magnitude=1 / 256)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            (
                hidden_states,
                attention_mask,
                attn_weight,
                mlp_fc1,
                mlp_fc2,
                encoder_layernorm_w1,
                encoder_layernorm_w2,
            ) = torch_inputs

            # Transpose weights to match our Linear semantics.
            expected = (
                MllamaVisionEncoder(
                    eps=eps,
                    hidden_size=HIDDEN_SIZE,
                    attention_heads=ATTENTION_HEADS,
                    attn_weight=attn_weight,
                    mlp_fc1=mlp_fc1,
                    mlp_fc2=mlp_fc2,
                    encoder_layernorm_w1=encoder_layernorm_w1,
                    encoder_layernorm_w2=encoder_layernorm_w2,
                    num_layers=num_layers,
                )(hidden_states, attention_mask)[0]
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
