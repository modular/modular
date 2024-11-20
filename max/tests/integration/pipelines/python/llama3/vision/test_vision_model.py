# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision model tests by comparing it against the transformers
package reference implementation.
"""

import numpy as np
import pytest
from llama3.vision.encoder import VisionEncoder, VisionEncoderLayer
from llama3.vision.hyperparameters import VisionHyperparameters
from llama3.vision.mlp import MLP
from llama3.vision.vision_model import VisionModel
from llama3.vision.positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from modular_graph_test import modular_graph_test
from nn import Conv2D, Linear, LPLayerNorm


def generate_test_vision_model() -> VisionModel:
    """
    This helper function generates a test vision model instance for testing purposes.
    """
    params = VisionHyperparameters()

    # TODO(AIPIPE-131): Implement this.
    patch_embedding = None
    # TODO: Check if permutation is needed here.
    # patch_embedding = Conv2D(
    #     # in_channels=params.num_channels,
    #     # out_channels=params.hidden_size,
    #     # kernel_size=params.patch_size,
    #     filter=patch_embedding_weight,
    #     stride=(params.patch_size, params.patch_size),
    #     padding=(0, 0, 0, 0),
    #     bias=False,
    # )

    # TODO(AIPIPE-136): Implement this.
    class_embedding = None
    # class_embedding = nn.Parameter(
    #     params.scale * torch.randn(params.hidden_size)
    # )

    # TODO: Reenable this.
    gated_positional_embedding = None
    # gated_positional_embedding = PrecomputedPositionEmbedding(
    #     params=params,
    #     gate=gate,
    #     embedding=embedding,
    #     tile_embedding=tile_embedding,
    # )

    # TODO: Reenable this.
    pre_tile_positional_embedding = None
    # pre_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
    #     params, is_gated=True
    # )
    post_tile_positional_embedding = None
    # post_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
    #     params, is_gated=True
    # )

    # The reference implementation does not specify an eps (so falls back to
    # default). We do this anyway by specifying eps=1e-5.
    layernorm_pre = None
    layernorm_post = None
    # layernorm_pre = LPLayerNorm(
    #     vision_model_pre_weight, eps=params.norm_eps
    # )
    # layernorm_post = LPLayerNorm(
    #     vision_model_post_weight, eps=params.norm_eps
    # )

    # encoders
    transformer = None
    # transformer = VisionEncoder(
    #     [
    #         VisionEncoderLayer(
    #             mlp=MLP(Linear(mlp_fc1), Linear(mlp_fc2)),
    #             input_layernorm=LPLayerNorm(encoder_layernorm_w1, eps),
    #             post_attention_layernorm=LPLayerNorm(
    #                 encoder_layernorm_w2, eps
    #             ),
    #             is_gated=False,
    #             gate_attn=None,
    #             gate_ffn=None,
    #         )
    #         for _ in range(num_hidden_layers)
    #     ]
    # )

    global_transformer = None
    # global_transformer = VisionEncoder(
    #     [
    #         VisionEncoderLayer(
    #             mlp=MLP(Linear(mlp_fc1), Linear(mlp_fc2)),
    #             input_layernorm=LPLayerNorm(encoder_layernorm_w1, eps),
    #             post_attention_layernorm=LPLayerNorm(
    #                 encoder_layernorm_w2, eps
    #             ),
    #             is_gated=True,
    #             # TODO(AIPIPE-137): Implement this.
    #             gate_attn=gate_attn,
    #             # TODO(AIPIPE-137): Implement this.
    #             gate_ffn=gate_ffn,
    #         )
    #         for _ in range(num_global_layers)
    #     ]
    # )

    # self.post_init()  # TODO: Needed?
    return VisionModel(
        params=params,
        patch_embedding=patch_embedding,  # type: ignore
        class_embedding=class_embedding,  # type: ignore
        gated_positional_embedding=gated_positional_embedding,  # type: ignore
        pre_tile_positional_embedding=pre_tile_positional_embedding,  # type: ignore
        post_tile_positional_embedding=post_tile_positional_embedding,  # type: ignore
        layernorm_pre=layernorm_pre,  # type: ignore
        layernorm_post=layernorm_post,  # type: ignore
        transformer=transformer,  # type: ignore
        global_transformer=global_transformer,  # type: ignore
    )


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"]),
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["x", "y", "z", "dim"]),
    ],
)
def test_vision_model(
    session: InferenceSession, input_type: TensorType
) -> None:
    dim = input_type.shape[-1]
    mlp_fc1_type = TensorType(input_type.dtype, ["hidden_dim", dim])
    mlp_fc2_type = TensorType(input_type.dtype, [dim, "hidden_dim"])
    encoder_layernorm_w1_type = TensorType(input_type.dtype, [dim])
    encoder_layernorm_w2_type = encoder_layernorm_w1_type

    # TODO: Complete implementation here!
