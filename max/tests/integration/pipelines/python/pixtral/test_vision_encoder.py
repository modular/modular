# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight
from nn import Conv2D, Linear
from nn.norm import RMSNorm
from pixtral.vision_encoder.attention import Attention
from pixtral.vision_encoder.rotary_embedding_2d import RotaryEmbedding2D
from pixtral.vision_encoder.transformer import (
    MLP,
    Transformer,
    TransformerBlock,
)
from pixtral.vision_encoder.vision_encoder import VisionEncoder
from torch import nn
from transformers import PixtralVisionConfig, PixtralVisionModel

ACCURACY_RTOL = 1e-1
ACCURACY_ATOL = 1e-1
num_channels = 3  # Number of input channels in the input images.
hidden_size = 1024  # Dimension of the hidden representations.
intermediate_size = 4096  # Dimension of the MLP representations.
patch_size = 16  # Size of the image patches.
rope_theta = 10000.0  # The base period of the RoPE embeddings.
num_attention_heads = (
    16  #  Number of attention heads in the Transformer encoder.
)
head_dim = (
    hidden_size // num_attention_heads
)  # dim of positional embeddings and attention heads
scale = head_dim**-0.5
num_hidden_layers = 24  # Number of hidden layers in the Transformer encoder.
hidden_act = "gelu"  # Activation function used in the hidden layers.
image_size = 1024  # Max dimension of the input images. Should be 1024
attention_dropout = 0.0  # Dropout probability for the attention layers.
variance_epsilon = 1e-5


@pytest.fixture
def img_sizes():
    return [(128, 64), (128, 256)]


@pytest.fixture
def img_dtype():
    return torch.float32


@pytest.fixture
def imgs(img_sizes, img_dtype):
    # generate imgs of shape (num_channels, height, width)
    return [
        torch.randint(low=0, high=255, size=(num_channels, height, width)).to(
            img_dtype
        )
        for height, width in img_sizes
    ]


@pytest.fixture
def pytorch_pixtral_vision_encoder():
    # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/pixtral/modeling_pixtral.py#L465
    config = PixtralVisionConfig()
    model = PixtralVisionModel(config)
    return model


@pytest.fixture
def vision_encoder(pytorch_pixtral_vision_encoder):
    ########################### Weights ####################################
    pytorch_model = pytorch_pixtral_vision_encoder

    # Collect all the weights into the weights registry.
    weights_registry: dict = {}

    def linear(name: str, array) -> Linear:
        """Creates a Linear layer backed by a weight."""
        weights_registry[name] = array
        return Linear(
            Weight(
                name=name,
                dtype=DType.from_numpy(array.numpy().dtype),
                shape=array.shape,
            )
        )

    rms_norm_weight = np.ones(hidden_size)
    filters = pytorch_model.patch_conv.weight.data
    filters = torch.permute(filters, (2, 3, 1, 0))

    mlp_gate_weights = [
        pytorch_model.transformer.layers[i].feed_forward.gate_proj.weight.data
        for i in range(num_hidden_layers)
    ]
    mlp_up_weights = [
        pytorch_model.transformer.layers[i].feed_forward.up_proj.weight.data
        for i in range(num_hidden_layers)
    ]
    mlp_down_weights = [
        pytorch_model.transformer.layers[i].feed_forward.down_proj.weight.data
        for i in range(num_hidden_layers)
    ]

    attention_k_proj_weights = [
        pytorch_model.transformer.layers[i].attention.k_proj.weight.data
        for i in range(num_hidden_layers)
    ]
    attention_v_proj_weights = [
        pytorch_model.transformer.layers[i].attention.v_proj.weight.data
        for i in range(num_hidden_layers)
    ]
    attention_q_proj_weights = [
        pytorch_model.transformer.layers[i].attention.q_proj.weight.data
        for i in range(num_hidden_layers)
    ]
    attention_o_proj_weights = [
        pytorch_model.transformer.layers[i].attention.o_proj.weight.data
        for i in range(num_hidden_layers)
    ]

    ###################### Graph-API VisionEncoder #########################

    graph_patch_conv = Conv2D(filters.numpy(), stride=(patch_size, patch_size))
    graph_ln_pre = RMSNorm(weight=rms_norm_weight, eps=1e-5)
    graph_rope = RotaryEmbedding2D(
        dim=hidden_size,
        n_heads=num_attention_heads,
        theta=rope_theta,
        max_patches_per_side=image_size // patch_size,
    )
    attention_layers = []
    for i in range(num_hidden_layers):
        gate_proj = linear(
            name=f"mlp_gate_weights_{i}", array=mlp_gate_weights[i]
        )
        down_proj = linear(
            name=f"mlp_down_weights_{i}", array=mlp_down_weights[i]
        )
        up_proj = linear(name=f"mlp_up_weights_{i}", array=mlp_up_weights[i])
        mlp = MLP(gate_proj, down_proj, up_proj)
        # TODO: init weights

        wq = linear(
            name=f"attention_q_proj_weights_{i}",
            array=attention_q_proj_weights[i],
        )
        wk = linear(
            name=f"attention_k_proj_weights_{i}",
            array=attention_k_proj_weights[i],
        )
        wv = linear(
            name=f"attention_v_proj_weights_{i}",
            array=attention_v_proj_weights[i],
        )
        wo = linear(
            name=f"attention_o_proj_weights_{i}",
            array=attention_o_proj_weights[i],
        )

        attention = Attention(
            n_heads=num_attention_heads,
            dim=hidden_size,
            head_dim=hidden_size // num_attention_heads,
            dropout=attention_dropout,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
        )
        attention_norm = RMSNorm(weight=np.ones(hidden_size), eps=1e-5)
        mlp_norm = RMSNorm(weight=np.ones(hidden_size), eps=1e-5)
        attention_layers.append(
            TransformerBlock(attention, mlp, attention_norm, mlp_norm)
        )

    graph_transformer = Transformer(num_attention_heads, attention_layers)

    graph_encoder = VisionEncoder(
        patch_conv=graph_patch_conv,
        layer_norm=graph_ln_pre,
        patch_positional_embedding=graph_rope,
        transformer=graph_transformer,
        patch_size=patch_size,
        max_image_size=image_size,
    )
    return graph_encoder, weights_registry


# TODO(KERN-1066): Fix and enable test
def test_patch_conv(imgs, img_sizes):
    # TODO: Check the values of pixels are expected to be in [0, 255]
    patch_conv = nn.Conv2d(
        in_channels=num_channels,
        out_channels=hidden_size,
        kernel_size=patch_size,
        stride=patch_size,
        bias=False,
    )

    filters = patch_conv.weight.data
    imgs = [img.unsqueeze(0) for img in imgs]

    with torch.no_grad():
        patch_embeds_list = [patch_conv(img) for img in imgs]

    patch_embeds_list = [
        torch.permute(img, (0, 2, 3, 1)) for img in patch_embeds_list
    ]
    filters = torch.permute(filters, (2, 3, 1, 0))

    graph_api_imgs = [torch.permute(img, (0, 2, 3, 1)) for img in imgs]

    session = InferenceSession()
    graph = Graph(
        "conv",
        Conv2D(filters.numpy(), stride=(patch_size, patch_size)),
        input_types=(
            TensorType(
                DType.float32, (1, "img_height", "img_width", num_channels)
            ),
        ),
    )

    compiled = session.load(graph)

    output = [
        compiled.execute(np.ascontiguousarray(img))[0].to_numpy()  # type: ignore
        for img in graph_api_imgs
    ]

    np.testing.assert_allclose(
        output[1],
        patch_embeds_list[1].detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def test_vision_encoder(
    imgs, img_sizes, pytorch_pixtral_vision_encoder, vision_encoder
):
    # TODO: Check the values of pixels are expected to be in [0, 255]
    pytorch_model = pytorch_pixtral_vision_encoder
    graph_encoder, weights_registry = vision_encoder

    with torch.no_grad():
        encoder_output = pytorch_model(imgs).last_hidden_state
        ####### Permute torch inputs for the graph API and init weights ########
        imgs = [
            np.ascontiguousarray(torch.permute(img, (1, 2, 0))) for img in imgs
        ]

        ########### Graph API Pixtral Layers #########
        session = InferenceSession()
        with Graph(
            "conv",
            input_types=[
                TensorType(DType.float32, (h, w, num_channels))
                for h, w in img_sizes
            ],
        ) as graph:
            graph_inputs = graph.inputs
            graph_encoder_output = graph_encoder(graph_inputs)

            graph.output(graph_encoder_output)
            compiled = session.load(graph, weights_registry=weights_registry)

            output = compiled.execute(*imgs)[0].to_numpy()  # type: ignore
            print("Vision Encoder Output shape = ", output.shape)

            np.testing.assert_allclose(
                output,
                encoder_output.detach().numpy(),
                equal_nan=True,
                rtol=ACCURACY_RTOL,
                atol=ACCURACY_ATOL,
            )
