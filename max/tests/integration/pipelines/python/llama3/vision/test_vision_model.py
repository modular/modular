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
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight
from max.nn import Embedding, LayerNorm, Linear
from max.pipelines.architectures.llama_vision.attention import Attention
from max.pipelines.architectures.llama_vision.encoder import (
    VisionEncoder,
    VisionEncoderLayer,
)
from max.pipelines.architectures.llama_vision.mlp import MLP
from max.pipelines.architectures.llama_vision.positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)
from max.pipelines.architectures.llama_vision.vision_model import (
    VisionConv2D,
    VisionModel,
)
from test_common.distance_metrics import is_euclidean_distance_close
from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
from transformers.models.mllama.modeling_mllama import MllamaVisionModel


class WrappedVisionModel:
    """(Test) Model containing vision model layers."""

    vision_model: VisionModel
    """Layer for computing vision model."""

    dtype: DType
    """DType of the model weights."""

    def __init__(
        self,
        config: MllamaVisionConfig,
        torch_vision_model: MllamaVisionModel,
        dtype: DType,
    ) -> None:
        """Inits position embedding layers using the torch model."""
        self.dtype = dtype

        # Use torch model weights to initialize MAX graph position embedding
        # shapes.
        self.vision_model = self.instantiate_vision_model(
            dtype=dtype,
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            max_num_tiles=config.max_num_tiles,
            norm_eps=config.norm_eps,
            attention_heads=config.attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            num_global_layers=config.num_global_layers,
            intermediate_layers_indices=config.intermediate_layers_indices,
            torch_vision_model=torch_vision_model,
        )

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
    ) -> TensorValue:
        return self.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )[0]

    def instantiate_vision_model(
        self,
        dtype: DType,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        max_num_tiles: int,
        norm_eps: float,
        attention_heads: int,
        num_hidden_layers: int,
        num_global_layers: int,
        intermediate_layers_indices: list[int],
        torch_vision_model: MllamaVisionModel,
    ) -> VisionModel:
        """
        Helper function to construct a VisionModel used exclusively for testing
        purposes here.
        """
        gated_positional_embedding = PrecomputedPositionEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            max_num_tiles=max_num_tiles,
            gate=Weight(
                name="gated_positional_embedding.gate",
                dtype=dtype,
                shape=torch_vision_model.gated_positional_embedding.gate.shape,
                device=DeviceRef.CPU(),
            ),
            embedding=Weight(
                name="gated_positional_embedding.embedding",
                dtype=dtype,
                shape=torch_vision_model.gated_positional_embedding.embedding.shape,
                device=DeviceRef.CPU(),
            ),
            tile_embedding=Embedding(
                Weight(
                    name="gated_positional_embedding.tile_embedding",
                    dtype=dtype,
                    shape=torch_vision_model.gated_positional_embedding.tile_embedding.weight.shape,
                    device=DeviceRef.CPU(),
                ),
                device=DeviceRef.CPU(),
            ),
        )

        pre_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            hidden_size=hidden_size,
            gate=Weight(
                name="pre_tile_positional_embedding.gate",
                dtype=dtype,
                shape=torch_vision_model.pre_tile_positional_embedding.gate.shape,
                device=DeviceRef.CPU(),
            ),
            embedding=Embedding(
                Weight(
                    name="pre_tile_positional_embedding.embedding",
                    dtype=dtype,
                    shape=torch_vision_model.pre_tile_positional_embedding.embedding.weight.shape,
                    device=DeviceRef.CPU(),
                ),
                device=DeviceRef.CPU(),
            ),
            is_gated=True,
        )

        post_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            hidden_size=hidden_size,
            gate=Weight(
                name="post_tile_positional_embedding.gate",
                dtype=dtype,
                shape=torch_vision_model.post_tile_positional_embedding.gate.shape,
                device=DeviceRef.CPU(),
            ),
            embedding=Embedding(
                Weight(
                    name="post_tile_positional_embedding.embedding",
                    dtype=dtype,
                    shape=torch_vision_model.post_tile_positional_embedding.embedding.weight.shape,
                    device=DeviceRef.CPU(),
                ),
                device=DeviceRef.CPU(),
            ),
            is_gated=True,
        )

        # patch_embedding filter has a shape of (1280, 3, 14, 14).
        patch_embedding = VisionConv2D(
            filter=Weight(
                name="patch_embedding",
                dtype=dtype,
                shape=torch_vision_model.patch_embedding.weight.shape,
                device=DeviceRef.CPU(),
            ),
            stride=patch_size,
            padding=(0, 0, 0, 0),
            bias=None,
        )

        class_embedding = Weight(
            name="class_embedding",
            dtype=dtype,
            shape=torch_vision_model.class_embedding.shape,
            device=DeviceRef.CPU(),
        )

        layernorm_pre = LayerNorm(
            Weight(
                name="layernorm_pre",
                dtype=dtype,
                shape=torch_vision_model.layernorm_pre.weight.shape,
                device=DeviceRef.CPU(),
            ),
            eps=norm_eps,
        )

        layernorm_post = LayerNorm(
            Weight(
                name="layernorm_post",
                dtype=dtype,
                shape=torch_vision_model.layernorm_post.weight.shape,
                device=DeviceRef.CPU(),
            ),
            eps=norm_eps,
        )

        transformer_encoder_layers: list[VisionEncoderLayer] = []

        head_dim = hidden_size // attention_heads

        for index in range(num_hidden_layers):
            curr_layer = torch_vision_model.transformer.layers[index]
            transformer_encoder_layers.append(
                VisionEncoderLayer(
                    mlp=MLP(
                        Linear(
                            Weight(
                                name=f"transformer.{index}.mlp.fc1",
                                dtype=dtype,
                                shape=curr_layer.mlp.fc1.weight.shape,
                                device=DeviceRef.CPU(),
                            ),
                            bias=None,
                        ),
                        Linear(
                            Weight(
                                name=f"transformer.{index}.mlp.fc2",
                                dtype=dtype,
                                shape=curr_layer.mlp.fc2.weight.shape,
                                device=DeviceRef.CPU(),
                            ),
                            bias=None,
                        ),
                    ),
                    input_layernorm=LayerNorm(
                        Weight(
                            name=f"transformer.{index}.input_layernorm",
                            dtype=dtype,
                            shape=curr_layer.input_layernorm.weight.shape,
                            device=DeviceRef.CPU(),
                        ),
                        eps=norm_eps,
                    ),
                    post_attention_layernorm=LayerNorm(
                        Weight(
                            name=f"transformer.{index}.post_attention_layernorm",
                            dtype=dtype,
                            shape=curr_layer.post_attention_layernorm.weight.shape,
                            device=DeviceRef.CPU(),
                        ),
                        eps=norm_eps,
                    ),
                    self_attn=Attention(
                        n_heads=attention_heads,
                        head_dim=head_dim,
                        wk=Linear(
                            Weight(
                                name=f"transformer.{index}.self_attn.k_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.k_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                        wv=Linear(
                            Weight(
                                name=f"transformer.{index}.self_attn.v_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.v_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                        wq=Linear(
                            Weight(
                                name=f"transformer.{index}.self_attn.q_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.q_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                        wo=Linear(
                            Weight(
                                name=f"transformer.{index}.self_attn.o_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.o_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                    ),
                    is_gated=False,
                    gate_attn=None,
                    gate_ffn=None,
                )
            )
        transformer = VisionEncoder(transformer_encoder_layers)

        global_transformer_layers: list[VisionEncoderLayer] = []

        for index in range(num_global_layers):
            curr_layer = torch_vision_model.global_transformer.layers[index]

            global_transformer_layers.append(
                VisionEncoderLayer(
                    mlp=MLP(
                        Linear(
                            Weight(
                                name=f"global_transformer.{index}.mlp.fc1",
                                dtype=dtype,
                                shape=curr_layer.mlp.fc1.weight.shape,
                                device=DeviceRef.CPU(),
                            ),
                            bias=None,
                        ),
                        Linear(
                            Weight(
                                name=f"global_transformer.{index}.mlp.fc2",
                                dtype=dtype,
                                shape=curr_layer.mlp.fc2.weight.shape,
                                device=DeviceRef.CPU(),
                            ),
                            bias=None,
                        ),
                    ),
                    input_layernorm=LayerNorm(
                        Weight(
                            name=f"global_transformer.{index}.input_layernorm",
                            dtype=dtype,
                            shape=curr_layer.input_layernorm.weight.shape,
                            device=DeviceRef.CPU(),
                        ),
                        eps=norm_eps,
                    ),
                    post_attention_layernorm=LayerNorm(
                        Weight(
                            name=f"global_transformer.{index}.post_attention_layernorm",
                            dtype=dtype,
                            shape=curr_layer.post_attention_layernorm.weight.shape,
                            device=DeviceRef.CPU(),
                        ),
                        eps=norm_eps,
                    ),
                    self_attn=Attention(
                        n_heads=attention_heads,
                        head_dim=head_dim,
                        wk=Linear(
                            Weight(
                                name=f"global_transformer.{index}.self_attn.k_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.k_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                        wv=Linear(
                            Weight(
                                name=f"global_transformer.{index}.self_attn.v_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.v_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                        wq=Linear(
                            Weight(
                                name=f"global_transformer.{index}.self_attn.q_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.q_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                        wo=Linear(
                            Weight(
                                name=f"global_transformer.{index}.self_attn.o_proj",
                                dtype=dtype,
                                shape=curr_layer.self_attn.o_proj.weight.shape,
                                device=DeviceRef.CPU(),
                            )
                        ),
                    ),
                    is_gated=True,
                    gate_attn=Weight(
                        name=f"global_transformer.{index}.gate_attn",
                        dtype=dtype,
                        shape=curr_layer.gate_attn.shape,
                        device=DeviceRef.CPU(),
                    ),
                    gate_ffn=Weight(
                        name=f"global_transformer.{index}.gate_ffn",
                        dtype=dtype,
                        shape=curr_layer.gate_ffn.shape,
                        device=DeviceRef.CPU(),
                    ),
                )
            )
        global_transformer = VisionEncoder(global_transformer_layers)

        return VisionModel(
            gated_positional_embedding=gated_positional_embedding,
            pre_tile_positional_embedding=pre_tile_positional_embedding,
            post_tile_positional_embedding=post_tile_positional_embedding,
            patch_embedding=patch_embedding,
            class_embedding=class_embedding,
            layernorm_pre=layernorm_pre,
            layernorm_post=layernorm_post,
            transformer=transformer,
            global_transformer=global_transformer,
            dtype=dtype,
            intermediate_layers_indices=intermediate_layers_indices,
            num_patches=(image_size // patch_size) ** 2 + 1,
        )


def _construct_transformer_weights_registry(
    prefix: str, layers: torch.nn.ModuleList, is_gated=False
) -> dict[str, torch.Tensor]:
    curr_weights_registry: dict[str, torch.Tensor] = {}

    for index, layer in enumerate(layers):
        curr_weights_registry[f"{prefix}.{index}.mlp.fc1"] = (
            layer.mlp.fc1.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.mlp.fc2"] = (
            layer.mlp.fc2.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.input_layernorm"] = (
            layer.input_layernorm.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.post_attention_layernorm"] = (
            layer.post_attention_layernorm.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.self_attn.k_proj"] = (
            layer.self_attn.k_proj.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.self_attn.v_proj"] = (
            layer.self_attn.v_proj.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.self_attn.q_proj"] = (
            layer.self_attn.q_proj.weight.detach()
        )
        curr_weights_registry[f"{prefix}.{index}.self_attn.o_proj"] = (
            layer.self_attn.o_proj.weight.detach()
        )

        if is_gated:
            curr_weights_registry[f"{prefix}.{index}.gate_attn"] = (
                layer.gate_attn.detach()
            )
            curr_weights_registry[f"{prefix}.{index}.gate_ffn"] = (
                layer.gate_ffn.detach()
            )

    return curr_weights_registry


@pytest.mark.parametrize(
    "hidden_size,max_num_tiles,patch_size,attention_heads,num_channels,image_size",
    [
        (1280, 4, 14, 16, 3, 448),
    ],
)
def test_vision_model(
    session: InferenceSession,
    hidden_size: int,
    max_num_tiles: int,
    patch_size: int,
    attention_heads: int,
    num_channels: int,
    image_size: int,
) -> None:
    # Globally disable saving activations for backprop.
    torch.set_grad_enabled(False)

    # Reduced set of vision configs for testing purposes.
    config = MllamaVisionConfig(
        hidden_size=hidden_size,
        max_num_tiles=max_num_tiles,
        model_type="mllama_vision_model",
        patch_size=patch_size,
        attention_heads=attention_heads,
        image_size=image_size,
        initializer_range=0.02,
        intermediate_layers_indices=[3, 7, 15, 23, 30],
        norm_eps=1e-05,
        num_global_layers=8,
        num_hidden_layers=32,
        vision_output_dim=7680,
    )

    # Set up PyTorch position embedding layer.
    torch_dtype = torch.float32
    torch_vision_model = MllamaVisionModel._from_config(
        config, attn_implementation="sdpa"
    )
    # torch_vision_model.to(torch_dtype)

    # Set up MAX graph position embedding layer.
    dtype = DType.float32

    pixel_values_type = TensorType(
        dtype,
        [
            "batch_size",
            "num_concurrent_media",
            max_num_tiles,
            image_size,
            image_size,
            num_channels,
        ],
        device=DeviceRef.CPU(),
    )
    aspect_ratio_ids_type = TensorType(
        DType.int64,
        ["batch_size", "num_concurrent_media"],
        device=DeviceRef.CPU(),
    )
    aspect_ratio_mask_type = TensorType(
        DType.int64,
        ["batch_size", "num_concurrent_media", max_num_tiles],
        device=DeviceRef.CPU(),
    )

    # Phase 1: op staging.
    graph = Graph(
        "test_vision_model",
        forward=WrappedVisionModel(
            config=config, torch_vision_model=torch_vision_model, dtype=dtype
        ),
        input_types=[
            pixel_values_type,
            aspect_ratio_ids_type,
            aspect_ratio_mask_type,
        ],
    )

    # Phase 2: model compilation and weight initialization.

    # Map torch weight values to their MAX graph counterparts.

    local_transformer_weights_registry = (
        _construct_transformer_weights_registry(
            prefix="transformer", layers=torch_vision_model.transformer.layers
        )
    )
    global_transformer_weights_registry = (
        _construct_transformer_weights_registry(
            prefix="global_transformer",
            layers=torch_vision_model.global_transformer.layers,
            is_gated=True,
        )
    )

    weights_registry: dict[str, torch.Tensor] = {
        "gated_positional_embedding.gate": torch_vision_model.gated_positional_embedding.gate.detach(),
        "gated_positional_embedding.embedding": torch_vision_model.gated_positional_embedding.embedding.detach(),
        "gated_positional_embedding.tile_embedding": torch_vision_model.gated_positional_embedding.tile_embedding.weight.detach(),
        "pre_tile_positional_embedding.gate": torch_vision_model.pre_tile_positional_embedding.gate.detach(),
        "pre_tile_positional_embedding.embedding": torch_vision_model.pre_tile_positional_embedding.embedding.weight.detach(),
        "post_tile_positional_embedding.gate": torch_vision_model.post_tile_positional_embedding.gate.detach(),
        "post_tile_positional_embedding.embedding": torch_vision_model.post_tile_positional_embedding.embedding.weight.detach(),
        "patch_embedding": torch_vision_model.patch_embedding.weight.detach(),
        "class_embedding": torch_vision_model.class_embedding.detach(),
        "layernorm_pre": torch_vision_model.layernorm_pre.weight.detach(),
        "layernorm_post": torch_vision_model.layernorm_post.weight.detach(),
    }
    weights_registry.update(local_transformer_weights_registry)
    weights_registry.update(global_transformer_weights_registry)

    vision_model = session.load(graph, weights_registry=weights_registry)

    batch_size = 1
    num_concurrent_media = 1
    pixel_values = torch.randn(
        [
            batch_size,
            num_concurrent_media,
            max_num_tiles,
            image_size,
            image_size,
            num_channels,
        ],
        dtype=torch_dtype,
    )

    # This needs to be within the range of [0, num_embeddings - 1].
    aspect_ratio_ids = torch.randint(
        0, 9, [batch_size, num_concurrent_media], dtype=torch.long
    )

    # This needs to be within the range of [0, 1].
    aspect_ratio_mask = torch.randint(
        0,
        1,
        [batch_size, num_concurrent_media, max_num_tiles],
        dtype=torch.long,
    )

    predicted = vision_model(
        pixel_values,
        aspect_ratio_ids,
        aspect_ratio_mask,
    )[0]
    assert isinstance(predicted, Tensor)

    pixel_values = pixel_values.permute((0, 1, 2, 5, 3, 4))
    cross_attention_states = torch_vision_model(
        pixel_values=pixel_values,
        aspect_ratio_ids=aspect_ratio_ids,
        aspect_ratio_mask=aspect_ratio_mask,
    )[0]
    expected = cross_attention_states.detach().numpy()

    np.testing.assert_array_equal(predicted.to_numpy().shape, expected.shape)

    assert is_euclidean_distance_close(
        predicted.to_numpy(), expected, rtol=0.01, atol=1e-4
    )
