# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision positional embedding tests by comparing it against the
transformers package reference implementation.
"""

import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue, Weight
from max.pipelines.llama_vision.positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)
from max.pipelines.nn import Embedding
from test_common.distance_metrics import is_euclidean_distance_close
from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaPrecomputedAspectRatioEmbedding,
    MllamaPrecomputedPositionEmbedding,
)


class PositionalEmbedding:
    """(Test) Model containing position embedding layers."""

    position_embedding: PrecomputedPositionEmbedding
    """Layer for computing positional embedding."""

    dtype: DType
    """DType of the model weights."""

    def __init__(
        self,
        config: MllamaVisionConfig,
        torch_pos_embed: MllamaPrecomputedPositionEmbedding,
        dtype: DType,
    ) -> None:
        """Inits position embedding layers using the torch model."""
        self.dtype = dtype

        # Use torch model weights to initialize MAX graph position embedding
        # shapes.
        self.position_embedding = PrecomputedPositionEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            max_num_tiles=config.max_num_tiles,
            hidden_size=config.hidden_size,
            gate=Weight(
                name="gate",
                dtype=self.dtype,
                shape=torch_pos_embed.gate.shape,
            ),
            embedding=Weight(
                name="embedding",
                dtype=self.dtype,
                shape=torch_pos_embed.embedding.shape,
            ),
            tile_embedding=Embedding(
                Weight(
                    name="tile_embedding",
                    dtype=self.dtype,
                    shape=torch_pos_embed.tile_embedding.weight.shape,
                )
            ),
        )

    def __call__(
        self,
        hidden_state: TensorValue,
        max_aspect_ratio_ids: TensorValue,
    ) -> TensorValue:
        return self.position_embedding(
            hidden_state,
            max_aspect_ratio_ids,
        )


class AspectRatioEmbedding:
    """(Test) Model containing aspect ratio embedding layers."""

    aspect_ratio_embedding: PrecomputedAspectRatioEmbedding
    """Layer for computing aspect ratio embedding."""

    dtype: DType
    """DType of the model weights."""

    def __init__(
        self,
        config: MllamaVisionConfig,
        is_gated: bool,
        torch_aspect_ratio_embed: MllamaPrecomputedAspectRatioEmbedding,
        dtype: DType,
    ) -> None:
        """Inits aspect ratio embedding layers using the torch model."""
        self.dtype = dtype

        # Use torch model weights to initialize MAX graph aspect ratio embedding
        # shapes.
        self.aspect_ratio_embedding = PrecomputedAspectRatioEmbedding(
            max_num_tiles=config.max_num_tiles,
            hidden_size=config.hidden_size,
            gate=Weight(
                name="gate",
                dtype=self.dtype,
                shape=torch_aspect_ratio_embed.gate.shape,
            ),
            embedding=Embedding(
                Weight(
                    name="embedding",
                    dtype=self.dtype,
                    shape=torch_aspect_ratio_embed.embedding.weight.shape,
                )
            ),
            is_gated=is_gated,
        )

    def __call__(
        self,
        hidden_state: TensorValue,
        max_aspect_ratio_ids: TensorValue,
    ) -> TensorValue:
        return self.aspect_ratio_embedding(
            hidden_state,
            max_aspect_ratio_ids,
        )


@pytest.mark.parametrize(
    "max_num_tiles,num_patches,hidden_size,patch_size,image_size",
    [
        (4, 1025, 1280, 14, 448),
    ],
)
def test_vision_precomputed_position_embedding(
    session: InferenceSession,
    max_num_tiles: int,
    num_patches: int,
    hidden_size: int,
    patch_size: int,
    image_size: int,
) -> None:
    # Globally disable saving activations for backprop.
    torch.set_grad_enabled(False)

    # Reduced set of vision configs for testing purposes.
    config = MllamaVisionConfig(
        hidden_size=hidden_size,
        image_size=image_size,
        max_num_tiles=max_num_tiles,
        model_type="mllama_vision_model",
        patch_size=patch_size,
    )

    # Set up PyTorch position embedding layer.
    torch_dtype = torch.float32
    torch_precomputed_pos_embed = MllamaPrecomputedPositionEmbedding(
        config=config
    )
    torch_precomputed_pos_embed.to(torch_dtype)

    # Set up MAX graph position embedding layer.
    dtype = DType.float32

    hidden_state_type = TensorType(
        dtype, [1, max_num_tiles, num_patches, hidden_size]
    )
    aspect_ratio_ids_type = TensorType(DType.int64, [1, 1])

    # Phase 1: op staging.
    graph = Graph(
        "test_precomputed_position_embedding",
        forward=PositionalEmbedding(config, torch_precomputed_pos_embed, dtype),
        input_types=[
            hidden_state_type,
            aspect_ratio_ids_type,
        ],
    )

    # Phase 2: model compilation and weight initialization.

    # Map torch weight values to their MAX graph counterparts.
    weights_registry = {
        "gate": torch_precomputed_pos_embed.gate.detach(),
        "embedding": torch_precomputed_pos_embed.embedding.detach(),
        "tile_embedding": torch_precomputed_pos_embed.tile_embedding.weight.detach(),
    }
    position_embed_model = session.load(
        graph, weights_registry=weights_registry
    )

    # Phase 3: execution.

    # Initialize model inputs.
    hidden_state = torch.randn(
        hidden_state_type.shape.static_dims, dtype=torch_dtype
    )

    # This needs to be within the range of [0, num_embeddings - 1].
    aspect_ratio_ids = torch.randint(
        0, 9, aspect_ratio_ids_type.shape.static_dims, dtype=torch.long
    )

    predicted = position_embed_model(
        hidden_state,
        aspect_ratio_ids,
    )[0]
    assert isinstance(predicted, Tensor)

    expected = (
        torch_precomputed_pos_embed(
            hidden_state=hidden_state,
            aspect_ratio_ids=aspect_ratio_ids,
        )[0]
        .detach()
        .numpy()
    )

    # Compare the outputs.
    assert is_euclidean_distance_close(
        result=predicted.to_numpy(), expected=expected, rtol=1e-4
    )


@pytest.mark.parametrize(
    "max_num_tiles,patch_size,hidden_size",
    [
        (4, 14, 1280),
    ],
)
def test_vision_precomputed_aspect_ratio_embedding(
    session: InferenceSession,
    max_num_tiles: int,
    patch_size: int,
    hidden_size: int,
) -> None:
    # Globally disable saving activations for backprop.
    torch.set_grad_enabled(False)

    # Reduced set of vision configs for testing purposes.
    config = MllamaVisionConfig(
        hidden_size=hidden_size,
        max_num_tiles=max_num_tiles,
        model_type="mllama_vision_model",
        patch_size=patch_size,
    )

    # Set up PyTorch position embedding layer.
    torch_dtype = torch.float32
    torch_aspect_ratio_embed = MllamaPrecomputedAspectRatioEmbedding(
        config=config,
        is_gated=True,
    )
    torch_aspect_ratio_embed.to(torch_dtype)

    # Set up MAX graph position embedding layer.
    dtype = DType.float32

    hidden_state_type = TensorType(
        dtype, [1, max_num_tiles, patch_size, hidden_size]
    )
    aspect_ratio_ids_type = TensorType(DType.int64, [1, 1])

    # Phase 1: op staging.
    graph = Graph(
        "test_precomputed_aspect_ratio_embedding",
        forward=AspectRatioEmbedding(
            config, True, torch_aspect_ratio_embed, dtype
        ),
        input_types=[
            hidden_state_type,
            aspect_ratio_ids_type,
        ],
    )

    # Phase 2: model compilation and weight initialization.

    # Map torch weight values to their MAX graph counterparts.
    weights_registry = {
        "gate": torch_aspect_ratio_embed.gate.detach(),
        "embedding": torch_aspect_ratio_embed.embedding.weight.detach(),
    }
    position_embed_model = session.load(
        graph, weights_registry=weights_registry
    )

    # Phase 3: execution.

    # Initialize model inputs.
    hidden_state = torch.randn(
        hidden_state_type.shape.static_dims, dtype=torch_dtype
    )

    # This needs to be within the range of [0, num_embeddings - 1].
    aspect_ratio_ids = torch.randint(
        0, 9, aspect_ratio_ids_type.shape.static_dims, dtype=torch.long
    )

    predicted = position_embed_model(
        hidden_state,
        aspect_ratio_ids,
    )[0]
    assert isinstance(predicted, Tensor)

    expected = (
        torch_aspect_ratio_embed(
            hidden_state=hidden_state,
            aspect_ratio_ids=aspect_ratio_ids,
        )[0]
        .detach()
        .numpy()
    )

    # Compare the outputs.
    assert is_euclidean_distance_close(
        result=predicted.to_numpy(), expected=expected, rtol=1e-4
    )
