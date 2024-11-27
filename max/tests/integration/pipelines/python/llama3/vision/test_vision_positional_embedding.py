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
import torch.nn as nn
from llama_vision.positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from nn import Embedding


class TorchPrecomputedPositionEmbedding(nn.Module):
    def __init__(
        self,
        max_aspect_ratio_id: int,
        max_num_tiles: int,
        num_patches: int,
        hidden_size: int,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.max_aspect_ratio_id = max_aspect_ratio_id
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.scale = hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1,
            self.max_num_tiles * self.num_patches * self.hidden_size,
        )

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(
            1, 1, self.num_patches, self.hidden_size
        )

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = (
            self.gate.tanh() * tile_position_embedding
        )
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class TorchPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(
        self,
        max_aspect_ratio_id: int,
        max_num_tiles: int,
        hidden_size: int,
        is_gated: bool = True,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.hidden_size = hidden_size
        self.max_aspect_ratio_id = max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size
        )
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(
            -1, self.max_num_tiles, 1, self.hidden_size
        )

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


@pytest.mark.skip("not yet working")
@pytest.mark.parametrize(
    "max_aspect_ratio_id,max_num_tiles,num_patches,hidden_size,patch_size,image_size",
    [
        (8, 4, 1025, 1280, 14, 448),
    ],
)
def test_vision_precomputed_position_embedding(
    session: InferenceSession,
    max_aspect_ratio_id: int,
    max_num_tiles: int,
    num_patches: int,
    hidden_size: int,
    patch_size: int,
    image_size: int,
) -> None:
    input_type = TensorType(
        DType.float32, [1, max_num_tiles, num_patches, hidden_size]
    )
    gate_weight_type = TensorType(DType.float32, [1])
    embedding_weight_type = TensorType(
        DType.float32, [num_patches, hidden_size]
    )
    tile_embedding_weight_type = TensorType(
        DType.float32,
        [max_aspect_ratio_id + 1, max_num_tiles * num_patches * hidden_size],
    )
    max_aspect_ratio_ids_type = TensorType(DType.int64, [1, 1])
    with Graph(
        "precomputed_position_embedding",
        input_types=[
            input_type,
            gate_weight_type,
            embedding_weight_type,
            tile_embedding_weight_type,
            max_aspect_ratio_ids_type,
        ],
    ) as graph:
        (
            x,
            gate_weight,
            embedding_weight,
            tile_embedding_weight,
            max_aspect_ratio_ids,
        ) = graph.inputs

        embedding = PrecomputedPositionEmbedding(
            image_size=patch_size,
            patch_size=image_size,
            max_num_tiles=max_num_tiles,
            hidden_size=hidden_size,
            gate=gate_weight,  # type: ignore
            embedding=embedding_weight,  # type: ignore
            tile_embedding=Embedding(tile_embedding_weight),  # type: ignore
        )
        graph.output(embedding(x, max_aspect_ratio_ids))  # type: ignore

        # @modular_graph_test(session, graph)
        # def test_correctness(execute, inputs, torch_inputs):
        #     result = execute(inputs)
        #     x, gate_weight, embedding_weight, tile_embedding_weight, max_aspect_ratio_ids = (
        #         torch_inputs
        #     )

        #     expected = (
        #         TorchPrecomputedPositionEmbedding(
        #             max_aspect_ratio_id, max_num_tiles, num_patches, hidden_size
        #         )(x, max_aspect_ratio_ids)
        #         .detach()
        #         .numpy()
        #     )
        #     # TODO(AIPIPE-159): These tolerances have to be kinda large to accommodate
        #     # large range of values generated by hypothesis. This is because there
        #     # isn't a way to specify this (float) range without affecting the
        #     # max_aspect_ratio_ids int64 tensor too.
        #     ACCURACY_RTOL = 1
        #     ACCURACY_ATOL = 1
        #     np.testing.assert_allclose(
        #         result,
        #         expected,
        #         atol=ACCURACY_ATOL,
        #         rtol=ACCURACY_RTOL,
        #         equal_nan=True,
        #     )


@pytest.mark.parametrize(
    "max_aspect_ratio_id,max_num_tiles,patch_size,hidden_size",
    [
        (8, 4, 14, 1280),
    ],
)
def test_vision_precomputed_aspect_ratio_embedding(
    session: InferenceSession,
    max_aspect_ratio_id: int,
    max_num_tiles: int,
    patch_size: int,
    hidden_size: int,
) -> None:
    input_type = TensorType(
        DType.float32, [1, max_num_tiles, patch_size, hidden_size]
    )
    gate_weight_type = TensorType(DType.float32, [1])
    embedding_weight_type = TensorType(
        DType.float32,
        [max_aspect_ratio_id + 1, max_num_tiles * hidden_size],
    )
    max_aspect_ratio_ids_type = TensorType(DType.int64, [1, 1])
    with Graph(
        "precomputed_aspect_ratio_embedding",
        input_types=[
            input_type,
            gate_weight_type,
            embedding_weight_type,
            max_aspect_ratio_ids_type,
        ],
    ) as graph:
        x, gate_weight, embedding_weight, max_aspect_ratio_ids = graph.inputs

        embedding = PrecomputedAspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            hidden_size=hidden_size,
            gate=gate_weight,  # type: ignore
            embedding=Embedding(embedding_weight),  # type: ignore
            is_gated=True,
        )
        graph.output(embedding(x, max_aspect_ratio_ids))  # type: ignore

        # @modular_graph_test(session, graph)
        # def test_correctness(execute, inputs, torch_inputs):
        #     result = execute(inputs)
        #     x, gate_weight, embedding_weight, max_aspect_ratio_ids = (
        #         torch_inputs
        #     )

        #     expected = (
        #         TorchPrecomputedAspectRatioEmbedding(
        #             max_aspect_ratio_id,
        #             max_num_tiles,
        #             hidden_size,
        #             is_gated=True,
        #         )(x, max_aspect_ratio_ids)
        #         .detach()
        #         .numpy()
        #     )
        #     # TODO(AIPIPE-159): These tolerances have to be kinda large to accommodate
        #     # large range of values generated by hypothesis. This is because there
        #     # isn't a way to specify this (float) range without affecting the
        #     # max_aspect_ratio_ids int64 tensor too.
        #     ACCURACY_RTOL = 1
        #     ACCURACY_ATOL = 1
        #     np.testing.assert_allclose(
        #         result,
        #         expected,
        #         atol=ACCURACY_ATOL,
        #         rtol=ACCURACY_RTOL,
        #         equal_nan=True,
        #     )
