# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Unit tests for InternVL multimodal embedding merging graph construction."""

from __future__ import annotations

from functools import partial

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)


def test_merge_multimodal_embeddings_graph_construction() -> None:
    """Test that the merge_multimodal_embeddings function can be used in a graph."""

    batch_size = 1
    seq_len = 10
    hidden_size = 768
    img_context_token_id = 100
    num_image_tokens = 4
    device = DeviceRef.GPU()

    graph = Graph(
        "test_merge",
        forward=partial(
            merge_multimodal_embeddings,
            image_context_token_id=img_context_token_id,
        ),
        input_types=[
            TensorType(
                dtype=DType.int32, shape=(batch_size, seq_len), device=device
            ),
            TensorType(
                dtype=DType.float32,
                shape=(batch_size, seq_len, hidden_size),
                device=device,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(batch_size, num_image_tokens, hidden_size),
                device=device,
            ),
        ],
    )
    # Test passes if graph construction succeeds.
