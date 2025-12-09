# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Unit tests for InternVL multimodal embedding merging graph construction."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)


def test_merge_multimodal_embeddings_graph_construction() -> None:
    """Test that the merge_multimodal_embeddings function can be used in a graph."""

    seq_len = 10
    hidden_size = 768
    num_image_tokens = 4
    device = DeviceRef.GPU()

    graph = Graph(
        "test_merge",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(num_image_tokens, hidden_size),
                device=device,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(num_image_tokens,),
                device=device,
            ),
        ],
    )
    # Test passes if graph construction succeeds.
