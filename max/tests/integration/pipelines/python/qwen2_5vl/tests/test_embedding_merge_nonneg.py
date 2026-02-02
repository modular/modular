# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Integration test for Qwen2.5VL multimodal embedding merging execution."""

from __future__ import annotations

import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, SymbolicDim, TensorType
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)


def merge_multimodal_embeddings_torch_reference(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: torch.Tensor,
    image_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation using pre-computed indices."""
    # Expect already flattened tensors.
    # inputs_embeds shape: [num_tokens, hidden_size]
    # multimodal_embeddings shape: [num_multimodal_tokens, hidden_size]
    # image_token_indices shape: [num_multimodal_tokens]

    # Verify count.
    if image_token_indices.shape[0] != multimodal_embeddings.shape[0]:
        raise ValueError(
            f"Index count mismatch: {image_token_indices.shape[0]} image token indices "
            f"but {multimodal_embeddings.shape[0]} multimodal embeddings provided"
        )

    # Replace tokens at specified indices with multimodal embeddings.
    result = inputs_embeds.clone()
    if image_token_indices.shape[0] > 0:
        mask = image_token_indices >= 0
        result[image_token_indices[mask]] = multimodal_embeddings[mask]

    return result


def test_embeddings_merge_with_negative_indices() -> None:
    """Test execution of embedding merge for a single image with flattened tensors."""
    torch.manual_seed(42)
    IMG = 100
    hidden_size = 32
    device = CPU()
    device_ref = DeviceRef.CPU()

    # Create test inputs with 2 images
    start_idx = 9  # aka cache_length
    input_ids = torch.tensor(
        #                    start_idx = 9 v
        #                                  |0  1    2    3    4  5
        [1, 2, 3, IMG, IMG, IMG, IMG, 4, 5, 6, IMG, IMG, IMG, 7, 8],
        dtype=torch.int32,
    )
    img0_size = 4
    img1_size = 3
    active_length = input_ids.shape[0] - start_idx
    # This is indices of both img0 and img1.
    num_image_tokens = torch.where(input_ids == IMG)[0].shape[0]

    # We want the image embeddings for the last 3 image tokens, skipping the first 4.
    image_token_indices = torch.tensor(
        [-1, -1, -1, -1, 1, 2, 3], dtype=torch.int32
    )

    text_embeds = torch.randn(active_length, hidden_size, dtype=torch.float32)
    vision_embeds = torch.randn(
        num_image_tokens, hidden_size, dtype=torch.float32
    )

    # Get reference output
    expected_output = merge_multimodal_embeddings_torch_reference(
        text_embeds, vision_embeds, image_token_indices
    )

    # Build the graph
    active_length_dim = SymbolicDim("active_length")
    hidden_size_dim = SymbolicDim("hidden_size")
    num_image_tokens_dim = SymbolicDim("num_image_tokens")
    graph = Graph(
        "test_merge_execution",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(active_length_dim, hidden_size_dim),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(num_image_tokens_dim, hidden_size_dim),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(num_image_tokens_dim,),
                device=device_ref,
            ),
        ],
    )

    # Create session and compile
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Convert inputs to MAX tensors
    text_embeds_tensor = Buffer.from_numpy(text_embeds.numpy()).to(device)
    vision_embeds_tensor = Buffer.from_numpy(vision_embeds.numpy()).to(device)
    image_token_indices_tensor = Buffer.from_numpy(
        image_token_indices.numpy()
    ).to(device)

    # Execute
    results = compiled.execute(
        text_embeds_tensor,
        vision_embeds_tensor,
        image_token_indices_tensor,
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Buffer)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Verify the output matches the reference implementation
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-5, atol=1e-5
    )

    # Specifically verify that vision embeddings replaced placeholders
    img1_embeds = vision_embeds[img0_size : img0_size + img1_size]
    assert torch.allclose(actual_output[1:4], img1_embeds)

    # And that other positions remained unchanged
    assert torch.allclose(actual_output[:1], text_embeds[:1])
    assert torch.allclose(actual_output[4:], text_embeds[4:])

    # Smoke test empty batch
    empty_results = compiled.execute(
        Buffer.zeros(shape=(0, 0), dtype=DType.float32),
        Buffer.zeros(shape=(0, 0), dtype=DType.float32),
        Buffer.zeros(shape=(0,), dtype=DType.int32),
    )
    assert empty_results[0].shape == (0, 0)
