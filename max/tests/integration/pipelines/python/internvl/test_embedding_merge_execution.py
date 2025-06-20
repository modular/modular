# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Integration test for InternVL multimodal embedding merging execution."""

from __future__ import annotations

import torch
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
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
    num_indices = image_token_indices.shape[0]
    num_embeddings = multimodal_embeddings.shape[0]
    if num_indices != num_embeddings:
        raise ValueError(
            f"Index count mismatch: {num_indices} indices "
            f"but {num_embeddings} multimodal embeddings provided"
        )

    # Replace tokens at specified indices with multimodal embeddings.
    result = inputs_embeds.clone()
    if num_indices > 0:
        result[image_token_indices] = multimodal_embeddings

    return result


def test_single_image_merge_execution() -> None:
    """Test execution of embedding merge for a single image with flattened tensors."""
    seq_len = 10
    hidden_size = 768
    img_context_token_id = 100
    num_image_tokens = 4
    device = CPU()
    device_ref = DeviceRef.CPU()

    # Create test inputs with image context tokens at positions 3-6 (flattened).
    input_ids = torch.tensor(
        [1, 2, 3, 100, 100, 100, 100, 4, 5, 6], dtype=torch.int32
    )
    # Pre-compute indices where image tokens should be inserted.
    image_token_indices = torch.where(input_ids == img_context_token_id)[0].to(
        torch.int32
    )

    text_embeds = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    vision_embeds = torch.randn(
        num_image_tokens, hidden_size, dtype=torch.float32
    )

    # Get reference output
    expected_output = merge_multimodal_embeddings_torch_reference(
        text_embeds, vision_embeds, image_token_indices
    )

    # Build the graph
    graph = Graph(
        "test_merge_execution",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(num_image_tokens, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(num_image_tokens,),
                device=device_ref,
            ),
        ],
    )

    # Create session and compile
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Convert inputs to MAX tensors
    text_embeds_tensor = Tensor.from_numpy(text_embeds.numpy()).to(device)
    vision_embeds_tensor = Tensor.from_numpy(vision_embeds.numpy()).to(device)
    indices_tensor = Tensor.from_numpy(image_token_indices.numpy()).to(device)

    # Execute
    results = compiled.execute(
        text_embeds_tensor,
        vision_embeds_tensor,
        indices_tensor,
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Tensor)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Verify the output matches the reference implementation
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-5, atol=1e-5
    )

    # Specifically verify that vision embeddings replaced placeholders
    assert torch.allclose(actual_output[3:7], vision_embeds)

    # And that other positions remained unchanged
    assert torch.allclose(actual_output[:3], text_embeds[:3])
    assert torch.allclose(actual_output[7:], text_embeds[7:])


def test_batch_merge_variable_positions_execution():
    """Test merging embeddings for batch with different placeholder positions using flattened tensors."""
    hidden_size = 768
    img_context_token_id = 100
    device = CPU()
    device_ref = DeviceRef.CPU()

    # Flatten batch with different image context token positions
    # First sequence: [1, 2, 100, 100, 100, 100, 3, 4, 5, 6]  # img at pos 2-5
    # Second sequence: [1, 2, 3, 4, 100, 100, 100, 100, 5, 6]  # img at pos 4-7
    input_ids = torch.tensor(
        [
            1,
            2,
            100,
            100,
            100,
            100,
            3,
            4,
            5,
            6,  # first sequence
            1,
            2,
            3,
            4,
            100,
            100,
            100,
            100,
            5,
            6,
        ],  # second sequence
        dtype=torch.int32,
    )

    # Total 20 tokens, 8 are image context tokens
    seq_len = 20
    num_image_tokens = 8

    # Pre-compute indices where image tokens should be inserted.
    image_token_indices = torch.where(input_ids == img_context_token_id)[0].to(
        torch.int32
    )

    text_embeds = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    vision_embeds = torch.randn(
        num_image_tokens, hidden_size, dtype=torch.float32
    )

    # Get reference output
    expected_output = merge_multimodal_embeddings_torch_reference(
        text_embeds, vision_embeds, image_token_indices
    )

    # Build and execute graph
    graph = Graph(
        "test_batch_merge",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(num_image_tokens, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(num_image_tokens,),
                device=device_ref,
            ),
        ],
    )

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Execute
    results = compiled.execute(
        Tensor.from_numpy(text_embeds.numpy()).to(device),
        Tensor.from_numpy(vision_embeds.numpy()).to(device),
        Tensor.from_numpy(image_token_indices.numpy()).to(device),
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Tensor)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Verify against reference
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-5, atol=1e-5
    )

    # Verify first sequence (positions 0-9)
    assert torch.allclose(actual_output[2:6], vision_embeds[0:4])
    assert torch.allclose(actual_output[:2], text_embeds[:2])
    assert torch.allclose(actual_output[6:10], text_embeds[6:10])

    # Verify second sequence (positions 10-19)
    assert torch.allclose(actual_output[14:18], vision_embeds[4:8])
    assert torch.allclose(actual_output[10:14], text_embeds[10:14])
    assert torch.allclose(actual_output[18:20], text_embeds[18:20])


def test_ragged_multimodal_embeddings_execution():
    """Test merging with ragged sequences of multimodal embeddings using flattened tensors."""
    hidden_size = 768
    img_context_token_id = 100
    device = CPU()
    device_ref = DeviceRef.CPU()

    # Flatten ragged sequences with different number of image context tokens
    # First: [1, 2, 100, 100, 100, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 3 image context tokens
    # Second: [1, 2, 3, 4, 5, 100, 100, 100, 100, 100, 6, 7, 8, 9, 10]  # 5 image context tokens
    input_ids = torch.tensor(
        [
            1,
            2,
            100,
            100,
            100,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,  # first
            1,
            2,
            3,
            4,
            5,
            100,
            100,
            100,
            100,
            100,
            6,
            7,
            8,
            9,
            10,
        ],  # second
        dtype=torch.int32,
    )

    # Total 30 tokens, 8 image context tokens
    seq_len = 30
    num_image_tokens = 8

    # Pre-compute indices where image tokens should be inserted.
    image_token_indices = torch.where(input_ids == img_context_token_id)[0].to(
        torch.int32
    )

    text_embeds = torch.randn(seq_len, hidden_size, dtype=torch.float32)

    # Ragged embeddings as a list that we'll concatenate
    vision_embeds_list = [
        torch.randn(3, hidden_size, dtype=torch.float32),  # For first sequence
        torch.randn(5, hidden_size, dtype=torch.float32),  # For second sequence
    ]
    vision_embeds = torch.cat(
        vision_embeds_list, dim=0
    )  # Shape: (8, hidden_size)

    # Get reference output
    expected_output = merge_multimodal_embeddings_torch_reference(
        text_embeds, vision_embeds, image_token_indices
    )

    # Build graph
    graph = Graph(
        "test_ragged_merge",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(num_image_tokens, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(num_image_tokens,),
                device=device_ref,
            ),
        ],
    )

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Execute
    results = compiled.execute(
        Tensor.from_numpy(text_embeds.numpy()).to(device),
        Tensor.from_numpy(vision_embeds.numpy()).to(device),
        Tensor.from_numpy(image_token_indices.numpy()).to(device),
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Tensor)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Verify against reference
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-5, atol=1e-5
    )


def test_no_image_context_tokens_fast_path_execution():
    """Test fast path when no image context tokens are present."""
    seq_len = 10
    hidden_size = 768
    img_context_token_id = 100
    device = CPU()
    device_ref = DeviceRef.CPU()

    # No image context tokens
    input_ids = torch.ones(seq_len, dtype=torch.int32) * 50
    # Pre-compute indices - should be empty tensor.
    image_token_indices = torch.where(input_ids == img_context_token_id)[0].to(
        torch.int32
    )

    text_embeds = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    vision_embeds = torch.empty(0, hidden_size, dtype=torch.float32)

    # Build graph
    graph = Graph(
        "test_no_placeholders",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(0, hidden_size),  # Empty multimodal embeddings
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(0,),  # Empty indices
                device=device_ref,
            ),
        ],
    )

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Execute
    results = compiled.execute(
        Tensor.from_numpy(text_embeds.numpy()).to(device),
        Tensor.from_numpy(vision_embeds.numpy()).to(device),
        Tensor.from_numpy(image_token_indices.numpy()).to(device),
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Tensor)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Should return inputs_embeds unchanged
    torch.testing.assert_close(actual_output, text_embeds, rtol=1e-5, atol=1e-5)

    # Get reference output to ensure consistency
    expected_output = merge_multimodal_embeddings_torch_reference(
        text_embeds, vision_embeds, image_token_indices
    )
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-5, atol=1e-5
    )


def test_count_mismatch_error():
    """Test that count mismatch is handled gracefully."""
    seq_len = 10
    hidden_size = 768
    img_context_token_id = 100
    device = CPU()
    device_ref = DeviceRef.CPU()

    input_ids = torch.tensor(
        [1, 2, 100, 100, 100, 3, 4, 5, 6, 7], dtype=torch.int32
    )  # 3 image context tokens
    # Pre-compute indices - should find 3 positions.
    image_token_indices = torch.where(input_ids == img_context_token_id)[0].to(
        torch.int32
    )
    # But we'll only pass 2 indices to create mismatch.
    mismatched_indices = image_token_indices[:2]

    text_embeds = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    vision_embeds = torch.randn(
        2, hidden_size, dtype=torch.float32
    )  # Only 2 embeddings!

    # Build graph
    graph = Graph(
        "test_count_mismatch",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(2, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(2,),  # Only 2 indices to match embeddings
                device=device_ref,
            ),
        ],
    )

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Execute
    results = compiled.execute(
        Tensor.from_numpy(text_embeds.numpy()).to(device),
        Tensor.from_numpy(vision_embeds.numpy()).to(device),
        Tensor.from_numpy(mismatched_indices.numpy()).to(device),
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Tensor)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Check that first two image context tokens were replaced
    assert not torch.allclose(actual_output[2:4], text_embeds[2:4])


def test_large_sequence_performance():
    """Test performance with large sequence lengths using flattened tensors."""
    seq_len = 2048
    hidden_size = 4096
    img_context_token_id = 100
    num_placeholders = 256
    device = CPU()
    device_ref = DeviceRef.CPU()

    # Create flattened input with scattered image context tokens
    input_ids = torch.ones(seq_len, dtype=torch.int32) * 50

    # Place image context tokens at regular intervals
    image_context_positions = torch.arange(
        0, seq_len, seq_len // num_placeholders
    )[:num_placeholders]
    input_ids[image_context_positions] = img_context_token_id

    # Pre-compute indices where image tokens should be inserted.
    image_token_indices = torch.where(input_ids == img_context_token_id)[0].to(
        torch.int32
    )

    text_embeds = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    vision_embeds = torch.randn(
        num_placeholders, hidden_size, dtype=torch.float32
    )

    # Get reference output
    expected_output = merge_multimodal_embeddings_torch_reference(
        text_embeds, vision_embeds, image_token_indices
    )

    # Build and execute graph
    graph = Graph(
        "test_large_sequence",
        forward=merge_multimodal_embeddings,
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=(seq_len, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.float32,
                shape=(num_placeholders, hidden_size),
                device=device_ref,
            ),
            TensorType(
                dtype=DType.int32,
                shape=(num_placeholders,),
                device=device_ref,
            ),
        ],
    )

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    # Execute
    results = compiled.execute(
        Tensor.from_numpy(text_embeds.numpy()).to(device),
        Tensor.from_numpy(vision_embeds.numpy()).to(device),
        Tensor.from_numpy(image_token_indices.numpy()).to(device),
    )

    # Convert result back to torch
    result_tensor = results[0]
    assert isinstance(result_tensor, Tensor)
    actual_output = torch.from_numpy(result_tensor.to_numpy())

    # Verify against reference
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-5, atol=1e-5
    )
