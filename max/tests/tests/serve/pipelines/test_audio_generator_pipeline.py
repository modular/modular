# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for AudioGeneratorPipeline."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import numpy as np
import pytest
from max.interfaces.pipeline_variants.audio_generation import (
    AudioGenerationMetadata,
)
from max.pipelines.core import AudioGenerationRequest, AudioGeneratorOutput
from max.serve.pipelines.llm import AudioGeneratorPipeline


class MockAudioGeneratorPipeline(AudioGeneratorPipeline):
    """Mock implementation of AudioGeneratorPipeline for testing."""

    def __init__(self, mock_chunks: list[AudioGeneratorOutput]) -> None:
        # Skip the parent constructor that requires real dependencies such as
        # `PipelineTokenizer`.
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False
        self._mock_chunks = mock_chunks
        self._background_tasks = set()

    async def next_chunk(
        self, request: AudioGenerationRequest
    ) -> AsyncGenerator[AudioGeneratorOutput, None]:
        """Mock implementation that yields predefined chunks."""
        for chunk in self._mock_chunks:
            yield chunk


def create_test_request() -> AudioGenerationRequest:
    """Create a test AudioGenerationRequest."""
    return AudioGenerationRequest(
        id="test-request-1",
        input="Hello, this is a test prompt",
        index=0,
        model="test-model",
        audio_prompt_tokens=[1, 2, 3],
        audio_prompt_transcription="test-transcription",
        streaming=False,
    )


def test_generate_full_audio_multiple_chunks() -> None:
    """Test generate_full_audio with multiple audio chunks."""
    # Create test audio data.
    chunk1_audio = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    chunk2_audio = np.array([[4.0, 5.0]], dtype=np.float32)
    chunk3_audio = np.array([[6.0, 7.0, 8.0, 9.0]], dtype=np.float32)

    # Create test chunks with the last one marked as done.
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk1_audio,
            metadata=AudioGenerationMetadata(sample_rate=44100, duration=0.1),
            is_done=False,
        ),
        AudioGeneratorOutput(
            audio_data=chunk2_audio,
            metadata=AudioGenerationMetadata(sample_rate=44100, duration=0.2),
            is_done=False,
        ),
        AudioGeneratorOutput(
            audio_data=chunk3_audio,
            metadata=AudioGenerationMetadata(
                sample_rate=44100, duration=0.3, final_chunk=True
            ),
            is_done=True,
        ),
    ]

    # Create mock pipeline.
    pipeline = MockAudioGeneratorPipeline(chunks)
    request = create_test_request()

    # Test the generate_full_audio method.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the result.
    assert result.is_done is True

    # Check that audio data is properly concatenated.
    expected_audio = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=np.float32
    )
    np.testing.assert_allclose(result.audio_data, expected_audio)

    # Check that metadata comes from the last chunk.
    assert result.metadata == {
        "sample_rate": 44100,
        "duration": 0.3,
        "final_chunk": True,
    }


def test_generate_full_audio_single_chunk() -> None:
    """Test generate_full_audio with a single audio chunk."""
    # Create test audio data.
    chunk_audio = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    # Create test chunk marked as done.
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk_audio,
            metadata=AudioGenerationMetadata(sample_rate=22050, duration=0.5),
            is_done=True,
        ),
    ]

    # Create mock pipeline.
    pipeline = MockAudioGeneratorPipeline(chunks)
    request = create_test_request()

    # Test the generate_full_audio method.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the result
    assert result.is_done is True
    np.testing.assert_allclose(result.audio_data, chunk_audio)
    assert result.metadata == {"sample_rate": 22050, "duration": 0.5}


def test_generate_full_audio_empty_chunks() -> None:
    """Test generate_full_audio with no audio chunks."""
    # Create mock pipeline with no chunks.
    pipeline = MockAudioGeneratorPipeline([])
    request = create_test_request()

    # Test the generate_full_audio method.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the result.
    assert result.is_done is True
    expected_empty_audio = np.array([])
    np.testing.assert_allclose(result.audio_data, expected_empty_audio)
    assert result.metadata == {}


def test_generate_full_audio_last_chunk_not_done() -> None:
    """Test that generate_full_audio asserts when last chunk is not done."""
    # Create test audio data.
    chunk_audio = np.array([[1.0, 2.0]], dtype=np.float32)

    # Create test chunk NOT marked as done - this should trigger the assertion.
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk_audio,
            metadata=AudioGenerationMetadata(sample_rate=44100),
            is_done=False,  # This should cause the assertion to fail
        ),
    ]

    # Create mock pipeline
    pipeline = MockAudioGeneratorPipeline(chunks)
    request = create_test_request()

    # Test that assertion error is raised
    with pytest.raises(AssertionError):
        asyncio.run(pipeline.generate_full_audio(request))


def test_generate_full_audio_different_tensor_shapes() -> None:
    """Test generate_full_audio with chunks of different shapes that can be concatenated."""
    # Create test audio data with different sequence lengths but same batch size
    chunk1_audio = np.array([[1.0, 2.0]], dtype=np.float32)  # (1, 2)
    chunk2_audio = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)  # (1, 3)
    chunk3_audio = np.array([[6.0]], dtype=np.float32)  # (1, 1)

    # Create test chunks.
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk1_audio,
            metadata=AudioGenerationMetadata(chunk_id=1),
            is_done=False,
        ),
        AudioGeneratorOutput(
            audio_data=chunk2_audio,
            metadata=AudioGenerationMetadata(chunk_id=2),
            is_done=False,
        ),
        AudioGeneratorOutput(
            audio_data=chunk3_audio,
            metadata=AudioGenerationMetadata(chunk_id=3, final_chunk=True),
            is_done=True,
        ),
    ]

    # Create mock pipeline.
    pipeline = MockAudioGeneratorPipeline(chunks)
    request = create_test_request()

    # Test the generate_full_audio method.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the result.
    assert result.is_done is True

    # Check that audio data is properly concatenated along the last dimension (-1).
    expected_audio = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32
    )
    np.testing.assert_allclose(result.audio_data, expected_audio)

    # Check that metadata comes from the last chunk.
    assert result.metadata == {"chunk_id": 3, "final_chunk": True}


def test_generate_full_audio_preserves_chunk_objects() -> None:
    """Test that generate_full_audio properly handles complete AudioGeneratorOutput objects."""
    # Create test audio data.
    chunk1_audio = np.array([[1.0]], dtype=np.float32)
    chunk2_audio = np.array([[2.0]], dtype=np.float32)

    # Create test chunks with different metadata.
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk1_audio,
            metadata=AudioGenerationMetadata(
                chunk_id=1, timestamp="2024-01-01"
            ),
            is_done=False,
        ),
        AudioGeneratorOutput(
            audio_data=chunk2_audio,
            metadata=AudioGenerationMetadata(
                chunk_id=2,
                timestamp="2024-01-02",
                final_chunk=True,
            ),
            is_done=True,
        ),
    ]

    # Create mock pipeline.
    pipeline = MockAudioGeneratorPipeline(chunks)
    request = create_test_request()

    # Test the generate_full_audio method.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the result.
    assert result.is_done is True

    # Check that audio data is properly concatenated.
    expected_audio = np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(result.audio_data, expected_audio)

    # Check that metadata comes from the last chunk (not the first).
    assert result.metadata == {
        "chunk_id": 2,
        "timestamp": "2024-01-02",
        "final_chunk": True,
    }
    assert result.metadata != {"chunk_id": 1, "timestamp": "2024-01-01"}
