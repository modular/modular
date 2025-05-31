# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for SamplingParams integration in AudioGeneratorPipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import torch
from max.pipelines.core import (
    AudioGenerationRequest,
    AudioGeneratorOutput,
    SamplingParams,
)
from max.serve.pipelines.llm import AudioGeneratorPipeline


class MockAudioGeneratorPipelineWithSamplingParams(AudioGeneratorPipeline):
    """Mock implementation of AudioGeneratorPipeline for testing SamplingParams."""

    def __init__(self, mock_chunks: list[AudioGeneratorOutput]):
        # Skip the parent constructor that requires real dependencies
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False
        self._mock_chunks = mock_chunks
        self._background_tasks = set()
        self.received_sampling_params: SamplingParams | None = None

    async def next_chunk(
        self, request: AudioGenerationRequest
    ) -> AsyncGenerator[AudioGeneratorOutput, None]:
        """Mock implementation that captures sampling_params and yields predefined chunks."""
        # Capture the sampling_params for verification
        self.received_sampling_params = request.sampling_params

        for chunk in self._mock_chunks:
            yield chunk


def create_test_request_with_sampling_params(
    sampling_params: SamplingParams, id: str = "test-request-sampling"
) -> AudioGenerationRequest:
    """Create a test AudioGenerationRequest with specific SamplingParams."""
    return AudioGenerationRequest(
        id=id,
        input="Test prompt for sampling params",
        index=0,
        model="test-model",
        voice="test-voice",
        sampling_params=sampling_params,
    )


def test_pipeline_receives_sampling_params() -> None:
    """Test that AudioGeneratorPipeline receives SamplingParams from request."""
    # Create custom sampling params.
    custom_params = SamplingParams(
        top_k=15,
        temperature=1.3,
        frequency_penalty=0.6,
        presence_penalty=0.8,
        repetition_penalty=1.05,
        enable_structured_output=False,
        enable_variable_logits=True,
        do_penalties=False,
    )

    # Create test audio data.
    chunk_audio = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk_audio,
            metadata={"sample_rate": 44100},
            is_done=True,
        ),
    ]

    # Create mock pipeline
    pipeline = MockAudioGeneratorPipelineWithSamplingParams(chunks)
    request = create_test_request_with_sampling_params(custom_params)

    # Execute the pipeline.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the pipeline received the correct sampling_params.
    assert pipeline.received_sampling_params is custom_params

    # Verify the audio generation still works correctly.
    assert result.is_done is True
    torch.testing.assert_close(result.audio_data, chunk_audio)


def test_pipeline_receives_default_sampling_params() -> None:
    """Test that pipeline receives default SamplingParams when none specified."""
    # Create test audio data.
    chunk_audio = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk_audio,
            metadata={"sample_rate": 22050},
            is_done=True,
        ),
    ]

    # Create mock pipeline.
    pipeline = MockAudioGeneratorPipelineWithSamplingParams(chunks)

    # Create request without explicit sampling_params (should use defaults).
    request = AudioGenerationRequest(
        id="test-request-default",
        input="Default sampling params test",
        index=0,
        model="test-model",
        voice="test-voice",
    )

    # Execute the pipeline.
    result = asyncio.run(pipeline.generate_full_audio(request))

    # Verify the pipeline received default sampling_params.
    assert isinstance(pipeline.received_sampling_params, SamplingParams)
    assert pipeline.received_sampling_params.top_k == 1
    assert pipeline.received_sampling_params.temperature == 1
    assert pipeline.received_sampling_params.frequency_penalty == 0.0
    assert pipeline.received_sampling_params.presence_penalty == 0.0
    assert pipeline.received_sampling_params.repetition_penalty == 1.0
    assert pipeline.received_sampling_params.enable_structured_output is False
    assert pipeline.received_sampling_params.enable_variable_logits is False
    assert pipeline.received_sampling_params.do_penalties is False

    # Verify the audio generation still works correctly.
    assert result.is_done is True
    torch.testing.assert_close(result.audio_data, chunk_audio)


def test_multiple_requests_different_sampling_params():
    """Test that different requests with different SamplingParams are handled correctly."""
    params_list = [
        SamplingParams(top_k=1, temperature=0.1),
        SamplingParams(top_k=10, temperature=1.0, do_penalties=True),
        SamplingParams(
            top_k=50, temperature=2.0, enable_structured_output=True
        ),
    ]

    # Create test audio data
    chunk_audio = torch.tensor([[1.0]], dtype=torch.float32)
    chunks = [
        AudioGeneratorOutput(
            audio_data=chunk_audio,
            metadata={"test": True},
            is_done=True,
        ),
    ]

    for i, params in enumerate(params_list):
        # Create fresh pipeline for each test
        pipeline = MockAudioGeneratorPipelineWithSamplingParams(chunks)
        request = create_test_request_with_sampling_params(
            params, id=f"test-request-{i}"
        )

        # Execute the pipeline
        result = asyncio.run(pipeline.generate_full_audio(request))

        # Verify each pipeline received the correct sampling_params
        assert pipeline.received_sampling_params is params
        assert pipeline.received_sampling_params.top_k == params.top_k
        assert (
            pipeline.received_sampling_params.temperature == params.temperature
        )
        assert (
            pipeline.received_sampling_params.do_penalties
            == params.do_penalties
        )
        assert (
            pipeline.received_sampling_params.enable_structured_output
            == params.enable_structured_output
        )

        # Verify the audio generation still works
        assert result.is_done is True
