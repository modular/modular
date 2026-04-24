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
"""Test Suite for Unit Testing the TextGenerationPipeline with structured output."""

import asyncio
import json
from typing import cast

import hf_repo_lock
import numpy as np
from max.driver import DeviceSpec
from max.interfaces import (
    RequestID,
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationResponseFormat,
)
from max.pipelines import PipelineConfig, TextGenerationPipeline
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    MAXModelConfig,
    OverlapTextGenerationPipeline,
    SamplingConfig,
    TextTokenizer,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.registry import PipelineRegistry

pytest_plugins = "test_common.registry"


def test_smollm_with_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    revision = hf_repo_lock.revision_for_hf_repo(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert revision is not None
    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=revision,
                    max_length=8192,
                )
            }
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Bob Dylan is 83 years old.' return with {"badnamey": "Bob Dylan", "badagey": 83}.
    Please extract the person's name and age from the following excerpt:
    'John Mayer is 47 years old.'
    """

    request_id = RequestID("request_0")
    sampling_params = SamplingParams(max_new_tokens=50, top_k=1)
    request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=sampling_params,
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "age": {
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
            },
        ),
    )

    # Get Context
    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    # Pipeline may be TextGenerationPipeline or OverlapTextGenerationPipeline
    # depending on auto-enable settings. Both support structured output.
    assert isinstance(
        pipeline, (TextGenerationPipeline, OverlapTextGenerationPipeline)
    )
    # Cast for type checker - both have the same interface for what we need.
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context.request_id, replica_idx=0)

    tokens: list[int] = []
    max_iterations = 60
    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]], num_steps=1
        )
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        response = pipeline.execute(inputs)

        # Handle overlap pipeline which may return empty response on first iteration
        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)

            if response[request_id].is_done:
                break

    # Flush remaining outputs with empty batch (for overlap pipeline)
    if isinstance(pipeline, OverlapTextGenerationPipeline):
        empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[]], num_steps=1
        )
        response = pipeline.execute(empty_inputs)
        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)

    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )

    result = json.loads(response_content)

    assert result["name"] == "John Mayer"
    assert result["age"] == 47


def test_multistep_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test that multi-step execution (num_steps > 1) produces valid JSON."""
    revision = hf_repo_lock.revision_for_hf_repo(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert revision is not None
    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=revision,
                    max_length=8192,
                )
            }
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        # Disable overlap scheduler: multi-step execution (num_steps > 1) is
        # only supported in TextGenerationPipeline, not OverlapTextGenerationPipeline.
        # Use force=True to prevent auto-enable from overriding our setting.
        runtime=PipelineRuntimeConfig(
            max_batch_size=1, enable_overlap_scheduler=False, force=True
        ),
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """Extract the person's name and age from: 'Alice Smith is 30 years old.'"""

    request_id = RequestID("multistep_request")
    sampling_params = SamplingParams(max_new_tokens=50, top_k=1)
    request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=sampling_params,
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        ),
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context.request_id, replica_idx=0)

    tokens = []
    num_steps = 4  # Use multi-step execution
    while True:
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]], num_steps=num_steps
        )
        kv_manager.alloc(context, replica_idx=0, num_steps=num_steps)
        response = pipeline.execute(inputs)

        for token in response[request_id].tokens:
            tokens.append(token)

        if response[request_id].is_done:
            break

    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )

    # Verify we got valid JSON matching the schema
    result = json.loads(response_content)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)


def test_multi_step_guided_decoding_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test that multi-step execution works correctly with guided decoding."""
    revision = hf_repo_lock.revision_for_hf_repo(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert revision is not None
    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=revision,
                    max_length=8192,
                )
            }
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        # Disable overlap scheduler: multi-step execution (num_steps > 1) is
        # only supported in TextGenerationPipeline, not OverlapTextGenerationPipeline.
        # Use force=True to prevent auto-enable from overriding our setting.
        runtime=PipelineRuntimeConfig(
            max_batch_size=1, enable_overlap_scheduler=False, force=True
        ),
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """Return JSON: 'Bob Jones is 25.'"""

    request_id = RequestID("multi_step_guided_test")
    request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=request_id,
        messages=[TextGenerationRequestMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(max_new_tokens=30, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        ),
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context.request_id, replica_idx=0)

    # Use multi-step execution (num_steps > 1) with guided decoding
    num_steps = 3
    max_iterations = 20

    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]], num_steps=num_steps
        )
        kv_manager.alloc(context, replica_idx=0, num_steps=num_steps)
        response = pipeline.execute(inputs)

        if response[request_id].is_done:
            break


def test_overlap_pipeline_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test overlap pipeline with structured output.

    This verifies that the OverlapTextGenerationPipeline correctly handles
    structured output (guided decoding) with single-step execution, producing
    valid JSON output conforming to the schema.
    """

    revision = hf_repo_lock.revision_for_hf_repo(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert revision is not None
    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=revision,
                    max_length=8192,
                )
            }
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            enable_overlap_scheduler=True,
        ),
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )
    assert isinstance(tokenizer, TextTokenizer)

    prompt = """Extract name and age from: 'Charlie Brown is 8 years old.'"""

    request_id = RequestID("overlap_structured")
    request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=request_id,
        messages=[TextGenerationRequestMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(max_new_tokens=50, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        ),
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    # Verify we got the overlap pipeline
    assert isinstance(pipeline, OverlapTextGenerationPipeline)
    pipeline = cast(OverlapTextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context.request_id, replica_idx=0)

    tokens: list[int] = []
    num_steps = 1  # Single-step execution for overlap pipeline
    max_iterations = 60  # More iterations needed with single-step

    for _ in range(max_iterations):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]], num_steps=num_steps
        )
        kv_manager.alloc(context, replica_idx=0, num_steps=num_steps)

        # For structured output, overlap pipeline syncs immediately (no overlap)
        # so results are returned in the same call, not delayed.
        response = pipeline.execute(inputs)

        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)
            if response[request_id].is_done:
                break

    # For normal overlap (non-structured-output), this would flush pending outputs.
    # For structured output with immediate sync, there are no pending outputs.
    empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[]], num_steps=1
    )
    response = pipeline.execute(empty_inputs)
    if request_id in response:
        for token in response[request_id].tokens:
            tokens.append(token)

    response_content = asyncio.run(
        tokenizer.decode(np.array(tokens), skip_special_tokens=True)
    )

    # Verify valid JSON matching schema
    result = json.loads(response_content)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)
