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
"""Integration tests for structured output with speculative decoding (EAGLE).

These tests verify that grammar constraints (JSON schema) are correctly applied
during speculative decoding target verification, producing valid JSON output.

NOTE: These tests require downloading EAGLE models (~7GB for 3B model pair) and
take significant time to compile. They are marked for the HF workflow.
"""

import asyncio
import json

import hf_repo_lock
import numpy as np
import pytest
from max.driver import DeviceSpec
from max.interfaces import (
    RequestID,
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationResponseFormat,
)
from max.pipelines import PipelineConfig
from max.pipelines.core import TextContext
from max.pipelines.lib import MAXModelConfig, SamplingConfig, TextTokenizer
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from max.pipelines.lib.registry import PipelineRegistry

pytest_plugins = "test_common.registry"


@pytest.mark.timeout(600)  # 10 minutes for model download + compile
def test_eagle_structured_output_json_schema_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test that EAGLE speculative decoding with structured output produces valid JSON.

    This test verifies the end-to-end integration of:
    1. EAGLE speculative decoding (draft + target verification)
    2. Structured output via JSON schema grammar constraints
    3. Bitmask-constrained acceptance sampling during target verification

    The grammar constraints are applied only during target model verification,
    not during draft token generation (following vLLM's approach).
    """
    # Use Llama-3.2-3B-Instruct as target with EAGLE-Llama-3.2-3B-Instruct-bf16
    # as the draft model. This is the smallest EAGLE model pair available.
    target_revision = hf_repo_lock.revision_for_hf_repo(
        "meta-llama/Llama-3.2-3B-Instruct"
    )
    assert target_revision is not None

    draft_revision = hf_repo_lock.revision_for_hf_repo(
        "atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16"
    )
    assert draft_revision is not None

    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="meta-llama/Llama-3.2-3B-Instruct",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=target_revision,
                    max_length=2048,
                ),
                "draft": MAXModelConfig(
                    model_path="atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=draft_revision,
                ),
            }
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=2,
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

    prompt = """Extract the person's name and age from: 'David Smith is 35 years old.'"""

    request_id = RequestID("eagle_structured")
    request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=request_id,
        messages=[TextGenerationRequestMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(max_new_tokens=50, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
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

    # Verify context has json_schema set (required for structured output)
    assert context.json_schema is not None

    pipeline = pipeline_factory()
    # The pipeline should support spec decode + structured output
    assert isinstance(pipeline, OverlapTextGenerationPipeline)

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

        if request_id in response:
            for token in response[request_id].tokens:
                tokens.append(token)
            if response[request_id].is_done:
                break

    # Flush any remaining outputs
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
    assert "name" in result, f"Missing 'name' in response: {response_content}"
    assert "age" in result, f"Missing 'age' in response: {response_content}"
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)


@pytest.mark.timeout(600)  # 10 minutes for model download + compile
def test_eagle_structured_output_heterogeneous_batch_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    """Test mixed batch with structured and non-structured requests under EAGLE.

    Verifies that when a batch contains both:
    - Requests with json_schema (structured output)
    - Requests without json_schema (free-form)

    Each request is handled correctly during speculative decoding:
    - Structured requests use grammar-constrained bitmasks during verification
    - Free-form requests use unconstrained (all-True) bitmasks
    """
    target_revision = hf_repo_lock.revision_for_hf_repo(
        "meta-llama/Llama-3.2-3B-Instruct"
    )
    assert target_revision is not None

    draft_revision = hf_repo_lock.revision_for_hf_repo(
        "atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16"
    )
    assert draft_revision is not None

    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="meta-llama/Llama-3.2-3B-Instruct",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=target_revision,
                    max_length=2048,
                ),
                "draft": MAXModelConfig(
                    model_path="atomicapple0/EAGLE-Llama-3.2-3B-Instruct-bf16",
                    quantization_encoding="bfloat16",
                    device_specs=[DeviceSpec.accelerator()],
                    huggingface_model_revision=draft_revision,
                ),
            }
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=2,
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(
            max_batch_size=2,
            enable_overlap_scheduler=True,
        ),
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )
    assert isinstance(tokenizer, TextTokenizer)

    # Request 1: Structured output with JSON schema
    structured_request_id = RequestID("eagle_structured_batch")
    structured_request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=structured_request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="Extract: 'Emma Wilson is 28 years old.'",
            )
        ],
        sampling_params=SamplingParams(max_new_tokens=50, top_k=1),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
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

    # Request 2: Non-structured output (no json_schema)
    freeform_request_id = RequestID("eagle_freeform_batch")
    freeform_request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=freeform_request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="Say hello in one sentence.",
            )
        ],
        sampling_params=SamplingParams(max_new_tokens=20, top_k=1),
    )

    structured_ctx: TextContext = asyncio.run(
        tokenizer.new_context(structured_request)
    )
    freeform_ctx: TextContext = asyncio.run(
        tokenizer.new_context(freeform_request)
    )

    # Verify one has json_schema and one doesn't
    assert structured_ctx.json_schema is not None
    assert freeform_ctx.json_schema is None

    pipeline = pipeline_factory()
    assert isinstance(pipeline, OverlapTextGenerationPipeline)

    kv_manager = pipeline.kv_manager
    kv_manager.claim(structured_ctx.request_id, replica_idx=0)
    kv_manager.claim(freeform_ctx.request_id, replica_idx=0)

    structured_tokens: list[int] = []
    freeform_tokens: list[int] = []
    max_iterations = 60

    # Keep both contexts in batch throughout execution.
    # Speculative decoding maintains state that expects consistent batch sizes.
    contexts: list[TextContext] = [structured_ctx, freeform_ctx]
    structured_done = False
    freeform_done = False

    for _ in range(max_iterations):
        if structured_done and freeform_done:
            break

        # Allocate KV cache for all contexts in the batch.
        # Even done contexts need consistent allocation for spec decode.
        for ctx in contexts:
            kv_manager.alloc(ctx, replica_idx=0, num_steps=1)

        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[contexts], num_steps=1
        )
        response = pipeline.execute(inputs)

        for ctx in contexts:
            if ctx.request_id in response:
                resp = response[ctx.request_id]
                if ctx.request_id == structured_request_id:
                    structured_tokens.extend(resp.tokens)
                    if resp.is_done:
                        structured_done = True
                else:
                    freeform_tokens.extend(resp.tokens)
                    if resp.is_done:
                        freeform_done = True

    # Flush remaining outputs
    empty_inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[]], num_steps=1
    )
    response = pipeline.execute(empty_inputs)
    if structured_request_id in response:
        structured_tokens.extend(response[structured_request_id].tokens)
    if freeform_request_id in response:
        freeform_tokens.extend(response[freeform_request_id].tokens)

    # Verify structured output produced valid JSON
    structured_response = asyncio.run(
        tokenizer.decode(np.array(structured_tokens), skip_special_tokens=True)
    )
    result = json.loads(structured_response)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)

    # Verify free-form output was generated (not blocked by bitmask)
    assert len(freeform_tokens) > 0, "Free-form request should generate tokens"
    freeform_response = asyncio.run(
        tokenizer.decode(np.array(freeform_tokens), skip_special_tokens=True)
    )
    assert len(freeform_response.strip()) > 0, (
        "Free-form request should produce non-empty output"
    )
