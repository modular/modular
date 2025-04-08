# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test Suite for Unit Testing the TextGenerationPipeline with structured output."""

import asyncio
import json

import hf_repo_lock
from max.driver import DeviceSpec
from max.pipelines import (
    PipelineConfig,
    SupportedEncoding,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
    TokenGeneratorResponseFormat,
)
from max.pipelines.core import TextContext

pytest_plugins = "test_common.registry"


def test_smollm_with_structured_output_gpu(pipeline_registry):
    pipeline_config = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        enable_structured_output=True,
        quantization_encoding=SupportedEncoding.bfloat16,
        device_specs=[DeviceSpec.accelerator()],
        huggingface_revision=hf_repo_lock.revision_for_hf_repo(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        ),
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Bob Dylan is 83 years old.' return with {"badnamey": "Bob Dylan", "badagey": 83}.
    Please extract the person's name and age from the following excerpt:
    'John Mayer is 47 years old.'
    """

    request_id = "request_0"
    request = TokenGeneratorRequest(
        model_name=pipeline_config.model_config.model_path,
        id=request_id,
        index=0,
        messages=[
            TokenGeneratorRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        max_new_tokens=50,
        response_format=TokenGeneratorResponseFormat(
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

    print("\nRaw Prompt:")
    print(context.prompt)
    print("Initializing Pipeline...")
    pipeline = pipeline_factory()

    print("Generating tokens...")

    tokens = []
    while True:
        response = pipeline.next_token({request_id: context}, num_steps=1)

        for token in response[request_id].tokens:
            tokens.append(token.next_token)

        if response[request_id].is_done:
            break

    print("Final Response: ")
    response_content = asyncio.run(
        tokenizer.decode(context, tokens, skip_special_tokens=True)
    )
    print(response_content)

    result = json.loads(response_content)

    assert result["name"] == "John Mayer"
    assert result["age"] == 47
