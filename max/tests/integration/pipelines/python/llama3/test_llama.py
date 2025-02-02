# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import asyncio

import pytest
from evaluate_llama import SupportedTestModels
from max.pipelines import (
    PipelineConfig,
    SupportedEncoding,
    TextContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
)
from test_common.evaluate import PROMPTS, compare_values, run_model
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path

pytest_plugins = "test_common.registry"


@pytest.mark.skip("Disabling tempoarily to update llama3 in a separate commit")
@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "q4_k"),
    ],
)
def test_llama(pipeline_registry, model, encoding, testdata_directory):
    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config()

    tokenizer, pipeline = pipeline_registry.retrieve(config)
    actual = run_model(
        pipeline._pipeline_model,
        tokenizer,
        prompts=PROMPTS[:1],
    )

    golden_data_path = find_runtime_path(
        test_model.golden_data_fname(), testdata_directory
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    with pytest.raises(AssertionError):
        # TODO(MSDK-968): Q4_K is currently expected not to match golden values.
        # This test will fail once we have fixed the accuracy issue.
        compare_values(actual, expected_results)


@pytest.mark.skip("loads llama model, which will download taking a while.")
def test_llama_eos_token_id(pipeline_registry):
    """This test is primarily written to be run in a bespoke fashion, as it is downloads llama-3.1, which can tax CI unnecessarily."""
    config = PipelineConfig(huggingface_repo_id="modularai/llama-3.1")
    _, pipeline = pipeline_registry.retrieve(config)

    # The llama3_1 huggingface config has three eos tokens I want to make sure these are grabbed appropriately.
    assert pipeline._eos_token_id == set([128001, 128008, 128009])
    assert len(pipeline._eos_token_id) == 3


@pytest.mark.skip("requires gated repo access and takes a long time")
def test_llama_with_tools(pipeline_registry):
    pipeline_config = PipelineConfig(
        huggingface_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        # huggingface_repo_id="meta-llama/Llama-3.2-3B-Instruct",
        quantization_encoding=SupportedEncoding.float32,
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )

    messages = [
        TokenGeneratorRequestMessage(
            role="user", content="What's the weather like in Toronto today?"
        )
    ]

    tools = [
        TokenGeneratorRequestTool(
            type="function",
            function=TokenGeneratorRequestFunction(
                name="get_current_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and province, eg. Toronto, ON.",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        TokenGeneratorRequestTool(
            type="function",
            function=TokenGeneratorRequestFunction(
                name="get_latitude",
                description="Get the latitude for a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and province, eg. Toronto, ON.",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
    ]

    request_id = "request_0"
    request = TokenGeneratorRequest(
        model_name=pipeline_config.huggingface_repo_id,
        id=request_id,
        index=0,
        messages=messages,
        tools=tools,
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
        next_token = pipeline.next_token({request_id: context}, num_steps=1)
        if request_id not in next_token[0]:
            break

        tokens.append(next_token[0][request_id].next_token)

    print("Generated Response: ")
    response_content = asyncio.run(
        tokenizer.decode(context, tokens, skip_special_tokens=True)
    )
    print(response_content)

    assert (
        response_content
        == '{"name": "get_current_weater", "parameters": {"location": "Toronto, ON", "unit": "celsius"}}'
    )
