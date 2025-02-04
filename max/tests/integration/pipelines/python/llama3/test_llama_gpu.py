# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 with full weights on GPU and compares it to previously generated
golden values.
"""

import asyncio
import json
from functools import partial
from pathlib import Path

import pytest
from evaluate_llama import SupportedTestModels
from max.driver import DeviceSpec
from max.pipelines import (
    PipelineConfig,
    SupportedEncoding,
    TextContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
    TokenGeneratorResponseFormat,
)
from test_common.distance_metrics import kl_divergence_verifier
from test_common.evaluate import (
    PROMPTS,
    compare_values,
    next_token_with_logits,
    run_model,
)
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path

pytest_plugins = "test_common.registry"


@pytest.mark.parametrize(
    "model_name,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
def test_llama(
    pipeline_registry, model_name: str, encoding: str, testdata_directory: Path
) -> None:
    test_model = SupportedTestModels.get(model_name, encoding)
    config = test_model.build_config(max_length=512)
    tokenizer, pipeline = pipeline_registry.retrieve(config)
    actual = run_model(
        pipeline._pipeline_model,
        tokenizer,
        prompts=PROMPTS[:1],
    )

    golden_data_path = find_runtime_path(
        "torch_llama3_1_bfloat16_golden.json",
        testdata_directory,
        bazel_dir=Path("torch_llama_golden"),
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())

    # Note that this threshold was set experimentally so that the test passes
    # with the existing Llama 3.1 implementation.
    compare_values(
        actual,
        expected_results,
        compare_fn=partial(kl_divergence_verifier, threshold=0.1),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
async def test_llama_ragged(
    pipeline_registry, model: str, encoding: str
) -> None:
    prompt_a = PROMPTS[0]
    prompt_b = PROMPTS[1]

    stored_logits: dict[str, list[TextContext]] = {"A": [], "B": [], "C": []}

    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config(max_batch_size=4, max_length=512)
    tokenizer, pipeline = pipeline_registry.retrieve(config)

    def request(prompt: str, idx: int) -> TokenGeneratorRequest:
        return TokenGeneratorRequest(
            id=str(idx), index=idx, prompt=prompt, model_name=model
        )

    # Send in A and B for context encoding.
    context_a = await tokenizer.new_context(request(prompt_a, idx=0))
    context_b = await tokenizer.new_context(request(prompt_b, idx=1))
    next_token_with_logits(
        pipeline._pipeline_model,
        {"A": context_a, "B": context_b},
        stored_logits,
    )

    # Send in both B and C for token generation.
    next_token_with_logits(
        pipeline._pipeline_model,
        {"A": context_a, "B": context_b},
        stored_logits,
    )


def test_smollm_with_constrained_decoding(pipeline_registry):
    pipeline_config = PipelineConfig(
        huggingface_repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        enable_constrained_decoding=True,
        quantization_encoding=SupportedEncoding.bfloat16,
        device_specs=[DeviceSpec.accelerator()],
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
        model_name=pipeline_config.huggingface_repo_id,
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
        next_token = pipeline.next_token({request_id: context}, num_steps=1)
        if request_id not in next_token[0]:
            break

        tokens.append(next_token[0][request_id].next_token)

    print("Final Response: ")
    response_content = asyncio.run(
        tokenizer.decode(context, tokens, skip_special_tokens=True)
    )
    print(response_content)

    result = json.loads(response_content)

    assert result["name"] == "John Mayer"
    assert result["age"] == 47
