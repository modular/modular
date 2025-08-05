# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import asyncio

import hf_repo_lock
import numpy as np
import pytest
import requests
from max.driver import DeviceSpec, accelerator_count
from max.interfaces import (
    SamplingParams,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TextGenerationResponseFormat,
)
from max.pipelines import (
    PipelineConfig,
    SupportedEncoding,
    TextAndVisionTokenizer,
    TextTokenizer,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from test_common.mocks import mock_estimate_memory_footprint

LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_3_1_HF_REVISION = hf_repo_lock.revision_for_hf_repo(LLAMA_3_1_HF_REPO_ID)


def convert_image_url_to_base64(image_url):  # noqa: ANN001
    """Fetches an image from a URL and converts it to Base64 encoded bytes."""
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None


@pytest.mark.skip("this requires authorized huggingface access")
def test_text_and_vision_tokenizer() -> None:
    """This test uses gated repos on huggingface, as such its not expected to run in CI.
    It is primarily written to test out the chat templating for multi-modal models.
    """

    VALID_REPOS = {
        # This is not currently working for pixtral.
        "mistral-community/pixtral-12b": "[IMG]",
        "meta-llama/Llama-3.2-11B-Vision-Instruct": "<|image|>",
    }
    img_url = "https://picsum.photos/id/237/200/300"
    img = convert_image_url_to_base64(img_url)
    imgs = [[], [img], [img, img]]
    for repo_id in VALID_REPOS:
        model_path = repo_id
        tokenizer = TextAndVisionTokenizer(model_path, trust_remote_code=True)
        for imgs_list in imgs:
            content = [
                {"type": "text", "text": "What is in this image?"},
            ] + [{"type": "image"} for _ in imgs_list]
            request = TextGenerationRequest(
                request_id="request...",
                model_name=repo_id,
                messages=[
                    TextGenerationRequestMessage(
                        role="user",
                        content=content,
                    )
                ],
                images=imgs_list,
            )

            context: TextAndVisionContext = asyncio.run(
                tokenizer.new_context(request)
            )

            if not imgs_list:
                assert context.pixel_values is None
            else:
                assert context.pixel_values is not None
                assert len(context.pixel_values) == len(imgs_list)


@pytest.mark.skip("CI does not appear to be working well with gated repos")
def test_text_tokenizer_with_tool_use(llama_3_1_8b_instruct_local_path) -> None:  # noqa: ANN001
    """This test uses gated repos on huggingface, as such its not expected to run in CI.
    It is written to test out chat templating and input features for tool use with Llama 3.2
    """

    model_path = llama_3_1_8b_instruct_local_path
    tokenizer = TextTokenizer(model_path)

    request = TextGenerationRequest(
        request_id="request_with_tools",
        model_name=model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="What is the weather in Toronto?",
            )
        ],
        tools=[
            TextGenerationRequestTool(
                type="function",
                function=TextGenerationRequestFunction(
                    name="get_current_weather",
                    description="Get the current weather for a given location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. Toronto.",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            )
        ],
    )

    context: TextContext = asyncio.run(tokenizer.new_context(request))


def test_tokenizer__truncates_to_max_length(
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    max_length = 12
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path,
        max_length=max_length,
    )

    short_request = TextGenerationRequest(
        request_id="request_with_short_message",
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Short message",
    )
    context: TextContext = asyncio.run(tokenizer.new_context(short_request))
    assert context.current_length < 12

    long_request = TextGenerationRequest(
        request_id="request_with_short_message",
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Longer message with lots of text with more words than max length for sure.",
    )
    with pytest.raises(ValueError, match="max length"):
        _ = asyncio.run(tokenizer.new_context(long_request))


def test_tokenizer_regression_MODELS_467() -> None:
    """Regression test for
    https://linear.app/modularml/issue/MODELS-467/[bug]-no-text-response-mistralaimistral-7b-instruct-v03
    """
    tokenizer = TextTokenizer(
        "mistralai/Mistral-7B-Instruct-v0.3",
        revision="e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
        enable_llama_whitespace_fix=True,
        trust_remote_code=True,
    )

    def rank1(items: list[int]) -> np.ndarray:
        return np.array(items, dtype=np.uint32)

    def rank0(item: int) -> np.ndarray:
        return rank1([item])[0]

    def decode(tokens: np.ndarray) -> str:
        return asyncio.run(tokenizer.decode(tokens))

    # Single token here needs preceding space, including rank-0.
    assert decode(rank1([23325])) == " Hello"
    assert decode(rank0(23325)) == " Hello"
    assert decode(rank1([23325, 2294])) == "Hello world"
    # But not all single tokens should have a preceding space, including rank-0.
    assert decode(rank1([1056])) == "ing"
    assert decode(rank0(1056)) == "ing"


@pytest.mark.asyncio
async def test_tokenizer__encode_and_decode(
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    tokenizer = TextTokenizer(model_path=llama_3_1_8b_instruct_local_path)

    test_string = "hi my name is"
    encoded = await tokenizer.encode(test_string, add_special_tokens=False)
    context = TextContext(
        max_length=10,
        tokens=np.array(encoded),
    )
    assert context.current_length == len(encoded)
    decoded = await tokenizer.decode(encoded)
    assert test_string == decoded


@pytest.mark.skip("TODO: Fix this flaky test")
@mock_estimate_memory_footprint
def test_text_tokenizer_with_constrained_decoding(
    modular_ai_llama_3_1_local_path,  # noqa: ANN001
) -> None:
    device_specs = []
    if accelerator_count() > 0:
        device_specs.append(DeviceSpec.accelerator(id=0))
    else:
        device_specs.append(DeviceSpec.cpu(id=0))
    pipeline_config = PipelineConfig(
        model_path=modular_ai_llama_3_1_local_path,
        quantization_encoding=SupportedEncoding.bfloat16,
        device_specs=device_specs,
        enable_structured_output=True,
    )

    tokenizer = TextTokenizer(pipeline_config.model_config.model_path)

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Joe Biden is 100 years old.' return with {"name": "Joe Biden", "age": 100}.

    Please extract the person's name and age from the following excerpt:
    'Donald Trump is 102 years old.'

    """

    request = TextGenerationRequest(
        request_id="request_with_tools",
        model_name=pipeline_config.model_config.model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
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

    context = asyncio.run(tokenizer.new_context(request))

    assert context.json_schema


def test_tokenizer_encode_stop_criteria(
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    tokenizer = TextTokenizer(model_path=llama_3_1_8b_instruct_local_path)

    prompt = "hi my name is"

    request = TextGenerationRequest(
        request_id="id_0",
        model_name=llama_3_1_8b_instruct_local_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=SamplingParams(stop=["!"]),
    )

    context = asyncio.run(tokenizer.new_context(request))
    # encoded stop criteria should equal [0]
    assert len(context.eos_sequences) == 1
    assert len(context.eos_sequences[0]) == 1
    assert np.array_equal(context.eos_sequences[0], [0])


@pytest.mark.asyncio
async def test_tokenizer__generate_prompt_and_token_ids(
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    """Test the _generate_prompt_and_token_ids method of TextTokenizer."""
    tokenizer = TextTokenizer(model_path=llama_3_1_8b_instruct_local_path)

    # Test with string prompt
    prompt = "Hello, how are you?"
    prompt_text, token_ids = await tokenizer._generate_prompt_and_token_ids(
        prompt=prompt,
        messages=None,
    )
    assert isinstance(prompt_text, str)
    assert prompt_text == prompt
    assert isinstance(token_ids, np.ndarray)
    expected_token_ids = await tokenizer.encode(prompt, add_special_tokens=True)
    assert np.array_equal(token_ids, expected_token_ids)

    # Test with list of messages
    messages = [
        TextGenerationRequestMessage(
            role="user",
            content="Hello, how are you?",
        ),
        TextGenerationRequestMessage(
            role="assistant",
            content="I'm doing well, thank you!",
        ),
    ]
    prompt_text, token_ids = await tokenizer._generate_prompt_and_token_ids(
        prompt=None,
        messages=messages,
    )
    assert isinstance(prompt_text, str)
    expected_token_ids = await tokenizer.encode(
        prompt_text, add_special_tokens=False
    )
    assert np.array_equal(token_ids, expected_token_ids)
    # Verify that the chat template was applied
    assert "Hello, how are you?" in prompt_text
    assert "I'm doing well, thank you!" in prompt_text

    # Test with both prompt and messages (should raise ValueError)
    with pytest.raises(
        ValueError, match="both prompt and messages cannot be provided"
    ):
        await tokenizer._generate_prompt_and_token_ids(
            prompt="test",
            messages=messages,
        )

    # Test with neither prompt nor messages (should raise ValueError)
    with pytest.raises(
        ValueError,
        match="either prompt must be provided as a list\\[int\\] or str, or messages must be provided as a list\\[TextGenerationRequestMessage\\]",
    ):
        await tokenizer._generate_prompt_and_token_ids(
            prompt=None,
            messages=None,
        )
