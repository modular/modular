# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import asyncio

import pytest
import requests
from max.pipelines import (
    TextAndVisionTokenizer,
    TextTokenizer,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
)
from max.pipelines.context import TextAndVisionContext, TextContext


def convert_image_url_to_base64(image_url):
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
def test_text_and_vision_tokenizer():
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
    for repo_id, check_str in VALID_REPOS.items():
        model_path = repo_id
        tokenizer = TextAndVisionTokenizer(model_path, trust_remote_code=True)
        for imgs_list in imgs:
            content = [
                {"type": "text", "text": "What is in this image?"},
            ] + [{"type": "image"} for _ in imgs_list]
            request = TokenGeneratorRequest(
                id="request...",
                index=0,
                model_name=repo_id,
                messages=[
                    TokenGeneratorRequestMessage(
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
                assert check_str not in context.prompt, context.prompt
            else:
                assert check_str in context.prompt, context.prompt
                assert context.pixel_values is not None
                assert len(context.pixel_values) == len(imgs_list)


@pytest.mark.skip("CI does not appear to be working well with gated repos")
def test_text_tokenizer_with_tool_use():
    """This test uses gated repos on huggingface, as such its not expected to run in CI.
    It is written to test out chat templating and input features for tool use with Llama 3.2
    """

    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = TextTokenizer(model_path)

    request = TokenGeneratorRequest(
        id="request_with_tools",
        index=0,
        model_name=model_path,
        messages=[
            TokenGeneratorRequestMessage(
                role="user",
                content="What is the weather in Toronto?",
            )
        ],
        tools=[
            TokenGeneratorRequestTool(
                type="function",
                function=TokenGeneratorRequestFunction(
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

    # This is just asserting that the prompt includes function context
    assert '"name": "get_current_weather"' in context.prompt


def test_tokenizer__truncates_to_max_length():
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    max_length = 12
    tokenizer = TextTokenizer(
        model_path,
        max_length=max_length,
    )

    short_request = TokenGeneratorRequest(
        id="request_with_short_message",
        index=0,
        model_name=model_path,
        prompt="Short message",
    )
    context: TextContext = asyncio.run(tokenizer.new_context(short_request))
    assert context.current_length < 12

    long_request = TokenGeneratorRequest(
        id="request_with_short_message",
        index=0,
        model_name=model_path,
        prompt="Longer message with lots of text with more words than max length for sure.",
    )
    with pytest.raises(ValueError, match="max length"):
        _ = asyncio.run(tokenizer.new_context(long_request))
