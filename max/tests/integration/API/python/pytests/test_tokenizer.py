# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import asyncio
import pytest
from max.pipelines import (
    PipelineConfig,
    TextAndVisionTokenizer,
    TokenGeneratorRequest,
    TextAndVisionContext,
)
from max.pipelines import TokenGeneratorRequestMessage


@pytest.mark.skip("this requires authorized huggingface access")
def test_text_and_vision_tokenizer():
    """This test uses gated repos on huggingface, as such its not expected to run in CI.
    It is primarily written to test out the chat templating for multi-modal models.
    """

    VALID_REPOS = {
        # This is not currently working for pixtral.
        # "mistral-community/pixtral-12b": "<|image|>",
        "meta-llama/Llama-3.2-11B-Vision-Instruct": "<|image|>",
    }

    for repo_id, check_str in VALID_REPOS.items():
        pipeline_config = PipelineConfig(
            huggingface_repo_id=repo_id, trust_remote_code=True
        )
        tokenizer = TextAndVisionTokenizer(pipeline_config)

        request = TokenGeneratorRequest(
            id="request...",
            index=0,
            model_name=repo_id,
            messages=[
                TokenGeneratorRequestMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                            },
                        },
                    ],
                )
            ],
        )

        context: TextAndVisionContext = asyncio.run(
            tokenizer.new_context(request)
        )

        assert check_str in context.prompt, context.prompt
