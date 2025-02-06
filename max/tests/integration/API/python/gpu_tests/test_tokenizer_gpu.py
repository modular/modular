# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import asyncio

from max.driver import DeviceSpec
from max.pipelines import (
    PipelineConfig,
    TextTokenizer,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
    TokenGeneratorResponseFormat,
)


def test_text_tokenizer_with_constrained_decoding():
    pipeline_config = PipelineConfig(
        huggingface_repo_id="HuggingFaceTB/SmolLM-135M",
        device_specs=[DeviceSpec.accelerator(id=0)],
        enable_structured_output=True,
    )

    tokenizer = TextTokenizer(pipeline_config)

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Joe Biden is 100 years old.' return with {"name": "Joe Biden", "age": 100}.

    Please extract the person's name and age from the following excerpt:
    'Donald Trump is 102 years old.'

    """

    request = TokenGeneratorRequest(
        id="request_with_tools",
        index=0,
        model_name=pipeline_config.huggingface_repo_id,
        messages=[
            TokenGeneratorRequestMessage(
                role="user",
                content=prompt,
            )
        ],
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

    context = asyncio.run(tokenizer.new_context(request))

    assert context.json_schema
    assert isinstance(context.prompt, str)
