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

"""Tests for Idefics3 tokenizer."""

import io

import numpy as np
import pytest
from max.pipelines.architectures.idefics3.tokenizer import Idefics3Tokenizer
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image
from test_common.mocks import DummyPipelineConfig

IDEFICS3_REPO_ID = "HuggingFaceM4/Idefics3-8B-Llama3"


def _make_tokenizer() -> Idefics3Tokenizer:
    pipeline_config = DummyPipelineConfig(
        model_path=IDEFICS3_REPO_ID,
        max_batch_size=None,
        max_length=None,
        quantization_encoding="float32",
    )
    # DummyPipelineConfig seeds a MagicMock HuggingFace config; set the
    # `image_token_id` that Idefics3Tokenizer reads.
    assert pipeline_config.model.huggingface_config is not None
    pipeline_config.model.huggingface_config.image_token_id = 128257
    return Idefics3Tokenizer(IDEFICS3_REPO_ID, pipeline_config=pipeline_config)


@pytest.mark.asyncio
async def test_idefics3_tokenizer_image_token_indices() -> None:
    """Test that the tokenizer correctly computes image token indices."""
    tokenizer = _make_tokenizer()
    assert tokenizer.vision_token_ids == [128257]
    assert tokenizer.enable_prefix_caching is True

    # Create a real image for the test using config dimensions
    img_buffer = io.BytesIO()
    image_size = 448
    Image.new("RGB", (image_size, image_size), color="red").save(
        img_buffer, format="PNG"
    )
    test_image = img_buffer.getvalue()

    # Create request with image
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    TextContentPart(text="test"),
                    ImageContentPart(),
                ],
            )
        ],
        images=[test_image],
        request_id=RequestID("test-id"),
        model_name=IDEFICS3_REPO_ID,
    )

    context = await tokenizer.new_context(request)
    assert context is not None
    # Idefics3 tokenizer turns this single image into 17 patch groups
    assert len(context.images) == 17


@pytest.mark.asyncio
async def test_idefics3_tokenizer_decode_accepts_token_list() -> None:
    """``max generate`` passes ``list[int]`` token ids; decode must accept it."""
    tokenizer = _make_tokenizer()
    tokens = tokenizer.delegate.encode("Hello world", add_special_tokens=False)
    assert isinstance(tokens, list)
    from_list = await tokenizer.decode(tokens)
    assert from_list == await tokenizer.decode(np.array(tokens))
    assert "Hello world" in from_list


@pytest.mark.asyncio
async def test_idefics3_tokenizer_token_id_prompt() -> None:
    tokenizer = _make_tokenizer()
    text_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID(),
            model_name=IDEFICS3_REPO_ID,
            prompt="What is this?",
        )
    )
    token_ids = text_context.tokens.prompt.tolist()

    context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID(),
            model_name=IDEFICS3_REPO_ID,
            prompt=token_ids,
        )
    )

    assert context.tokens.prompt.tolist() == token_ids
    assert context.images == []
    assert context.max_length == text_context.max_length
    assert context.eos_tracker == text_context.eos_tracker
    assert (
        context.extra_model_args.keys() == text_context.extra_model_args.keys()
    )


@pytest.mark.asyncio
async def test_idefics3_tokenizer_token_id_range() -> None:
    tokenizer = _make_tokenizer()
    # The mocked config has no vocab size, so the bound is the tokenizer's
    # length, which matches the checkpoint's text_config.vocab_size.
    bound = 128259
    assert tokenizer._token_id_bound == bound

    def request(prompt: list[int]) -> TextGenerationRequest:
        return TextGenerationRequest(
            request_id=RequestID(), model_name=IDEFICS3_REPO_ID, prompt=prompt
        )

    context = await tokenizer.new_context(request([0, bound - 1]))
    assert context.tokens.prompt.tolist() == [0, bound - 1]

    for bad_id in (-1, bound):
        with pytest.raises(InputError, match="out of range"):
            await tokenizer.new_context(request([0, bad_id]))
