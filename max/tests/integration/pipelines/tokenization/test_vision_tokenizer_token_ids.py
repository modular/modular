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
"""Token-id prompts on vision tokenizers, as a completion request sends them.

A token-id prompt must build the same row as the text it encodes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.architectures.qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.tokenizer import TextAndVisionTokenizer
from max.pipelines.modeling.types import RequestID, TextGenerationRequest
from transformers import AutoConfig

GEMMA3 = "google/gemma-3-4b-it"
QWEN2_5VL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = "The capital of France is"


def _pipeline_config(model_path: str) -> MagicMock:
    pipeline_config = MagicMock()
    pipeline_config.model.huggingface_config = AutoConfig.from_pretrained(
        model_path
    )
    pipeline_config.model.generation_config.eos_token_id = None
    pipeline_config.model.kv_cache.enable_prefix_caching = False
    pipeline_config.draft_model = None
    pipeline_config.runtime.max_vision_preprocess_cache_bytes = 0
    pipeline_config.runtime.max_media_preprocess_cache_idle_seconds = 0.0
    return pipeline_config


def _request(prompt: str | list[int]) -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id=RequestID(), model_name="model", prompt=prompt
    )


def _assert_same_row(
    context: TextAndVisionContext, expected: TextAndVisionContext
) -> None:
    np.testing.assert_array_equal(context.tokens.prompt, expected.tokens.prompt)
    assert context.max_length == expected.max_length
    assert context.vocab_size == expected.vocab_size
    assert context.eos_tracker == expected.eos_tracker
    assert context.vision_token_ids == expected.vision_token_ids
    assert context.images == expected.images == []
    assert context.extra_model_args.keys() == expected.extra_model_args.keys()
    for name, value in expected.extra_model_args.items():
        np.testing.assert_array_equal(context.extra_model_args[name], value)


@pytest.mark.asyncio
async def test_text_and_vision_tokenizer_token_id_prompt() -> None:
    tokenizer = TextAndVisionTokenizer(
        GEMMA3, pipeline_config=_pipeline_config(GEMMA3)
    )
    text_context = await tokenizer.new_context(_request(PROMPT))
    token_ids = text_context.tokens.prompt.tolist()

    context = await tokenizer.new_context(_request(token_ids))

    _assert_same_row(context, text_context)


@pytest.mark.asyncio
async def test_text_and_vision_tokenizer_rejects_placeholder_without_image() -> (
    None
):
    tokenizer = TextAndVisionTokenizer(
        GEMMA3, pipeline_config=_pipeline_config(GEMMA3)
    )
    text_context = await tokenizer.new_context(_request(PROMPT))
    token_ids = text_context.tokens.prompt.tolist()
    token_ids.insert(1, tokenizer.vision_token_ids[0])

    with pytest.raises(ValueError, match="1 image placeholder runs"):
        await tokenizer.new_context(_request(token_ids))


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_token_id_prompt() -> None:
    tokenizer = Qwen2_5VLTokenizer(
        QWEN2_5VL, pipeline_config=_pipeline_config(QWEN2_5VL)
    )
    text_context = await tokenizer.new_context(_request(PROMPT))
    token_ids = text_context.tokens.prompt.tolist()

    context = await tokenizer.new_context(_request(token_ids))

    _assert_same_row(context, text_context)
    assert context.rope_delta == text_context.rope_delta
    np.testing.assert_array_equal(
        context.decoder_position_ids, text_context.decoder_position_ids
    )


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_rejects_placeholder_without_image() -> None:
    tokenizer = Qwen2_5VLTokenizer(
        QWEN2_5VL, pipeline_config=_pipeline_config(QWEN2_5VL)
    )
    text_context = await tokenizer.new_context(_request(PROMPT))
    token_ids = text_context.tokens.prompt.tolist()
    token_ids.insert(1, tokenizer.image_token_id)

    with pytest.raises(ValueError, match="image placeholder tokens"):
        await tokenizer.new_context(_request(token_ids))
