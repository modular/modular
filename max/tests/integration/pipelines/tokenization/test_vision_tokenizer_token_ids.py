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

A token-id prompt must build the same row as the text it encodes, and each
id must name a row of the model's embedding.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.architectures.qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.context import TextAndVisionContext
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import PipelineRuntimeConfig
from max.pipelines.lib.tokenizer import TextAndVisionTokenizer
from max.pipelines.modeling.types import RequestID, TextGenerationRequest
from transformers import AutoConfig

GEMMA3 = "google/gemma-3-4b-it"
QWEN2_5VL = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN3_5 = "Qwen/Qwen3.5-9B"
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


async def _assert_token_id_range(
    tokenizer: TextAndVisionTokenizer, model_path: str
) -> None:
    """Checks the bound is the embedding size and every tokenizer id fits."""
    bound = AutoConfig.from_pretrained(model_path).get_text_config().vocab_size
    assert tokenizer._token_id_bound == bound
    # Special and added tokens included.
    assert max(tokenizer.delegate.get_vocab().values()) < bound

    context = await tokenizer.new_context(_request([0, bound - 1]))
    assert context.tokens.prompt.tolist() == [0, bound - 1]

    for bad_id in (-1, bound):
        with pytest.raises(InputError, match="out of range"):
            await tokenizer.new_context(_request([0, bad_id]))


@pytest.mark.asyncio
async def test_text_and_vision_tokenizer_token_id_range() -> None:
    tokenizer = TextAndVisionTokenizer(
        GEMMA3, pipeline_config=_pipeline_config(GEMMA3)
    )

    await _assert_token_id_range(tokenizer, GEMMA3)
    # DeepSeek-V4.1 builds its rows through encode.
    with pytest.raises(InputError, match="out of range"):
        await tokenizer.encode([tokenizer._token_id_bound])


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_token_id_range() -> None:
    tokenizer = Qwen2_5VLTokenizer(
        QWEN2_5VL, pipeline_config=_pipeline_config(QWEN2_5VL)
    )

    await _assert_token_id_range(tokenizer, QWEN2_5VL)


@pytest.mark.asyncio
async def test_qwen3_5_tokenizer_token_id_range() -> None:
    pipeline_config = _pipeline_config(QWEN3_5)
    pipeline_config.runtime = PipelineRuntimeConfig()
    tokenizer = Qwen3_5Tokenizer(QWEN3_5, pipeline_config=pipeline_config)

    await _assert_token_id_range(tokenizer, QWEN3_5)
