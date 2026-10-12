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
"""Tests how tokenizers resolve the token ids that end generation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.context import SamplingParams
from max.pipelines.lib.tokenizer import (
    build_eos_tracker_for_request,
    resolve_eos_token_ids,
)
from max.pipelines.modeling.types import RequestID, TextGenerationRequest
from transformers import GenerationConfig, PretrainedConfig

TOKENIZER_EOS = 151645
# MiMo-V2.6-Flash: ``config.json`` and the tokenizer name only
# ``<|im_end|>``; ``generation_config.json`` lists three stop tokens.
GENERATION_EOS = [151643, 151645, 151672]


def _pipeline_config(
    hf_eos: int | list[int] | None = None,
    generation_eos: int | list[int] | None = None,
    draft_eos: int | list[int] | None = None,
) -> MagicMock:
    pipeline_config = MagicMock()
    pipeline_config.model.huggingface_config = PretrainedConfig(
        eos_token_id=hf_eos
    )
    pipeline_config.model.generation_config = GenerationConfig(
        eos_token_id=generation_eos
    )
    if draft_eos is None:
        pipeline_config.draft_model = None
    else:
        pipeline_config.draft_model.huggingface_config = PretrainedConfig(
            eos_token_id=draft_eos
        )
    return pipeline_config


@pytest.mark.parametrize(
    "generation_eos, expected",
    [
        (GENERATION_EOS, set(GENERATION_EOS)),
        (151672, {TOKENIZER_EOS, 151672}),
        (None, {TOKENIZER_EOS}),
    ],
)
def test_unions_generation_config_eos(
    generation_eos: int | list[int] | None, expected: set[int]
) -> None:
    pipeline_config = _pipeline_config(
        hf_eos=TOKENIZER_EOS, generation_eos=generation_eos
    )
    assert resolve_eos_token_ids(TOKENIZER_EOS, pipeline_config) == expected


def test_unions_every_source() -> None:
    pipeline_config = _pipeline_config(
        hf_eos=[1, 2], generation_eos=3, draft_eos=4
    )
    assert resolve_eos_token_ids(0, pipeline_config) == {0, 1, 2, 3, 4}


def test_without_pipeline_config_uses_tokenizer_eos() -> None:
    assert resolve_eos_token_ids(TOKENIZER_EOS, None) == {TOKENIZER_EOS}
    assert resolve_eos_token_ids(None, None) == set()


def test_tolerates_missing_huggingface_config() -> None:
    pipeline_config = _pipeline_config(generation_eos=GENERATION_EOS)
    pipeline_config.model.huggingface_config = None
    assert resolve_eos_token_ids(None, pipeline_config) == set(GENERATION_EOS)


async def _encode(
    text: str, add_special_tokens: bool
) -> npt.NDArray[np.integer[Any]]:
    return np.array([ord(c) for c in text], dtype=np.int64)


def _eos_tracker_ids(
    eos_token_ids: set[int], sampling_params: SamplingParams
) -> set[int]:
    request = TextGenerationRequest(
        request_id=RequestID(),
        model_name="test-model",
        prompt="hi",
        sampling_params=sampling_params,
    )
    tracker = asyncio.run(
        build_eos_tracker_for_request(eos_token_ids, request, _encode)
    )
    return tracker.eos_token_ids


def test_user_stop_token_ids_extend_generation_config_eos() -> None:
    eos_token_ids = resolve_eos_token_ids(
        TOKENIZER_EOS, _pipeline_config(generation_eos=GENERATION_EOS)
    )
    assert _eos_tracker_ids(
        eos_token_ids, SamplingParams(stop_token_ids=[7])
    ) == {*GENERATION_EOS, 7}


def test_ignore_eos_drops_generation_config_eos() -> None:
    eos_token_ids = resolve_eos_token_ids(
        TOKENIZER_EOS, _pipeline_config(generation_eos=GENERATION_EOS)
    )
    assert (
        _eos_tracker_ids(eos_token_ids, SamplingParams(ignore_eos=True))
        == set()
    )
