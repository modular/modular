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
"""Every id in a token-id prompt must name a row of the model's embedding.

An id outside that range fails the whole batch the request lands in, so the
tokenizer rejects it with an ``InputError``, which serve returns as a 400.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.tokenizer import (
    TextTokenizer,
    checked_token_ids,
    resolve_token_id_bound,
)
from max.pipelines.modeling.types import RequestID, TextGenerationRequest

TOKENIZER_LEN = 100
VOCAB_SIZE = 128


class _StubDelegate:
    """Encodes text as its characters' code points."""

    def __len__(self) -> int:
        return TOKENIZER_LEN

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in prompt]


def _pipeline_config(huggingface_config: object) -> MagicMock:
    pipeline_config = MagicMock()
    pipeline_config.model.huggingface_config = huggingface_config
    return pipeline_config


@pytest.mark.parametrize(
    "huggingface_config",
    [
        SimpleNamespace(vocab_size=VOCAB_SIZE),
        SimpleNamespace(
            vocab_size=1, text_config=SimpleNamespace(vocab_size=VOCAB_SIZE)
        ),
        SimpleNamespace(llm_config=SimpleNamespace(vocab_size=VOCAB_SIZE)),
    ],
    ids=["text", "text_config", "llm_config"],
)
def test_bound_is_the_text_config_vocab_size(
    huggingface_config: object,
) -> None:
    bound = resolve_token_id_bound(
        _StubDelegate(), _pipeline_config(huggingface_config)
    )

    assert bound == VOCAB_SIZE


def test_bound_is_the_config_vocab_size_when_the_tokenizer_is_longer() -> None:
    """Ids past the embedding have no row, even if the tokenizer knows them."""
    config = _pipeline_config(SimpleNamespace(vocab_size=TOKENIZER_LEN - 1))

    assert resolve_token_id_bound(_StubDelegate(), config) == TOKENIZER_LEN - 1


@pytest.mark.parametrize(
    "pipeline_config",
    [None, _pipeline_config(SimpleNamespace()), _pipeline_config(MagicMock())],
    ids=["no_pipeline_config", "no_vocab_size", "mock_config"],
)
def test_bound_falls_back_to_the_tokenizer_length(
    pipeline_config: MagicMock | None,
) -> None:
    assert resolve_token_id_bound(_StubDelegate(), pipeline_config) == (
        TOKENIZER_LEN
    )


def test_checked_token_ids_keeps_every_id_in_range() -> None:
    ids = checked_token_ids(range(VOCAB_SIZE), VOCAB_SIZE)

    assert ids.dtype == np.int64
    np.testing.assert_array_equal(ids, np.arange(VOCAB_SIZE))


@pytest.mark.parametrize(
    ("bad_id", "message"),
    [
        (-1, "id -1 at position 1"),
        (VOCAB_SIZE, f"id {VOCAB_SIZE} at position 1"),
        (2**64, f"id {2**64} at position 1"),
        (-(2**64), f"id {-(2**64)} at position 1"),
    ],
)
def test_checked_token_ids_rejects_an_id_out_of_range(
    bad_id: int, message: str
) -> None:
    with pytest.raises(InputError, match=message):
        checked_token_ids([0, bad_id, 1], VOCAB_SIZE)


def _text_tokenizer() -> TextTokenizer:
    """Builds a ``TextTokenizer`` without loading a checkpoint."""
    tokenizer = TextTokenizer.__new__(TextTokenizer)
    tokenizer.delegate = _StubDelegate()
    tokenizer.max_length = 4096
    tokenizer._eos_token_ids = {0}
    tokenizer._token_id_bound = VOCAB_SIZE
    return tokenizer


def test_text_tokenizer_encode_accepts_the_highest_id() -> None:
    tokenizer = _text_tokenizer()
    assert tokenizer.token_id_bound == VOCAB_SIZE

    encoded = asyncio.run(tokenizer.encode([0, VOCAB_SIZE - 1]))

    np.testing.assert_array_equal(encoded, [0, VOCAB_SIZE - 1])


@pytest.mark.parametrize("bad_id", [-1, VOCAB_SIZE])
def test_text_tokenizer_new_context_rejects_an_id_out_of_range(
    bad_id: int,
) -> None:
    request = TextGenerationRequest(
        request_id=RequestID(), model_name="model", prompt=[1, bad_id]
    )

    with pytest.raises(InputError, match="out of range"):
        asyncio.run(_text_tokenizer().new_context(request))


def test_text_tokenizer_checks_text_when_the_tokenizer_outgrows_the_embedding() -> (
    None
):
    """A tokenizer longer than the embedding can encode text past its rows."""
    tokenizer = _text_tokenizer()
    tokenizer._token_id_bound = TOKENIZER_LEN - 1

    np.testing.assert_array_equal(asyncio.run(tokenizer.encode("a")), [97])
    with pytest.raises(
        InputError, match=f"id {TOKENIZER_LEN - 1} at position 1"
    ):
        asyncio.run(tokenizer.encode("a" + chr(TOKENIZER_LEN - 1)))


def test_stop_strings_skip_the_range_check() -> None:
    """A stop string never reaches the embedding, so its ids are not checked."""
    tokenizer = _text_tokenizer()
    tokenizer._token_id_bound = TOKENIZER_LEN - 1

    stop = asyncio.run(
        tokenizer._encode_stop_criteria(["a" + chr(TOKENIZER_LEN - 1)])
    )

    assert stop == [[97, TOKENIZER_LEN - 1]]


class _StubChatEncoder:
    """Encodes text as its characters' code points, as the native encoder does."""

    def tokenizer_json_sha256(self) -> str:
        return ""

    def encode(self, text: str) -> bytes:
        return np.array([ord(c) for c in text], dtype="<u4").tobytes()


def test_chat_encoder_ids_get_the_text_range_check() -> None:
    tokenizer = _text_tokenizer()
    tokenizer._token_id_bound = TOKENIZER_LEN - 1
    tokenizer._chat_encoder = _StubChatEncoder()
    tokenizer._chat_encoder_outcomes = Counter()
    tokenizer._chat_encoder_failed = False

    np.testing.assert_array_equal(
        asyncio.run(tokenizer._encode_chat_prompt("a")), [97]
    )
    with pytest.raises(
        InputError, match=f"id {TOKENIZER_LEN - 1} at position 1"
    ):
        asyncio.run(tokenizer._encode_chat_prompt("a" + chr(TOKENIZER_LEN - 1)))
