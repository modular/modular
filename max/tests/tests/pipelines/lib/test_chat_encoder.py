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
"""TextTokenizer's ``--tokenizer-impl`` chat-encoder seam, with a fake encoder.

The fake claims the ``tokenizer.json`` of a tiny byte-level checkpoint written
to disk, and answers every prompt with fixed ids HuggingFace never produces,
so a test can tell which path encoded it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib.tokenizer import TextTokenizer
from max.pipelines.modeling.types import (
    ChatEncoderOutcomesProbe,
    ImageContentPart,
    TextGenerationRequestMessage,
)
from transformers.convert_slow_tokenizer import bytes_to_unicode

_TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
_SPECIAL = ["<|user|>", "<|assistant|>"]
_BYTES = list(bytes_to_unicode())
_FAST_IDS = [3, 1, 4, 1, 5]


def _tokenizer_json(normalizer: object = None) -> str:
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True,
    }
    tokenizer_json: dict[str, object] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": len(_BYTES) + i,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, token in enumerate(_SPECIAL)
        ],
        "normalizer": normalizer,
        "pre_tokenizer": byte_level,
        "post_processor": None,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {
                char: i for i, char in enumerate(bytes_to_unicode().values())
            },
            "merges": [],
        },
    }
    return json.dumps(tokenizer_json)


class FakeChatEncoder:
    """Encodes every prompt as ``_FAST_IDS``, recording the prompts."""

    raises: ClassVar[Exception | None] = None
    prompts: ClassVar[list[str]] = []

    def tokenizer_json_sha256(self) -> str:
        return hashlib.sha256(_tokenizer_json().encode()).hexdigest()

    def encode(self, text: str) -> bytes:
        FakeChatEncoder.prompts.append(text)
        if FakeChatEncoder.raises is not None:
            raise FakeChatEncoder.raises
        return np.array(_FAST_IDS, dtype="<u4").tobytes()


class OtherTokenizerJson(FakeChatEncoder):
    def tokenizer_json_sha256(self) -> str:
        return hashlib.sha256(b"{}").hexdigest()


class DigestRaises(FakeChatEncoder):
    def tokenizer_json_sha256(self) -> str:
        raise RuntimeError("no digest")


class TornOutput(FakeChatEncoder):
    def encode(self, text: str) -> bytes:
        return super().encode(text)[:-1]


class NotAnEncoder:
    pass


def _impl(name: str) -> str:
    return f"{__name__}:{name}"


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeChatEncoder.raises = None
    FakeChatEncoder.prompts = []


def _checkpoint(
    path: Path,
    tokenizer_json: str | None = None,
    **tokenizer_config: object,
) -> Path:
    (path / "tokenizer.json").write_text(tokenizer_json or _tokenizer_json())
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "chat_template": _TEMPLATE,
                **tokenizer_config,
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _checkpoint(tmp_path_factory.mktemp("checkpoint"))


def _tokenizer(
    checkpoint: Path,
    tokenizer_impl: str | None,
    *,
    chat_template: str | None = None,
    max_length: int | None = None,
) -> TextTokenizer:
    config = MagicMock()
    config.tokenizer_impl = tokenizer_impl
    config.draft_model = None
    return TextTokenizer(
        str(checkpoint),
        config,
        max_length=max_length,
        chat_template=chat_template,
    )


def _hi() -> list[TextGenerationRequestMessage]:
    return [TextGenerationRequestMessage(role="user", content="hi")]


async def _ids(
    tokenizer: TextTokenizer, messages: list[TextGenerationRequestMessage]
) -> list[int]:
    _, ids = await tokenizer._generate_prompt_and_token_ids(
        None, messages, None
    )
    return np.asarray(ids).tolist()


_HF_HI = [
    len(_BYTES),
    _BYTES.index(ord("h")),
    _BYTES.index(ord("i")),
    len(_BYTES) + 1,
]


@pytest.mark.asyncio
async def test_unset_encodes_with_hf(checkpoint: Path) -> None:
    tokenizer = _tokenizer(checkpoint, None)
    assert tokenizer._chat_encoder is None
    assert await _ids(tokenizer, _hi()) == _HF_HI
    assert tokenizer.take_chat_encoder_outcomes() == {}


@pytest.mark.asyncio
async def test_a_matching_encoder_encodes_the_hf_render(
    checkpoint: Path,
) -> None:
    tokenizer = _tokenizer(checkpoint, _impl("FakeChatEncoder"))
    assert await _ids(tokenizer, _hi()) == _FAST_IDS
    assert FakeChatEncoder.prompts == ["<|user|>hi<|assistant|>"]
    assert isinstance(tokenizer, ChatEncoderOutcomesProbe)
    assert tokenizer.take_chat_encoder_outcomes() == {"custom": 1}
    assert tokenizer.take_chat_encoder_outcomes() == {}


@pytest.mark.asyncio
async def test_a_chat_template_override_is_what_it_encodes(
    checkpoint: Path,
) -> None:
    override = "{% for m in messages %}{{ m.content }}!{% endfor %}"
    tokenizer = _tokenizer(
        checkpoint, _impl("FakeChatEncoder"), chat_template=override
    )
    assert await _ids(tokenizer, _hi()) == _FAST_IDS
    assert FakeChatEncoder.prompts == ["hi!"]


@pytest.mark.asyncio
async def test_lone_surrogates_reach_the_encoder_replaced(
    checkpoint: Path,
) -> None:
    tokenizer = _tokenizer(checkpoint, _impl("FakeChatEncoder"))
    messages = [TextGenerationRequestMessage(role="user", content="a\ud83d")]
    await _ids(tokenizer, messages)
    assert FakeChatEncoder.prompts == ["<|user|>a�<|assistant|>"]


@pytest.mark.parametrize(
    "impl",
    [
        "no_such_module:FakeChatEncoder",
        "no_class_separator",
        _impl("NotAnEncoder"),
        _impl("OtherTokenizerJson"),
        _impl("DigestRaises"),
    ],
)
def test_an_encoder_that_cannot_serve_is_not_used(
    checkpoint: Path, impl: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        assert _tokenizer(checkpoint, impl)._chat_encoder is None
    (record,) = caplog.records
    assert "HuggingFace tokenizer" in record.getMessage()


@pytest.mark.parametrize(
    "tokenizer_config",
    [
        # Another tokenizer.json than the one the encoder claims.
        {},
        # tokenizer_config.json adding a special token, or splitting them.
        {"extra_special_tokens": ["<x>"]},
        {"split_special_tokens": True},
    ],
    ids=["normalizer", "extra_special_token", "split_special_tokens"],
)
def test_a_checkpoint_that_encodes_otherwise_is_not_served(
    tmp_path: Path, tokenizer_config: dict[str, object]
) -> None:
    tokenizer_json = (
        _tokenizer_json({"type": "Lowercase"}) if not tokenizer_config else None
    )
    path = _checkpoint(tmp_path, tokenizer_json, **tokenizer_config)
    assert _tokenizer(path, _impl("FakeChatEncoder"))._chat_encoder is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("impl", "raises"),
    [("FakeChatEncoder", RuntimeError("boom")), ("TornOutput", None)],
    ids=["raises", "not_whole_u32s"],
)
async def test_an_encoder_error_falls_back_to_hf(
    checkpoint: Path,
    impl: str,
    raises: Exception | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tokenizer = _tokenizer(checkpoint, _impl(impl))
    FakeChatEncoder.raises = raises
    with caplog.at_level(logging.DEBUG, logger="max.pipelines"):
        assert await _ids(tokenizer, _hi()) == _HF_HI
        assert tokenizer.take_chat_encoder_outcomes() == {"fallback": 1}
        assert await _ids(tokenizer, _hi()) == _HF_HI
    records = [r for r in caplog.records if "HuggingFace encodes" in r.message]
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG]
    assert records[0].exc_info


@pytest.mark.asyncio
async def test_a_chat_hf_cannot_render_never_reaches_the_encoder(
    checkpoint: Path,
) -> None:
    tokenizer = _tokenizer(checkpoint, _impl("FakeChatEncoder"))
    messages = [
        TextGenerationRequestMessage(role="user", content=[ImageContentPart()])
    ]
    with pytest.raises(ValueError):
        await _ids(tokenizer, messages)
    assert FakeChatEncoder.prompts == []
    assert tokenizer.take_chat_encoder_outcomes() == {}


@pytest.mark.asyncio
async def test_a_raw_prompt_skips_the_encoder(checkpoint: Path) -> None:
    tokenizer = _tokenizer(checkpoint, _impl("FakeChatEncoder"))
    _, ids = await tokenizer._generate_prompt_and_token_ids("hi", [], None)
    assert np.asarray(ids).tolist() != _FAST_IDS
    assert FakeChatEncoder.prompts == []
    assert tokenizer.take_chat_encoder_outcomes() == {}


@pytest.mark.asyncio
async def test_a_prompt_past_max_length_is_refused(checkpoint: Path) -> None:
    tokenizer = _tokenizer(
        checkpoint, _impl("FakeChatEncoder"), max_length=len(_FAST_IDS) - 1
    )
    with pytest.raises(PromptTooLongError):
        await _ids(tokenizer, _hi())


@pytest.mark.asyncio
async def test_pickling_drops_the_encoder(checkpoint: Path) -> None:
    # The pipeline factory pickles the tokenizer into the model worker.
    tokenizer = _tokenizer(checkpoint, _impl("FakeChatEncoder"))
    clone: Any = pickle.loads(ForkingPickler.dumps(tokenizer))
    assert clone._chat_encoder is None
    assert isinstance(tokenizer._chat_encoder, FakeChatEncoder)
    assert await _ids(clone, _hi()) == _HF_HI
