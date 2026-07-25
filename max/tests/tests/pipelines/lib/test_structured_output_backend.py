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
"""Tests for the structured-output grammar backends."""

import json
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.modeling.types import PipelineTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

_N_VOCAB = 256

# An ordinary printable byte ("~") the runtime additionally stops on. The
# declared EOS (byte 0) is deliberately distinct: the runtime EOS set layers
# extra terminators (chat turn-end tokens and the like) on top of the
# declared EOS, which is exactly the split the grammar backend's stop set
# has to respect. Unless the backend registers the extra terminator as a
# stop token, the grammar admits it as ordinary string content.
_EXTRA_TERMINATOR_ID = 126


class _FakeTikTokenTokenizer:
    """Prod-shaped TikToken delegate over a byte-level vocab."""

    eos_token_id: int = 0
    bos_token_id: int | None = None
    all_special_ids: list[int] = []

    def __init__(self) -> None:
        self.byte_decoder = {chr(b): b for b in range(256)}

    def __len__(self) -> int:
        return _N_VOCAB

    def get_vocab(self) -> dict[str, int]:
        return {chr(i): i for i in range(256)}

    def convert_ids_to_tokens(self, idx: int) -> str:
        return chr(idx)

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return [ord(c) for c in text]


def _fast_tokenizer_delegate() -> PreTrainedTokenizerFast:
    """``PreTrainedTokenizerFast`` delegate over the same byte-level vocab."""
    vocab = {chr(i): i for i in range(256)}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token=chr(1))),
        eos_token=chr(0),
        unk_token=chr(1),
    )


def _allowed_tokens(backend: Any, matcher: Any) -> np.ndarray:
    """Bool ``[vocab]`` mask of the tokens ``matcher`` currently allows."""
    packed = backend.allocate_token_bitmask(1, _N_VOCAB)
    backend.fill_next_token_bitmask(matcher, packed, 0)
    masks = np.int32(1) << np.arange(32, dtype=np.int32)
    bits = (packed[..., np.newaxis] & masks) != 0
    return bits.reshape(*packed.shape[:-1], -1)[0, :_N_VOCAB]


@pytest.mark.parametrize(
    "delegate_factory",
    [_FakeTikTokenTokenizer, _fast_tokenizer_delegate],
    ids=["tiktoken", "hf_fast"],
)
def test_xgrammar_stop_tokens_cover_runtime_eos_set(
    delegate_factory: Callable[[], Any],
) -> None:
    """The grammar's stop set must match the runtime EOS set, not just EOS."""
    delegate = delegate_factory()
    runtime_eos = {delegate.eos_token_id, _EXTRA_TERMINATOR_ID}
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = runtime_eos

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="xgrammar",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )

    # Stop inside the string value: ordinary content bytes are allowed, but
    # no terminator may be sampled, or the runtime would end the request
    # mid-structure (silent truncation).
    for char in '{"a":"x':
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert not matcher.is_accepting()
    mid = _allowed_tokens(helper.backend, matcher)
    assert mid[ord("y")]
    assert not mid[sorted(runtime_eos)].any(), (
        "a terminator is sampleable mid-structure — the runtime would stop "
        "generation inside the constrained response and truncate it"
    )

    for char in '"}':
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()

    done = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert done == runtime_eos, (
        "a completed grammar must permit exactly the runtime EOS set: a "
        "missing terminator forces an unnatural declared-EOS ending, an "
        "extra token would leak unconstrained output"
    )
