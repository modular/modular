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
"""Tests for structured output support with TikToken-based tokenizers.

Verifies that the _TikTokenAdapter correctly wraps TikToken tokenizers
(like Kimi K2.5's TikTokenTokenizer) for use with llguidance structured
output / grammar-guided decoding.
"""

import json

import hf_repo_lock
import llguidance
import llguidance.numpy
import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.lib.pipeline_variants.utils import _TikTokenAdapter
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

KIMI_K25_HF_REPO_ID = "nvidia/Kimi-K2.5-NVFP4"


@pytest.fixture(scope="module")
def kimi_tokenizer() -> PreTrainedTokenizerBase:
    """Load Kimi K2.5 tokenizer (TikTokenTokenizer)."""
    revision = hf_repo_lock.revision_for_hf_repo(KIMI_K25_HF_REPO_ID)
    return AutoTokenizer.from_pretrained(
        KIMI_K25_HF_REPO_ID,
        revision=revision,
        trust_remote_code=True,
    )


def test_tiktoken_adapter_accepts_kimi_tokenizer(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify _TikTokenAdapter successfully wraps Kimi's tokenizer."""

    # First, verify Kimi K2.5 uses TikTokenTokenizer, not PreTrainedTokenizerFast.
    assert "TikToken" in type(kimi_tokenizer).__name__
    assert not isinstance(kimi_tokenizer, PreTrainedTokenizerFast)

    adapter = _TikTokenAdapter(kimi_tokenizer)

    assert adapter.eos_token_id == kimi_tokenizer.eos_token_id
    assert adapter.bos_token_id == kimi_tokenizer.bos_token_id
    assert len(adapter.tokens) == len(kimi_tokenizer)
    assert all(isinstance(t, bytes) for t in adapter.tokens)


def test_tiktoken_adapter_encodes_special_tokens(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify adapter correctly encodes text containing special tokens."""
    adapter = _TikTokenAdapter(kimi_tokenizer)

    # Encode text with a special token
    text_with_special = "Hello <|im_end|> world"
    encoded = adapter(text_with_special)

    # Verify encoding worked and special token was recognized
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    # Decode and verify round-trip
    decoded = kimi_tokenizer.decode(encoded)
    assert "<|im_end|>" in decoded or "im_end" in decoded


def test_tiktoken_adapter_handles_bytes_input(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify adapter correctly handles bytes input."""
    adapter = _TikTokenAdapter(kimi_tokenizer)

    text = "Hello world"
    encoded_from_str = adapter(text)
    encoded_from_bytes = adapter(text.encode("utf-8"))

    assert encoded_from_str == encoded_from_bytes


def test_llguidance_integration_with_kimi(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify full llguidance integration works with Kimi's TikToken tokenizer."""
    # Create the adapter chain
    adapter = _TikTokenAdapter(kimi_tokenizer)
    wrapper = TokenizerWrapper(adapter)
    vocab_size = len(kimi_tokenizer)
    ll_tokenizer = LLTokenizer(wrapper, n_vocab=vocab_size)

    # Create a JSON schema grammar
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    grammar = LLMatcher.grammar_from_json_schema(json.dumps(json_schema))
    matcher = LLMatcher(ll_tokenizer, grammar)

    # Allocate and fill bitmask
    bitmask = llguidance.numpy.allocate_token_bitmask(1, vocab_size)
    llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, index=0)

    # Verify bitmask has expected shape and contains data
    assert bitmask.shape == (1, (vocab_size + 31) // 32)
    # Bitmask should have some bits set (allowed tokens for JSON start)
    assert bitmask.any()


def test_tiktoken_adapter_rejects_non_tiktoken() -> None:
    """Verify _TikTokenAdapter raises ValueError for non-TikToken tokenizers."""
    # Load a fast tokenizer (not TikToken)
    fast_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert isinstance(fast_tokenizer, PreTrainedTokenizerFast)

    with pytest.raises(ValueError, match="Structured output requires"):
        _TikTokenAdapter(fast_tokenizer)
