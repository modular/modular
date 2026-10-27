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
"""CPU tests for the ModuleV3 Engram token-map adapter.

The builder is main's; these tests pin the behavior the ModuleV3 loader
relies on (normalization collisions, partial-UTF-8 byte tokens, the
array-with-size entry point and the recorded digest assertion) using a
hand-built stand-in backend tokenizer, which the Protocol supports by
design.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41_modulev3.layers.engram_token_map import (
    EXPECTED_COMPRESSED_VOCAB_SIZE,
    EXPECTED_MAP_SHA256,
    BackendTokenizerLike,
    build_compressed_token_map,
    check_map_digest,
    compressed_token_map_array,
    compressed_token_map_with_size,
)


class _Standin:
    """A minimal ``TokenizerLike`` with a table-driven backend."""

    def __init__(self, tokens: dict[int, str]) -> None:
        self._tokens = tokens
        self.backend_tokenizer: BackendTokenizerLike = self

    def __len__(self) -> int:
        return len(self._tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return self._tokens[ids[0]]

    def id_to_token(self, token_id: int) -> str:
        # The raw byte-piece string: distinct per token id, as real byte
        # tokens are.
        return f"<byte:{token_id}>"


def _tokenizer() -> _Standin:
    return _Standin(
        {
            0: "The",
            1: " the",
            2: "the",
            3: "Hello",
            4: "hello",
            5: " ",  # exactly one space: must not collapse into ""
            6: "  ",  # two spaces normalize to one space, same as id 5
            7: "\t\n",  # whitespace run, also the same after collapse
            8: "ä",  # NFD strips the accent to "a"
            9: "a",
            10: "�",  # partial UTF-8 byte token: keyed raw, unnormalized
            11: "�",  # a different raw byte token, same decoded text
            12: "Wörld",
            13: "world",
        }
    )


def test_normalization_collapses_ids() -> None:
    lookup, size = build_compressed_token_map(_tokenizer())
    assert size == len(set(lookup))
    # Case, leading space, tabs and accents all merge.
    assert lookup[0] == lookup[1] == lookup[2]  # "The", " the", "the"
    assert lookup[3] == lookup[4]  # "Hello" / "hello"
    assert lookup[5] == lookup[6] == lookup[7]  # one-space survivors
    assert lookup[8] == lookup[9]  # accent-stripped to "a"
    assert lookup[12] == lookup[13]  # "Wörld" -> "world"
    # First-seen order: the earliest token of each class keeps the lowest
    # id, and later distinct classes get strictly higher ids.
    assert lookup[0] == 0
    assert lookup[3] == 1  # "Hello" opens the second class
    assert lookup[5] == 2  # the one-space survivor opens the third
    assert lookup[9] == 3  # "a" (with id 8) opens the fourth


def test_partial_byte_tokens_key_raw() -> None:
    """Two byte tokens with the same U+FFFD decode stay distinct."""
    lookup, size = build_compressed_token_map(_tokenizer())
    # Both decode to U+FFFD, so both are keyed by their raw token string.
    assert lookup[10] != lookup[11]
    # ...and they do not collide with any normalized class.
    assert size == len(set(lookup))


def test_array_forms() -> None:
    tokenizer = _tokenizer()
    array = compressed_token_map_array(tokenizer)
    assert array.dtype == np.int32
    assert array.shape[0] == len(tokenizer)

    with_size, size = compressed_token_map_with_size(tokenizer)
    assert with_size.dtype == np.int32
    np.testing.assert_array_equal(with_size, array)
    assert int(array.max()) + 1 == size


def test_digest_check_rejects_wrong_data() -> None:
    """The recorded digest is the real model's; anything else must fail."""
    wrong = np.zeros(129280, dtype=np.int32)
    with pytest.raises(ValueError, match="sha256"):
        check_map_digest(wrong)
    # The recorded constants are the real map's, not placeholders.
    assert EXPECTED_COMPRESSED_VOCAB_SIZE == 99092
    assert len(EXPECTED_MAP_SHA256) == 64
