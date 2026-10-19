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
"""Maps tokenizer ids to the compressed vocabulary the Engram hash uses.

Tokens whose text normalizes to the same string share one compressed id. The
mapping must be reproduced exactly, since the hash of every n-gram depends
on it.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from tokenizers import Regex, normalizers


class BackendTokenizerLike(Protocol):
    """The slice of a Rust ``tokenizers`` backend this module calls.

    Public so test stand-ins can annotate ``backend_tokenizer`` with it
    (mypy checks Protocol data attributes invariantly).
    """

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str: ...
    def id_to_token(self, token_id: int) -> str: ...


class TokenizerLike(Protocol):
    """The slice of an outer tokenizer this module calls."""

    backend_tokenizer: BackendTokenizerLike

    def __len__(self) -> int: ...


def build_compressed_token_map(
    tokenizer: TokenizerLike,
) -> tuple[list[int], int]:
    """Maps token ids to compressed ids that merge tokens normalizing alike.

    Each id is decoded with the Rust backend and normalized (NFKC, NFD,
    strip accents, lowercase, collapse whitespace, strip); compressed ids
    are assigned in first-seen order. A partial UTF-8 byte token (decode
    contains U+FFFD) is keyed by its raw token string.

    Args:
        tokenizer: A tokenizer with ``__len__`` and a ``backend_tokenizer``.

    Returns:
        ``(token_map, compressed_vocab_size)``, where ``token_map[token_id]``
        is the compressed id for every ``token_id < len(tokenizer)``.
    """
    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of merging into the empty string.
    sentinel = ""
    # The tokenizers stubs declare no constructor for ``Sequence``, though
    # it takes the list of normalizers to run in order.
    sequence: Any = normalizers.Sequence
    normalizer: normalizers.Normalizer = sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with (no
    # clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # A partial UTF-8 byte token: key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized or text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compressed_token_map_array(
    tokenizer: TokenizerLike,
) -> npt.NDArray[np.int32]:
    """:func:`build_compressed_token_map`'s lookup as an int32 array.

    Returns:
        ``[len(tokenizer)]`` int32 compressed ids, indexed by token id.
    """
    lookup, _compressed_vocab_size = build_compressed_token_map(tokenizer)
    return np.asarray(lookup, dtype=np.int32)
