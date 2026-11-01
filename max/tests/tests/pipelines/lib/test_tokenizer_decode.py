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
"""The base tokenizers decode every token form their callers pass.

``max generate`` hands ``TextGenerationOutput.tokens`` (a ``list[int]``)
straight to ``decode``, log-probability responses pass a single ``int``, and
the serving paths pass NumPy arrays. Every architecture that inherits
``decode`` from a base tokenizer relies on all three decoding the same way.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from max.pipelines.lib.tokenizer import TextAndVisionTokenizer, TextTokenizer


class _StubDelegate:
    """Stands in for the HuggingFace tokenizer, which takes Python ids."""

    def __len__(self) -> int:
        return 1000

    def decode(self, ids: int | list[int], **kwargs: Any) -> str:
        assert isinstance(ids, (int, list))
        return "".join(f"[{i}]" for i in np.atleast_1d(ids).tolist())


def _tokenizer(cls: type[Any]) -> Any:
    """Builds a tokenizer without loading a HuggingFace checkpoint."""
    tokenizer = object.__new__(cls)
    tokenizer.delegate = _StubDelegate()
    tokenizer._enable_llama_whitespace_fix = False
    return tokenizer


def _decode(tokenizer: Any, encoded: Any) -> str:
    return asyncio.run(tokenizer.decode(encoded))


@pytest.mark.parametrize("cls", [TextTokenizer, TextAndVisionTokenizer])
def test_decode_accepts_token_list(cls: type[Any]) -> None:
    tokenizer = _tokenizer(cls)
    tokens = [100, 200, 300]

    from_list = _decode(tokenizer, tokens)
    from_array = _decode(tokenizer, np.array(tokens, dtype=np.int64))

    assert from_list == from_array == "[100][200][300]"


@pytest.mark.parametrize("cls", [TextTokenizer, TextAndVisionTokenizer])
def test_decode_accepts_single_token_id(cls: type[Any]) -> None:
    tokenizer = _tokenizer(cls)

    assert _decode(tokenizer, 42) == _decode(tokenizer, np.array(42)) == "[42]"
