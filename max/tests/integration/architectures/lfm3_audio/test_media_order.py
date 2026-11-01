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

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.architectures.lfm3_vl.tokenizer import (
    Frame,
    Lfm3VlTokenizer,
)

_IMAGE, _AUDIO = 1, 2


def _tokenizer() -> Lfm3VlTokenizer:
    tokenizer = object.__new__(Lfm3VlTokenizer)
    ids = {"<image>": _IMAGE, "<audio>": _AUDIO}
    tokenizer.delegate = MagicMock(convert_tokens_to_ids=ids.__getitem__)
    return tokenizer


def _item(n: int) -> tuple[str, list[Frame]]:
    return "", [(np.zeros(1, np.float32), 0, n)]


def test_token_id_media_claim_their_runs_in_order() -> None:
    ids = np.array([0, *[_IMAGE] * 3, 0, *[_AUDIO] * 2, 0])
    frames = _tokenizer()._in_prompt_order(
        ids, {"<image>": [_item(3)], "<audio>": [_item(2)]}
    )
    assert [n for *_, n in frames] == [3, 2]


def test_a_run_short_of_its_media_is_refused() -> None:
    ids = np.array([*[_IMAGE] * 3, _AUDIO, _IMAGE, _AUDIO])
    with pytest.raises(ValueError, match="fewer media placeholders"):
        _tokenizer()._in_prompt_order(
            ids, {"<image>": [_item(4)], "<audio>": [_item(2)]}
        )
