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
"""Tests for PreTrainedPipelineTokenizer."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from max.pipelines.lib.tokenizer import PreTrainedPipelineTokenizer


@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_encode_forwards_add_special_tokens(
    add_special_tokens: bool,
) -> None:
    delegate = MagicMock()
    delegate.encode.return_value = [1, 2, 3]
    tok = object.__new__(PreTrainedPipelineTokenizer)  # type: ignore[type-abstract]
    tok.delegate = delegate
    asyncio.run(tok.encode("hello", add_special_tokens=add_special_tokens))
    delegate.encode.assert_called_once_with(
        "hello", add_special_tokens=add_special_tokens
    )
