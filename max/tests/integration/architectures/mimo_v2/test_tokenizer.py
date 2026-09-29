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
"""Tests the MiMo-V2 tokenizer's stop tokens."""

from __future__ import annotations

import pytest
from max.pipelines.architectures.mimo_v2.tokenizer import (
    generation_eos_token_ids,
)
from transformers import GenerationConfig


@pytest.mark.parametrize(
    "eos_token_id, expected",
    [
        # The published generation_config.json's list.
        ([151643, 151645, 151672], {151643, 151645, 151672}),
        (151645, {151645}),
        (None, set()),
    ],
)
def test_generation_eos_token_ids(
    eos_token_id: int | list[int] | None, expected: set[int]
) -> None:
    config = GenerationConfig(eos_token_id=eos_token_id)
    assert generation_eos_token_ids(config) == expected
