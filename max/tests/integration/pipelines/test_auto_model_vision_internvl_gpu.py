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
"""AutoModel runs text and image rows on InternVL3-1B."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from _auto_model_vision_gpu import (
    check_image_rows,
    check_pil_image_rows,
    check_text_rows,
    load,
)
from max.pipelines import AutoModel


@pytest.fixture(scope="module")
def model() -> Iterator[AutoModel]:
    yield from load("OpenGVLab/InternVL3-1B-Instruct")


def test_text_prompts_match_the_tokenizer_built_rows(model: AutoModel) -> None:
    check_text_rows(model)


def test_pil_image_rows_match_the_bytes_rows(model: AutoModel) -> None:
    check_pil_image_rows(model)


def test_image_rows_match_the_tokenizer_built_rows(model: AutoModel) -> None:
    check_image_rows(model)
