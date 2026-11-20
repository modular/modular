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
"""AutoModel runs text and image rows on Gemma 3 4B's ModuleV3 version.

It registers its own architecture, apart from the graph version.
"""

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
    yield from load("google/gemma-3-4b-it", prefer_module_v3=True)


def test_text_prompts_match_the_tokenizer_built_rows(model: AutoModel) -> None:
    check_text_rows(model)


def test_pil_image_rows_match_the_bytes_rows(model: AutoModel) -> None:
    check_pil_image_rows(model)


def test_image_rows_match_the_tokenizer_built_rows(model: AutoModel) -> None:
    check_image_rows(model)
