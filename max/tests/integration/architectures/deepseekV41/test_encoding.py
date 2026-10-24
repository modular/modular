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
"""Tests flattening multi-part request messages before prompt rendering."""

from __future__ import annotations

from max.pipelines.architectures.deepseekV41.tokenizer import (
    _flatten_message,
    render_prompt,
)
from max.pipelines.modeling.types import TextGenerationRequestMessage

_MULTI_PART_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "first part"},
        {"type": "text", "text": "second part"},
    ],
}


def test_flatten_message_joins_multi_part_text_with_blank_line() -> None:
    """Checks multi-part text joins with a blank line (``\\n\\n``)."""
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)
    flattened = _flatten_message(message)
    assert flattened["content"] == "first part\n\nsecond part"


def test_render_prompt_renders_flattened_multi_part_message() -> None:
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)
    prompt = render_prompt([_flatten_message(message)], None, {})
    assert "first part\n\nsecond part" in prompt
