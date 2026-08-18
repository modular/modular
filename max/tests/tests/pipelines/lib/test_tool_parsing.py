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

import pytest
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser
from max.pipelines.architectures.minimax_m2.tool_parser import (
    MinimaxM2ToolParser,
)
from max.pipelines.lib.tool_parsing import create
from max.serve.parser.llama_tool_parser import LlamaToolParser


def test_create_returns_registered_llama_parser() -> None:
    assert isinstance(create("llama"), LlamaToolParser)


def test_create_returns_registered_kimi_parser() -> None:
    assert isinstance(create("kimik2_5"), KimiToolParser)


def test_create_returns_registered_minimax_parser() -> None:
    assert isinstance(create("minimax_m2"), MinimaxM2ToolParser)


def test_create_unknown_parser_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tool parser"):
        create("unknown_parser")


def test_llama_parser_parse_complete_smoke() -> None:
    parser = create("llama")
    parsed = parser.parse_complete(
        'text {"name":"get_weather","parameters":{"location":"Boston"}}'
    )

    assert parsed.content is None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
