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
"""Tests the ModuleV3 Laguna registration mirrors the graph one.

The parsers are shared serving-layer code; only the ``SupportedArchitecture``
defaults differ per registration, and a drift there changes what a request
returns depending on which architecture happened to serve it.
"""

from __future__ import annotations

from max.pipelines.architectures.laguna.arch import laguna_arch
from max.pipelines.architectures.laguna_modulev3.arch import (
    laguna_modulev3_arch,
)
from max.pipelines.lib import reasoning, tool_parsing


def test_request_handling_matches_the_graph_architecture() -> None:
    assert laguna_modulev3_arch.name == f"{laguna_arch.name}_ModuleV3"
    assert laguna_modulev3_arch.tool_parser == laguna_arch.tool_parser
    assert laguna_modulev3_arch.reasoning_parser == laguna_arch.reasoning_parser
    assert laguna_modulev3_arch.tokenizer is laguna_arch.tokenizer
    assert laguna_modulev3_arch.config is laguna_arch.config
    assert (
        laguna_modulev3_arch.supported_encodings
        == laguna_arch.supported_encodings
    )


def test_named_parsers_are_registered() -> None:
    assert isinstance(laguna_modulev3_arch.tool_parser, str)
    assert (
        tool_parsing.get_parser_cls(laguna_modulev3_arch.tool_parser)
        is not None
    )
    assert (
        reasoning.get_parser_cls(laguna_modulev3_arch.reasoning_parser)
        is not None
    )
