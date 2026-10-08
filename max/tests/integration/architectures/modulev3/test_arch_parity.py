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
"""Request-handling registration parity between graph and ModuleV3 arches.

The ModuleV3 arch must default the same serving-layer request-handling
features as its graph counterpart: tool-call parsing, reasoning parsing, and
the structured-output (constrained decoding) backend. The parsers themselves
live at the serving layer and are shared; only the ``SupportedArchitecture``
defaults differ per registration.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures.deepseekV3_2.arch import deepseekV3_2_arch
from max.pipelines.architectures.deepseekV3_2_modulev3.arch import (
    deepseekV3_2_modulev3_arch,
)
from max.pipelines.architectures.gemma4.arch import (
    gemma4_arch,
    gemma4_unified_arch,
)
from max.pipelines.architectures.gemma4_modulev3.arch import (
    gemma4_modulev3_arch,
    gemma4_unified_modulev3_arch,
)
from max.pipelines.lib import reasoning, tool_parsing
from max.pipelines.lib.registry import SupportedArchitecture


@pytest.mark.parametrize(
    ("graph_arch", "modulev3_arch"),
    [
        pytest.param(gemma4_arch, gemma4_modulev3_arch, id="gemma4"),
        pytest.param(
            gemma4_unified_arch,
            gemma4_unified_modulev3_arch,
            id="gemma4_unified",
        ),
        pytest.param(
            deepseekV3_2_arch,
            deepseekV3_2_modulev3_arch,
            id="deepseekV3_2",
        ),
    ],
)
def test_request_handling_parity(
    graph_arch: SupportedArchitecture,
    modulev3_arch: SupportedArchitecture,
) -> None:
    """ModuleV3 registrations mirror the graph arch's request-handling defaults."""
    assert modulev3_arch.tool_parser == graph_arch.tool_parser
    assert modulev3_arch.reasoning_parser == graph_arch.reasoning_parser
    assert (
        modulev3_arch.default_structured_output_backend
        == graph_arch.default_structured_output_backend
    )


def test_gemma4_modulev3_default_parsers_are_registered() -> None:
    """Importing the ModuleV3 arch registers the parsers it names.

    Parser registration is an import side effect (the gemma4 package's
    ``__init__`` imports ``tool_parser``/``reasoning``); a missing import
    would surface as "Unknown ... parser" per request at serve time.
    """
    assert isinstance(gemma4_modulev3_arch.tool_parser, str)
    assert (
        tool_parsing.get_parser_cls(gemma4_modulev3_arch.tool_parser)
        is not None
    )
    assert (
        reasoning.get_parser_cls(gemma4_modulev3_arch.reasoning_parser)
        is not None
    )


def test_deepseekv3_2_modulev3_registration_parity() -> None:
    """The V3.2 ModuleV3 arch mirrors the graph arch's serving contract.

    Importing it here also exercises the package's import graph: the arch is
    registered lazily, so a bad import would otherwise only surface the first
    time someone tries to serve the model.
    """
    assert deepseekV3_2_modulev3_arch.name == "DeepseekV32ForCausalLM_ModuleV3"
    assert (
        deepseekV3_2_modulev3_arch.default_encoding
        == deepseekV3_2_arch.default_encoding
    )
    assert (
        deepseekV3_2_modulev3_arch.supported_encodings
        == deepseekV3_2_arch.supported_encodings
    )
    # Sparse MLA shards the same way the graph arch does; a V3 port that
    # silently dropped multi-GPU would not fit the model on any node.
    assert deepseekV3_2_modulev3_arch.multi_gpu_supported
    assert (
        deepseekV3_2_modulev3_arch.example_repo_ids
        == deepseekV3_2_arch.example_repo_ids
    )
