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
"""Tests the Phi-3 ModuleV3 registration."""

from __future__ import annotations

from unittest.mock import NonCallableMock, patch

from max.pipelines.architectures.llama3_modulev3.llama3 import Llama3Decoder
from max.pipelines.architectures.llama3_modulev3.model_config import (
    Llama3DecoderConfig,
)
from max.pipelines.architectures.phi3_modulev3 import (
    Phi3Tokenizer,
    phi3_modulev3_arch,
)
from max.pipelines.lib import GraphRole, TextTokenizer
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    graph_name,
)


def test_registers_the_llama3_decoder() -> None:
    assert dict(phi3_modulev3_arch.modules) == {
        GraphRole.DECODER: Llama3Decoder
    }
    assert phi3_modulev3_arch.config is Llama3DecoderConfig
    assert phi3_modulev3_arch.pipeline_model is not None
    assert issubclass(
        phi3_modulev3_arch.pipeline_model, ModuleV3DecoderPipelineModel
    )
    # The graph name keys precompiled MEFs; it predates registration.
    assert graph_name(Llama3Decoder) == "Llama3"


def test_tokenizer_enables_the_llama_whitespace_fix_by_default() -> None:
    assert phi3_modulev3_arch.tokenizer is Phi3Tokenizer
    with patch.object(TextTokenizer, "__init__", return_value=None) as init:
        Phi3Tokenizer("repo", NonCallableMock(), max_length=8)
    assert init.call_args.kwargs["enable_llama_whitespace_fix"] is True
    assert init.call_args.kwargs["max_length"] == 8


def test_tokenizer_keeps_an_explicit_whitespace_fix_setting() -> None:
    with patch.object(TextTokenizer, "__init__", return_value=None) as init:
        Phi3Tokenizer(
            "repo", NonCallableMock(), enable_llama_whitespace_fix=False
        )
    assert init.call_args.kwargs["enable_llama_whitespace_fix"] is False
