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

"""Tests the Phi-3 graph-API registration."""

from __future__ import annotations

from unittest.mock import NonCallableMock, patch

from max.pipelines.architectures.llama3.decoder import (
    Llama3Decoder,
    Llama3DecoderConfig,
)
from max.pipelines.architectures.phi3 import Phi3Tokenizer, phi3_arch
from max.pipelines.lib import GraphRole, TextTokenizer
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
)


def test_registers_the_llama3_decoder() -> None:
    assert dict(phi3_arch.modules) == {GraphRole.DECODER: Llama3Decoder}
    assert phi3_arch.config is Llama3DecoderConfig
    model = phi3_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)


def test_tokenizer_enables_the_llama_whitespace_fix_by_default() -> None:
    assert phi3_arch.tokenizer is Phi3Tokenizer
    with patch.object(TextTokenizer, "__init__", return_value=None) as init:
        Phi3Tokenizer("repo", NonCallableMock(), max_length=8)
    assert init.call_args.kwargs["enable_llama_whitespace_fix"] is True
    assert init.call_args.kwargs["max_length"] == 8
