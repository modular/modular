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

"""Tests the OLMo 2 graph-API registration."""

from __future__ import annotations

from max.pipelines.architectures.llama3.decoder import Llama3DecoderConfig
from max.pipelines.architectures.olmo2 import Olmo2Config, olmo2_arch
from max.pipelines.architectures.olmo2.decoder import Olmo2Decoder
from max.pipelines.lib import GraphRole
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
    graph_name,
)


def test_registers_the_olmo2_decoder() -> None:
    assert dict(olmo2_arch.modules) == {GraphRole.DECODER: Olmo2Decoder}
    model = olmo2_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert model.decoder_cls is Olmo2Decoder
    # Precompiled MEFs are keyed by the graph name the pipeline model used.
    assert graph_name(Olmo2Decoder) == "olmo2"


def test_config_finalizes_with_the_pipeline_outputs() -> None:
    # The framework finalizes with a ModelOutputConfig, which the Llama 3
    # decoder config accepts.
    assert issubclass(Olmo2Config, Llama3DecoderConfig)
    assert olmo2_arch.config is Olmo2Config
