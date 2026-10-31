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

"""Tests the Granite graph-API registration."""

from __future__ import annotations

from max.pipelines.architectures.granite import granite_arch
from max.pipelines.architectures.llama3.decoder import (
    Llama3Decoder,
    Llama3DecoderConfig,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
)


def test_registers_the_llama3_decoder() -> None:
    # Granite is Llama 3 plus scalar multipliers, which Llama3Config reads
    # from the checkpoint config.
    assert dict(granite_arch.modules) == {GraphRole.DECODER: Llama3Decoder}
    assert granite_arch.config is Llama3DecoderConfig
    model = granite_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
