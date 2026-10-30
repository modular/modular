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

"""Tests the Granite ModuleV3 registration."""

from __future__ import annotations

from max.pipelines.architectures.granite_modulev3 import granite_modulev3_arch
from max.pipelines.architectures.llama3_modulev3 import (
    Llama3Decoder,
    Llama3DecoderConfig,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    graph_name,
)


def test_registers_the_llama3_decoder_module() -> None:
    # Granite is Llama 3 plus scalar multipliers, which Llama3Config reads
    # from the checkpoint config.
    assert dict(granite_modulev3_arch.modules) == {
        GraphRole.DECODER: Llama3Decoder
    }
    assert granite_modulev3_arch.config is Llama3DecoderConfig
    assert granite_modulev3_arch.pipeline_model is not None
    assert issubclass(
        granite_modulev3_arch.pipeline_model, ModuleV3DecoderPipelineModel
    )
    # The graph name keys precompiled MEFs; it predates registration.
    assert graph_name(Llama3Decoder) == "Llama3"
