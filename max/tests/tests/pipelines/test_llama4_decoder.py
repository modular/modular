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
"""Tests the registered Llama 4 decoder."""

from __future__ import annotations

import pytest
from max.pipelines.architectures.llama4 import (
    Llama4Decoder,
    llama4_arch,
    llama4_conditional_arch,
)
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
    graph_name,
)


@pytest.mark.parametrize("arch", [llama4_arch, llama4_conditional_arch])
def test_llama4_registers_its_decoder(arch: SupportedArchitecture) -> None:
    assert dict(arch.modules) == {GraphRole.DECODER: Llama4Decoder}
    model = arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)


def test_llama4_keeps_its_graph_name() -> None:
    assert graph_name(Llama4Decoder) == "llama4"
