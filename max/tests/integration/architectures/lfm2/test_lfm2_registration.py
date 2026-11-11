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
"""Tests LFM2's registration."""

from __future__ import annotations

from max.pipelines.architectures.lfm2 import LFM2, lfm2_arch
from max.pipelines.lib import GraphRole
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
)


def test_lfm2_registers_its_decoder() -> None:
    model = lfm2_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert lfm2_arch.modules == {GraphRole.DECODER: LFM2}
    assert LFM2.graph_name == "lfm2"
