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
"""Tests the registered GPT OSS decoder."""

from __future__ import annotations

from max.pipelines.architectures.gpt_oss import GptOssDecoder, gpt_oss_arch
from max.pipelines.lib import AlwaysSignalBuffersMixin, GraphRole
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderBatchProcessor,
    GraphApiDecoderPipelineModel,
    graph_name,
    supports_data_parallel,
)


def test_gpt_oss_registers_its_decoder() -> None:
    assert dict(gpt_oss_arch.modules) == {GraphRole.DECODER: GptOssDecoder}
    model = gpt_oss_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    # Every device takes signals, even alone.
    assert issubclass(model, AlwaysSignalBuffersMixin)
    assert model.batch_processor_cls is GraphApiDecoderBatchProcessor
    assert not supports_data_parallel(GptOssDecoder)


def test_gpt_oss_keeps_its_graph_name() -> None:
    assert graph_name(GptOssDecoder) == "gpt_oss"
