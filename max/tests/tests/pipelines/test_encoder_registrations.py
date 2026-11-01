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
"""Tests the embedding architectures that register an ``ENCODER``."""

from __future__ import annotations

from typing import Any

import pytest
from max.pipelines.architectures.bert import BertModel, bert_arch
from max.pipelines.architectures.mpnet import MPNetModel, mpnet_arch
from max.pipelines.architectures.mpnet.batch_processor import (
    MPNetBatchProcessor,
)
from max.pipelines.architectures.mpnet_modulev3 import (
    MPNetModel as MPNetModuleV3Model,
)
from max.pipelines.architectures.mpnet_modulev3 import mpnet_modulev3_arch
from max.pipelines.architectures.qwen3_embedding import (
    Qwen3EmbeddingEncoder,
    qwen3_embedding_arch,
)
from max.pipelines.architectures.qwen3_embedding_modulev3 import (
    Qwen3Embedding,
    qwen3_embedding_modulev3_arch,
)
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.lib.registration_encoder import (
    EncoderPipelineModel,
    GraphApiEncoderPipelineModel,
    ModuleV3EncoderPipelineModel,
    PaddedEncoderProcessor,
    RaggedEncoderProcessor,
)

_GRAPH_API = GraphApiEncoderPipelineModel
_MODULEV3 = ModuleV3EncoderPipelineModel


@pytest.mark.parametrize(
    ("arch", "encoder", "graph_name", "batching", "served_by"),
    [
        (bert_arch, BertModel, "bert", PaddedEncoderProcessor, _GRAPH_API),
        (mpnet_arch, MPNetModel, "mpnet", MPNetBatchProcessor, _GRAPH_API),
        (
            qwen3_embedding_arch,
            Qwen3EmbeddingEncoder,
            "qwen3_embedding",
            RaggedEncoderProcessor,
            _GRAPH_API,
        ),
        (
            mpnet_modulev3_arch,
            MPNetModuleV3Model,
            "MPNetModel",
            PaddedEncoderProcessor,
            _MODULEV3,
        ),
        (
            qwen3_embedding_modulev3_arch,
            Qwen3Embedding,
            "Qwen3Embedding",
            RaggedEncoderProcessor,
            _MODULEV3,
        ),
    ],
)
def test_an_embedding_architecture_registers_its_encoder(
    arch: SupportedArchitecture,
    encoder: type[Any],
    graph_name: str,
    batching: type,
    served_by: type,
) -> None:
    assert arch.modules == {GraphRole.ENCODER: encoder}
    model = arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, EncoderPipelineModel)
    assert issubclass(model, served_by)
    assert model.batch_processor_cls is batching
    # The graph keeps the name its precompiled MEFs are keyed by.
    assert encoder.graph_name == graph_name
