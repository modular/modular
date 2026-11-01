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

"""Tests the dense Qwen3 graph-API registration."""

from __future__ import annotations

from max.pipelines.architectures.qwen3 import (
    Qwen3Config,
    Qwen3Decoder,
    Qwen3Model,
    qwen3_arch,
    qwen3_moe_arch,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
    supports_data_parallel,
)


def test_dense_qwen3_registers_its_decoder() -> None:
    assert dict(qwen3_arch.modules) == {GraphRole.DECODER: Qwen3Decoder}
    assert qwen3_arch.config is Qwen3Config
    model = qwen3_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert model.decoder_cls is Qwen3Decoder
    assert supports_data_parallel(Qwen3Decoder)


def test_the_data_parallel_graph_keeps_its_input_order() -> None:
    assert Qwen3Decoder.data_parallel_batch_inputs == (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "host_input_row_offsets",
        "data_parallel_splits",
    )


def test_qwen3_moe_keeps_its_pipeline_model() -> None:
    assert qwen3_moe_arch.pipeline_model is Qwen3Model
