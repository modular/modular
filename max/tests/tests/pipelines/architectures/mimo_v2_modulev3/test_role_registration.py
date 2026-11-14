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
"""Checks MiMo's role boundary without compiling a model."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, NonCallableMock

import pytest
from max.pipelines.architectures.mimo_v2_modulev3.arch import (
    mimo_v2_modulev3_arch,
)
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2Decoder
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    ModuleV3DecoderPipelineModel,
)
from max.pipelines.lib.role_contracts import DecoderInputs
from max.tensor import Tensor


def _tensor() -> Tensor:
    return cast(Tensor, NonCallableMock(spec=Tensor))


@pytest.mark.parametrize("data_parallel", [False, True])
@pytest.mark.parametrize("with_ep", [False, True])
def test_semantic_inputs_preserve_the_positional_abi(
    data_parallel: bool,
    with_ep: bool,
) -> None:
    tokens, count, offsets, splits, host, kv0, kv1, ep = (
        _tensor() for _ in range(8)
    )
    inputs = DecoderInputs[Tensor](
        tokens=tokens,
        return_n_logits=count,
        input_row_offsets=offsets,
        data_parallel_splits=splits if data_parallel else None,
        host_input_row_offsets=host if data_parallel else None,
        kv=(kv0, kv1),
        ep=(ep,) if with_ep else None,
    )
    decoder: Any = SimpleNamespace(dp_attention=data_parallel, _forward=Mock())
    MiMoV2Decoder.forward(decoder, inputs)
    expected = (
        tokens,
        count,
        offsets,
        *((splits, host) if data_parallel else ()),
        kv0,
        kv1,
        *((ep,) if with_ep else ()),
    )
    decoder._forward.assert_called_once_with(*expected)


def test_registration_uses_the_shared_decoder_batch_processor() -> None:
    model = mimo_v2_modulev3_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.batch_processor_cls is DecoderBatchProcessor
    assert model.decoder_cls is MiMoV2Decoder
    # Set by module_pipeline_model's class dict, not declared on the base.
    assert vars(model)["ep_dispatch_token_layout"] == "dp_replicated"
    assert MiMoV2Decoder.input_layout == (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "data_parallel_splits",
        "host_input_row_offsets",
        "kv",
        "ep",
    )
