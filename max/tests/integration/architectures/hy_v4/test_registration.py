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
"""Checks Hy4's registered decoder graph contract."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import pytest
from max import tree
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.architectures.hy_v4.arch import hy_v4_arch
from max.pipelines.architectures.hy_v4.decoder import HYV4Decoder
from max.pipelines.architectures.hy_v4.model_config import (
    HYV4Config,
    check_parallelism,
)
from max.pipelines.architectures.hy_v4.spec_adapters import verify
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    decoder_input_types,
)
from max.pipelines.lib.registration_runtime import batch_context_length_types


def test_the_registration_uses_the_shared_decoder_runtime() -> None:
    model = hy_v4_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is HYV4Decoder
    assert HYV4Config.expert_parallel_communication is False


def test_the_registered_input_order_keeps_the_prefill_length_before_kv() -> (
    None
):
    cache = BufferType(DType.float32, [2], device=DeviceRef.CPU())
    types = decoder_input_types(
        NonCallableMock(get_symbolic_inputs=lambda: (cache,)),
        DeviceRef.CPU(),
        batch_context_lengths=batch_context_length_types(1),
        input_layout=HYV4Decoder.input_layout,
    )
    leaves = tree.leaves(types)
    assert tuple(types) == HYV4Decoder.input_layout
    assert isinstance(leaves[3], TensorType)
    assert leaves[3].dtype == DType.int32
    assert leaves[3].shape == [1]
    assert leaves[3].device == DeviceRef.CPU()
    assert leaves[4] == cache


@pytest.mark.parametrize("ep_size", [1, 4])
def test_local_expert_sharding_accepts_both_existing_ep_settings(
    ep_size: int,
) -> None:
    pipeline: Any = SimpleNamespace(
        model=SimpleNamespace(data_parallel_degree=1),
        runtime=SimpleNamespace(ep_size=ep_size),
    )
    check_parallelism(pipeline, 4)


@pytest.mark.parametrize(("dp", "ep_size"), [(2, 4), (1, 2)])
def test_local_expert_sharding_rejects_other_multi_device_layouts(
    dp: int, ep_size: int
) -> None:
    pipeline: Any = SimpleNamespace(
        model=SimpleNamespace(data_parallel_degree=dp),
        runtime=SimpleNamespace(ep_size=ep_size),
    )
    with pytest.raises(ValueError, match="tensor-parallel attention"):
        check_parallelism(pipeline, 4)


def test_the_mtp_target_adapter_accepts_the_registered_decoder() -> None:
    target = object.__new__(HYV4Decoder)
    language = NonCallableMock()
    object.__setattr__(target, "language_model", language)
    verified = verify(target, None, ReturnHiddenStates.NONE, None)
    assert verified.target is language
