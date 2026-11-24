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
"""Pins the dtypes of the weights a fused ModuleV3 spec-decode graph takes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn import Module, module_dataclass
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _Modules,
    _SpecDecodeModule,
    _SpecDecodeModuleV3PipelineModel,
)
from max.pipelines.speculative import spec_adapters
from max.pipelines.weights import AUTO_CAST_ENV_VAR
from max.tensor import Tensor


class _Stop(Exception):
    pass


@module_dataclass
class _Weight(Module[[], Tensor]):
    weight: Tensor

    def forward(self) -> Tensor:
        return self.weight


def _bfloat16_module() -> _Weight:
    # A driver buffer backs the parameter, so building it runs no graph.
    return _Weight(Tensor(storage=Buffer.zeros([2, 2], DType.bfloat16, CPU())))


def _float32(name: str) -> WeightData:
    return WeightData.from_numpy(np.eye(2, dtype=np.float32), name)


def test_fused_weights_compile_in_their_parameters_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights that entries name straight from a float32 checkpoint compile
    as the bfloat16 parameters they bind, as checkpoint-named ones do."""
    monkeypatch.delenv(AUTO_CAST_ENV_VAR, raising=False)
    # A virtual device casts to an uninitialized buffer, without compiling
    # the cast.
    monkeypatch.setattr(
        "max.graph.weights.weights.is_virtual_device_mode", lambda: True
    )
    cls = _SpecDecodeModuleV3PipelineModel
    modules = _Modules(
        target=_bfloat16_module(),
        target_config=MagicMock(),
        draft=_bfloat16_module(),
        draft_config=MagicMock(),
        hidden_states=ReturnHiddenStates.LAST,
        weights={
            "target.weight": _float32("target.weight"),
            "draft.weight": _float32("draft.weight"),
        },
    )
    compiled_with: dict[str, Any] = {}

    def compile(self: object, *input_types: object, **kwargs: Any) -> None:
        compiled_with.update(kwargs["weights"])
        raise _Stop

    monkeypatch.setattr(
        spec_decode_pipeline_model,
        "start_checkpoint_prefetch",
        lambda *configs: None,
    )
    monkeypatch.setattr(spec_adapters, "signal_buffer_index", lambda arch: None)
    monkeypatch.setattr(cls, "arch_config_as", lambda self, kind: MagicMock())
    monkeypatch.setattr(cls, "_build_modules", lambda self, session: modules)
    monkeypatch.setattr(
        _SpecDecodeModule, "driver_call", lambda self: MagicMock()
    )
    monkeypatch.setattr(_SpecDecodeModule, "compile", compile)
    model = object.__new__(cls)
    model.pipeline_config = MagicMock()
    model.device_refs = []
    model.__dict__["pairing"] = MagicMock()
    model.__dict__["takes_signal_buffers"] = False

    with pytest.raises(_Stop):
        model.load_model(MagicMock())

    assert {name: weight.dtype for name, weight in compiled_with.items()} == {
        "target.weight": DType.bfloat16,
        "draft.weight": DType.bfloat16,
    }
