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
"""Pins how a ModuleV3 vision-language model registers and loads its towers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.nn import Module as ModuleV3
from max.experimental.nn import module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.pipelines.lib import (
    GraphRole,
    GraphSpec,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
)

X = TensorType(DType.float32, [2], device=DeviceRef.CPU())


@module_dataclass
class Scale(ModuleV3[[Tensor], Tensor]):
    """Multiplies its input by a weight."""

    weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


def _tower(name: str, state_dict: dict[str, Any]) -> GraphSpec:
    module = Scale(Tensor.zeros([2], dtype=DType.float32, device=CPU()))
    return GraphSpec(module, name, [X], weights=state_dict)


class _Model(ModuleV3MultiGraphPipelineModelWithKVCache[Any]):
    has_vision: bool

    def _load_state_dict(self) -> dict[str, Any]:
        self._vision_weights_dict = {"weight": np.full(2, 3.0, np.float32)}
        self._language_weights_dict = {"weight": np.full(2, 5.0, np.float32)}
        return {}

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        return SimpleNamespace()

    def _vision_spec(
        self, model_config: Any, state_dict: dict[str, Any]
    ) -> GraphSpec | None:
        return _tower("vision", state_dict) if self.has_vision else None

    def _language_spec(
        self, model_config: Any, state_dict: dict[str, Any]
    ) -> GraphSpec:
        return _tower("language", state_dict)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


def _model(has_vision: bool) -> _Model:
    model = object.__new__(_Model)
    model._batch_processor = None
    model.has_vision = has_vision
    return model


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


def _run(tower: Any) -> list[float]:
    out = tower(Buffer.from_numpy(np.ones(2, dtype=np.float32)))
    return out.to_numpy().tolist()


def test_the_vision_tower_registers_before_the_decoder() -> None:
    model = _model(has_vision=True)
    model._load_state_dict()

    specs = model.graph_specs(SimpleNamespace())

    assert list(specs) == [GraphRole.VISION, GraphRole.DECODER]
    assert specs[GraphRole.VISION].weights is model._vision_weights_dict
    assert specs[GraphRole.DECODER].weights is model._language_weights_dict


def test_each_tower_runs_with_its_own_weights(
    session: InferenceSession,
) -> None:
    vision, language = _model(has_vision=True)._load_towers(session)

    assert vision is not None
    assert _run(vision) == [3.0, 3.0]
    assert _run(language) == [5.0, 5.0]


def test_a_checkpoint_without_a_vision_tower_loads_the_decoder_alone(
    session: InferenceSession,
) -> None:
    model = _model(has_vision=False)

    vision, language = model._load_towers(session)

    assert vision is None
    assert list(model.compiled_modules) == [GraphRole.DECODER]
    assert _run(language) == [5.0, 5.0]
