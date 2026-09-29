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
"""Pins where a ModuleV3 KV-cache model may rewrite its weights."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from max.dtype import DType
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
)


class _Module:
    def compile(self, *input_types: object, weights: object) -> object:
        return weights


class _Model(ModuleV3PipelineModelWithKVCache[Any]):
    configured_from: dict[str, Any]

    def _load_state_dict(self) -> dict[str, Any]:
        return {"weight": "stored"}

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        self.configured_from = dict(state_dict)
        return SimpleNamespace(dtype=DType.float32, weight="prepared")

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        return {"weight": model_config.weight}

    def _instantiate_module(self, model_config: Any) -> Any:
        return _Module()

    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        return ()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


def test_compile_takes_the_weights_the_config_prepared() -> None:
    model = object.__new__(_Model)
    model._lora_manager = None
    model._batch_processor = None

    assert model.load_model() == {"weight": "prepared"}
    assert model.configured_from == {"weight": "stored"}
