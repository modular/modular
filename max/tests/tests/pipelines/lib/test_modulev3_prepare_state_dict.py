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

import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.nn import Linear
from max.experimental.tensor import default_device
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData
from max.pipelines.lib import (
    GraphRole,
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
)
from max.pipelines.weights import AUTO_CAST_ENV_VAR


class _Module:
    parameters: list[tuple[str, Any]] = []

    def compile(self, *input_types: object, weights: object) -> object:
        return weights


class _Model(ModuleV3PipelineModelWithKVCache[Any]):
    configured_from: dict[str, Any]

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


def test_the_decoder_takes_the_weights_the_config_prepared() -> None:
    model = object.__new__(_Model)
    model._lora_manager = None
    model._batch_processor = None
    # What `_load_state_dict` keeps for `graph_specs`.
    model._checkpoint = {"weight": "stored"}

    specs = model.graph_specs(model._create_model_config(model._checkpoint))

    assert specs[GraphRole.DECODER].weights == {"weight": "prepared"}
    assert specs[GraphRole.DECODER].name == "_Module"
    assert model.configured_from == {"weight": "stored"}


class _BFloat16Model(ModuleV3PipelineModelWithKVCache[Any]):
    """A bfloat16 module whose checkpoint stores float32 weights."""

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        return SimpleNamespace(dtype=DType.bfloat16)

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        return state_dict

    def _instantiate_module(self, model_config: Any) -> Any:
        with default_device(CPU()):
            return Linear(2, 2, bias=False)

    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        return (TensorType(DType.bfloat16, [1, 2], DeviceRef.CPU()),)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


def _float32_checkpoint() -> dict[str, Any]:
    return {
        "weight": WeightData.from_numpy(np.eye(2, dtype=np.float32), "weight")
    }


def _decoder_weight(specs: dict[str, Any], name: str) -> WeightData:
    """The decoder spec's checkpoint weight, narrowed for mypy."""
    weights = specs[GraphRole.DECODER].weights
    assert weights is not None
    # GraphSpec.weights is typed as Mapping[str, DLPackArray]; graph_specs
    # hands it the checkpoint's WeightData objects unchanged.
    return cast(WeightData, weights[name])


def test_float32_checkpoint_loads_into_bfloat16_module(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A float32 checkpoint casts to bfloat16 parameters at graph build.

    Checkpoints saved in float32 (such as ``allenai/OLMo-2-0425-1B``) serve
    as bfloat16 on a GPU, and the cast logs the precision loss.
    """
    monkeypatch.delenv(AUTO_CAST_ENV_VAR, raising=False)
    model = object.__new__(_BFloat16Model)
    model._lora_manager = None
    model._batch_processor = None
    model._checkpoint = _float32_checkpoint()

    with caplog.at_level(
        logging.WARNING, logger="max.experimental.nn._weight_utils"
    ):
        specs = model.graph_specs(model._create_model_config(model._checkpoint))

    assert (
        f"from {DType.float32} to {DType.bfloat16} (precision loss)"
        in caplog.text
    )
    assert _decoder_weight(specs, "weight").dtype == DType.bfloat16


def test_auto_cast_env_var_turns_the_cast_off(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The env var leaves float32 weights untouched."""
    monkeypatch.setenv(AUTO_CAST_ENV_VAR, "false")
    model = object.__new__(_BFloat16Model)
    model._lora_manager = None
    model._batch_processor = None
    model._checkpoint = _float32_checkpoint()

    with caplog.at_level(
        logging.WARNING, logger="max.experimental.nn._weight_utils"
    ):
        specs = model.graph_specs(model._create_model_config(model._checkpoint))

    assert (
        f"from {DType.float32} to {DType.bfloat16} (precision loss)"
        not in caplog.text
    )
    assert _decoder_weight(specs, "weight").dtype == DType.float32
