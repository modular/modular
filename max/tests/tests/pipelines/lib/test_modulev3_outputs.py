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
"""Pins how a ModuleV3 KV-cache model maps its graph outputs by name."""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental.compilation import compile
from max.experimental.sharding import TensorLayout
from max.experimental.tensor import Tensor
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    ModuleV3Outputs,
    ModuleV3PipelineModelWithKVCache,
)


class _Model(ModuleV3PipelineModelWithKVCache[Any]):
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


def _map(outputs_fn: Any) -> ModelOutputs:
    run = compile(outputs_fn)(TensorLayout(DType.float32, ["n", 2], CPU()))
    x = Tensor.from_dlpack(np.arange(6, dtype=np.float32).reshape(3, 2))
    return object.__new__(_Model)._to_model_outputs(run(x))


def _numpy(buffer: Buffer | None) -> np.ndarray:
    assert buffer is not None
    return buffer.to_numpy()


def test_next_token_logits_also_serve_as_logits() -> None:
    outputs = _map(lambda x: ModuleV3Outputs(next_token_logits=x + 1))

    np.testing.assert_array_equal(
        _numpy(outputs.next_token_logits), np.arange(1, 7).reshape(3, 2)
    )
    assert outputs.logits is outputs.next_token_logits
    assert outputs.logit_offsets is None
    assert outputs.hidden_states is None


def test_each_field_maps_by_name() -> None:
    def forward(x: Tensor) -> ModuleV3Outputs:
        return ModuleV3Outputs(
            next_token_logits=x + 1,
            logits=x + 2,
            logit_offsets=x + 3,
            hidden_states=x + 4,
        )

    outputs = _map(forward)

    base = np.arange(6).reshape(3, 2)
    np.testing.assert_array_equal(_numpy(outputs.next_token_logits), base + 1)
    np.testing.assert_array_equal(_numpy(outputs.logits), base + 2)
    np.testing.assert_array_equal(_numpy(outputs.logit_offsets), base + 3)
    np.testing.assert_array_equal(_numpy(outputs.hidden_states), base + 4)


def test_hidden_states_without_all_logits() -> None:
    outputs = _map(
        lambda x: ModuleV3Outputs(next_token_logits=x, hidden_states=x * 2)
    )

    assert outputs.logits is outputs.next_token_logits
    assert outputs.logit_offsets is None
    np.testing.assert_array_equal(
        _numpy(outputs.hidden_states), np.arange(0, 12, 2).reshape(3, 2)
    )
