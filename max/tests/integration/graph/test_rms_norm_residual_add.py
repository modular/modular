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
"""Exercises fused residual-add RMSNorm through graph dispatch across ranks."""

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

_SHAPES = ((4096,), (1, 2048, 128))
_EPSILON = 1e-6


@pytest.fixture(scope="module")
def models() -> dict[tuple[int, ...], Model]:
    session = InferenceSession(devices=[Accelerator(0)])
    gpu = DeviceRef.GPU(0)
    compiled: dict[tuple[int, ...], Model] = {}
    for shape in _SHAPES:
        with Graph(
            "rms_norm_residual_add_rank_" + str(len(shape)),
            input_types=[
                TensorType(DType.float32, shape, device=gpu),
                TensorType(DType.float32, shape, device=gpu),
                TensorType(DType.float32, (shape[-1],), device=gpu),
            ],
        ) as graph:
            x, residual, gamma = (value.tensor for value in graph.inputs)
            summed = x + residual
            normalized = ops.rms_norm(summed, gamma, epsilon=_EPSILON)
            graph.output(normalized, summed)
        model = session.load(graph)
        assert any(
            "mo.composite.rms_norm_residual_add" in summary
            for summary in model.kernel_summaries
        ), model.kernel_summaries
        compiled[shape] = model
    return compiled


@pytest.mark.parametrize("shape", _SHAPES)
def test_rms_norm_residual_add_dispatch(
    models: dict[tuple[int, ...], Model], shape: tuple[int, ...]
) -> None:
    model = models[shape]
    rng = np.random.default_rng(5)
    x = rng.standard_normal(shape).astype(np.float32)
    residual = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, shape[-1]).astype(np.float32)
    inputs = [
        Buffer.from_numpy(value).to(model.input_devices[i])
        for i, value in enumerate((x, residual, gamma))
    ]
    normalized, summed = model.execute(*inputs)
    expected_sum = x + residual
    expected = (
        expected_sum
        / np.sqrt(
            np.mean(expected_sum * expected_sum, axis=-1, keepdims=True)
            + _EPSILON
        )
        * gamma
    )
    np.testing.assert_array_equal(summed.to_numpy(), expected_sum)
    np.testing.assert_allclose(
        normalized.to_numpy(), expected, rtol=1e-4, atol=1e-5, equal_nan=False
    )
