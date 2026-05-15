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
"""Tests for the InferenceSession compile/init API split.

These tests exercise the public ``compile()`` / ``init()`` / ``init_all()``
methods that expose the two halves of ``load()`` as separate steps. They
also confirm ``load()`` / ``load_all()`` still behave the same way after
the internal refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max.driver import CPU
from max.dtype import DType
from max.engine import CompiledModel, InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


@dataclass
class Unity:
    """Graph body that returns a scalar 1.0 (input is unused)."""

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.constant(1.0, dtype=DType.float32, device=DeviceRef.CPU())


def _unity_graph(name: str = "unity") -> Graph:
    return Graph(
        name,
        forward=Unity(),
        input_types=[
            TensorType(DType.float32, ["batch", "dim"], device=DeviceRef.CPU())
        ],
    )


def test_compile_returns_compiled_model() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    assert isinstance(compiled, CompiledModel)
    # A CompiledModel is not executable: it has no `execute` attribute.
    assert not hasattr(compiled, "execute")


def test_compiled_model_repr() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    assert repr(compiled) == "CompiledModel()"


def test_init_produces_runnable_model() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())
    model = session.init(compiled)

    assert isinstance(model, Model)
    output = model.execute(np.zeros((1, 4), dtype=np.float32))
    assert len(output) == 1
    assert output[0].to_numpy().item() == pytest.approx(1.0)


def test_init_all_returns_single_model() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())
    models = session.init_all(compiled)

    assert len(models) == 1
    assert isinstance(next(iter(models.values())), Model)


def test_compile_init_matches_load() -> None:
    """compile()+init() should produce the same output as load()."""
    session = InferenceSession(devices=[CPU()])
    x = np.zeros((1, 4), dtype=np.float32)

    via_load = session.load(_unity_graph()).execute(x)
    compiled = session.compile(_unity_graph())
    via_compile = session.init(compiled).execute(x)

    assert via_load[0].to_numpy().item() == via_compile[0].to_numpy().item()


def test_compiled_model_can_be_initialized_twice() -> None:
    """A single compiled artifact can produce multiple independent Models."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    model_a = session.init(compiled)
    model_b = session.init(compiled)

    assert isinstance(model_a, Model)
    assert isinstance(model_b, Model)
    assert model_a is not model_b

    x = np.zeros((1, 4), dtype=np.float32)
    assert (
        model_a.execute(x)[0].to_numpy().item()
        == model_b.execute(x)[0].to_numpy().item()
    )


def test_init_rejects_non_contiguous_weights() -> None:
    """init() applies the same contiguity check that load() does today."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    weight_tensor = torch.randn(4, 4).t()
    assert not weight_tensor.is_contiguous()

    with pytest.raises(ValueError, match="non-contiguous tensors"):
        session.init(
            compiled, weights_registry={"extra": weight_tensor.numpy()}
        )


def test_load_after_refactor_still_works() -> None:
    """Regression: load() still returns a single executable Model."""
    session = InferenceSession(devices=[CPU()])
    model = session.load(_unity_graph())

    assert isinstance(model, Model)
    output = model.execute(np.zeros((1, 4), dtype=np.float32))
    assert output[0].to_numpy().item() == pytest.approx(1.0)


def test_load_all_after_refactor_still_works() -> None:
    """Regression: load_all() still returns a dict of executable models."""
    session = InferenceSession(devices=[CPU()])
    models = session.load_all(_unity_graph())

    assert len(models) == 1
    assert isinstance(next(iter(models.values())), Model)
