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

"""End-to-end values for loops `map-flatten-shapes` merges the dims of.

The pass rewrites a loop to run at a lower rank over the same memory, reading
and writing through collapse views. Whether it fired, and which dims it
merged, is covered by the MLIR suite under
`MAPDialect/Transforms/FlattenShapes`; these are the e2e checks that a
flattened loop still computes the same values, with numpy as the oracle.

Each graph is a shape the pass treats differently: dims already contiguous, a
bias broadcast over the outer dims, a window cut on a middle dim, a transpose
that keeps one pair adjacent, a concatenation on the outer dim, a window taken
from a loop's own result, and a symbolic shape whose merged extent becomes a
graph parameter.

Every graph is built into one `Module` at import and compiled by a single
`load_all`, since the compile dominates the cost of graphs this small.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_utils import execute
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Module, TensorType, ops

_MODULE = Module()


def _cpu(shape: list[int | str]) -> TensorType:
    """A float32 CPU tensor; a string dim is symbolic."""
    return TensorType(DType.float32, shape, device=DeviceRef.CPU())


def _graph(name: str, *shapes: list[int | str]) -> Graph:
    return Graph(name, input_types=[_cpu(s) for s in shapes], module=_MODULE)


@pytest.fixture(scope="module")
def models(session: InferenceSession) -> dict[str, Model]:
    return session.load_all(_MODULE)


with _graph("contiguous", [2, 3]) as contiguous:
    (x,) = (v.tensor for v in contiguous.inputs)
    contiguous.output(x + x)


def test_contiguous(models: dict[str, Model]) -> None:
    """Every pair of dims is one run, so the whole loop merges to rank 1."""
    a = np.random.randn(2, 3).astype(np.float32)
    (out,) = execute(models[contiguous.name], a)
    np.testing.assert_allclose(out, a + a, rtol=1e-5, atol=1e-5)


with _graph("dynamic", ["T", 4], ["T", 4]) as dynamic:
    first, second = (v.tensor for v in dynamic.inputs)
    dynamic.output(first + second)


def test_dynamic(models: dict[str, Model]) -> None:
    """A symbolic shape merges too: the extent `T * 4` becomes a parameter the
    kernel receives as a dim."""
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    (out,) = execute(models[dynamic.name], a, b)
    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


with _graph("bias", [2, 2, 3], [3]) as bias:
    x, b = (v.tensor for v in bias.inputs)
    bias.output(x + ops.broadcast_to(b, [2, 2, 3]))


def test_bias(models: dict[str, Model]) -> None:
    """A bias broadcast over the outer dims keeps those dims mergeable."""
    a = np.random.randn(2, 2, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    (out,) = execute(models[bias.name], a, b)
    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


with _graph("window", [2, 4, 3]) as window:
    (x,) = (v.tensor for v in window.inputs)
    window.output(ops.abs(x[:, 1:3, :]))


def test_window(models: dict[str, Model]) -> None:
    """A window cut on a middle dim leaves the rows apart from each other."""
    a = np.random.randn(2, 4, 3).astype(np.float32)
    (out,) = execute(models[window.name], a)
    np.testing.assert_allclose(out, np.abs(a[:, 1:3, :]), rtol=1e-5, atol=1e-5)


with _graph("transpose", [2, 2, 3]) as transpose:
    (x,) = (v.tensor for v in transpose.inputs)
    transpose.output(ops.abs(ops.permute(x, [1, 2, 0])))


def test_transpose(models: dict[str, Model]) -> None:
    """Out dims 0 and 1 come from in dims 1 and 2, which stay contiguous."""
    a = np.random.randn(2, 2, 3).astype(np.float32)
    (out,) = execute(models[transpose.name], a)
    np.testing.assert_allclose(
        out, np.abs(np.transpose(a, [1, 2, 0])), rtol=1e-5, atol=1e-5
    )


with _graph("concat_outer", [3, 16], [5, 16]) as concat_outer:
    first, second = (v.tensor for v in concat_outer.inputs)
    concat_outer.output(ops.concat([first, second], axis=0))


def test_concat_outer(models: dict[str, Model]) -> None:
    """A concatenation on the outer dim keeps each piece's rows one run."""
    a = np.random.randn(3, 16).astype(np.float32)
    b = np.random.randn(5, 16).astype(np.float32)
    (out,) = execute(models[concat_outer.name], a, b)
    np.testing.assert_allclose(
        out, np.concatenate([a, b], axis=0), rtol=1e-5, atol=1e-5
    )


with _graph("window_select", [2, 4, 8, 16]) as window_select:
    (x,) = (v.tensor for v in window_select.inputs)
    doubled = x + x
    window_select.output(doubled, ops.abs(doubled[:, 1:3, :, :]))


def test_window_select(models: dict[str, Model]) -> None:
    """A window taken from a loop's own result, which the loop also outputs."""
    a = np.random.randn(2, 4, 8, 16).astype(np.float32)
    whole, cut = execute(models[window_select.name], a)
    np.testing.assert_allclose(whole, a + a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        cut, np.abs((a + a)[:, 1:3, :, :]), rtol=1e-5, atol=1e-5
    )
