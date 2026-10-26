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
"""The graph API's entry points are marked as boundaries for source tracebacks."""

import inspect
import sys
import types

import pytest
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import BufferValue, Graph, TensorValue, ops
from max.graph._api_boundary import (
    api_boundary,
    callers_of_api,
    is_api_boundary,
)


def _public_functions(namespace: dict[str, object]) -> list[str]:
    return [
        name
        for name, value in namespace.items()
        if not name.startswith("_") and isinstance(value, types.FunctionType)
    ]


# `ops` and every submodule it holds, so a new submodule is checked too.
_OP_MODULES = [ops, F] + [
    value
    for value in vars(ops).values()
    if isinstance(value, types.ModuleType)
    and value.__name__.startswith("max.graph.ops.")
]


@pytest.mark.parametrize(
    "module", _OP_MODULES, ids=[m.__name__ for m in _OP_MODULES]
)
def test_every_op_function_is_a_boundary(module: types.ModuleType) -> None:
    missing = [
        name
        for name in _public_functions(vars(module))
        if getattr(vars(module)[name], "__module__", "").startswith("max.")
        and not is_api_boundary(vars(module)[name])
    ]
    assert not missing, f"not marked as API boundaries: {missing}"


@pytest.mark.parametrize("cls", [TensorValue, BufferValue, Tensor])
def test_value_operators_are_boundaries(cls: type) -> None:
    for name in ("__add__", "__mul__", "__getitem__", "reshape"):
        if hasattr(cls, name):
            assert is_api_boundary(inspect.getattr_static(cls, name)), name


def test_graph_output_is_a_boundary() -> None:
    assert is_api_boundary(Graph.output)
    assert is_api_boundary(Graph.add_weight)


@api_boundary
def _fake_op() -> list[types.FrameType] | None:
    return callers_of_api(sys._getframe())


@api_boundary
def _fake_composite_op() -> list[types.FrameType] | None:
    # A boundary that calls another one: the cut is at the outer call.
    return _fake_op()


def _not_an_op() -> list[types.FrameType] | None:
    return callers_of_api(sys._getframe())


def test_callers_end_at_the_call_into_the_api() -> None:
    callers = _fake_op()
    assert callers is not None
    assert (
        callers[-1].f_code is test_callers_end_at_the_call_into_the_api.__code__
    )


def test_nested_boundaries_cut_at_the_outermost() -> None:
    callers = _fake_composite_op()
    assert callers is not None
    assert (
        callers[-1].f_code
        is test_nested_boundaries_cut_at_the_outermost.__code__
    )


def test_no_boundary_means_no_cut() -> None:
    assert callers_of_api(sys._getframe()) is None
    assert _not_an_op() is None
