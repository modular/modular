# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from max.driver import Tensor
from max.dtype import DType
from max.graph import (
    Dim,
    StaticDim,
    SymbolicDim,
    TensorType,
)
from test_common.graph_utils import (
    are_all_tensor_values_iterable as _are_all_tensor_values_iterable,
)

MAX_INPUT_MAGNITUDE = 1e5
MIN_SHAPE_DIM = 1
MAX_SHAPE_DIM = 16

ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-8


# TODO: Swap all imports of this to use test_common.graph_utils directly.
are_all_tensor_values = _are_all_tensor_values_iterable


def elements(dtype, max_magnitude=MAX_INPUT_MAGNITUDE, **kwargs):
    if max_magnitude:
        kwargs.setdefault("max_value", max_magnitude)

    if "min_value" not in kwargs:
        if np.issubdtype(dtype, np.integer):
            kwargs.setdefault("min_value", 0)
        else:
            kwargs.setdefault("min_value", -max_magnitude)

    return nps.from_dtype(
        dtype,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        **kwargs,
    )


@st.composite
def shapes(
    draw: st.DrawFn,
    tensor_type: TensorType,
    static_dims: Mapping[str, int] = {},
):
    """Defines a strategy for generating a concrete shape from a TensorType.

    Args:
        draw: Used with @st.composite to write type hints.
        tensor_type: Input tensor type.
        static_dims: Map of symbolic dimension name to static dimensions. This
            is used when there are dimensions in the test input arrays that must
            be hardcoded. For example, if the input has dims ["a", "b", "c"],
            and "c" must be "a" + "b".

    Returns:
        Shape tuple
    """

    def draw_dim(dim: Dim):
        if isinstance(dim, SymbolicDim):
            if (static := static_dims.get(dim.name)) is not None:
                return static

            sides = st.integers(
                min_value=MIN_SHAPE_DIM, max_value=MAX_SHAPE_DIM
            )
            # shared ensures symbolic dimensions match
            return draw(st.shared(sides, key=dim.name))
        elif isinstance(dim, StaticDim):
            return int(dim)
        raise NotImplementedError

    return tuple(draw_dim(dim) for dim in tensor_type.shape)


def arrays(tensor_type: TensorType, static_dims={}, **kwargs):
    if tensor_type.dtype == DType.bfloat16:
        tensor_type = TensorType(DType.float32, tensor_type.shape)
        return arrays(tensor_type, static_dims=static_dims, **kwargs).map(
            lambda t: Tensor.from_dlpack(
                torch.from_dlpack(t).to(torch.bfloat16)
            )
        )

    dtype = tensor_type.dtype.to_numpy()
    return nps.arrays(
        dtype,
        shapes(tensor_type, static_dims),
        elements=elements(dtype, **kwargs),
    ).map(Tensor.from_dlpack)


def given_input_types(
    input_types,
    static_dims: Mapping[str, int] = {},
    provided_inputs: Mapping[int, np.ndarray | Tensor] = {},
    max_magnitude: Optional[float] = None,
    **kwargs,
):
    input_arrays = []
    for i, input_type in enumerate(input_types):
        if i in provided_inputs:
            input_arrays.append(st.just(provided_inputs[i]))
        elif max_magnitude is not None:
            input_arrays.append(
                arrays(
                    input_type,
                    static_dims=static_dims,
                    max_magnitude=max_magnitude,
                    **kwargs,
                )
            )
        else:
            input_arrays.append(
                arrays(input_type, static_dims=static_dims, **kwargs)
            )

    return given(st.tuples(*input_arrays))


def execute(model, inputs):
    tensor_inputs = [
        t if isinstance(t, Tensor) else Tensor.from_dlpack(t) for t in inputs
    ]
    results = model.execute(*tensor_inputs)
    if results:
        return results[0]


def modular_graph_test(
    session,
    graph,
    *,
    static_dims: Mapping[str, int] = {},
    provided_inputs: Mapping[int, np.ndarray | Tensor] = {},
    hypothesis_settings: Optional[settings] = None,
    max_magnitude: Optional[float] = None,
    **kwargs,
):
    def decorator(test_fn):
        model = session.load(graph)

        @settings(suppress_health_check=[HealthCheck.data_too_large])
        @given_input_types(
            (input.type for input in graph.inputs),
            static_dims=static_dims,
            provided_inputs=provided_inputs,
            max_magnitude=max_magnitude,
            **kwargs,
        )
        def test_correctness(inputs):
            model_execute = functools.partial(execute, model)

            torch_inputs = [torch.from_dlpack(t) for t in inputs]
            test_fn(model_execute, inputs, torch_inputs)

        if hypothesis_settings is not None:
            test_correctness = hypothesis_settings(test_correctness)
        test_correctness()
        return test_correctness

    return decorator
