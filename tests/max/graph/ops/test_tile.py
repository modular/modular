# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.outer tests."""

import pytest
from conftest import shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, TensorType, ops

shared_shapes = st.shared(shapes().filter(lambda shape: 0 not in shape))
tensor_types_nd = tensor_types(shapes=shared_shapes)


def valid_repeat_counts(dim):
    if dim.is_symbolic():
        return st.just(1)
    else:
        return st.integers(min_value=1, max_value=(2**63 - 1) // dim.dim)


valid_repeats = shared_shapes.flatmap(
    lambda shape: st.tuples(*(valid_repeat_counts(dim) for dim in shape))
).map(list)


@given(input_type=tensor_types_nd, repeats=valid_repeats)
def test_tile__valid(input_type: TensorType, repeats: list[int]):
    with Graph("tiles", input_types=[input_type]) as graph:
        out = ops.tile(graph.inputs[0], repeats)
        expected_shape = [
            dim if r == 1 else dim.dim * r
            for r, dim in zip(repeats, input_type.shape)
        ]
        assert out.shape == expected_shape
        graph.output(out)


def invalid_symbolic_repeat_counts(dim):
    if dim.is_symbolic():
        return st.integers().filter(lambda x: x != 1)
    else:
        return st.integers(min_value=1, max_value=(2**63 - 1) // dim.dim)


invalid_symbolic_repeats = (
    shared_shapes.filter(lambda shape: any(dim.is_symbolic() for dim in shape))
    .flatmap(
        lambda shape: st.tuples(
            *(invalid_symbolic_repeat_counts(dim) for dim in shape)
        )
    )
    .map(list)
)

invalid_static_repeats = shared_shapes.flatmap(
    lambda shape: st.lists(
        st.integers(min_value=-(2**63), max_value=0),
        min_size=len(shape),
        max_size=len(shape),
    )
)

invalid_len = shared_shapes.flatmap(
    lambda shape: st.lists(st.just(1)).filter(lambda l: len(l) != len(shape))
)

invalid_repeats = st.one_of(
    invalid_symbolic_repeats, invalid_static_repeats, invalid_len
)


@given(input_type=tensor_types_nd, repeats=invalid_repeats)
def test_tile__invalid(input_type: TensorType, repeats: list[int]):
    assume(len(input_type.shape) != 0)
    with Graph("tiles", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            out = ops.tile(graph.inputs[0], repeats)
