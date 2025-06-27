# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for ops.band_part."""

import pytest
from conftest import shapes, static_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Dim, StaticDim, TensorType, ops


@given(base_type=tensor_types(shapes=shapes(min_rank=2)))
def test_band_part__trivial(graph_builder, base_type: TensorType) -> None:
    """Test band_part with no lower or upper bounds, corresponding to the full matrix."""
    with graph_builder(input_types=[base_type]) as graph:
        out = ops.band_part(graph.inputs[0])
        assert out.type == base_type
        graph.output(out)


@given(base_type=tensor_types(shapes=shapes(min_rank=2)))
def test_band_part__main_diagonal(graph_builder, base_type: TensorType) -> None:
    """Test band_part with only the main diagonal (num_lower=0, num_upper=0)."""
    *_, m, n = base_type.shape
    assume(m != 0 and n != 0)
    with graph_builder(input_types=[base_type]) as graph:
        out = ops.band_part(graph.inputs[0], num_lower=0, num_upper=0)
        assert out.type == base_type
        graph.output(out)


@given(base_type=tensor_types(shapes=shapes(min_rank=2)))
def test_band_part__lower_triangle(
    graph_builder, base_type: TensorType
) -> None:
    """Test band_part with lower triangle (num_lower=None, num_upper=0)."""
    *_, m, n = base_type.shape
    assume(n != 0)
    with graph_builder(input_types=[base_type]) as graph:
        out = ops.band_part(graph.inputs[0], num_upper=0)
        assert out.type == base_type
        graph.output(out)


@given(base_type=tensor_types(shapes=shapes(min_rank=2)))
def test_band_part__upper_triangle(
    graph_builder, base_type: TensorType
) -> None:
    """Test band_part with upper triangle (num_lower=0, num_upper=None)."""
    *_, m, n = base_type.shape
    assume(m != 0)
    with graph_builder(input_types=[base_type]) as graph:
        out = ops.band_part(graph.inputs[0], num_lower=0)
        assert out.type == base_type
        graph.output(out)


shared_tensor_types = st.shared(tensor_types(shapes=shapes(min_rank=2)))


def upper_bound(d: Dim) -> int:
    if isinstance(d, StaticDim):
        return int(d)
    return 2**63 - 1


@given(
    base_type=shared_tensor_types,
    # We allow -1 as a historical alternative for None
    num_lower=shared_tensor_types.flatmap(
        lambda t: st.integers(
            min_value=-1, max_value=upper_bound(t.shape[-2]) - 1
        )
    ),
    # We allow -1 as a historical alternative for None
    num_upper=shared_tensor_types.flatmap(
        lambda t: st.integers(
            min_value=-1, max_value=upper_bound(t.shape[-1]) - 1
        )
    ),
    exclude=...,
)
def test_band_part__general(
    graph_builder,
    base_type: TensorType,
    num_lower: int,
    num_upper: int,
    exclude: bool,
) -> None:
    """Test band_part with valid inputs."""
    assert base_type.rank >= 2
    *_, m, n = base_type.shape
    assert not isinstance(m, StaticDim) or num_lower < int(m)
    assert not isinstance(n, StaticDim) or num_upper < int(n)

    with graph_builder(input_types=[base_type]) as graph:
        out = ops.band_part(
            graph.inputs[0],
            num_lower=num_lower,
            num_upper=num_upper,
            exclude=exclude,
        )
        assert out.type == base_type
        graph.output(out)


@given(base_type=tensor_types(shapes=shapes(max_rank=1)))
def test_band_part__error__low_rank(
    graph_builder, base_type: TensorType
) -> None:
    """Test that band_part raises an error for tensors with rank < 2."""
    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(ValueError, match="must have at least 2 dimensions"):
            ops.band_part(graph.inputs[0])


@given(
    base_type=tensor_types(shapes=shapes(min_rank=2)),
    # we allow -1 as a historical alternative for None
    num_lower=st.integers(max_value=-2),
)
def test_band_part__error__negative_num_lower(
    graph_builder, base_type: TensorType, num_lower: int
) -> None:
    """Test that band_part raises an error for num_lower < -1."""
    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(ValueError, match="must be non-negative"):
            ops.band_part(graph.inputs[0], num_lower=num_lower)


@given(
    base_type=tensor_types(shapes=shapes(min_rank=2)),
    # we allow -1 as a historical alternative for None
    num_upper=st.integers(min_value=-10, max_value=-2),
)
def test_band_part__error__negative_num_upper(
    graph_builder, base_type: TensorType, num_upper: int
) -> None:
    """Test that band_part raises an error for num_upper < -1."""
    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(ValueError, match="must be non-negative"):
            ops.band_part(graph.inputs[0], num_upper=num_upper)


shared_static_dim = st.shared(static_dims())


@given(
    base_type=tensor_types(shapes=shapes(min_rank=1)),
    static_dim=shared_static_dim,
    num_lower=shared_static_dim.flatmap(
        lambda d: st.integers(min_value=int(d))
    ),
)
def test_band_part__error__out_of_bounds_num_lower(
    graph_builder, base_type: TensorType, static_dim: StaticDim, num_lower: int
) -> None:
    """Test that band_part raises an error when num_lower is out of bounds."""
    *broadcast, n = base_type.shape
    input_type = TensorType(
        base_type.dtype, [*broadcast, static_dim, n], base_type.device
    )
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="is out of bounds"):
            ops.band_part(graph.inputs[0], num_lower=num_lower)


@given(
    base_type=tensor_types(shapes=shapes(min_rank=1)),
    static_dim=shared_static_dim,
    num_upper=shared_static_dim.flatmap(
        lambda d: st.integers(min_value=int(d))
    ),
)
def test_band_part__error__out_of_bounds_num_upper(
    graph_builder, base_type: TensorType, static_dim: StaticDim, num_upper: int
) -> None:
    """Test that band_part raises an error when num_upper is out of bounds."""
    *broadcast, m = base_type.shape
    input_type = TensorType(
        base_type.dtype, [*broadcast, m, static_dim], base_type.device
    )
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="is out of bounds"):
            ops.band_part(graph.inputs[0], num_upper=num_upper)
