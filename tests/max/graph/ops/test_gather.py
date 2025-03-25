# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.gather tests."""

import pytest
from conftest import axes, dims, shapes, static_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, StaticDim, TensorType, ops

# gather not meaningful for scalar inputs
input_types = tensor_types(shapes=st.lists(dims, min_size=1))


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(dtypes=st.just(DType.int64)),
    axis=axes(st.shared(input_types, key="input")),
)
def test_gather(input_type: TensorType, indices_type: TensorType, axis: int):
    assume(indices_type.rank > 0)
    with Graph("gather", input_types=[input_type, indices_type]) as graph:
        input, indices = graph.inputs
        out = ops.gather(input, indices, axis)
        target_shape = [
            *input.shape[:axis],
            *indices.shape,
            *input.shape[axis + 1 :],
        ]
        assert out.type == TensorType(input.dtype, target_shape)
        graph.output(out)


@given(input=..., indices=tensor_types(dtypes=st.just(DType.uint64)))
def test_gather_nd(input: TensorType, indices: TensorType):
    assume(isinstance(indices.shape[-1], StaticDim))
    index_size = int(indices.shape[-1])
    assume(index_size <= input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        out = ops.gather_nd(*graph.inputs)
        assert out.dtype == input.dtype
        assert out.shape == [*indices.shape[:-1], *input.shape[index_size:]]


n_batch_dims = st.shared(st.integers(min_value=0, max_value=10))
batch_dims = st.shared(
    n_batch_dims.flatmap(lambda n: shapes(min_rank=n, max_rank=n))
)
input_types = st.shared(
    batch_dims.flatmap(
        lambda dims: tensor_types().map(
            lambda t: TensorType(t.dtype, dims + t.shape)
        )
    )
)
indices_types = st.shared(
    batch_dims.flatmap(
        lambda dims: static_dims(max=10).flatmap(
            lambda index_size: tensor_types(dtypes=st.just(DType.uint64)).map(
                lambda t: TensorType(t.dtype, [*dims, *t.shape, index_size])
            )
        )
    )
)


@given(input=input_types, indices=indices_types, batch_dims=n_batch_dims)
@pytest.mark.skip("MAXPLAT-152")
def test_gather_nd__batch_dims(
    input: TensorType, indices: TensorType, batch_dims: int
):
    assert isinstance(indices.shape[-1], StaticDim)
    index_size = int(indices.shape[-1])
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)
    assert input.shape[:batch_dims] == indices.shape[:batch_dims]
    assume(batch_dims + index_size <= input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        out = ops.gather_nd(*graph.inputs, batch_dims=batch_dims)
        assert out.dtype == input.dtype
        assert out.shape == [
            *input.shape[:batch_dims],
            *indices.shape[batch_dims:-1],
            *input.shape[batch_dims + index_size :],
        ]


@given(input=input_types, indices=indices_types, batch_dims=...)
@pytest.mark.skip("MAXPLAT-152")
def test_gather_nd__invalid_batch_dims(
    input: TensorType, indices: TensorType, batch_dims: int
):
    assert isinstance(indices.shape[-1], StaticDim)
    index_size = int(indices.shape[-1])

    # valid so far, now assume bad batch dim condition
    assume(batch_dims < 0 or index_size + batch_dims > input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)


# TODO: what happens if batch_dims + 1 > indices.rank?


@given(
    input=input_types.flatmap(
        lambda t: tensor_types(shapes=shapes(min_rank=t.rank, max_rank=t.rank))
    ),
    indices=indices_types,
    batch_dims=n_batch_dims,
)
@pytest.mark.skip("MAXPLAT-152")
def test_gather_nd__mismatching_batch_dims(
    input: TensorType, indices: TensorType, batch_dims: int
):
    assert isinstance(indices.shape[-1], StaticDim)
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)
    index_size = int(indices.shape[-1])
    assume(index_size + batch_dims <= input.rank)

    # valid so far, now assume mismatching batch dims condition
    assume(input.shape[:batch_dims] != indices.shape[:batch_dims])

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)


@given(input=input_types, indices=..., batch_dims=n_batch_dims)
@pytest.mark.skip("MAXPLAT-152")
def test_gather_nd__symbolic_index(
    input: TensorType, indices: TensorType, batch_dims: int
):
    # error condition
    assume(not isinstance(indices.shape[-1], StaticDim))

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)


@given(input=input_types, indices=indices_types, batch_dims=n_batch_dims)
@pytest.mark.skip("MAXPLAT-152")
def test_gather_nd__index_too_long(
    input: TensorType, indices: TensorType, batch_dims: int
):
    assert isinstance(indices.shape[-1], StaticDim)
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)

    index_size = int(indices.shape[-1])
    # error condition
    assume(index_size + batch_dims > input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)


@given(
    input=input_types,
    indices=indices_types,
    batch_dims=n_batch_dims,
    dtype=st.sampled_from([dtype for dtype in DType if dtype.is_float()]),
)
@pytest.mark.skip("MAXPLAT-152")
def test_gather_nd__non_int_indices(
    input: TensorType,
    indices: TensorType,
    batch_dims: int,
    dtype: DType,
):
    assert isinstance(indices.shape[-1], StaticDim)
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)
    index_size = int(indices.shape[-1])
    assume(index_size + batch_dims <= input.rank)

    # error condition
    assert dtype.is_float()
    indices.dtype = dtype

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)
